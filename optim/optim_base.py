"""
Set up the optimizer and scheduler manager
"""

import os
import sys
import logging
from abc import ABC
from colorama import Fore, Style

import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

from optim_utils import *

# -------------------------------------------------------------------------------------------------

class OptimManager(object):
    """
    Manages optimizer and scheduler
    """
    
    def __init__(self, config, model, train_set) -> None:
        """
        @args:
            - config (Namespace): nested namespace containing all args
            - model (nn.Module): model we aim to optimize
            - train_set (Dataset): training dataset, needed for computing total updates
        """
        super().__init__()

        # Save vars
        self.config = config
        self.model = model
        
        # Compute total steps and samples (needed for some schedulers)
        self.total_num_samples, self.total_num_updates = compute_total_updates(self.config, train_set)
        logging.info(f"Total number of samples for this run: {self.total_num_samples}")
        logging.info(f"Total number of updates for this run: {self.total_num_updates}")
        
        # Set up optimizer and scheduler
        self.set_up_optim_and_scheduling(total_updates=self.total_num_updates)

        # Load optim and scheduler states, if desired
        if self.config.continued_training and self.config.model_load_path is not None:
            logging.info(f"{Fore.YELLOW}Loading optimizers and schedulers from {self.config.model_load_path}{Style.RESET_ALL}")
            self.load_optim_and_sched(self.config.model_load_path)
        elif self.config.model_load_path is not None:
            logging.info(f"{Fore.YELLOW}No optimizers or schedulers loaded this run{Style.RESET_ALL}")

    def configure_optim_groups(self):
        """
        This function can split up parameters into different parameter groups if needed (e.g., to separate encoder and decoder params)
        """

        optim_groups = []

        optim_groups += [{"params": list(self.model.parameters()), "lr": self.config.optim.lr, "weight_decay": self.config.optim.weight_decay}]

        return optim_groups

    def set_up_optim_and_scheduling(self, total_updates):
        """
        Sets up the optimizer and the learning rate scheduler using the config
        @args:
            - total_updates (int): total number of updates in training (used for OneCycleLR)
        @outputs:
            - self.optim: optimizer
            - self.sched: scheduler
        """
        c = self.config # short config name because of multiple uses

        self.optim = None
        self.sched = None
        self.curr_epoch = 0

        optim_groups = self.configure_optim_groups() 

        if c.optim_type == "adam":
            self.optim = optim.Adam(optim_groups, lr=c.optim.lr, betas=(c.optim.beta1, c.optim.beta2), weight_decay=c.optim.weight_decay)
        elif c.optim_type == "adamw":
            self.optim = optim.AdamW(optim_groups, lr=c.optim.lr, betas=(c.optim.beta1, c.optim.beta2), weight_decay=c.optim.weight_decay)
        elif c.optim_type == "sgd":
            self.optim = optim.SGD(optim_groups, lr=c.optim.lr, momentum=0.9, weight_decay=c.optim.weight_decay)
        elif c.optim_type == "nadam":
            self.optim = optim.NAdam(optim_groups, lr=c.optim.lr, betas=(c.optim.beta1, c.optim.beta2), weight_decay=c.optim.weight_decay)
        elif c.optim_type == "lbfgs":
            self.optim = optim.LBFGS(optim_groups, lr=c.optim.lr, max_iter=c.optim.max_iter, history_size=c.optim.history_size, line_search_fn=c.optim.line_search_fn)
        elif c.optim_type is None:
            self.optim = None
        else:
            raise NotImplementedError(f"Optimizer not implemented: {c.optim_type}")

        if c.scheduler_type == "ReduceLROnPlateau":
            self.sched = optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode="min", factor=c.scheduler.factor,
                                                                    patience=c.scheduler.patience, 
                                                                    cooldown=c.scheduler.cooldown, 
                                                                    min_lr=c.scheduler.min_lr,
                                                                    verbose=True)
        elif c.scheduler_type == "StepLR":
            self.sched = optim.lr_scheduler.StepLR(self.optim, 
                                                   step_size=c.scheduler.step_size, 
                                                   gamma=c.scheduler.gamma, 
                                                   last_epoch=-1,
                                                   verbose=True)
        elif c.scheduler_type == "OneCycleLR":
            self.sched = optim.lr_scheduler.OneCycleLR(self.optim, max_lr=c.optim.lr, total_steps=total_updates,
                                                            pct_start=c.scheduler.pct_start, anneal_strategy="cos", verbose=False)
        elif c.scheduler_type is None:
            self.sched = None

        else:
            raise NotImplementedError(f"Scheduler not implemented: {c.scheduler_type}")
        
    def load_optim_and_sched(self, full_load_path):
        logging.info(f"{Fore.YELLOW}Loading optim and scheduler from {full_load_path}{Style.RESET_ALL}")

        status = torch.load(full_load_path, map_location=self.config.device)
        
        if 'optim_state' in status:
            self.optim.load_state_dict(status['optim_state'])
            optimizer_to(self.optim, device=self.config.device)
            logging.info(f"{Fore.GREEN} Optimizer loading successful {Style.RESET_ALL}")
        else:
            logging.warning(f"{Fore.YELLOW} Optim state is not available in specified load_path {Style.RESET_ALL}")
        
        if 'sched_state' in status:    
            self.sched.load_state_dict(status['sched_state'])
            logging.info(f"{Fore.GREEN} Scheduler loading successful {Style.RESET_ALL}")
        else:
            logging.warning(f"{Fore.YELLOW} Scheduler state is not available in specified load_path {Style.RESET_ALL}")
        
        if 'epoch' in status:
            self.curr_epoch = status['epoch']
            logging.info(f"{Fore.GREEN} Epoch loading successful {Style.RESET_ALL}")
        else:
            logging.warning(f"{Fore.YELLOW} Epoch is not available in specified load_path {Style.RESET_ALL}")           
