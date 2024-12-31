"""
Base infrastructure for training, supporting multi-node and multi-gpu training
"""

import os
import sys
import logging
import warnings

from colorama import Fore, Style

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
# torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

Project_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Project_DIR))

from trainer_utils import *
from model.model_utils import save_model, load_model
from utils.status import model_info, start_timer, end_timer, support_bfloat16
from setup.setup_utils import setup_logger

class TrainManager(object):
    """
    Base Runtime model for training. This class supports:
        - single node, single process, single gpu training
        - single node, multiple process, multiple gpu training
        - multiple nodes, multiple processes, multiple gpu training
    """
    def __init__(self, 
                 config,
                 datasets,
                 model, 
                 loss_func,
                 optim_manager, 
                 metric_manager):
    
        """
        @args:
            - config (Namespace): runtime namespace for setup
            - datasets (List[torch Dataset]): list of datasets for train, val, and test sets
            - model (nn.Module): object that contains the model with a forward function
            - loss_func (function): loss function to be used for training
            - optim_manager (OptimManager): OptimManager object that contains optimizer and scheduler
            - metric_manager (MetricManager): MetricManager object that tracks metrics and checkpoints models during training
        """
        super().__init__()

        assert len(datasets) == 3, "Need to provide train, val, and test datasets"

        self.config = config
        self.train_set = datasets[0]
        self.val_set = datasets[1]
        self.test_set = datasets[2]
        self.model = model
        self.loss_func = loss_func
        self.optim_manager = optim_manager
        self.metric_manager = metric_manager

        if self.config.use_amp:
            if support_bfloat16(self.config.device):
                self.cast_type = torch.bfloat16
            else:
                self.cast_type = torch.float16
        else:
            self.cast_type = torch.float32

    def _train_and_eval_model(self, rank, global_rank):
        """
        @args:
            - rank (int): for distributed data parallel (ddp) -1 if running on cpu or only one gpu
            - global_rank (int): for distributed data parallel (ddp)
        """
        c = self.config # shortening due to numerous uses     

        # All metrics are handled by the metric manager
        self.metric_manager.setup_wandb_and_metrics(rank)

        # Send models to device
        if c.ddp:
            dist.barrier()
            device = torch.device(f"cuda:{rank}")
            self.model = self.model.to(device)
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=False)
        else:
            device = c.device
            self.model = self.model.to(device)

        # Print out model summary
        if rank<=0:
            logging.info(f"Configuration for this run:\n{c}")
            model_info(self.model, c)
            logging.info(f"Wandb name:\n{self.metric_manager.wandb_run.name}")
            self.metric_manager.wandb_run.watch(self.model)
            if c.ddp: 
                setup_logger(self.config) # setup master process logging; I don't know if this needs to be here, it is also in setup.py
        
        # Extracting optim and sched for convenience
        optim = self.optim_manager.optim
        sched = self.optim_manager.sched
        curr_epoch = self.optim_manager.curr_epoch
        scaler = torch.cuda.amp.GradScaler(enabled=c.use_amp)
        logging.info(f"{Fore.RED}{'-'*20}Local Rank:{rank}, global rank {global_rank}{'-'*20}{Style.RESET_ALL}")

        # Zero gradient before training
        optim.zero_grad(set_to_none=True)

        # Training loop
        if self.config.train_model:

            # Create train dataloaders
            if c.ddp:
                shuffle = False
                sampler = DistributedSampler(self.train_set)
            else:
                shuffle = True
                sampler = None
            
            train_loader = DataLoader(dataset=self.train_set, batch_size=c.batch_size, shuffle=shuffle, 
                                        sampler=sampler, num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, 
                                        drop_last=False, persistent_workers=self.config.num_workers>0)
                                
            # Compute total iters
            total_iters = len(train_loader) if not c.debug else 3
            if c.percent_data<1.0: 
                raise ValueError("percent_data is not implemented yet")
                # total_iters = int(total_iters*c.percent_data) # Incorrect, will not work with shuffled loaders or things like one cycle LR scheduler
            
            # Start training
            logging.info(f"{Fore.CYAN}OPTIMIZER PARAMETERS: {optim} {Style.RESET_ALL}")
            for epoch in range(curr_epoch, c.num_epochs):
                logging.info(f"{Fore.GREEN}{'-'*20}Epoch:{epoch}/{c.num_epochs}, rank {rank} {'-'*20}{Style.RESET_ALL}")

                self.model.train()
                if self.config.ddp: train_loader.sampler.set_epoch(epoch)
                self.metric_manager.on_train_epoch_start()
                train_loader_iter = iter(train_loader)

                with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:
                    for idx in range(total_iters):

                        inputs, outputs, ids = next(train_loader_iter, None)
                        inputs = inputs.to(device)
                        outputs = outputs.to(device)

                        # If batch size is 1, duplicate the batch to avoid batch norm issues
                        if len(ids) == 1:
                            inputs = torch.cat([inputs]*2, dim=0)
                            outputs = torch.cat([outputs]*2, dim=0)
                            ids = ids*2

                        with torch.autocast(device_type='cuda', dtype=self.cast_type, enabled=self.config.use_amp):
                            model_outputs = self.model(inputs)
                            loss = self.loss_func(model_outputs,outputs)
                            loss = loss / self.config.iters_to_accumulate

                        scaler.scale(loss).backward()
                        if (idx + 1) % self.config.iters_to_accumulate == 0 or (idx + 1 == total_iters):
                            if(self.config.clip_grad_norm>0):
                                scaler.unscale_(optim)
                                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

                            scaler.step(optim)
                            optim.zero_grad(set_to_none=True)
                            scaler.update()

                            if c.scheduler_type == "OneCycleLR": 
                                sched.step()
                        
                        curr_lr = optim.param_groups[0]['lr']

                        self.metric_manager.on_train_step_end(loss.item(), model_outputs, outputs, rank, curr_lr)

                        pbar.update(1)
                        pbar.set_description(f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} tra, rank {rank}, {inputs.shape}, lr {curr_lr:.8f}, loss {loss.item():.4f}{Style.RESET_ALL}")
                        
                    # Run metric logging for each epoch 
                    self.metric_manager.on_train_epoch_end(self.model, optim, sched, epoch, rank)

                    # Print out metrics from this epoch
                    pbar_str = f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} tra, rank {rank},  {inputs.shape}, lr {curr_lr:.8f}"
                    if hasattr(self.metric_manager, 'average_train_metrics'):
                        if isinstance(self.metric_manager.average_train_metrics, dict):
                            for metric_name, metric_value in self.metric_manager.average_train_metrics.items():
                                try: pbar_str += f", {Fore.CYAN} {metric_name} {metric_value:.4f}"
                                except: pass
                    pbar_str += f"{Style.RESET_ALL}"
                    pbar.set_description(pbar_str)

                    # Write training status to log file
                    if rank<=0: 
                        logging.getLogger("file_only").info(pbar_str)

                if epoch % c.eval_frequency==0 or epoch==c.num_epochs:
                    self._eval_model(rank=rank, model=self.model, epoch=epoch, device=device, optim=optim, sched=sched, id="", split="val", final_eval=False)
                
                if c.scheduler_type not in ["OneCycleLR", "none", "None", None]:
                    if c.scheduler_type == "ReduceLROnPlateau":
                        try: 
                            sched.step(self.metric_manager.average_eval_metrics['total_loss'])
                        except:
                            warnings.warn("Average loss not available, using step loss to step scheduler.")
                            sched.step(loss.item())
                    elif c.scheduler_type == "StepLR":
                        sched.step()

                    # if c.ddp:
                    #     self.distribute_learning_rates(rank, optim, src=0)

            # Load the best model from training
            dist.barrier()
            if self.config.eval_train_set or self.config.eval_val_set or self.config.eval_test_set:
                logging.info(f"{Fore.CYAN}Loading the best models from training for final evaluation...{Style.RESET_ALL}")
                self.model = load_model(self.model, os.path.join(self.config.log_dir,self.config.run_name,'models','model_best_checkpoint.pth'))
       
        else: # Not training
            epoch = 0

        # Evaluate models of each split
        if self.config.eval_train_set: 
            logging.info(f"{Fore.CYAN}Evaluating train set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model=self.model, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="train", final_eval=True)
        if self.config.eval_val_set: 
            logging.info(f"{Fore.CYAN}Evaluating val set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model=self.model, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="val", final_eval=True)
        if self.config.eval_test_set: 
            logging.info(f"{Fore.CYAN}Evaluating test set...{Style.RESET_ALL}")
            self._eval_model(rank=rank, model=self.model, epoch=self.config.num_epochs, device=device, optim=optim, sched=sched, id="", split="test", final_eval=True)

        # Finish up training
        self.metric_manager.on_training_end(rank, epoch, self.model, optim, sched, self.config.train_model)
        
    def _eval_model(self, rank, model, epoch, device, optim, sched, id, split, final_eval):
        """
        Model evaluation.
        @args:
            - rank (int): used for ddp
            - model (nn.Module): model to be validated
            - epoch (int): the current epoch
            - device (torch.device): the device to run eval on
            - optim: optimizer for training
            - sched: scheduler for optimizer
            - id: identifier for ddp runs
            - split: one of {train, val, test}
            - final_eval: whether this is the final evaluation being run at the end of training
        @rets:
            - None; logs and checkpoints within this function
        """
        c = self.config # shortening due to numerous uses
        curr_lr = optim.param_groups[0]['lr']
                
        # Determine if we will save the predictions to files for thie eval 
        if split=='train': save_samples = final_eval and self.config.save_train_samples
        elif split=='val': save_samples = final_eval and self.config.save_val_samples
        elif split=='test': save_samples = final_eval and self.config.save_test_samples
        else: raise ValueError(f"Unknown split {split} specified, should be in [train, val, test]")

        # Set up eval data loaders
        if split=='train': task_dataset = self.train_set
        elif split=='val': task_dataset = self.val_set
        elif split=='test': task_dataset = self.test_set
        if c.ddp:
            sampler = DistributedSampler(task_dataset)
        else:
            sampler = None
        
        eval_loader = DataLoader(dataset=task_dataset, batch_size=c.batch_size, shuffle=False, 
                                    sampler=sampler,num_workers=c.num_workers, prefetch_factor=c.prefetch_factor, 
                                    drop_last=False, persistent_workers=c.num_workers>0)
        data_loader_iters = iter(eval_loader)

        # Set up a few things before starting eval loop
        self.metric_manager.on_eval_epoch_start()
        model.eval()
        total_iters = len(eval_loader) if not c.debug else 3
        
        # Evaluation loop
        with torch.inference_mode():
            with tqdm(total=total_iters, bar_format=get_bar_format()) as pbar:

                for idx in range(total_iters):

                    # Sample data from the dataloaders
                    inputs, labels, ids = next(data_loader_iters, None)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Run inference
                    with torch.autocast(device_type='cuda', dtype=self.cast_type, enabled=c.use_amp):
                        adjusted_batch = False
                        if inputs.shape[0] == 1: 
                            inputs = inputs.repeat(2,1,1,1,1) # Take care of batch size = 1 case so batch norm doesn't throw an error
                            adjusted_batch = True
                        output = model(inputs)
                        if adjusted_batch:
                            output = output[0:1]
                            inputs = inputs[0:1]
                        loss = self.loss_func(output, labels)

                    # Update evaluation metrics
                    self.metric_manager.on_eval_step_end(loss.item(), output, labels, ids, rank, save_samples, split)

                    # Print evaluation metrics to terminal
                    pbar.update(1)
                    pbar.set_description(f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} {split}, rank {rank}, {id} {inputs.shape}, lr {curr_lr:.8f}, loss {loss.item():.4f}{Style.RESET_ALL}")

                # Update evaluation metrics 
                self.metric_manager.on_eval_epoch_end(rank, epoch, model, optim, sched, split, final_eval)

                # Print evaluation metrics to terminal
                pbar_str = f"{Fore.GREEN}Epoch {epoch}/{c.num_epochs},{Style.RESET_ALL} {split}, rank {rank}, {id} {inputs.shape}, lr {curr_lr:.8f}"
                if hasattr(self.metric_manager, 'average_eval_metrics'):
                    if isinstance(self.metric_manager.average_eval_metrics, dict):
                        for metric_name, metric_value in self.metric_manager.average_eval_metrics.items():
                            try: pbar_str += f", {Fore.MAGENTA} {metric_name} {metric_value:.4f}"
                            except: pass

                        # Save final evaluation metrics to a text file
                        if final_eval and rank<=0:
                            metric_file = os.path.join(self.config.log_dir,self.config.run_name,f'{split}_metrics.txt')
                            with open(metric_file, 'w') as f:
                                for metric_name, metric_value in self.metric_manager.average_eval_metrics.items():
                                    try: f.write(f"{split}_{metric_name}: {metric_value:.4f}, ")
                                    except: pass

                pbar_str += f"{Style.RESET_ALL}"
                pbar.set_description(pbar_str)

                if rank<=0: 
                    logging.getLogger("file_only").info(pbar_str)
                        
        return 
       
    def run(self):

        # -------------------------------------------------------
        # Get the rank and runtime info
        if self.config.ddp:
            rank = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

        else:
            rank = -1
            global_rank = -1
            print(f"---> ddp is off <---", flush=True)

        print(f"--------> run training on local rank {rank}", flush=True)

        # -------------------------------------------------------
        # Initialize wandb

        if global_rank<=0:
            self.metric_manager.init_wandb()
            
        # -------------------------------------------------------
        # If ddp is used, broadcast the parameters from rank0 to all other ranks (originally used for sweep, commented out for now)

        if self.config.ddp:

            # if rank<=0:
            #     c_list = [self.config]
            #     print(f"{Fore.RED}--->before, on local rank {rank}, {c_list[0].run_name}{Style.RESET_ALL}", flush=True)
            # else:
            #     c_list = [None]
            #     print(f"{Fore.RED}--->before, on local rank {rank}, {self.config.run_name}{Style.RESET_ALL}", flush=True)

            # if world_size > 1:
            #     torch.distributed.broadcast_object_list(c_list, src=0, group=None, device=rank)

            # print(f"{Fore.RED}--->after, on local rank {rank}, {c_list[0].run_name}{Style.RESET_ALL}", flush=True)
            # if rank>0:
            #     self.config = c_list[0]

            # print(f"---> config synced for the local rank {rank}")
            # if world_size > 1: dist.barrier()

            print(f"{Fore.RED}---> Ready to run on local rank {rank}, {self.config.run_name}{Style.RESET_ALL}", flush=True)

            # self.config.device = torch.device(f'cuda:{rank}')

        # -------------------------------------------------------
        # Run the training and evaluation loops for each rank
        try: 
            self._train_and_eval_model(rank=rank, global_rank=global_rank)
            print(f"{Fore.RED}---> Run finished on local rank {rank} <---{Style.RESET_ALL}", flush=True)

        except KeyboardInterrupt:
            print(f"{Fore.YELLOW}Interrupted from the keyboard ...{Style.RESET_ALL}", flush=True)

            if self.config.ddp:
                torch.distributed.destroy_process_group()

            if self.metric_manager.wandb_run is not None: 
                print(f"{Fore.YELLOW}Remove {self.metric_manager.wandb_run.name} ...{Style.RESET_ALL}", flush=True)

        # -------------------------------------------------------
        # After the run, release the process groups
        if self.config.ddp:
            if dist.is_initialized():
                print(f"---> dist.destory_process_group on local rank {rank}", flush=True)
                dist.destroy_process_group()

