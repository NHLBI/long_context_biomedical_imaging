"""
Provides basic save/load utils
"""
import os
import sys
import logging
from colorama import Fore, Style

import torch
import torch.nn as nn
from pathlib import Path

def save_model(config, model, save_dir=None, save_filename=None, epoch=None, optim=None, sched=None):
    """
    Save entire model, including pre/backbone/post components
    @args:
        - config (Namespace): nested namespace containing all args
        - model (nn.Module): model to save
        - save_dir (str): directory where model will be saved
        - save_filename (str): name of the file to save within the directory (do not include file extension)
        - epoch (int): current epoch of the training cycle
        - optim (torch.optim): optimizer to save
        - sched (torch.optim.lr_scheduler): scheduler to save
    """

    if save_dir is None: save_dir = os.path.join(config.log_dir, config.run_name, "models")
    if save_filename is None and epoch is not None: save_filename = f"model_epoch_{epoch}"
    elif save_filename is None and epoch is None: save_filename = "model"

    os.makedirs(save_dir, exist_ok=True)
    full_save_path = os.path.join(save_dir, save_filename+'.pth')
    logging.info(f"{Fore.YELLOW}Saving entire model at {full_save_path}{Style.RESET_ALL}")
    
    save_dict = {
        "epoch":epoch,
        "config": config,
    }
    if optim is not None: 
        save_dict["optim_state"] = optim.state_dict()
    if sched is not None:
        save_dict["sched_state"] = sched.state_dict()
    save_dict["model_state"] = model.state_dict()
    torch.save(save_dict, full_save_path)

def load_model(model, full_load_path, device=torch.device('cpu')):
    """
    Load a model's weights
    @args:
        - model (nn.Module): model to load weights into
        - full_load_path (str): path to load model
        - device (torch.device): device to setup the model on
    """

    assert os.path.exists(full_load_path), f"{Fore.YELLOW} Specified load path {full_load_path} does not exist {Style.RESET_ALL}"
    logging.info(f"{Fore.YELLOW}Loading model weights from {full_load_path}{Style.RESET_ALL}")

    saved_model = torch.load(full_load_path, map_location=device)
    saved_model_state = saved_model['model_state']
    current_model_state = model.state_dict()

    # Hack, make models compatible
    ex_saved_key = list(saved_model_state.keys())[0]
    ex_current_key = list(current_model_state.keys())[0]
    if 'module' in ex_current_key and 'module' in ex_saved_key: 
        pass
    elif 'module' in ex_current_key and 'module' not in ex_saved_key:
        saved_model_state = {f"module.{k}":v for k,v in saved_model_state.items()}
    elif 'module' not in ex_current_key and 'module' in ex_saved_key:
        saved_model_state = {k.replace('module.',''):v for k,v in saved_model_state.items()}
    else:
        pass

    model.load_state_dict(saved_model_state)

    logging.info(f"{Fore.GREEN} Entire model loading from {full_load_path.split('/')[-1]} was successful {Style.RESET_ALL}")

    return model


