"""
Utility functions to measure system status
"""
import os
import torch
from collections import OrderedDict
from datetime import datetime
from torchinfo import summary
from colorama import Fore, Style
import numpy as np
import logging

# -------------------------------------------------------------------------------------------------
    
def start_timer(enable=False):
    
    if enable:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return (start, end)
    else:
        return None

def end_timer(enable=False, t=None, msg=""):
    if enable:
        t[1].record()
        torch.cuda.synchronize()
        print(f"{Fore.LIGHTBLUE_EX}{msg} {t[0].elapsed_time(t[1])} ms ...{Style.RESET_ALL}", flush=True)
                               
# -------------------------------------------------------------------------------------------------

def get_cuda_info(device):
	return {
		"PyTorch_version": torch.__version__,
		"CUDA_version": torch.version.cuda,
		"cuDNN_version": torch.backends.cudnn.version(),
		"Arch_version": torch._C._cuda_getArchFlags(),
		"device_count": torch.cuda.device_count(),
		"device_name": torch.cuda.get_device_name(device=device),
		"device_id": torch.cuda.current_device(),
		"cuda_capability": torch.cuda.get_device_capability(device=device),
		"device_properties": torch.cuda.get_device_properties(device=device),
		"reserved_memory": torch.cuda.memory_reserved(device=device) / 1024**3,
		"allocated_memory": torch.cuda.memory_allocated(device=device) / 1024**3,
		"max_allocated_memory": torch.cuda.max_memory_allocated(device=device) / 1024**3,
        "gpu_name": torch.cuda.get_device_name()
	}

def support_bfloat16(device):
    DISABLE_FLOAT16_INFERENCE = os.environ.get("DISABLE_FLOAT16_INFERENCE", "False")
    if DISABLE_FLOAT16_INFERENCE == "True": return False

    info =  get_cuda_info(device)
    if info["gpu_name"].find("A100") >= 0 or info["gpu_name"].find("H100") >= 0:
        return True
    else:
        return False

# -------------------------------------------------------------------------------------------------
# Model info

def model_info(model, config):
    """
    Prints model info and sets total and trainable parameters in the config
    @args:
        - model (torch model): the model to check parameters of
        - config (Namespace): runtime namespace for setup
    @rets:
        - model_summary (ModelStatistics object): the model summary
            see torchinfo/model_statistics.py for more information.
    """
    c = config
    col_names=("num_params", "params_percent", "mult_adds", "input_size", "output_size", "trainable")
    row_settings=["var_names", "depth"]
    dtypes=[torch.float32]
    model = model.module if config.ddp else model 
    model.train()
    mod_batch_size = 2

    example_input = torch.ones((mod_batch_size, c.no_in_channel, c.time, c.height, c.width)).to(c.device)
    model_summary = summary(model, \
                            verbose=0, mode="train", depth=c.summary_depth,\
                            input_data=example_input, 
                            col_names=col_names,row_settings=row_settings, dtypes=dtypes,\
                            device=config.device)
    
    logging.info(f"{Fore.MAGENTA}{'-'*40}Summary of model{'-'*40}{Style.RESET_ALL}")
    logging.info(f"\n{str(model_summary)}") 

    torch.cuda.empty_cache()

    return 

# -------------------------------------------------------------------------------------------------
def get_device(device=None):
    """
    Wrapper around getting device
    @args:
        - device (torch.device): if not None this device will be returned
            otherwise check if cuda is available
    @rets:
        - device (torch.device): the device to be used
    """

    return device if device is not None else \
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__=="__main__":
    pass