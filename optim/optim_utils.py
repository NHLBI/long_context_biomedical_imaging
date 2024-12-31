"""
Defines helper functions for optimizer
"""

import torch 
import numpy as np
import torch.distributed as dist

#-------------------------------------------------------------------------------------------
def compute_total_updates(config, train_set):

    num_samples = len(train_set)

    if config.ddp:
        num_gpus = dist.get_world_size()
    else:
        num_gpus = 1

    num_updates = int(np.ceil(num_samples/(config.batch_size*config.iters_to_accumulate*num_gpus))*config.num_epochs)
    
    return num_samples, num_updates

# -------------------------------------------------------------------------------------------------
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

# -------------------------------------------------------------------------------------------------    
