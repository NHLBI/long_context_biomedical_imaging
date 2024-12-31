"""
Defines the loss function specified in the config
"""

import sys
from torch import nn
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

from loss_functions import *

# -------------------------------------------------------------------------------------------------
def get_loss_func(loss_name):
    """
    Sets up the loss function.
    By default, the loss function will be passed (model_outputs, datalodaer_labels) and should return a float
    @args:
        - loss_name: str defining the loss function
    @rets:
        - loss_f: loss function
    """
    if loss_name=='CrossEntropy':
        loss_f = nn.CrossEntropyLoss()
    elif loss_name=='MSE':
        loss_f = nn.MSELoss()
    elif loss_name=='CombinationEnhance':
        loss_f = Combined_Loss(["mse", "charbonnier", "gaussian3D"], [1, 1, 1])
    else:
        raise NotImplementedError(f"Loss function not implemented: {loss_name}")
    return loss_f
        
