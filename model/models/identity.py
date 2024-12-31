
import torch
from collections import OrderedDict
import numpy as np
import torch.nn as nn


# -------------------------------------------------------------------------------------------------
class IdentityModel(nn.Module):
    """
    Simple class to implement identity model in format required by codebase
    """
    def __init__(self):
        super().__init__()
        self.identity_layer = nn.Identity()

    def forward(self, x):
        return self.identity_layer(x)

def identity_model(config, input_feature_channels):
    """
    Simple function to return identity model and feature channels in format requierd by codebase
    """
    model = IdentityModel()
    output_feature_channels = input_feature_channels
    return model, output_feature_channels
    

                        
