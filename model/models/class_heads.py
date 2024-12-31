"""
Decoders for classification tasks
"""

import sys
import torch
import torch.nn as nn

from pathlib import Path


#----------------------------------------------------------------------------------------------------------------
class ViTLinear(nn.Module):
    def __init__(
        self,
        config,
        input_feature_channels,
        output_feature_channels
    ):
        """
        Takes in features from backbone model and produces a classification vector using only the last feature tensor; assumes a class token from ViT if using attn
        @args:
            config (namespace): contains all parsed args
            input_feature_channels (List[int]): contains a list of the number of feature channels in each tensor input into this task head (i.e., returned by the backbone)
            output_feature_channels (int): the number of feature channels in each tensor expected to be returned by this task head
            forward pass, x (List[tensor]): contains a list of torch tensors output by the encoder
        @rets:
            forward pass, x (tensor): output from the classification task head, size B x output_feature_channels
        """
        super().__init__()

        if config.encoder_name == 'ViT':
            if config.ViT.use_hyena or config.ViT.use_mamba:
                self.cls_token = False
            else:
                self.cls_token = True
        else:
            raise ValueError(f"Invalid backbone component for ViTLinear head: {config.backbone_component}")
        
        self.classification_head = nn.Sequential(nn.Linear(input_feature_channels[-1], output_feature_channels), nn.Tanh())

    def forward(self, x):
        x = x[-1]
        if self.cls_token: # cls token in ViT with attention
            x = x[:, 0]
        else: # no cls token in ViT with hyena or mamba
            x = x.mean(1)
        x = self.classification_head(x)
        return x
    
#----------------------------------------------------------------------------------------------------------------
class SwinLinear(nn.Module):
    def __init__(
        self,
        config,
        input_feature_channels,
        output_feature_channels
    ):
        """
        Takes in features from backbone model and produces a classification vector using only the last feature tensor
        @args:
            config (namespace): contains all parsed args
            input_feature_channels (List[int]): contains a list of the number of feature channels in each tensor input into this task head (i.e., returned by the backbone)
            output_feature_channels (int): the number of feature channels in each tensor expected to be returned by this task head
            forward pass, x (List[tensor]): contains a list of torch tensors output by the encoder
        @rets:
            forward pass, x (tensor): output from the classification task head, size B x output_feature_channels
        """
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classification_head = nn.Sequential(nn.Linear(input_feature_channels[-1], output_feature_channels), nn.Tanh())

    def forward(self, x):
        x = x[-1]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classification_head(x)
        return x
    
