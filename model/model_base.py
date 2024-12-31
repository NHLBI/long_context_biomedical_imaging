"""
Creates an encoder/decoder model based on args specified in the config file
"""

import os
import sys
import logging
from colorama import Fore, Style

import torch
import torch.nn as nn

from models import *
from model_utils import *

# -------------------------------------------------------------------------------------------------

class EncoderDecoderModel(nn.Module):
    """
    Connects an encoder and decoder together to form a full model
    """

    def __init__(self, config, encoder_name, decoder_name, input_feature_channels, output_feature_channels):
        """
        @args:
            - config (Namespace): nested namespace containing all args
            - encoder_name (str): name of the encoder to create
            - decoder_name (str): name of the decoder to create
            - input_feature_channels (int): int specifying the channel dimension of the inputs to the model
            - output_feature_channels (int): int specifying the channel dimensions of the output of the model       
        """
        
        super().__init__()

        self.config = config
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.input_feature_channels = input_feature_channels
        self.output_feature_channels = output_feature_channels

        # Encoders
        if self.encoder_name=='Identity':
            self.encoder, self.encoder_feature_channels = identity_model(self.config, self.input_feature_channels)
        elif self.encoder_name=='ViT':
            self.encoder, self.encoder_feature_channels = custom_ViT(self.config, self.input_feature_channels)
        elif self.encoder_name=='Swin':
            self.encoder, self.encoder_feature_channels = custom_Swin(self.config, self.input_feature_channels)
        else:
            raise NotImplementedError(f"Encoder not implemented: {self.encoder_name}")

        # Decoders
        if self.decoder_name=='Identity':
            self.decoder, _ = identity_model(self.config, self.encoder_feature_channels)
        elif self.decoder_name=='UperNet2D': # 2D enhancement or seg
            self.decoder = UperNet2D(self.config, self.encoder_feature_channels, self.output_feature_channels)
        elif self.decoder_name=='UperNet3D': # 3D enhancement or seg
            self.decoder = UperNet3D(self.config, self.encoder_feature_channels, self.output_feature_channels)
        elif self.decoder_name=='ViTLinear': # 2D or 3D class
            self.decoder = ViTLinear(self.config, self.encoder_feature_channels, self.output_feature_channels)
        elif self.decoder_name=='SwinLinear': # 2D or 3D class
            self.decoder = SwinLinear(self.config, self.encoder_feature_channels, self.output_feature_channels)
        elif self.decoder_name=='SwinUNETR': # 2D or 3D enhancement or seg
            self.decoder = SwinUNETR(self.config, self.encoder_feature_channels, self.output_feature_channels)
        elif self.decoder_name=='ViTUNETR': # 2D or 3d enhancement or seg
            self.decoder = ViTUNETR(self.config, self.encoder_feature_channels, self.output_feature_channels)
        else:
            raise NotImplementedError(f"Decoder not implemented: {self.decoder_name}")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        """
        Define the forward pass through the model
        @args:
            - x (5D torch.Tensor): input image, B C D/T H W
        @rets:
            - output (tensor): final output from model for this task
        """
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(encoder_output)
        return decoder_output
