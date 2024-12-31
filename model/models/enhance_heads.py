"""
Decoders for enhancement tasks
UNETR CODE MODIFIED FROM MONAI GITHUB: https://github.com/Project-MONAI/MONAI
License found at: https://github.com/Project-MONAI/MONAI/blob/dev/LICENSE
"""

from __future__ import annotations

from collections.abc import Sequence

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from pathlib import Path

Model_DIR = Path(__file__).parents[1].resolve()
sys.path.append(str(Model_DIR))

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock, UnetrPrUpBlock
from monai.utils import optional_import

rearrange, _ = optional_import("einops", name="rearrange")

#----------------------------------------------------------------------------------------------------------------
class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    UNETR code modified from monai
    """
    def __init__(
        self,
        config,
        input_feature_channels,
        output_feature_channels
    ) -> None:

        super().__init__()

        if input_feature_channels[0] % 12 != 0:
            raise ValueError("Features should be divisible by 12 to use current UNETR config.")
        
        if config.encoder_name!='Swin': 
            raise ValueError(f"Invalid backbone component for SwinUNETR head: {config.encoder_name}")

        input_image_channels = config.no_in_channel
        if config.time==1:
            spatial_dims=2
            self.spatial_dims=2
            upsample_kernel_size=2
            mod_patch_size=config.Swin.patch_size[1:]
        else: 
            spatial_dims=3
            self.spatial_dims=3
            upsample_kernel_size=(2,2,2)
            mod_patch_size=config.Swin.patch_size
        

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_image_channels,
            out_channels=input_feature_channels[0],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[0],
            out_channels=input_feature_channels[0],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[1],
            out_channels=input_feature_channels[1],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[2],
            out_channels=input_feature_channels[2],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[4],
            out_channels=input_feature_channels[4],
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[4],
            out_channels=input_feature_channels[3],
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size, #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[3],
            out_channels=input_feature_channels[2],
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size, #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[2],
            out_channels=input_feature_channels[1],
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size, #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[1],
            out_channels=input_feature_channels[0],
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size, #These all should reflect the patchmerging ops in the backbone
            norm_name="instance",
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=input_feature_channels[0],
            out_channels=input_feature_channels[0],
            kernel_size=3,
            upsample_kernel_size=mod_patch_size, #This should be the patch embedding kernel size
            norm_name="instance",
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=input_feature_channels[0], out_channels=output_feature_channels)

    def forward(self, input_data):
        if self.spatial_dims==2:
            input_data = [i.squeeze(2) for i in input_data]
        x_in = input_data[0]
        backbone_features = input_data[1:]
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(backbone_features[0])
        enc2 = self.encoder3(backbone_features[1])
        enc3 = self.encoder4(backbone_features[2])
        dec4 = self.encoder10(backbone_features[4])
        dec3 = self.decoder5(dec4, backbone_features[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        out = self.out(out)
        if self.spatial_dims==2:
            out = out.unsqueeze(2)
        return out

#----------------------------------------------------------------------------------------------------------------
class ViTUNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    Code modified from monai
    """
    def __init__(
        self,
        config,
        input_feature_channels,
        output_feature_channels
    ) -> None:

        super().__init__()
        
        feature_size = 32

        input_image_channels = config.no_in_channel
        if config.encoder_name=='ViT': hidden_size = config.ViT.hidden_size
        else: raise ValueError(f"Invalid encoder_name for ViTUNETR head: {config.encoder_name}")
        if config.time==1:
            self.spatial_dims=2
            img_size = [config.height, config.width]
            if config.encoder_name=='ViT': patch_size = config.ViT.patch_size[1:]
            else: raise ValueError(f"Invalid encoder_name for ViTUNETR head: {config.encoder_name}")
        else: 
            self.spatial_dims=3
            img_size = [config.time, config.height, config.width]
            if config.encoder_name=='ViT': patch_size = config.ViT.patch_size
            else: raise ValueError(f"Invalid encoder_name for ViTUNETR head: {config.encoder_name}")
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.hidden_size = hidden_size

        three_dim_patch_size = patch_size if len(patch_size)==3 else (1,)+tuple(patch_size)
        if three_dim_patch_size[1]==2 and three_dim_patch_size[2]==2:
            n_us2, n_us3, n_us4 = 0, 0, 0
            enc_us2, enc_us3, enc_us4 = 2, 2, 2
            dec_us1, dec_us2, dec_us3, dec_us4 = 1, 1, 1, 2
        elif three_dim_patch_size[1]==4 and three_dim_patch_size[2]==4:
            n_us2, n_us3, n_us4 = 1, 1, 0
            enc_us2, enc_us3, enc_us4 = 2, 2, 2
            dec_us1, dec_us2, dec_us3, dec_us4 = 1, 1, 2, 2
        elif three_dim_patch_size[1]==8 and three_dim_patch_size[2]==8:
            n_us2, n_us3, n_us4 = 2, 1, 0
            enc_us2, enc_us3, enc_us4 = 2, 2, 2
            dec_us1, dec_us2, dec_us3, dec_us4 = 1, 2, 2, 2
        elif three_dim_patch_size[1]==16 and three_dim_patch_size[2]==16:
            n_us2, n_us3, n_us4 = 2, 1, 0
            enc_us2, enc_us3, enc_us4 = 2, 2, 2
            dec_us1, dec_us2, dec_us3, dec_us4 = 2, 2, 2, 2
        elif three_dim_patch_size[1]==32 and three_dim_patch_size[2]==32:
            n_us2, n_us3, n_us4 = 2, 1, 0
            enc_us2, enc_us3, enc_us4 = 2, 2, 2
            dec_us1, dec_us2, dec_us3, dec_us4 = 4, 2, 2, 2
        else:
            raise ValueError(f'ViT UNETR patch size {three_dim_patch_size} not yet supported')
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=self.spatial_dims,
            in_channels=input_image_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=n_us2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=enc_us2,
            norm_name="instance",
            conv_block=True,
            res_block=True,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=n_us3,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=enc_us3,
            norm_name="instance",
            conv_block=True,
            res_block=True,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=n_us4,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=enc_us4,
            norm_name="instance",
            conv_block=True,
            res_block=True,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=dec_us4,
            norm_name="instance",
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=dec_us3,
            norm_name="instance",
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=dec_us2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=dec_us1,
            norm_name="instance",
            res_block=True,
        )
        self.out = UnetOutBlock(spatial_dims=self.spatial_dims, in_channels=feature_size, out_channels=output_feature_channels)
        self.proj_axes = (0, self.spatial_dims + 1) + tuple(d + 1 for d in range(self.spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, input_data):
        x_in = input_data[0]
        if self.spatial_dims==2:
            x_in = x_in.squeeze(2)
        hidden_states_out = input_data[1:-1]
        x = input_data[-1]
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        out = self.out(out)
        if self.spatial_dims==2:
            out = out.unsqueeze(2)
        return out
