"""
Decoders for segmentation tasks
Upernet head code adapted from https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py
License here: https://github.com/yassouali/pytorch-segmentation/blob/master/LICENSE
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

#----------------------------------------------------------------------------------------------------------------


class PSPModule2D(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule2D, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class FPN_fuse2D(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse2D, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

class UperNet2D(nn.Module):
    """
    UperNet3D head, used for segmentation. Incorporates features from four different depths in backbone.
    @args:
        config (namespace): contains all args
        input_feature_channels (List[int]): contains a list of the number of feature channels in each tensor input into this task head (i.e., returned by the encoder)
        output_feature_channels (int): the number of feature channels in each tensor expected to be returned by this task head
        forward pass, features (torch tensor): features we will process, size B C H W
    @rets:
        forward pass, x (torch tensor): output tensor, size B C H W

    """
    def __init__(self, 
                 config, 
                 input_feature_channels,
                 output_feature_channels):
        super(UperNet2D, self).__init__()

        if config.encoder_name=='Swin':
            self.upernet_feature_channels = [-4, -3, -2, -1]
        elif config.encoder_name=='ViT':
            self.upernet_feature_channels = [4, 7, 10, -1]
        else:
            raise ValueError(f'encoder_name {config.encoder_name} not recognized or comaptible with UperNet3D')
        input_feature_channels = [input_feature_channels[c] for c in self.upernet_feature_channels]

        self.config = config
        self.fpn_out = input_feature_channels[0]
        self.input_size = (config.height,config.width)
        self.PPN = PSPModule2D(input_feature_channels[-1])
        self.FPN = FPN_fuse2D(input_feature_channels, fpn_out=self.fpn_out)
        self.head = nn.Conv2d(self.fpn_out, output_feature_channels, kernel_size=3, padding=1)

        if config.encoder_name=='ViT':
            spatial_dims = 2
            patch_size = [config.ViT.patch_size[1], config.ViT.patch_size[2]]
            self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(self.input_size, patch_size))
            self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        
        
    def _reshape_vit_output(self, x):

        hidden_size = x.shape[-1]
        proj_view_shape = list(self.feat_size) + [hidden_size]
        
        new_view = [x.size(0)] + proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()

        return x
    
    def forward(self, features):
        features = [features[c] for c in self.upernet_feature_channels]
        if self.config.encoder_name=='ViT':
            features = [self._reshape_vit_output(f) for f in features]
        else:
            features = [f[:,:,0,:,:] for f in features] # Remove time dim for 2d convoultional upernet
        features[-1] = self.PPN(features[-1])
        x = self.FPN(features)
        x = F.interpolate(x, size=self.input_size, mode='bilinear')
        x = self.head(x)
        x = torch.unsqueeze(x,2) # Add back in time dim for main codebase 
        
        return x

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


#----------------------------------------------------------------------------------------------------------------
class PSPModule3D(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule3D, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=1, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool3d(output_size=bin_sz)
        conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm3d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        d, h, w = features.size()[2], features.size()[3], features.size()[4]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(d, h, w), mode='trilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

def up_and_add3D(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3), y.size(4)), mode='trilinear', align_corners=True) + y

class FPN_fuse3D(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse3D, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv3d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv3d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv3d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add3D(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        D, H, W = P[0].size(2), P[0].size(3), P[0].size(4)
        P[1:] = [F.interpolate(feature, size=(D, H, W), mode='trilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

class UperNet3D(nn.Module):
    """
    UperNet3D head, used for segmentation. Incorporates features from four different depths in backbone.
    @args:
        config (namespace): contains all args
        input_feature_channels (List[int]): contains a list of the number of feature channels in each tensor input into this task head (i.e., returned by the encoder)
        output_feature_channels (int): the number of feature channels in each tensor expected to be returned by this task head
        forward pass, features (torch tensor): features we will process, size B C D H W
    @rets:
        forward pass, x (torch tensor): output tensor, size B C D H W

    """
    def __init__(self, 
                 config, 
                 input_feature_channels,
                 output_feature_channels):
        super(UperNet3D, self).__init__()

        if config.encoder_name=='Swin':
            self.upernet_feature_channels = [-4, -3, -2, -1]
        elif config.encoder_name=='ViT':
            self.upernet_feature_channels = [4, 7, 10, -1]
        else:
            raise ValueError(f'encoder_name {config.encoder_name} not recognized or comaptible with UperNet3D')
        input_feature_channels = [input_feature_channels[c] for c in self.upernet_feature_channels]

        self.config = config
        self.fpn_out = input_feature_channels[0]
        self.input_size = (config.time,config.height,config.width)
        self.PPN = PSPModule3D(input_feature_channels[-1])
        self.FPN = FPN_fuse3D(input_feature_channels, fpn_out=self.fpn_out)
        self.head = nn.Conv3d(self.fpn_out, output_feature_channels, kernel_size=3, padding=1)

        if config.encoder_name=='ViT':
            spatial_dims = 3
            patch_size = config.ViT.patch_size
            self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(self.input_size, patch_size))
            self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        
        
    def _reshape_vit_output(self, x):

        hidden_size = x.shape[-1]
        proj_view_shape = list(self.feat_size) + [hidden_size]
        
        new_view = [x.size(0)] + proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()

        return x

    def forward(self, features, output_size=None):
        features = [features[c] for c in self.upernet_feature_channels]
        if self.config.encoder_name=='ViT':
            features = [self._reshape_vit_output(f) for f in features]
        features[-1] = self.PPN(features[-1])
        x = self.FPN(features)
        if output_size is None:
            x = F.interpolate(x, size=self.input_size, mode='trilinear')
        else:
            x = F.interpolate(x, size=output_size, mode='trilinear')
        x = self.head(x)
        return x

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d): module.eval()

