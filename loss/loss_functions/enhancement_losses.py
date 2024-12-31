"""
Defines additional losses for image enhancement applications

"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

Current_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Current_DIR))

# -------------------------------------------------------------------------------------------------
# MSE loss

class MSE_Loss:
    """
    Weighted MSE loss
    """
    def __init__(self, rmse_mode=False, complex_i=False):
        """
        @args:
            - rmse_mode (bool): whether to turn root mean sequare error
            - complex_i (bool): whether images are 2 channelled for complex data
        """
        self.rmse_mode = rmse_mode
        self.complex_i = complex_i

    def __call__(self, outputs, targets, weights=None):

        B, C, T, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            diff_mag_square = torch.square(outputs[:,0]-targets[:,0]) + torch.square(outputs[:,1]-targets[:,1])
        else:
            diff_mag_square = torch.square(outputs-targets)

        if self.rmse_mode: diff_mag_square = torch.sqrt(diff_mag_square)

        if(weights is not None):

            if(weights.ndim==1):
                weights = weights.reshape(B,1,1,1,1)
            elif weights.ndim==2:
                weights = weights.reshape(B,T,1,1,1)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for MSE_Loss")

            v_l2 = torch.sum(weights*diff_mag_square) / (torch.sum(weights) + torch.finfo(torch.float32).eps)
        else:
            v_l2 = torch.sum(diff_mag_square)

        if(torch.any(torch.isnan(v_l2))):
            raise NotImplementedError(f"nan in MSE_Loss")

        return v_l2 / diff_mag_square.numel()

# -------------------------------------------------------------------------------------------------
# Charbonnier Loss

class Charbonnier_Loss:
    """
    Charbonnier Loss (L1)
    """
    def __init__(self, complex_i=False, eps=1e-3):
        """
        @args:
            - complex_i (bool): whether images are 2 channelled for complex data
            - eps (float): epsilon, different values can be tried here
        """
        self.complex_i = complex_i
        self.eps = eps

    def __call__(self, outputs, targets, weights=None):

        B, C, T, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            diff_L1_real = torch.abs(outputs[:,0]-targets[:,0])
            diff_L1_imag = torch.abs(outputs[:,1]-targets[:,1])
            loss = torch.sqrt(diff_L1_real * diff_L1_real + diff_L1_imag * diff_L1_imag + self.eps * self.eps)
        else:
            diff_L1 = torch.abs(outputs-targets)
            loss = torch.sqrt(diff_L1 * diff_L1 + self.eps * self.eps)

        if(weights is not None):

            if(weights.ndim==1):
                weights = weights.reshape(B,1,1,1,1)
            elif weights.ndim==2:
                weights = weights.reshape(B,T,1,1,1)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for L1_Loss")

            v_l1 = torch.sum(weights*loss) / torch.sum(weights)
        else:
            v_l1 = torch.sum(loss)

        return v_l1 / loss.numel()

# -------------------------------------------------------------------------------------------------


def get_gaussionand_derivatives_1D(sigma, halfwidth, voxelsize):
    """compute guassian kernels

    Args:
        sigma (float): sigma in the unit of physical world
        halfwidth (float): sampled halfwidth
        voxelsize (float): voxel size, in the same unit of sigma

    Returns:
        kernelSamplePoints, Gx, Dx, Dxx: sampled locations, gaussian and its derivatives
    """
    s = np.arange( 2 * round(halfwidth*sigma/voxelsize) + 1)
    kernelSamplePoints=(s-round(halfwidth*sigma/voxelsize))*voxelsize

    Gx, Dx, Dxx = gaussian_fucntion(kernelSamplePoints, sigma)

    return kernelSamplePoints, Gx, Dx, Dxx

def gaussian_fucntion(kernelSamplePoints, sigma):
    """compute gaussian and its derivatives

    Args:
        kernelSamplePoints (np array): sampled kernal points
        sigma (float): guassian sigma

    Returns:
        G, D, DD: guassian kernel, guassian 1st and 2nd order derivatives
    """

    N = 1/np.sqrt(2*np.pi*sigma*sigma)
    T = np.exp(-(kernelSamplePoints*kernelSamplePoints)/(2*sigma*sigma))

    G = N * T
    G = G / np.sum(G)

    D = N * (-kernelSamplePoints / (sigma*sigma)) * T
    D = D / np.sum(np.abs(D))

    DD = N * ((-1/(sigma*sigma)*T) + ((-kernelSamplePoints / (sigma*sigma)) * (-kernelSamplePoints / (sigma*sigma)) * T))
    DD = DD / np.sum(np.abs(DD))

    return G, D, DD

def create_window_3d(sigma=(1.25, 1.25, 1.25), halfwidth=(3, 3, 3), voxelsize=(1.0, 1.0, 1.0), order=(1,1,1)):
    """
    Creates a 3D gauss kernel
    """
    k_0 = get_gaussionand_derivatives_1D(sigma[0], halfwidth[0], voxelsize[0])
    k_1 = get_gaussionand_derivatives_1D(sigma[1], halfwidth[1], voxelsize[1])
    k_2 = get_gaussionand_derivatives_1D(sigma[2], halfwidth[2], voxelsize[2])
    window = k_0[order[0]+1][:, np.newaxis] * k_1[order[1]+1][:, np.newaxis].T

    window = window[:, :, np.newaxis] * np.expand_dims(k_2[order[2]+1], axis=(0,1))

    window /= np.sum(np.abs(window))
    
    return window

class GaussianDeriv3D_Loss:
    """
    Weighted gaussian derivative loss for 3D
    For every sigma, the gaussian derivatives are computed for outputs and targets along the magnitude of T, H, W
    The l1 loss are computed to measure the agreement of gaussian derivatives
    
    If sigmas have more than one value, every sigma in sigmas are used to compute a guassian derivative tensor
    The mean l1 is returned
    """
    def __init__(self, sigmas=[0.5, 1.0, 1.25], sigmas_T=[0.5, 1.0, 1.25], complex_i=False, device='cpu'):
        """
        @args:
            - sigmas (list): sigma for every scale along H and W
            - sigmas_T (list): sigma for every scale along T
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on
        """
        self.complex_i = complex_i
        self.sigmas = sigmas
        self.sigmas_T = sigmas_T

        assert len(self.sigmas_T) == len(self.sigmas)

        # compute kernels
        self.kernels = []
        for sigma, sigma_T in zip(sigmas, sigmas_T):
            k_3d = create_window_3d(sigma=(sigma, sigma, sigma_T), halfwidth=(3, 3, 3), voxelsize=(1.0, 1.0, 1.0), order=(1,1,1))
            kx, ky, kz = k_3d.shape
            k_3d = torch.from_numpy(np.reshape(k_3d, (1, 1, kx, ky, kz))).to(torch.float32)
            k_3d = torch.permute(k_3d, [0, 1, 4, 2, 3])
            self.kernels.append(k_3d.to(device=device))

    def __call__(self, outputs, targets, weights=None):

        B, C, T, H, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            outputs_im = torch.sqrt(outputs[:,:1]*outputs[:,:1] + outputs[:,1:]*outputs[:,1:])
            targets_im = torch.sqrt(targets[:,:1]*targets[:,:1] + targets[:,1:]*targets[:,1:])
        else:
            outputs_im = outputs
            targets_im = targets

        B, C, T, H, W = targets_im.shape
        
        loss = 0
        for k_3d in self.kernels:
            k_3d = k_3d.to(device=outputs_im.device)
            grad_outputs_im = F.conv3d(outputs_im, k_3d, bias=None, stride=1, padding='same', groups=C)
            grad_targets_im = F.conv3d(targets_im, k_3d, bias=None, stride=1, padding='same', groups=C)
            loss += torch.mean(torch.abs(grad_outputs_im-grad_targets_im), dim=(1, 2, 3, 4), keepdim=True)

        loss /= len(self.kernels)

        if weights is not None:
            if not weights.ndim==1:
                raise NotImplementedError(f"Only support 1D(Batch) weights for GaussianDeriv3D_Loss")
            v = torch.sum(weights*loss) / (torch.sum(weights) + torch.finfo(torch.float32).eps)
        else:
            v = torch.mean(loss)

        if(torch.any(torch.isnan(v))):
            raise NotImplementedError(f"nan in GaussianDeriv3D_Loss")

        return v

# -------------------------------------------------------------------------------------------------
# Combined loss class

class Combined_Loss:
    """
    Combined loss for image enhancement
    Sums multiple loss with their respective weights
    """
    def __init__(self, losses, loss_weights, complex_i=False, device="cpu") -> None:
        """
        @args:
            - losses (list of "ssim", "ssim3D", "l1", "mse"):
                list of losses to be combined
            - loss_weights (list of floats)
                weights of the losses in the combined loss
            - complex_i (bool): whether images are 2 channelled for complex data
            - device (torch.device): device to run the loss on
        """
        assert len(losses)>0, f"At least one loss is required to setup"
        assert len(losses)<=len(loss_weights), f"Each loss should have its weight"

        self.complex_i = complex_i
        self.device = device

        losses = [self.str_to_loss(loss) for loss in losses]
        self.losses = list(zip(losses, loss_weights))

    def str_to_loss(self, loss_name):

        if loss_name=="mse":
            loss_f = MSE_Loss(rmse_mode=False, complex_i=self.complex_i)
        elif loss_name=="charbonnier":
            loss_f = Charbonnier_Loss(complex_i=self.complex_i)
        elif loss_name=="gaussian3D":
            loss_f = GaussianDeriv3D_Loss(sigmas=[0.25, 0.5, 1.0], sigmas_T=[0.25, 0.5, 0.5], complex_i=self.complex_i, device=self.device)
        else:
            raise NotImplementedError(f"Loss type not implemented: {loss_name}")

        return loss_f
    
    def __call__(self, outputs, targets, weights=None):

        combined_loss = 0
        for loss_f, weight in self.losses:
            v = weight*loss_f(outputs=outputs, targets=targets, weights=weights)
            if not torch.isnan(v):
                combined_loss += v

        return combined_loss

