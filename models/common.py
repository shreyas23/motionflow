import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_pose_ms, intrinsic_scale, disp2depth_kitti
from utils.inverse_warp import inverse_warp
from utils.helpers import upsample

import numbers, math

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        return self.conv1(x)

class UpConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale, pad_mode='zeros', use_bn=False):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv1 = Conv(num_in_layers, num_out_layers, kernel_size, stride=1, use_bn=use_bn, pad_mode=pad_mode)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        return self.conv1(x)


class Conv(nn.Module):
    """ Conv layer with options for padding and ELU/LeakyReLU nonlinearity
    """
    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=1, nonlin='leakyrelu', pad_mode='zeros', use_bn=False, bias=True):
        super(Conv, self).__init__()

        nonlin_dict = {
            'leakyrelu' : nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True),
            'relu': nn.ReLU(inplace=True)
        }

        padding = ((kernel_size - 1) * dilation) // 2

        layers = []
        layers.append(nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding, padding_mode=pad_mode, bias=bias))

        if use_bn:
            layers.append(nn.BatchNorm2d(out_chs))

        if nonlin in nonlin_dict:
            nonlin_layer = nonlin_dict[nonlin]
            layers.append(nonlin_layer)

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return out


class WarpingLayer_SF(nn.Module):
    def __init__(self):
        super(WarpingLayer_SF, self).__init__()
 
    def forward(self, x, sceneflow, disp, k1, input_size):

        _, _, h_x, w_x = x.size()
        disp = interpolate2d_as(disp, x) * w_x

        local_scale = torch.zeros_like(input_size)
        local_scale[:, 0] = h_x
        local_scale[:, 1] = w_x

        pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
        _, _, coord1 = pts2pixel_ms(k1_scale, pts1, sceneflow, [h_x, w_x])

        grid = coord1.transpose(1, 2).transpose(2, 3)
        x_warp = tf.grid_sample(x, grid)

        mask = torch.ones_like(x, requires_grad=False)
        mask = tf.grid_sample(mask, grid)
        mask = (mask >= 1.0).float()

        return x_warp * mask

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()

        self.kernel_size = kernel_size

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = tf.conv1d
        elif dim == 2:
            self.conv = tf.conv2d
        elif dim == 3:
            self.conv = tf.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        padding = ((self.kernel_size - 1)) // 2
        # padding = [padding] * self.dim
        # input = tf.pad(input, pad=padding, mode='reflection')
        return self.conv(input, weight=self.weight, groups=self.groups, padding=padding)