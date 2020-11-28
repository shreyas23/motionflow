import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_pose_ms, intrinsic_scale, disp2depth_kitti
from utils.inverse_warp import inverse_warp
from utils.helpers import upsample


class UpConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale, use_bn=False):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv1 = Conv(num_in_layers, num_out_layers, kernel_size, stride=1, use_bn=use_bn)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
        return self.conv1(x)


class Conv(nn.Module):
    """ Conv layer with options for padding and ELU/LeakyReLU nonlinearity
    """
    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=1, nonlin='leakyrelu', pad_mode='reflection', use_bn=False, bias=True):
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
