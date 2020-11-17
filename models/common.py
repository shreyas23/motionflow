import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_pose_ms, intrinsic_scale, disp2depth_kitti
from utils.inverse_warp import inverse_warp
from utils.helpers import upsample, sample_grid


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
    def __init__(self, in_chs, out_chs, kernel_size=3, stride=1, dilation=1, nonlin='elu', pad_mode='reflection', use_bn=False, bias=True):
        super(Conv, self).__init__()

        nonlin_dict = {
            'leakyrelu' : nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True),
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


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        return pix_coords