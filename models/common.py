import torch
import torch.nn as nn
import torch.nn.functional as tf

from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_pose_ms


""" some functions borrowed and modified from SENSE
    https://github.com/NVlabs/SENSE/blob/master/sense/models/common.py
"""

def post_process(pose, depth, mask, sf):
    post_sf = None
    return post_sf


def conv(in_chs, out_chs, kernel_size=3, stride=1, dilation=1, bias=True, use_relu=True, use_bn=False):
  layers = []
  layers.append(nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2, bias=bias))
  if use_bn:
    layers.append(nn.BatchNorm2d(out_chs))
  if use_relu:
    layers.append(nn.LeakyReLU(0.1, inplace=True))
  
  return nn.Sequential(*layers)


def upconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.1, inplace=True)
    )


class WarpingLayer_Pose(nn.Module):
    def __init__(self):
        super(WarpingLayer_Pose, self).__init__()
 
    def forward(self, x, pose, disp, k1, input_size):

        _, _, h_x, w_x = x.size()
        disp = interpolate2d_as(disp, x) * w_x

        local_scale = torch.zeros_like(input_size)
        local_scale[:, 0] = h_x
        local_scale[:, 1] = w_x

        pts1, k1_scale = pixel2pts_ms(k1, disp, local_scale / input_size)
        _, coord1 = pts2pixel_pose_ms(k1_scale, pts1, None, [h_x, w_x], pose_mat=pose)

        grid = coord1.transpose(1, 2).transpose(2, 3)
        x_warp = tf.grid_sample(x, grid)

        mask = torch.ones_like(x, requires_grad=False)
        mask = tf.grid_sample(mask, grid)
        mask = (mask >= 1.0).float()

        return x_warp * mask


def flow_warp(x, flo):
    """warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.ones(x.size()).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)

    mask[mask.data < 0.9999] = 0
    mask[mask.data > 0] = 1
    
    return output*mask

def disp_warp(rim, disp):
    """
    warp stereo image (right image) with disparity
    rim: [B, C, H, W] image/tensor
    disp: [B, 1, H, W] (left) disparity
    for disparity (left), we have
        left_image(x,y) = right_image(x-d(x,y),y)
    """
    B, C, H, W = rim.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if rim.is_cuda:
        grid = grid.cuda()
    vgrid = grid

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*(vgrid[:,0,:,:]-disp[:,0,:,:])/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    return nn.functional.grid_sample(rim, vgrid.permute(0,2,3,1))