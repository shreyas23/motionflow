import torch
import torch.nn as nn
import torch.nn.functional as tf
import math

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

class PyramidPoolingModule(nn.Module):
    def __init__(self, args, encoder_planes, pool_scales=[1, 2, 3, 6], ppm_last_conv_planes=256, ppm_inter_conv_planes=128):
        super(PyramidPoolingModule, self).__init__()

        self.ppm_pooling = []
        self.ppm_conv = []

        self.ppm_last_conv_planes = ppm_last_conv_planes

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(encoder_planes[-1], ppm_inter_conv_planes, kernel_size=1, bias=False),
                    nn.BatchNorm2d(ppm_inter_conv_planes),
                    nn.ReLU(inplace=True)
                )
            )
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv(
            encoder_planes[-1] + len(pool_scales)*128, 
            self.ppm_last_conv_planes, 
            bias=False, use_bn=True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, conv5):
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(tf.upsample(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False
            )))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)
        return f

class Hourglass(nn.Module):
    def __init__(self, in_planes, do_flow=False, no_output=False, bn_type='plain'):
        super(Hourglass, self).__init__()
        self.no_output = no_output
        # in 1/2, out: 1/4
        self.conv1 = convbn(
            in_planes, 
            in_planes * 2, 
            kernel_size=3, 
            stride=2, 
            padding=1,
            bn_type=bn_type,
        )
        # in: 1/4, out: 1/8
        self.conv2 = convbn(
            in_planes * 2,
            in_planes * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bn_type=bn_type
        )
        # in: 1/8, out : 1/8
        self.conv3 = convbn(
            in_planes * 2,
            in_planes * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bn_type=bn_type
        )
        # in: 1/8, out: 1/4
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(in_planes * 2, in_planes * 2, 3, padding=1, output_padding=1, stride=2,bias=False),
            make_bn_layer(bn_type, in_planes * 2),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_planes * 2, in_planes, 3, padding=1, output_padding=1, stride=2,bias=False),
            make_bn_layer(bn_type, in_planes),
            nn.ReLU()
        )
        if not no_output:
            if do_flow:
                self.output = predict_flow(in_planes)
            else:
                self.output = predict_class(in_planes, 1, bias=False)

        weight_init(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if not self.no_output:
            y = self.output(x)
        else:
            y = None
        return y, x
