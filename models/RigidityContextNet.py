import torch
import torch.nn as nn
import torch.nn.functional as tf

from .common import conv, upconv

from utils.inverse_warp import pose2sceneflow
from utils.sceneflow_util import pixel2pts_ms_depth


class RigidityContextNet(nn.Module):

    def __init__(self, args, in_ch):
        super(RigidityContextNet, self).__init__()
        self.args = args

        self.convs = nn.Sequential(
            conv(in_ch, 128, 3, 1, 1, use_bn=args.use_bn),
            conv(128, 128, 3, 1, 2, use_bn=args.use_bn),
            conv(128, 128, 3, 1, 4, use_bn=args.use_bn),
            conv(128, 96, 3, 1, 8, use_bn=args.use_bn),
            conv(96, 64, 3, 1, 16, use_bn=args.use_bn),
            conv(64, 32, 3, 1, 1, use_bn=args.use_bn)
        )
        self.conv_sf = conv(32, 3, use_relu=False)
        self.conv_d1 = nn.Sequential(
            conv(32, 1, use_relu=False), 
            torch.nn.Sigmoid()
        )

    def forward(self, pose_sf, sf, disp):
        x = torch.cat([pose_sf, sf, disp], dim=1) 

        x_out = self.convs(x)
        sf_refine = self.conv_sf(x_out)
        disp_refine = self.conv_d1(x_out) * 0.3

        return sf_refine, disp_refine