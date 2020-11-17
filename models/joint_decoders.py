import torch
import torch.nn as nn
import torch.nn.functional as tf

from .common import conv, upconv


class FlowDispPoseDecoder(nn.Module):
    def __init__(self, ch_in, conv_chs=None, num_refs=1, use_mask=True, use_bn=False):
        super(FlowDispPoseDecoder, self).__init__()

        self.use_mask = use_mask

        if conv_chs is None:
            conv_chs = [ch_in, 128, 128, 96, 64, 32]

        layers = []
        for in_, out_ in zip(conv_chs[:-1], conv_chs[1:]):
            layers.append(conv(in_, out_, use_bn=use_bn))
        self.convs = nn.Sequential(*layers)
        
        self.conv_sf = conv(conv_chs[-1], 3, use_relu=False)
        self.conv_d1 = conv(conv_chs[-1], 1, use_relu=False)

        self.convs_pose = conv(conv_chs[-1], num_refs * 6, kernel_size=1, use_relu=False)
        if use_mask:
            self.convs_mask = conv(conv_chs[-1], 1, use_relu=False)

    def forward(self, x):
        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out)
        pose_out = self.convs_pose(x_out)
        pred_pose = pose_out.mean(3).mean(2) * 0.01

        if self.use_mask:
            mask = self.convs_mask(x_out)
        else:
            mask = None

        return x_out, sf, disp1, mask, pred_pose, pose_out


class JointContextNetwork(nn.Module):
    def __init__(self, ch_in, conv_chs=None, num_refs=1, use_mask=True, use_bn=False):
        super(JointContextNetwork, self).__init__()

        self.use_mask = use_mask

        if conv_chs is None:
            conv_chs = [ch_in, 128, 128, 128, 96, 64, 32]

        layers = []
        for in_, out_ in zip(conv_chs[:-1], conv_chs[1:]):
            layers.append(conv(in_, out_, use_bn=use_bn))
        self.convs = nn.Sequential(*layers)

        self.conv_sf = conv(32, 3, use_relu=False)
        self.conv_d1 = nn.Sequential(conv(32, 1, use_relu=False), torch.nn.Sigmoid())
        self.convs_pose = conv(conv_chs[-1], num_refs * 6, kernel_size=1, use_relu=False)

        if use_mask:
            self.convs_mask = nn.Sequential(
                conv(conv_chs[-1], 1, use_relu=False, use_bn=False),
                torch.nn.Sigmoid()
            )

    def forward(self, x):

        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out) * 0.3

        pose_out = self.convs_pose(x_out)
        pred_pose = pose_out.mean(3).mean(2) * 0.01

        if self.use_mask:
            mask = self.convs_mask(x_out)
        else:
            mask = None

        return sf, disp1, mask, pred_pose
