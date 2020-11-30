import torch
import torch.nn as nn
import torch.nn.functional as tf

from .common import Conv

class JointDecoder(nn.Module):
    def __init__(self, args, ch_in, conv_chs=None, num_refs=1, use_mask=True, use_bn=False):
        super(JointDecoder, self).__init__()

        self.use_mask = use_mask

        if conv_chs is None:
            conv_chs = [ch_in, 128, 128, 96, 64, 32]

        layers = []
        for in_, out_ in zip(conv_chs[:-1], conv_chs[1:]):
            layers.append(Conv(in_, out_, use_bn=use_bn))
        self.convs = nn.Sequential(*layers)
        
        self.conv_sf = Conv(conv_chs[-1], 3, nonlin='none')
        self.conv_d1 = Conv(conv_chs[-1], 1, nonlin='none')

        self.convs_pose = Conv(conv_chs[-1], num_refs * 6, kernel_size=1, nonlin='none')
        if use_mask:
            self.convs_mask = Conv(conv_chs[-1], 1, nonlin='none')

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
    def __init__(self, args, in_chs, num_refs=1, use_bn=False):
        super(JointContextNetwork, self).__init__()

        self.use_mask = args.use_mask

        self.convs = nn.Sequential(
            Conv(in_chs, 128, 3, 1, 1, use_bn=use_bn),
            Conv(128, 128, 3, 1, 2, use_bn=use_bn),
            Conv(128, 128, 3, 1, 4, use_bn=use_bn),
            Conv(128, 96, 3, 1, 8, use_bn=use_bn),
            Conv(96, 64, 3, 1, 16, use_bn=use_bn),
            Conv(64, 32, 3, 1, 1, use_bn=use_bn)
        )

        self.conv_sf = Conv(32, 3, nonlin='none')
        self.conv_d1 = nn.Sequential(
            Conv(32, 1, nonlin='none'), 
            torch.nn.Sigmoid()
        )
        self.convs_pose = Conv(32, num_refs * 6, kernel_size=1, nonlin='none')
        if self.use_mask:
            self.convs_mask = Conv(32, 1, nonlin='none')

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

        return sf, disp1, pred_pose, mask