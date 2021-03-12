import torch
import torch.nn as nn

from .common import conv, Conv

class MonoSceneFlowDecoder(nn.Module):
    def __init__(self, ch_in):
        super(MonoSceneFlowDecoder, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_sf = conv(32, 3, isReLU=False)
        self.conv_d1 = conv(32, 1, isReLU=False)

    def forward(self, x):
        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out)

        return x_out, sf, disp1


class MonoSceneFlowDecoderLarge(nn.Module):
    def __init__(self, ch_in):
        super(MonoSceneFlowDecoder, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 192),
            conv(192, 128),
            conv(128, 128),
            conv(128, 96),
            conv(96, 64),
            conv(64, 32)
        )
        self.conv_sf = conv(32, 3, isReLU=False)
        self.conv_d1 = conv(32, 1, isReLU=False)

    def forward(self, x):
        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out)

        return x_out, sf, disp1


class ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1),
            conv(128, 128, 3, 1, 2),
            conv(128, 128, 3, 1, 4),
            conv(128, 96, 3, 1, 8),
            conv(96, 64, 3, 1, 16),
            conv(64, 32, 3, 1, 1)
        )
        self.conv_sf = conv(32, 3, isReLU=False)
        self.conv_d1 = nn.Sequential(
            conv(32, 1, isReLU=False), 
            torch.nn.Sigmoid()
        )

    def forward(self, x):

        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out) * 0.3

        return sf, disp1

class ContextNetworkSF(nn.Module):
    def __init__(self, in_chs, use_bn=False):
        super(ContextNetworkSF, self).__init__()

        self.convs = nn.Sequential(
            Conv(in_chs, 128, 3, 1, 1, use_bn=use_bn),
            Conv(128, 128, 3, 1, 2, use_bn=use_bn),
            Conv(128, 128, 3, 1, 4, use_bn=use_bn),
            Conv(128, 96, 3, 1, 8, use_bn=use_bn),
            Conv(96, 64, 3, 1, 16, use_bn=use_bn),
            Conv(64, 32, 3, 1, 1, use_bn=use_bn)
        )

        self.conv_sf = Conv(32, 3, nonlin='none')

    def forward(self, x):

        x_out = self.convs(x)
        sf = self.conv_sf(x_out)

        return sf