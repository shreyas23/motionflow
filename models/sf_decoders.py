import torch
import torch.nn as nn
from .common import Conv

class SFDecoder(nn.Module):
    def __init__(self, in_ch, use_bn=False):
        super(SFDecoder, self).__init__()

        self.convs = nn.Sequential(
            Conv(in_ch, 128, use_bn=use_bn, nonlin='leakyrelu'),
            Conv(128, 128, use_bn=use_bn, nonlin='leakyrelu'),
            Conv(128, 128, use_bn=use_bn, nonlin='leakyrelu'),
            Conv(128, 96, use_bn=use_bn, nonlin='leakyrelu'),
            Conv(96, 64, use_bn=use_bn, nonlin='leakyrelu'),
            Conv(64, 32, use_bn=use_bn, nonlin='leakyrelu')
        )

        self.conv_sf = Conv(32, 3, nonlin='none')

    def forward(self, x):
        x_out = self.convs(x)
        sf = self.conv_sf(x_out)

        return x_out, sf