import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict

from .common import conv

class SFDecoder(nn.Module):
    def __init__(self, ch_in, num_refs=1, use_mask=True, use_bn=False):
        super(SFDecoder, self).__init__()

        self.use_mask = use_mask

        self.convs = nn.Sequential(
            conv(ch_in, 256, use_bn=use_bn),
            conv(256, 128, use_bn=use_bn),
            conv(128, 128, use_bn=use_bn),
            conv(128, 96, use_bn=use_bn),
            conv(96, 64, use_bn=use_bn),
            conv(64, 32, use_bn=use_bn),
            conv(32, 3, use_relu=False, use_bn=False)
        )

    def forward(self, x):
        return self.convs(x)
