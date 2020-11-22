import torch
import torch.nn as nn
from .common import Conv

class SFDecoder(nn.Module):
    def __init__(self, in_ch, use_bn=False):
        super(SFDecoder, self).__init__()

        self.convs = nn.Sequential(
            Conv(in_ch, 128, use_bn=use_bn, nonlin='leakyrelu'),
            Conv(128, 128, use_bn=use_bn, nonlin='leakyrelu'),
            Conv(128, 96, use_bn=use_bn, nonlin='leakyrelu'),
            Conv(96, 64, use_bn=use_bn, nonlin='leakyrelu'),
            Conv(64, 32, use_bn=use_bn, nonlin='leakyrelu')
        )

        self.conv_sf = Conv(32, 3, nonlin='none')

        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.LeakyReLU):
                pass

            elif isinstance(layer, nn.Sequential):
                pass
        
    def forward(self, x):
        x_out = self.convs(x)
        sf = self.conv_sf(x_out)

        return x_out, sf