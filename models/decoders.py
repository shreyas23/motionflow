import torch
import torch.nn as nn
import torch.nn.functional as tf

from .common import conv

class PoseNet(nn.Module):

    def __init__(self, nb_ref_imgs=1):
        super(PoseNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv0 = conv(3*(1+self.nb_ref_imgs), 3*(1+self.nb_ref_imgs), kernel_size=3)
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1(out_conv0)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)
        pose = pose.mean(3).mean(2)
        pose = pose.view(pose.size(0), self.nb_ref_imgs, 6) * 0.01

        return pose


class FlowDispDecoder(nn.Module):
  def __init__(self, ch_in, use_bn=True):
    super(FlowDispDecoder, self).__init__()

    self.convs = nn.Sequential(
      conv(ch_in, 128, use_bn=use_bn),
      conv(128, 128, use_bn=use_bn),
      conv(128, 96, use_bn=use_bn),
      conv(96, 64, use_bn=use_bn),
      conv(64, 32, use_bn=use_bn))

    self.conv_sf = conv(32, 3, use_relu=False, use_bn=False)
    self.conv_d1 = conv(32, 1, use_relu=False, use_bn=False)

  def forward(self, x):
    x_out = self.convs(x)
    sf = self.conv_sf(x_out)
    disp1 = self.conv_d1(x_out)

    return x_out, sf, disp1