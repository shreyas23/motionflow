
import torch
import torch.nn as nn
import torch.nn.functional as tf

from collections import OrderedDict

from .common import Conv

class PoseNet(nn.Module):

    def __init__(self, num_input_images=2, in_ch=3, use_bn=False):
        super(PoseNet, self).__init__()
        self.num_input_images = num_input_images

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv0 = Conv(in_ch*(self.num_input_images-1), in_ch*(self.num_input_images-1), kernel_size=3, stride=2, use_bn=use_bn)
        self.conv1 = Conv(in_ch*(self.num_input_images-1), conv_planes[0], kernel_size=7, stride=2, use_bn=use_bn)
        self.conv2 = Conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=2, use_bn=use_bn)
        self.conv3 = Conv(conv_planes[1], conv_planes[2], stride=2, use_bn=use_bn)
        self.conv4 = Conv(conv_planes[2], conv_planes[3], stride=2, use_bn=use_bn)
        self.conv5 = Conv(conv_planes[3], conv_planes[4], stride=2, use_bn=use_bn)
        self.conv6 = Conv(conv_planes[4], conv_planes[5], stride=2, use_bn=use_bn)
        self.conv7 = Conv(conv_planes[5], conv_planes[6], stride=2, use_bn=use_bn)

        self.pose_pred = Conv(conv_planes[6], 6*(self.num_input_images-1), kernel_size=1, stride=1, nonlin='none', use_bn=False)


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
        pose = pose.mean(3).mean(2) * 0.01

        return pose


class PoseExpNet(nn.Module):

    def __init__(self, nb_ref_imgs=1, output_exp=True, in_ch=3, use_bn=False):
        super(PoseExpNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv1 = conv(in_ch*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7, stride=2, use_bn=use_bn)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=2, use_bn=use_bn)
        self.conv3 = conv(conv_planes[1], conv_planes[2], stride=2, use_bn=use_bn)
        self.conv4 = conv(conv_planes[2], conv_planes[3], stride=2, use_bn=use_bn)
        self.conv5 = conv(conv_planes[3], conv_planes[4], stride=2, use_bn=use_bn)
        self.conv6 = conv(conv_planes[4], conv_planes[5], stride=2, use_bn=use_bn)
        self.conv7 = conv(conv_planes[5], conv_planes[6], stride=2, use_bn=use_bn)

        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

        if args.use_mask:
            upconv_planes = [256, 256, 128, 64, 32, 16]
            self.upconv6 = upconv(conv_planes[5], upconv_planes[0], kernel_size=4, stride=2)
            self.upconv5 = upconv(upconv_planes[0]+conv_planes[4], upconv_planes[1], kernel_size=4, stride=2, use_bn=False)
            self.upconv4 = upconv(upconv_planes[1]+conv_planes[3], upconv_planes[2], kernel_size=4, stride=2, use_bn=False)
            self.upconv3 = upconv(upconv_planes[2]+conv_planes[2], upconv_planes[3], kernel_size=4, stride=2, use_bn=False)
            self.upconv2 = upconv(upconv_planes[3]+conv_planes[1], upconv_planes[4], kernel_size=4, stride=2, use_bn=False)
            self.upconv1 = upconv(upconv_planes[4]+conv_planes[0], upconv_planes[5], kernel_size=4, stride=2, use_bn=False)

            self.predict_mask5 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask4 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(upconv_planes[5], self.nb_ref_imgs, kernel_size=3, padding=1)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose_feats = self.pose_pred(out_conv7)
        pose = pose_feats.mean(3).mean(2) * 0.01

        if self.output_exp:
            out_upconv6 = self.upconv6(out_conv6)
            out_upconv5 = self.upconv5(torch.cat((out_upconv6, out_conv5), dim=1))
            out_upconv4 = self.upconv4(torch.cat((out_upconv5, out_conv4), dim=1))
            out_upconv3 = self.upconv3(torch.cat((out_upconv4, out_conv3), dim=1))
            out_upconv2 = self.upconv2(torch.cat((out_upconv3, out_conv2), dim=1))
            out_upconv1 = self.upconv1(torch.cat((out_upconv2, out_conv1), dim=1))

            exp_mask5 = torch.sigmoid(self.predict_mask5(out_upconv5))
            exp_mask4 = torch.sigmoid(self.predict_mask4(out_upconv4))
            exp_mask3 = torch.sigmoid(self.predict_mask3(out_upconv3))
            exp_mask2 = torch.sigmoid(self.predict_mask2(out_upconv2))
            exp_mask1 = torch.sigmoid(self.predict_mask1(out_upconv1))
        else:
            exp_mask5 = None
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        return [exp_mask1, exp_mask2, exp_mask3, exp_mask4, exp_mask5], pose, pose_feats


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        feats = cat_features
        for i in range(3):
            feats = self.convs[("pose", i)](feats)
            if i != 2:
                feats = self.relu(feats)

        out = feats.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 6)

        # axisangle = out[..., :3]
        # translation = out[..., 3:]

        # return axisangle, translation
        return out, feats