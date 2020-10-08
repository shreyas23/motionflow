import torch
import torch.nn as nn
import torch.nn.functional as tf

from .common import conv, upconv


class PoseNet(nn.Module):

    def __init__(self, nb_ref_imgs=1, in_ch=3, use_bn=False):
        super(PoseNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        self.conv0 = conv(in_ch*(1+self.nb_ref_imgs), in_ch*(1+self.nb_ref_imgs), kernel_size=3, stride=2, use_bn=use_bn)
        self.conv1 = conv(in_ch*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7, stride=2, use_bn=use_bn)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=2, use_bn=use_bn)
        self.conv3 = conv(conv_planes[1], conv_planes[2], stride=2, use_bn=use_bn)
        self.conv4 = conv(conv_planes[2], conv_planes[3], stride=2, use_bn=use_bn)
        self.conv5 = conv(conv_planes[3], conv_planes[4], stride=2, use_bn=use_bn)
        self.conv6 = conv(conv_planes[4], conv_planes[5], stride=2, use_bn=use_bn)
        self.conv7 = conv(conv_planes[5], conv_planes[6], stride=2, use_bn=use_bn)

        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0, stride=1)

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

        if self.output_exp:
            upconv_planes = [256, 256, 128, 64, 32, 16]
            # self.upconv5 = upconv(conv_planes[4],   upconv_planes[0], kernel_size=4, stride=2, use_bn=use_bn)
            # self.upconv4 = upconv(upconv_planes[0], upconv_planes[1], kernel_size=4, stride=2, use_bn=use_bn)
            # self.upconv3 = upconv(upconv_planes[1], upconv_planes[2], kernel_size=4, stride=2, use_bn=use_bn)
            # self.upconv2 = upconv(upconv_planes[2], upconv_planes[3], kernel_size=4, stride=2, use_bn=use_bn)
            # self.upconv1 = upconv(upconv_planes[3], upconv_planes[4], kernel_size=4, stride=2, use_bn=use_bn)
            self.upconv6 = upconv(conv_planes[5], upconv_planes[0], kernel_size=4, stride=2)
            self.upconv5 = upconv(upconv_planes[0]+conv_planes[4], upconv_planes[1], kernel_size=4, stride=2)
            self.upconv4 = upconv(upconv_planes[1]+conv_planes[3], upconv_planes[2], kernel_size=4, stride=2)
            self.upconv3 = upconv(upconv_planes[2]+conv_planes[2], upconv_planes[3], kernel_size=4, stride=2)
            self.upconv2 = upconv(upconv_planes[3]+conv_planes[1], upconv_planes[4], kernel_size=4, stride=2)
            self.upconv1 = upconv(upconv_planes[4]+conv_planes[0], upconv_planes[5], kernel_size=4, stride=2)

            self.predict_mask5 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask4 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(upconv_planes[5], self.nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

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
            # out_upconv5 = self.upconv5(out_conv5  )[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
            # out_upconv4 = self.upconv4(out_upconv5)[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
            # out_upconv3 = self.upconv3(out_upconv4)[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
            # out_upconv2 = self.upconv2(out_upconv3)[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
            # out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:x.size(2), 0:x.size(3)]
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

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4, exp_mask5], pose, pose_feats
        else:
            return exp_mask1, pose


class MotionNet(nn.Module):
    def __init__(self, nb_ref_imgs=1, use_bn=False):
        super(MotionNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs

        conv_planes = [16, 32, 64, 128, 256, 256, 256, 256]
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7, stride=2)
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5, stride=2)
        self.conv3 = conv(conv_planes[1], conv_planes[2], stride=2)
        self.conv4 = conv(conv_planes[2], conv_planes[3], stride=2)
        self.conv5 = conv(conv_planes[3], conv_planes[4], stride=2)
        self.conv6 = conv(conv_planes[4], conv_planes[5], stride=2)

        upconv_planes = [256, 256, 128, 64, 32, 16]
        self.deconv6 = upconv(conv_planes[5], upconv_planes[0], kernel_size=4, stride=2)
        self.deconv5 = upconv(upconv_planes[0]+conv_planes[4], upconv_planes[1], kernel_size=4, stride=2)
        self.deconv4 = upconv(upconv_planes[1]+conv_planes[3], upconv_planes[2], kernel_size=4, stride=2)
        self.deconv3 = upconv(upconv_planes[2]+conv_planes[2], upconv_planes[3], kernel_size=4, stride=2)
        self.deconv2 = upconv(upconv_planes[3]+conv_planes[1], upconv_planes[4], kernel_size=4, stride=2)
        self.deconv1 = upconv(upconv_planes[4]+conv_planes[0], upconv_planes[5], kernel_size=4, stride=2)

        # self.pred_mask6 = conv(upconv_planes[0], self.nb_ref_imgs, use_relu=False)
        self.pred_mask5 = conv(upconv_planes[1], self.nb_ref_imgs, use_relu=False)
        self.pred_mask4 = conv(upconv_planes[2], self.nb_ref_imgs, use_relu=False)
        self.pred_mask3 = conv(upconv_planes[3], self.nb_ref_imgs, use_relu=False)
        self.pred_mask2 = conv(upconv_planes[4], self.nb_ref_imgs, use_relu=False)
        self.pred_mask1 = conv(upconv_planes[5], self.nb_ref_imgs, use_relu=False)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def init_mask_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        # for module in [self.pred_mask1, self.pred_mask2, self.pred_mask3, self.pred_mask4, self.pred_mask5, self.pred_mask6]:
        for module in [self.pred_mask1, self.pred_mask2, self.pred_mask3, self.pred_mask4, self.pred_mask5]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)

        out_upconv6 = self.deconv6(out_conv6)
        out_upconv5 = self.deconv5(torch.cat((out_upconv6, out_conv5), 1))
        out_upconv4 = self.deconv4(torch.cat((out_upconv5, out_conv4), 1))
        out_upconv3 = self.deconv3(torch.cat((out_upconv4, out_conv3), 1))
        out_upconv2 = self.deconv2(torch.cat((out_upconv3, out_conv2), 1))
        out_upconv1 = self.deconv1(torch.cat((out_upconv2, out_conv1), 1))

        # exp_mask6 = torch.sigmoid(self.pred_mask6(out_upconv6))
        exp_mask5 = torch.sigmoid(self.pred_mask5(out_upconv5))
        exp_mask4 = torch.sigmoid(self.pred_mask4(out_upconv4))
        exp_mask3 = torch.sigmoid(self.pred_mask3(out_upconv3))
        exp_mask2 = torch.sigmoid(self.pred_mask2(out_upconv2))
        exp_mask1 = torch.sigmoid(self.pred_mask1(out_upconv1))

        if self.training:
            return exp_mask1, exp_mask2, exp_mask3, exp_mask4, exp_mask5
        else:
            return exp_mask1


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


class FlowDispPoseDecoder(nn.Module):
    def __init__(self, ch_in, num_refs=1, use_mask=True, use_bn=False):
        super(FlowDispPoseDecoder, self).__init__()

        self.use_mask = use_mask

        self.convs = nn.Sequential(
            conv(ch_in, 256, use_bn=use_bn),
            conv(256, 128, use_bn=use_bn),
            conv(128, 128, use_bn=use_bn),
            conv(128, 96, use_bn=use_bn),
            conv(96, 64, use_bn=use_bn),
            conv(64, 32, use_bn=use_bn)
        )

        # self.conv_sf = conv(32, 3, isReLU=False)
        # self.conv_d1 = conv(32, 1, isReLU=False)
        self.conv_sf = conv(32, 3, use_relu=False)
        self.conv_d1 = conv(32, 1, use_relu=False)
        self.convs_pose = nn.Sequential(
            conv(32, 32, use_bn=use_bn),
            conv(32, 16, use_bn=use_bn),
            conv(16, num_refs * 6, use_relu=False)
        )

        if use_mask:
            self.convs_mask = nn.Sequential(
                conv(32, 32, use_bn=use_bn),
                conv(32, 16, use_bn=use_bn),
                conv(16, 1, use_relu=False),
                torch.nn.Sigmoid()
            )

    def forward(self, x):
        x_out = self.convs(x)
        sf = self.conv_sf(x_out)
        disp1 = self.conv_d1(x_out)
        pose_out = self.convs_pose(x_out)
        pred_pose = pose_out.mean(3).mean(2) * 0.01

        if self.use_mask:
            mask = self.convs_mask(x_out)

        return x_out, sf, disp1, mask, pred_pose, pose_out


class JointContextNetwork(nn.Module):
    def __init__(self, ch_in, num_refs = 1, use_mask=True, use_bn=False):
        super(JointContextNetwork, self).__init__()

        self.use_mask = use_mask

        self.convs = nn.Sequential(
            conv(ch_in, 128, 3, 1, 1, use_bn=use_bn),
            conv(128, 128, 3, 1, 2, use_bn=use_bn),
            conv(128, 128, 3, 1, 4, use_bn=use_bn),
            conv(128, 96, 3, 1, 8, use_bn=use_bn),
            conv(96, 64, 3, 1, 16, use_bn=use_bn),
            conv(64, 32, 3, 1, 1, use_bn=use_bn)
        )
        # self.conv_sf = conv(32, 3, isReLU=False)
        # self.conv_d1 = nn.Sequential(
        #     conv(32, 1, isReLU=False), 
        #     torch.nn.Sigmoid()
        # )
        self.conv_sf = conv(32, 3, use_relu=False)
        self.conv_d1 = nn.Sequential(
            conv(32, 1, use_relu=False), 
            torch.nn.Sigmoid()
        )
        self.convs_pose = nn.Sequential(
            conv(32, 32, use_bn=use_bn),
            conv(32, 16, use_bn=use_bn),
            conv(16, num_refs * 6, use_relu=False)
        )

        if use_mask:
            self.convs_mask = nn.Sequential(
                conv(32, 32, use_bn=use_bn),
                conv(32, 16, use_bn=use_bn),
                conv(16, 1, use_relu=False),
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

        return sf, disp1, mask, pred_pose
