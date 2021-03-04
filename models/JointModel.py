from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from sys import exit

from .correlation_package.correlation import Correlation
from .encoders import PoseBottleNeck, PoseBottleNeck3D
from .modules_sceneflow import get_grid, WarpingLayer_SF, WarpingLayer_Pose
from .modules_sceneflow import initialize_msra, upsample_outputs_as
from .joint_decoders import JointDecoder, JointContextNetwork
from .encoders import PWCEncoder, ResnetEncoder

from utils.inverse_warp import pose_vec2mat
from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing, pose_process_flow
from utils.helpers import Warp_SceneFlow, Warp_Pose, add_pose, invert_pose
from .common import UpConv


def make_leaky(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.LeakyReLU(inplace=True))
        else:
            make_leaky(child)


class JointModel(nn.Module):
    def __init__(self, args):
        super(JointModel, self).__init__()

        self.args = args
        self.use_mask = args.train_exp_mask or args.train_census_mask
        self.search_range = 4
        self.output_level = 4
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        if self.args.encoder_name == "resnet":
            self.feature_pyramid_extractor = ResnetEncoder(args, num_layers=18, pretrained=args.pt_encoder, num_input_images=1)
            self.num_chs = self.feature_pyramid_extractor.num_ch_enc
            make_leaky(self.feature_pyramid_extractor)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif self.args.encoder_name == 'pwc':
            self.num_chs = [3, 32, 64, 96, 128, 192, 256]
            self.feature_pyramid_extractor = PWCEncoder(self.num_chs, use_bn=args.use_bn)
        else:
            raise NotImplementedError

        if args.use_bottleneck:
            self.bottlenecks = nn.ModuleList()
            bottleneck_out_ch = 96

        self.warping_layer_sf = WarpingLayer_SF()
        self.warping_layer_pose = WarpingLayer_Pose()
        
        self.flow_estimators = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.dim_corr = (self.search_range * 2 + 1) ** 2
        
        self.out_ch_size = 32

        for l, ch in enumerate(self.num_chs[::-1]):
            if l > self.output_level:
                break
            if l == 0:
                num_ch_in = self.dim_corr + ch
                if args.use_bottleneck:
                    num_ch_in += bottleneck_out_ch*2
                if args.use_pose_corr:
                    num_ch_in += self.dim_corr
                if not (args.use_bottleneck or args.use_pose_corr):
                    num_ch_in += ch
            else:
                num_ch_in = self.dim_corr + ch + self.out_ch_size + 3 + 1 + 6 + 1
                if args.use_bottleneck:
                    num_ch_in += bottleneck_out_ch*2
                if args.use_pose_corr:
                    num_ch_in += self.dim_corr
                if not (args.use_bottleneck or args.use_pose_corr):
                    num_ch_in += ch
                self.upconv_layers.append(UpConv(self.out_ch_size, self.out_ch_size, 3, 2, use_bn=args.use_bn))

            layer_sf = JointDecoder(args, num_ch_in, use_bn=args.use_bn)
            self.flow_estimators.append(layer_sf)
            if args.use_bottleneck:
                bottleneck = PoseBottleNeck3D(in_ch=ch)
                self.bottlenecks.append(bottleneck)

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks = JointContextNetwork(args, self.out_ch_size + 3 + 1 + 6 + 1, use_bn=args.use_bn)
        self.sigmoid = torch.nn.Sigmoid()

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.LeakyReLU):
                pass

            elif isinstance(layer, nn.Sequential):
                pass

    def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
        output_dict = {}

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw)
        x2_pyramid = self.feature_pyramid_extractor(x2_raw)

        x1_pyramid = [input_dict['input_l1_aug']] + x1_pyramid
        x2_pyramid = [input_dict['input_l2_aug']] + x2_pyramid

        # outputs
        sceneflows_f = []
        sceneflows_b = []
        poses_f = []
        poses_b = []
        disps_1 = []
        disps_2 = []
        masks_1 = []
        masks_2 = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid[::-1], x2_pyramid[::-1])):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1

                x2_warp_pose = x2
                x1_warp_pose = x1

            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
                disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
                pose_f_out = interpolate2d_as(pose_f_out, x1, mode="bilinear")
                pose_b_out = interpolate2d_as(pose_b_out, x1, mode="bilinear")
                mask_l1 = interpolate2d_as(mask_l1, x1, mode="bilinear")
                mask_l2 = interpolate2d_as(mask_l2, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)

                x2_warp = self.warping_layer_sf(x2, flow_f, disp_l1, k1, input_dict['aug_size'])
                x1_warp = self.warping_layer_sf(x1, flow_b, disp_l2, k2, input_dict['aug_size'])

                if self.args.use_pose_corr:
                    x2_warp_pose = self.warping_layer_pose(x2, pose_mat_f, disp_l1, k1, input_dict['aug_size'])
                    x1_warp_pose = self.warping_layer_pose(x1, pose_mat_b, disp_l2, k2, input_dict['aug_size'])

            if self.args.use_bottleneck:
                aux_in_f = torch.cat([x1.unsqueeze(dim=2), x2_warp.unsqueeze(dim=2)], dim=2)
                aux_in_b = torch.cat([x2.unsqueeze(dim=2), x1_warp.unsqueeze(dim=2)], dim=2)
                aux_f = self.bottlenecks[l](aux_in_f)
                aux_b = self.bottlenecks[l](aux_in_b)
                aux_f = torch.cat([aux_f[:, :, 0, :, :], aux_f[:, :, 1, :, :]], dim=1)
                aux_b = torch.cat([aux_b[:, :, 0, :, :], aux_b[:, :, 1, :, :]], dim=1)
            
            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            if self.args.use_pose_corr:
                pose_out_corr_f = Correlation.apply(x1, x2_warp_pose, self.corr_params)
                pose_out_corr_b = Correlation.apply(x2, x1_warp_pose, self.corr_params)
                pose_out_corr_relu_f = self.leakyRELU(pose_out_corr_f)
                pose_out_corr_relu_b = self.leakyRELU(pose_out_corr_b)

            # monosf estimator
            if l == 0:
                input_f = torch.cat([out_corr_relu_f, x1], dim=1)
                input_b = torch.cat([out_corr_relu_b, x2], dim=1)
                if self.args.use_bottleneck:
                    input_f = torch.cat([input_f, aux_f], dim=1)
                    input_b = torch.cat([input_b, aux_b], dim=1)
                if self.args.use_pose_corr:
                    input_f = torch.cat([input_f, pose_out_corr_relu_f], dim=1)
                    input_b = torch.cat([input_b, pose_out_corr_relu_b], dim=1)
                if not (self.args.use_bottleneck or self.args.use_pose_corr):
                    input_f = torch.cat([input_f, x2])
                    input_b = torch.cat([input_b, x1])

                x1_out, flow_f, disp_l1, mask_l1, pose_f, pose_f_out = self.flow_estimators[l](input_f)
                x2_out, flow_b, disp_l2, mask_l2,      _, pose_b_out = self.flow_estimators[l](input_b)

                pose_mat_f = pose_vec2mat(pose_f)
                pose_mat_b = invert_pose(pose_mat_f)
            else:
                input_f = torch.cat([out_corr_relu_f, x1, x1_out, flow_f, disp_l1, mask_l1, pose_f_out], dim=1)
                input_b = torch.cat([out_corr_relu_b, x2, x2_out, flow_b, disp_l2, mask_l2, pose_b_out], dim=1)
                if self.args.use_bottleneck:
                    input_f = torch.cat([input_f, aux_f], dim=1)
                    input_b = torch.cat([input_b, aux_b], dim=1)
                if self.args.use_pose_corr:
                    input_f = torch.cat([input_f, pose_out_corr_relu_f], dim=1)
                    input_b = torch.cat([input_b, pose_out_corr_relu_b], dim=1)
                if not (self.args.use_bottleneck or self.args.use_pose_corr):
                    input_f = torch.cat([input_f, x2], dim=1)
                    input_b = torch.cat([input_b, x1], dim=1)

                x1_out, flow_f_res, disp_l1, mask_l1, pose_f_res, pose_f_out = self.flow_estimators[l](input_f)
                x2_out, flow_b_res, disp_l2, mask_l2,          _, pose_b_out = self.flow_estimators[l](input_b)

                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

                pose_mat_f_res = pose_vec2mat(pose_f_res)
                pose_mat_b_res = invert_pose(pose_mat_f_res)

                pose_mat_f = add_pose(pose_mat_f, pose_mat_f_res)
                pose_mat_b = add_pose(pose_mat_b, pose_mat_b_res)

            # upsampling or post-processing
            if l != self.output_level:
                disp_l1 = self.sigmoid(disp_l1) * 0.3
                disp_l2 = self.sigmoid(disp_l2) * 0.3
                mask_l1 = self.sigmoid(mask_l1)
                mask_l2 = self.sigmoid(mask_l2)
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                masks_1.append(mask_l1)
                masks_2.append(mask_l2)
                poses_f.append(pose_mat_f)
                poses_b.append(pose_mat_b)
            else:
                flow_res_f, disp_l1, pose_f_res, mask_l1 = self.context_networks(torch.cat([x1_out, flow_f, disp_l1, pose_f_out, mask_l1], dim=1))
                flow_res_b, disp_l2,          _, mask_l2 = self.context_networks(torch.cat([x2_out, flow_b, disp_l2, pose_b_out, mask_l2], dim=1))

                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b

                pose_mat_f_res = pose_vec2mat(pose_f_res)
                pose_mat_b_res = invert_pose(pose_mat_f_res)

                pose_mat_f = add_pose(pose_mat_f, pose_mat_f_res)
                pose_mat_b = add_pose(pose_mat_b, pose_mat_b_res)

                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)
                disps_1.append(disp_l1)
                disps_2.append(disp_l2)
                masks_1.append(mask_l1)
                masks_2.append(mask_l2)
                poses_f.append(pose_mat_f)
                poses_b.append(pose_mat_b)

                break
        
        x1_rev = x1_pyramid
        
        output_dict['flows_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flows_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
        output_dict['disps_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
        output_dict['disps_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
        output_dict['masks_l1'] = upsample_outputs_as(masks_1[::-1], x1_rev)
        output_dict['masks_l2'] = upsample_outputs_as(masks_2[::-1], x1_rev)
        output_dict["pose_f"] = poses_f[::-1]
        output_dict["pose_b"] = poses_b[::-1]
        # output_dict['feats_l1'] = upsample_outputs_as(x1_pyramid[1:], x1_pyramid)
        # output_dict['feats_l2'] = upsample_outputs_as(x2_pyramid[1:], x2_pyramid)

        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self.args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flows_f'])):
                output_dict_r['flows_f'][ii] = flow_horizontal_flip(output_dict_r['flows_f'][ii])
                output_dict_r['flows_b'][ii] = flow_horizontal_flip(output_dict_r['flows_b'][ii])
                output_dict_r['disps_l1'][ii] = torch.flip(output_dict_r['disps_l1'][ii], [3])
                output_dict_r['disps_l2'][ii] = torch.flip(output_dict_r['disps_l2'][ii], [3])
                output_dict_r['masks_l1'][ii] = torch.flip(output_dict_r['masks_l1'][ii], [3])
                output_dict_r['masks_l2'][ii] = torch.flip(output_dict_r['masks_l2'][ii], [3])

            output_dict['output_dict_r'] = output_dict_r

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self.args.evaluation or (not self.training):

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flows_f_pp = []
            flows_b_pp = []
            disps_l1_pp = []
            disps_l2_pp = []
            masks_l1_pp = []
            masks_l2_pp = []
            pose_flows_f_pp = []
            pose_flows_b_pp = []
            rigidity_masks_l1 = []
            rigidity_masks_l2 = []

            for ii in range(0, len(output_dict_flip['flows_f'])):

                flow_f_pp = post_processing(output_dict['flows_f'][ii], flow_horizontal_flip(output_dict_flip['flows_f'][ii]))
                flow_b_pp = post_processing(output_dict['flows_b'][ii], flow_horizontal_flip(output_dict_flip['flows_b'][ii]))
                flows_f_pp.append(flow_f_pp)
                flows_b_pp.append(flow_b_pp)
                disps_l1_pp.append(post_processing(output_dict['disps_l1'][ii], torch.flip(output_dict_flip['disps_l1'][ii], [3])))
                disps_l2_pp.append(post_processing(output_dict['disps_l2'][ii], torch.flip(output_dict_flip['disps_l2'][ii], [3])))
                masks_l1_pp.append(post_processing(output_dict['masks_l1'][ii], torch.flip(output_dict_flip['masks_l1'][ii], [3])))
                masks_l2_pp.append(post_processing(output_dict['masks_l2'][ii], torch.flip(output_dict_flip['masks_l2'][ii], [3])))

                K1 = input_dict['input_k_l1_aug']
                K2 = input_dict['input_k_l2_aug']
                aug_size = input_dict['aug_size']

                pose_flow_f_pp, rigidity_mask_l1 = pose_process_flow(self.args, output_dict['pose_f'][ii], flow_f_pp, disps_l1_pp[ii], masks_l1_pp[ii], K1, aug_size, self.args.mask_thresh, self.args.flow_diff_thresh)
                pose_flow_b_pp, rigidity_mask_l2 = pose_process_flow(self.args, output_dict['pose_b'][ii], flow_b_pp, disps_l2_pp[ii], masks_l2_pp[ii], K2, aug_size, self.args.mask_thresh, self.args.flow_diff_thresh)
                pose_flows_f_pp.append(pose_flow_f_pp)
                pose_flows_b_pp.append(pose_flow_b_pp)
                rigidity_masks_l1.append(rigidity_mask_l1)
                rigidity_masks_l2.append(rigidity_mask_l2)

            output_dict['flows_f_pp'] = flows_f_pp
            output_dict['flows_b_pp'] = flows_b_pp
            output_dict['disps_l1_pp'] = disps_l1_pp
            output_dict['disps_l2_pp'] = disps_l2_pp
            output_dict['pose_flows_f_pp'] = pose_flows_f_pp
            output_dict['pose_flows_b_pp'] = pose_flows_b_pp
            output_dict['masks_l1_pp'] = masks_l1_pp
            output_dict['masks_l2_pp'] = masks_l2_pp
            output_dict['rigidity_masks_l1_pp'] = rigidity_masks_l1
            output_dict['rigidity_masks_l2_pp'] = rigidity_masks_l2

        return output_dict