# from sys import exit

# import torch
# import torch.nn as nn
# import torch.nn.functional as tf

# from .encoders import ResnetEncoder, FeatureExtractor
# from .disp_decoders import DispDecoder
# from .pose_decoders import PoseExpNet, PoseNet, PoseDecoder
# from .sf_decoders import SFDecoder
# from .mask_decoders import MaskDecoder
# from .joint_decoders import JointDecoder, JointContextNetwork
# from .correlation_package.correlation import Correlation

# from .common import Conv, UpConv
# from utils.helpers import invert_pose, Warp_SceneFlow, Warp_Pose, add_pose
# from utils.sceneflow_util import flow_horizontal_flip, post_processing
# from utils.interpolation import interpolate2d_as
# from .modules_sceneflow import upsample_outputs_as

# from utils.inverse_warp import pose_vec2mat


# class JointModel(nn.Module):
#     def __init__(self, args):
#         super(JointModel, self).__init__()

#         self.args = args
#         self.search_range = 4
#         self.output_level = 4
#         self.num_levels = 7
        
#         self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

#         # self.encoder = ResnetEncoder(args, num_layers=18, pretrained=args.pt_encoder, num_input_images=1)
#         # self.encoder_chs = self.encoder.num_ch_enc
#         self.encoder_chs = [3, 32, 64, 96, 128, 192, 256]
#         self.encoder = FeatureExtractor(num_chs=self.encoder_chs)

#         self.warping_layer_sf = Warp_SceneFlow()
#         self.warping_layer_pose = Warp_Pose()
        
#         self.flow_estimators = nn.ModuleList()
#         self.upconv_layers = nn.ModuleList()

#         self.dim_corr = (self.search_range * 2 + 1) ** 2
        
#         self.out_ch_size = 32 

#         for l, ch in enumerate(self.encoder_chs[::-1]):
#             if l > self.output_level:
#                 break
#             if l == 0:
#                 num_ch_in = self.dim_corr + self.dim_corr + ch
#             else:
#                 num_ch_in = self.dim_corr + self.dim_corr + ch + self.out_ch_size + 3 + 1 + 6 + 1
#                 self.upconv_layers.append(UpConv(self.out_ch_size, self.out_ch_size, 3, 2))

#             layer_sf = JointDecoder(args, num_ch_in)

#             self.flow_estimators.append(layer_sf)

#         self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
#         self.context_networks = JointContextNetwork(args, self.out_ch_size + 3 + 1 + 6 + 1, use_bn=args.use_bn)

#         self.sigmoid = torch.nn.Sigmoid()

#         self.initialize_weights()

#     def initialize_weights(self):
#         for layer in self.modules():
#             if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
#                 # nn.init.kaiming_normal_(layer.weight)
#                 if layer.bias is not None:
#                     nn.init.constant_(layer.bias, 0)

#             elif isinstance(layer, nn.LeakyReLU):
#                 pass

#             elif isinstance(layer, nn.Sequential):
#                 pass

#     def run_pwc(self, input_dict, x1_raw, x2_raw, k1, k2):
            
#         output_dict = {}

#         # on the bottom level are original images
#         x1_pyramid = self.encoder(x1_raw)
#         x2_pyramid = self.encoder(x2_raw)

#         x1_pyramid = [input_dict['input_l1_aug']] + x1_pyramid
#         x2_pyramid = [input_dict['input_l2_aug']] + x2_pyramid

#         # outputs
#         sceneflows_f = []
#         sceneflows_b = []
#         poses_f = []
#         poses_b = []
#         disps_1 = []
#         disps_2 = []
#         masks_1 = []
#         masks_2 = []

#         for l, (x1, x2) in enumerate(zip(x1_pyramid[::-1], x2_pyramid[::-1])):

#             # warping
#             if l == 0:
#                 x2_warp = x2
#                 x1_warp = x1

#                 x2_warp_pose = x2
#                 x1_warp_pose = x1

#             else:
#                 flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
#                 flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
#                 disp_l1 = interpolate2d_as(disp_l1, x1, mode="bilinear")
#                 disp_l2 = interpolate2d_as(disp_l2, x1, mode="bilinear")
#                 pose_f_out = interpolate2d_as(pose_f_out, x1, mode="bilinear")
#                 pose_b_out = interpolate2d_as(pose_b_out, x1, mode="bilinear")
#                 mask_l1 = interpolate2d_as(mask_l1, x1, mode="bilinear")
#                 mask_l2 = interpolate2d_as(mask_l2, x1, mode="bilinear")
#                 x1_out = self.upconv_layers[l-1](x1_out)
#                 x2_out = self.upconv_layers[l-1](x2_out)

#                 x2_warp = self.warping_layer_sf(x2, flow_f, disp_l1, k1, input_dict['aug_size'])
#                 x1_warp = self.warping_layer_sf(x1, flow_b, disp_l2, k2, input_dict['aug_size'])

#                 x2_warp_pose = self.warping_layer_pose(x2, pose_mat_f, disp_l1, k1, input_dict['aug_size'])
#                 x1_warp_pose = self.warping_layer_pose(x1, pose_mat_b, disp_l2, k2, input_dict['aug_size'])

#             # correlation
#             out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
#             out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
#             out_corr_relu_f = self.leakyRELU(out_corr_f)
#             out_corr_relu_b = self.leakyRELU(out_corr_b)

#             out_corr_pose_f = Correlation.apply(x1, x2_warp_pose, self.corr_params)
#             out_corr_pose_b = Correlation.apply(x2, x1_warp_pose, self.corr_params)
#             out_corr_pose_relu_f = self.leakyRELU(out_corr_pose_f)
#             out_corr_pose_relu_b = self.leakyRELU(out_corr_pose_b)

#             # monosf estimator
#             if l == 0:
#                 x1_out, flow_f, disp_l1, mask_l1, pose_f, pose_f_out = self.flow_estimators[l](torch.cat([out_corr_relu_f, out_corr_pose_relu_f, x1], dim=1))
#                 x2_out, flow_b, disp_l2, mask_l2, _, pose_b_out = self.flow_estimators[l](torch.cat([out_corr_relu_b, out_corr_pose_relu_b, x2], dim=1))
#                 pose_mat_f = pose_vec2mat(pose_f)
#                 pose_mat_b = invert_pose(pose_mat_f)
#                 # pose_mat_b = pose_vec2mat(pose_b)
#             else:
#                 x1_out, flow_f_res, disp_l1, mask_l1, pose_f_res, pose_f_out = self.flow_estimators[l](torch.cat([
#                     out_corr_relu_f, out_corr_pose_relu_f, x1, x1_out, flow_f, disp_l1, mask_l1, pose_f_out], dim=1))
#                 x2_out, flow_b_res, disp_l2, mask_l2, _, pose_b_out = self.flow_estimators[l](torch.cat([
#                     out_corr_relu_b, out_corr_pose_relu_b, x2, x2_out, flow_b, disp_l2, mask_l2, pose_b_out], dim=1))

#                 flow_f = flow_f + flow_f_res
#                 flow_b = flow_b + flow_b_res

#                 pose_mat_f_res = pose_vec2mat(pose_f_res)
#                 pose_mat_b_res = invert_pose(pose_mat_f_res)
#                 pose_mat_f = add_pose(pose_mat_f, pose_mat_f_res)
#                 pose_mat_b = add_pose(pose_mat_b, pose_mat_b_res)
#                 # pose_mat_b = add_pose(pose_mat_b, pose_b_res)

#             # upsampling or post-processing
#             if l != self.output_level:
#                 disp_l1 = self.sigmoid(disp_l1) * 0.3
#                 disp_l2 = self.sigmoid(disp_l2) * 0.3
#                 mask_l1 = self.sigmoid(mask_l1)
#                 mask_l2 = self.sigmoid(mask_l2)
#                 sceneflows_f.append(flow_f)
#                 sceneflows_b.append(flow_b)                
#                 disps_1.append(disp_l1)
#                 disps_2.append(disp_l2)
#                 masks_1.append(mask_l1)
#                 masks_2.append(mask_l2)
#                 poses_f.append(pose_mat_f)
#                 poses_b.append(pose_mat_b)

#             else:
#                 flow_res_f, disp_l1, pose_f_res, mask_l1 = self.context_networks(torch.cat([x1_out, flow_f, disp_l1, pose_f_out, mask_l1], dim=1))
#                 flow_res_b, disp_l2, _, mask_l2 = self.context_networks(torch.cat([x2_out, flow_b, disp_l2, pose_b_out, mask_l2], dim=1))
#                 flow_f = flow_f + flow_res_f
#                 flow_b = flow_b + flow_res_b

#                 pose_mat_f_res = pose_vec2mat(pose_f_res)
#                 pose_mat_b_res = invert_pose(pose_mat_f_res)
#                 pose_mat_f = add_pose(pose_mat_f, pose_mat_f_res)
#                 pose_mat_b = add_pose(pose_mat_b, pose_mat_b_res)

#                 sceneflows_f.append(flow_f)
#                 sceneflows_b.append(flow_b)
#                 disps_1.append(disp_l1)
#                 disps_2.append(disp_l2)
#                 masks_1.append(mask_l1)
#                 masks_2.append(mask_l2)
#                 poses_f.append(pose_mat_f)
#                 poses_b.append(pose_mat_b)

#                 break

#         x1_rev = x1_pyramid

#         output_dict['flows_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
#         output_dict['flows_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)
#         output_dict['disps_l1'] = upsample_outputs_as(disps_1[::-1], x1_rev)
#         output_dict['disps_l2'] = upsample_outputs_as(disps_2[::-1], x1_rev)
#         if self.args.use_mask:
#             output_dict['masks_l1'] = upsample_outputs_as(masks_1[::-1], x1_rev)
#             output_dict['masks_l2'] = upsample_outputs_as(masks_2[::-1], x1_rev)
#         output_dict["pose_f"] = poses_f[::-1]
#         output_dict["pose_b"] = poses_b[::-1]

#         return output_dict


#     def forward(self, input_dict):

#         output_dict = {}

#         ## Left
#         output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

#         ## Right
#         ## ss: train val 
#         ## ft: train 
#         if self.training or (not self.args.evaluation):
#             input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
#             input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
#             k_r1_flip = input_dict["input_k_r1_flip_aug"]
#             k_r2_flip = input_dict["input_k_r2_flip_aug"]

#             output_dict_r = self.run_pwc(input_dict, input_r1_flip, input_r2_flip, k_r1_flip, k_r2_flip)

#             for ii in range(0, len(output_dict_r['flows_f'])):
#                 output_dict_r['flows_f'][ii] = flow_horizontal_flip(output_dict_r['flows_f'][ii])
#                 output_dict_r['flows_b'][ii] = flow_horizontal_flip(output_dict_r['flows_b'][ii])
#                 output_dict_r['disps_l1'][ii] = torch.flip(output_dict_r['disps_l1'][ii], [3])
#                 output_dict_r['disps_l2'][ii] = torch.flip(output_dict_r['disps_l2'][ii], [3])
#                 if self.args.use_mask:
#                     output_dict_r['masks_l1'][ii] = torch.flip(output_dict_r['masks_l1'][ii], [3])
#                     output_dict_r['masks_l2'][ii] = torch.flip(output_dict_r['masks_l2'][ii], [3])

#             output_dict['output_dict_r'] = output_dict_r

#         ## Post Processing 
#         ## ss:           eval
#         ## ft: train val eval
#         if self.args.evaluation:

#             input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
#             input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
#             k_l1_flip = input_dict["input_k_l1_flip_aug"]
#             k_l2_flip = input_dict["input_k_l2_flip_aug"]

#             output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

#             flow_f_pp = []
#             flow_b_pp = []
#             disp_l1_pp = []
#             disp_l2_pp = []
#             mask_l1_pp = []
#             mask_l2_pp = []

#             for ii in range(0, len(output_dict_flip['flows_f'])):

#                 flow_f_pp.append(post_processing(output_dict['flows_f'][ii], flow_horizontal_flip(output_dict_flip['flows_f'][ii])))
#                 flow_b_pp.append(post_processing(output_dict['flows_b'][ii], flow_horizontal_flip(output_dict_flip['flows_b'][ii])))
#                 disp_l1_pp.append(post_processing(output_dict['disps_l1'][ii], torch.flip(output_dict_flip['disps_l1'][ii], [3])))
#                 disp_l2_pp.append(post_processing(output_dict['disps_l2'][ii], torch.flip(output_dict_flip['disps_l2'][ii], [3])))
#                 mask_l1_pp.append(post_processing(output_dict['masks_l1'][ii], torch.flip(output_dict_flip['masks_l1'][ii], [3])))
#                 mask_l2_pp.append(post_processing(output_dict['masks_l2'][ii], torch.flip(output_dict_flip['masks_l2'][ii], [3])))

#             output_dict['flow_f_pp'] = flow_f_pp
#             output_dict['flow_b_pp'] = flow_b_pp
#             output_dict['disp_l1_pp'] = disp_l1_pp
#             output_dict['disp_l2_pp'] = disp_l2_pp
#             output_dict['mask_l1_pp'] = mask_l1_pp
#             output_dict['mask_l2_pp'] = mask_l2_pp

#         return output_dict

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as tf
import logging

from sys import exit

from .correlation_package.correlation import Correlation

from .modules_sceneflow import get_grid, WarpingLayer_SF, WarpingLayer_Pose
from .modules_sceneflow import initialize_msra, upsample_outputs_as
from .joint_decoders import JointDecoder, JointContextNetwork
from .encoders import FeatureExtractor, ResnetEncoder

from utils.inverse_warp import pose_vec2mat
from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import flow_horizontal_flip, intrinsic_scale, get_pixelgrid, post_processing
from utils.helpers import Warp_SceneFlow, Warp_Pose, add_pose, invert_pose
from.common import UpConv


class JointModel(nn.Module):
    def __init__(self, args):
        super(JointModel, self).__init__()

        self._args = args
        self.search_range = 4
        self.output_level = 5
        self.num_levels = 7
        
        self.leakyRELU = nn.LeakyReLU(0.1, inplace=True)

        self.num_chs = [3, 32, 64, 96, 128, 192, 256]
        self.feature_pyramid_extractor = FeatureExtractor(self.num_chs, use_bn=args.use_bn)

        # self.feature_pyramid_extractor = ResnetEncoder(args, num_layers=18, pretrained=args.pt_encoder, num_input_images=1)
        # self.num_chs = self.feature_pyramid_extractor.num_ch_enc

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
                num_ch_in = self.dim_corr + ch + ch
            else:
                num_ch_in = self.dim_corr + ch + ch + self.out_ch_size + 3 + 1 + 6 + 1
                self.upconv_layers.append(UpConv(self.out_ch_size, self.out_ch_size, 3, 2))

            layer_sf = JointDecoder(args, num_ch_in, use_bn=args.use_bn)

            self.flow_estimators.append(layer_sf)

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.context_networks = JointContextNetwork(args, self.out_ch_size + 3 + 1 + 6 + 1, use_bn=args.use_bn)
        self.sigmoid = torch.nn.Sigmoid()

        self.initialize_weights()

    def initialize_weights(self):
        logging.info("Initializing weights")
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(layer.weight)
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

                x2_warp_pose = self.warping_layer_pose(x2, pose_mat_f, disp_l1, k1, input_dict['aug_size'])
                x1_warp_pose = self.warping_layer_pose(x1, pose_mat_b, disp_l2, k2, input_dict['aug_size'])

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # out_corr_pose_f = Correlation.apply(x1, x2_warp_pose, self.corr_params)
            # out_corr_pose_b = Correlation.apply(x2, x1_warp_pose, self.corr_params)
            # out_corr_pose_relu_f = self.leakyRELU(out_corr_pose_f)
            # out_corr_pose_relu_b = self.leakyRELU(out_corr_pose_b)

            # monosf estimator
            if l == 0:
                x1_out, flow_f, disp_l1, mask_l1, pose_f, pose_f_out = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x2_warp_pose], dim=1))
                x2_out, flow_b, disp_l2, mask_l2,      _, pose_b_out = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x1_warp_pose], dim=1))
                # x2_out, flow_b, disp_l2, mask_l2, pose_b, pose_b_out = self.flow_estimators[l](torch.cat([out_corr_relu_b, out_corr_pose_relu_b, x2], dim=1))
                pose_mat_f = pose_vec2mat(pose_f)
                pose_mat_b = invert_pose(pose_mat_f)
                # pose_mat_b = pose_vec2mat(pose_b)
            else:
                x1_out, flow_f_res, disp_l1, mask_l1, pose_f_res, pose_f_out = self.flow_estimators[l](torch.cat([
                    out_corr_relu_f, x1, x2_warp_pose, x1_out, flow_f, disp_l1, mask_l1, pose_f_out], dim=1))
                x2_out, flow_b_res, disp_l2, mask_l2,          _, pose_b_out = self.flow_estimators[l](torch.cat([
                    out_corr_relu_b, x2, x1_warp_pose, x2_out, flow_b, disp_l2, mask_l2, pose_b_out], dim=1))
                # x2_out, flow_b_res, disp_l2, mask_l2, pose_b_res, pose_b_out = self.flow_estimators[l](torch.cat([
                #     out_corr_relu_b, out_corr_pose_relu_b, x2, x2_out, flow_b, disp_l2, mask_l2, pose_b_out], dim=1))

                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

                pose_mat_f_res = pose_vec2mat(pose_f_res)
                pose_mat_b_res = invert_pose(pose_mat_f_res)
                # pose_mat_b_res = pose_vec2mat(pose_b_res)

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
                # flow_res_b, disp_l2, pose_b_res, mask_l2 = self.context_networks(torch.cat([x2_out, flow_b, disp_l2, pose_b_out, mask_l2], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b

                pose_mat_f_res = pose_vec2mat(pose_f_res)
                pose_mat_b_res = invert_pose(pose_mat_f_res)
                # pose_mat_b_res = pose_vec2mat(pose_b_res)

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
        if self._args.use_mask:
            output_dict['masks_l1'] = upsample_outputs_as(masks_1[::-1], x1_rev)
            output_dict['masks_l2'] = upsample_outputs_as(masks_2[::-1], x1_rev)
        output_dict["pose_f"] = poses_f[::-1]
        output_dict["pose_b"] = poses_b[::-1]

        return output_dict


    def forward(self, input_dict):

        output_dict = {}

        ## Left
        output_dict = self.run_pwc(input_dict, input_dict['input_l1_aug'], input_dict['input_l2_aug'], input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.evaluation):
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
                if self._args.use_mask:
                    output_dict_r['masks_l1'][ii] = torch.flip(output_dict_r['masks_l1'][ii], [3])
                    output_dict_r['masks_l2'][ii] = torch.flip(output_dict_r['masks_l2'][ii], [3])

            output_dict['output_dict_r'] = output_dict_r

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        if self._args.evaluation:

            input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
            input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
            k_l1_flip = input_dict["input_k_l1_flip_aug"]
            k_l2_flip = input_dict["input_k_l2_flip_aug"]

            output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

            flow_f_pp = []
            flow_b_pp = []
            disp_l1_pp = []
            disp_l2_pp = []
            mask_l1_pp = []
            mask_l2_pp = []

            for ii in range(0, len(output_dict_flip['flows_f'])):

                flow_f_pp.append(post_processing(output_dict['flows_f'][ii], flow_horizontal_flip(output_dict_flip['flows_f'][ii])))
                flow_b_pp.append(post_processing(output_dict['flows_b'][ii], flow_horizontal_flip(output_dict_flip['flows_b'][ii])))
                disp_l1_pp.append(post_processing(output_dict['disps_l1'][ii], torch.flip(output_dict_flip['disps_l1'][ii], [3])))
                disp_l2_pp.append(post_processing(output_dict['disps_l2'][ii], torch.flip(output_dict_flip['disps_l2'][ii], [3])))
                if self._args.use_mask:
                    mask_l1_pp.append(post_processing(output_dict['masks_l1'][ii], torch.flip(output_dict_flip['masks_l1'][ii], [3])))
                    mask_l2_pp.append(post_processing(output_dict['masks_l2'][ii], torch.flip(output_dict_flip['masks_l2'][ii], [3])))

            output_dict['flow_f_pp'] = flow_f_pp
            output_dict['flow_b_pp'] = flow_b_pp
            output_dict['disp_l1_pp'] = disp_l1_pp
            output_dict['disp_l2_pp'] = disp_l2_pp
            if self._args.use_mask:
                output_dict['mask_l1_pp'] = disp_l1_pp
                output_dict['mask_l2_pp'] = disp_l2_pp

        return output_dict