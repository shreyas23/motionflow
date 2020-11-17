import torch
import torch.nn as nn
import torch.nn.functional as tf

from .encoders import ResnetEncoder
from .disp_decoders import DispDecoder
from .pose_decoders import PoseExpNet, PoseNet, PoseDecoder
from .sf_decoders import SFDecoder
from .joint_decoders import JointContextNetwork
from .modules_sceneflow import WarpingLayer_SF
from .correlation_package.correlation import Correlation

from .common import Conv, UpConv
from utils.helpers import invert_pose
from utils.sceneflow_util import flow_horizontal_flip, post_processing, upsample_outputs_as
from utils.interpolation import interpolate2d_as


class MonoSep(nn.Module):

    def __init__(self, args):
        super(MonoSep, self).__init__()

        self.args = args

        self.encoder = ResnetEncoder(num_layers=18, pretrained=args.pt_encoder, num_input_images=1)
        self.encoder_chs = self.encoder.num_ch_enc

        self.disp_decoder = DispDecoder(num_ch_enc=self.encoder_chs)
        self.pose_decoder = PoseDecoder(self.encoder_chs, 2)

        self.sf_out_ch = 32
        self.sf_layers = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()

        self.context_network = JointContextNetwork(3 + 6 + 1 + 1)

        self.corr_params = {"pad_size": self.search_range, "kernel_size": 1, "max_disp": self.search_range, "stride1": 1, "stride2": 1, "corr_multiply": 1}        
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.sigmoid = torch.nn.Sigmoid()

        for l, ch in enumerate(self.encoder_chs[::-1]):
            if l > self.output_level:
                break
            if l == 0:
                num_ch_in = self.dim_corr + self.dim_corr + ch
            else:
                num_ch_in = self.dim_corr + self.dim_corr + ch + self.out_ch_size + 3 + 1 + 6 + 1
                self.upconv_layers.append(UpConv(self.out_ch_size, self.out_ch_size, 3, 2))

            layer_sf = SFDecoder(num_ch_in)
            self.flow_estimators.append(layer_sf)
        

    def run_pwc(self, input_dict, x1_pyramid, x2_pyramid, img_x1, img_x2, disp_l1, disp_l2, k1, k2):
            
        output_dict = {}

        input_x1_features = self.encoder(img_x1)
        input_x2_features = self.encoder(img_x2)

        output_dict["out_disps_l1"] = self.disp_decoder(input_x1_features)
        output_dict["out_disps_l2"] = self.disp_decoder(input_x2_features)

        pose_feats = [input_x1_features, input_x2_features]
        output_dict["out_pose_f"] = self.pose_decoder(pose_feats)
        output_dict["out_pose_b"] = invert_pose(output_dict["out_pose_f"])

        x1_pyramid.append(img_x1)
        x2_pyramid.append(img_x2)

        # outputs
        sceneflows_f = []
        sceneflows_b = []

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):

            # warping
            if l == 0:
                x2_warp = x2
                x1_warp = x1
            else:
                flow_f = interpolate2d_as(flow_f, x1, mode="bilinear")
                flow_b = interpolate2d_as(flow_b, x1, mode="bilinear")
                x1_out = self.upconv_layers[l-1](x1_out)
                x2_out = self.upconv_layers[l-1](x2_out)

                x2_warp = self.warping_layer_sf(x2, flow_f, disp_l1, k1, input_dict['aug_size'])
                x1_warp = self.warping_layer_sf(x1, flow_b, disp_l2, k2, input_dict['aug_size'])

            # correlation
            out_corr_f = Correlation.apply(x1, x2_warp, self.corr_params)
            out_corr_b = Correlation.apply(x2, x1_warp, self.corr_params)
            out_corr_relu_f = self.leakyRELU(out_corr_f)
            out_corr_relu_b = self.leakyRELU(out_corr_b)

            # monosf estimator
            if l == 0:
                x1_out, flow_f = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1], dim=1))
                x2_out, flow_b = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2], dim=1))
            else:
                x1_out, flow_f_res = self.flow_estimators[l](torch.cat([out_corr_relu_f, x1, x1_out, flow_f], dim=1))
                x2_out, flow_b_res = self.flow_estimators[l](torch.cat([out_corr_relu_b, x2, x2_out, flow_b], dim=1))

                flow_f = flow_f + flow_f_res
                flow_b = flow_b + flow_b_res

            # upsampling or post-processing
            if l != self.output_level:
                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)                
            else:
                flow_res_f, disp_l1, mask_l1, pose_f_res = self.context_networks(torch.cat([x1_out, flow_f, disp_l1, pose_f_out, mask_l1], dim=1))
                flow_res_b, disp_l2, mask_l2, pose_b_res = self.context_networks(torch.cat([x2_out, flow_b, disp_l2, pose_b_out, mask_l2], dim=1))
                flow_f = flow_f + flow_res_f
                flow_b = flow_b + flow_res_b

                sceneflows_f.append(flow_f)
                sceneflows_b.append(flow_b)

                break

        x1_rev = x1_pyramid[::-1]

        output_dict['flow_f'] = upsample_outputs_as(sceneflows_f[::-1], x1_rev)
        output_dict['flow_b'] = upsample_outputs_as(sceneflows_b[::-1], x1_rev)

        return output_dict


    def forward(self, input_dict):

        output_dict = {}


        ## Left
        output_dict = self.run_pwc(input_dict, 
                                   input_l1_features[::-1], input_l2_features[::-1], 
                                   input_dict['input_l1_aug'], input_dict['input_l2_aug'],
                                   input_dict['input_k_l1_aug'], input_dict['input_k_l2_aug'])

        flow_res_f, disp_l1, mask_l1, pose_f = self.context_networks(torch.cat([x1_out, flow_f, disp_l1, pose_f_out, mask_l1], dim=1))
        flow_res_b, disp_l2, mask_l2, pose_b = self.context_networks(torch.cat([x2_out, flow_b, disp_l2, pose_b_out, mask_l2], dim=1))

        flow_f = flow_f + flow_res_f
        flow_b = flow_b + flow_res_b

        ## Right
        ## ss: train val 
        ## ft: train 
        if self.training or (not self._args.evaluation):
            input_r1_flip = torch.flip(input_dict['input_r1_aug'], [3])
            input_r2_flip = torch.flip(input_dict['input_r2_aug'], [3])
            k_r1_flip = input_dict["input_k_r1_flip_aug"]
            k_r2_flip = input_dict["input_k_r2_flip_aug"]

            input_r1_features = self.encoder(input_r1_flip)
            input_r2_features = self.encoder(input_r2_flip)

            output_dict_r = self.run_pwc(input_dict, 
                                         input_r1_features, input_r2_features, 
                                         input_r1_flip, input_r2_flip, 
                                         k_r1_flip, k_r2_flip)

            for ii in range(0, len(output_dict_r['flow_f'])):
                output_dict_r['flow_f'][ii] = flow_horizontal_flip(output_dict_r['flow_f'][ii])
                output_dict_r['flow_b'][ii] = flow_horizontal_flip(output_dict_r['flow_b'][ii])

            output_dict['output_dict_r'] = output_dict_r

        ## Post Processing 
        ## ss:           eval
        ## ft: train val eval
        # if self._args.evaluation:

        #     input_l1_flip = torch.flip(input_dict['input_l1_aug'], [3])
        #     input_l2_flip = torch.flip(input_dict['input_l2_aug'], [3])
        #     k_l1_flip = input_dict["input_k_l1_flip_aug"]
        #     k_l2_flip = input_dict["input_k_l2_flip_aug"]

        #     output_dict_flip = self.run_pwc(input_dict, input_l1_flip, input_l2_flip, k_l1_flip, k_l2_flip)

        #     flow_f_pp = []
        #     flow_b_pp = []
        #     disp_l1_pp = []
        #     disp_l2_pp = []
        #     mask_l1_pp = []
        #     mask_l2_pp = []

        #     for ii in range(0, len(output_dict_flip['flow_f'])):

        #         flow_f_pp.append(post_processing(output_dict['flow_f'][ii], flow_horizontal_flip(output_dict_flip['flow_f'][ii])))
        #         flow_b_pp.append(post_processing(output_dict['flow_b'][ii], flow_horizontal_flip(output_dict_flip['flow_b'][ii])))
        #         disp_l1_pp.append(post_processing(output_dict['disp_l1'][ii], torch.flip(output_dict_flip['disp_l1'][ii], [3])))
        #         disp_l2_pp.append(post_processing(output_dict['disp_l2'][ii], torch.flip(output_dict_flip['disp_l2'][ii], [3])))
        #         mask_l1_pp.append(post_processing(output_dict['mask_l1'][ii], torch.flip(output_dict_flip['mask_l1'][ii], [3])))
        #         mask_l2_pp.append(post_processing(output_dict['mask_l2'][ii], torch.flip(output_dict_flip['mask_l2'][ii], [3])))

        #     output_dict['flow_f_pp'] = flow_f_pp
        #     output_dict['flow_b_pp'] = flow_b_pp
        #     output_dict['disp_l1_pp'] = disp_l1_pp
        #     output_dict['disp_l2_pp'] = disp_l2_pp
        #     output_dict['mask_l1_pp'] = disp_l1_pp
        #     output_dict['mask_l2_pp'] = disp_l2_pp

        return output_dict