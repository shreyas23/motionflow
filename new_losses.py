import torch
import torch.nn as nn
import torch.nn.functional as tf

from sys import exit
import matplotlib.pyplot as plt

from utils.interpolation import interpolate2d_as
from utils.helpers import BackprojectDepth, Project3D
from utils.loss_utils import _generate_image_left, _generate_image_right, _smoothness_motion_2nd, disp_smooth_loss
from utils.loss_utils import _SSIM, _reconstruction_error, _disp2depth_kitti_K, logical_or
from utils.loss_utils import _adaptive_disocc_detection, _adaptive_disocc_detection_disp
from utils.sceneflow_util import projectSceneFlow2Flow
from utils.inverse_warp import pose2flow
from utils.sceneflow_util import intrinsic_scale


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.args = args

        self.flow_min_w = args.flow_min_w
        self.ssim_w = args.ssim_w
        self.flow_reduce_mode = args.flow_reduce_mode

        # dis weights
        self.disp_sm_w = args.disp_sm_w

        #sf wights
        self.sf_3d_pts = args.sf_pts_w
        self.sf_3d_sm = args.sf_sm_w

        # pos weights
        self.pose_pts_w = args.pose_pts_w
        self.pose_sm_w = args.pose_sm_w

        # mas weights
        self.mask_reg_w = args.mask_reg_w
        self.mask_sm_w = args.mask_sm_w
        self.mask_cons_w = args.mask_cons_w
        self.flow_diff_thresh = args.flow_diff_thresh

        # conistency weights 
        self.mask_lr_w = args.mask_lr_w
        self.disp_lr_w = args.disp_lr_w
        self.static_cons_w = args.static_cons_w

        self.use_flow_mask = args.use_flow_mask

        # self.scale_weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self.scale_weights = [1.0, 1.0, 1.0, 1.0, 1.0]


    def depth_loss(self, disp_l, disp_r, img_l, img_r, scale):
        """ Calculate the difference between the src and tgt images 
        Inputs:
        disp_l: disparity from left to right (B, 1, H, W)
        disp_r: disparity from right to left (B, 1, H, W)
        img_l/img_r: stereo images
        """

        # occlusion detection
        left_occ = _adaptive_disocc_detection_disp(disp_r).float()
        left_occ = interpolate2d_as(left_occ, img_l).detach().bool()

        img_r_ds = interpolate2d_as(img_r, disp_l)
        img_r_warp = _generate_image_left(img_r_ds, disp_l)
        img_r_warp = interpolate2d_as(img_r_warp, img_l)
        img_diff = _reconstruction_error(img_l, img_r_warp, self.ssim_w)
        img_diff[~left_occ].detach_()

        img_l_ds = interpolate2d_as(img_l, disp_l)

        # Smoothness loss
        # mean_disp = disp_l.mean(2, True).mean(3, True)
        # norm_disp = disp_l / (mean_disp + 1e-7)
        # img_l_ds = interpolate2d_as(img_l, norm_disp)
        # smooth_loss = disp_smooth_loss(norm_disp, img_l_ds) / (2 ** scale)

        smooth_loss = _smoothness_motion_2nd(disp_l, img_l_ds).mean() / (2**scale)

        return img_diff, left_occ, smooth_loss

    
    def flow_loss(self, disp, src, tgt, K, sf=None, T=None, mode='pose'):
        """ Calculate the difference between the src and tgt images 
        Inputs:
        disp: disparity from left to right (B, 1, H, W)
        src/tgt: consecutive images from left camera
        flow: scene flow from tgt to src (B, 3, H, W)
        pose: pose transform from tgt to src (B, 3, 3)
        """

        b, _, h, w = disp.shape
        depth = _disp2depth_kitti_K(disp, K[:, 0, 0])

        # downsample image to estimated scale for warping
        src_ds = interpolate2d_as(src, disp)

        if mode == 'pose':
            assert (T is not None), "T cannot be None when mode=pose"
            of = pose2flow(depth.squeeze(dim=1), None, K, torch.inverse(K), pose_mat=T)
        elif mode == 'sf':
            assert (sf is not None), "sf cannot be None when mode=sf"
            of = projectSceneFlow2Flow(K, sf, disp)

        back_proj = BackprojectDepth(b, h, w).to(device=disp.device)
        proj = Project3D(b, h, w).to(device=disp.device)

        occ_mask = _adaptive_disocc_detection(of).detach()

        cam_points = back_proj(depth, torch.inverse(K), mode=mode)
        grid = proj(cam_points, K, T=T, sf=sf, mode=mode)

        # upsample warped image back to input resolution for diff 
        ref_warp = tf.grid_sample(src_ds, grid, padding_mode="zeros")
        ref_warp_us = interpolate2d_as(ref_warp, tgt)
        occ_mask = interpolate2d_as(occ_mask.float(), tgt).bool()

        img_diff = _reconstruction_error(tgt, ref_warp_us, self.ssim_w)

        # TODO: add a point reconstruction loss as well

        #

        return img_diff, occ_mask


    def explainability_loss(self, mask):
        loss = tf.binary_cross_entropy(mask, torch.ones_like(mask, device=mask.device))
        return loss


    def forward(self, output, target):
        depth_loss_sum = 0
        flow_loss_sum = 0
        disp_sm_sum = 0
        exp_loss_sum = 0

        img_l1 = target['input_l1_orig']
        img_l2 = target['input_l2_orig']
        img_r1 = target['input_r1_orig']
        img_r2 = target['input_r2_orig']
        K_l1 = target['input_k_l1_aug']
        K_l2 = target['input_k_l2_aug']

        aug_size = target['aug_size']

        disps_l1 = output['disps_l1']
        disps_l2 = output['disps_l2']
        disps_r1 = output['output_dict_r']['disps_l1']
        disps_r2 = output['output_dict_r']['disps_l2']
        flows_f = output['flows_f']
        flows_b = output['flows_b']
        pose_f = output['pose_f']
        pose_b = output['pose_b']

        if self.args.use_mask:
            masks_l1 = output['masks_l1']
            masks_l2 = output['masks_l2']

        num_scales = len(disps_l1)
        for s in range(num_scales):
            flow_f = flows_f[s]
            flow_b = flows_b[s]
            disp_l1 = disps_l1[s]
            disp_l2 = disps_l2[s]
            disp_r1 = disps_r1[s]
            disp_r2 = disps_r2[s]

            if self.args.use_mask:
                mask_l1 = interpolate2d_as(masks_l1[s], img_l1)
                mask_l2 = interpolate2d_as(masks_l2[s], img_l1)

                if self.args.use_flow_mask:
                    flow_mask_l1 = 1.0 - mask_l1
                    flow_mask_l2 = 1.0 - mask_l2
                else:
                    flow_mask_l1 = None
                    flow_mask_l2 = None
            else:
                mask_l1 = None
                mask_l2 = None
                flow_mask_l1 = None
                flow_mask_l2 = None

            # depth diffs
            disp_diff1, left_occ1, loss_disp_sm1 = self.depth_loss(disp_l1, disp_r1, img_l1, img_r1, s)
            disp_diff2, left_occ2, loss_disp_sm2 = self.depth_loss(disp_l2, disp_r2, img_l2, img_r2, s)
            loss_disp_sm = (loss_disp_sm1 + loss_disp_sm2) * self.disp_sm_w

            # un-normalize disparity
            _, _, h, w = disp_l1.shape 
            disp_l1 = disp_l1 * w
            disp_l2 = disp_l2 * w

            local_scale = torch.zeros_like(aug_size)
            local_scale[:, 0] = h
            local_scale[:, 1] = w

            rel_scale = local_scale / aug_size

            K_l1_s = intrinsic_scale(K_l1, rel_scale[:, 0], rel_scale[:, 1])
            K_l2_s = intrinsic_scale(K_l2, rel_scale[:, 0], rel_scale[:, 1])

            # pose diffs
            pose_diff1, pose_occ_b = self.flow_loss(disp_l1, img_l2, img_l1, K_l1_s, T=pose_f, mode='pose')
            pose_diff2, pose_occ_f = self.flow_loss(disp_l2, img_l1, img_l2, K_l2_s, T=pose_b, mode='pose')

            # sf diffs
            sf_diff1, sf_occ_b = self.flow_loss(disp_l1, img_l2, img_l1, K_l1_s, sf=flow_f, mode='sf')
            sf_diff2, sf_occ_f = self.flow_loss(disp_l2, img_l1, img_l2, K_l2_s, sf=flow_b, mode='sf')

            # explainability mask loss
            if self.args.use_mask:
                exp_loss = (self.explainability_loss(mask_l1) + self.explainability_loss(mask_l2)) * self.mask_reg_w
                pose_diff1 = pose_diff1 * mask_l1
                pose_diff2 = pose_diff2 * mask_l2
            else:
                exp_loss = torch.tensor(0, requires_grad=False)

            if self.args.use_flow_mask:
                flow_diff1 = flow_diff1 * flow_mask_l1
                flow_diff2 = flow_diff2 * flow_mask_l2

            flow_diffs1 = torch.cat([pose_diff1, sf_diff1], dim=1)
            flow_diffs2 = torch.cat([pose_diff2, sf_diff2], dim=1)
            min_flow_diff1, _ = flow_diffs1.min(dim=1, keepdim=True)
            min_flow_diff2, _ = flow_diffs2.min(dim=1, keepdim=True)
            min_flow_diff1.detach_()
            min_flow_diff2.detach_()

            # mask_disp_diff1 = (disp_diff1 <= min_flow_diff1).detach()
            # mask_disp_diff2 = (disp_diff2 <= min_flow_diff2).detach()

            # depth_loss1 = disp_diff1[mask_disp_diff1 * left_occ1].mean()
            # depth_loss2 = disp_diff2[mask_disp_diff2 * left_occ2].mean()

            depth_loss1 = disp_diff1[left_occ1].mean()
            depth_loss2 = disp_diff2[left_occ2].mean()
            depth_loss = depth_loss1 + depth_loss2

            occ_f = pose_occ_f * sf_occ_f * left_occ1
            occ_b = pose_occ_b * sf_occ_b * left_occ2

            if self.flow_reduce_mode == 'min':
                flow_loss1 = min_flow_diff1[occ_f].mean()
                flow_loss2 = min_flow_diff2[occ_b].mean()
            elif self.flow_reduce_mode == 'avg':
                flow_loss1 = flow_diffs1.mean(dim=1, keepdim=True)[occ_f].mean()
                flow_loss2 = flow_diffs2.mean(dim=1, keepdim=True)[occ_b].mean()
            elif self.flow_reduce_mode == 'sum':
                flow_loss1 = pose_diff1.mean(dim=1, keepdim=True)[occ_f].mean() + sf_diff1.mean(dim=1, keepdim=True)[occ_f].mean()
                flow_loss2 = pose_diff2.mean(dim=1, keepdim=True)[occ_b].mean() + sf_diff2.mean(dim=1, keepdim=True)[occ_b].mean()

            pose_diff1[~occ_f].detach_()
            sf_diff1[~occ_f].detach_()
            pose_diff2[~occ_b].detach_()
            sf_diff2[~occ_b].detach_()

            flow_loss = flow_loss1 + flow_loss2

            depth_loss_sum = depth_loss_sum + (depth_loss + loss_disp_sm * self.disp_sm_w) * self.scale_weights[s]
            flow_loss_sum = flow_loss_sum + flow_loss * self.scale_weights[s] 
            exp_loss_sum = exp_loss_sum + exp_loss * self.scale_weights[s]
            disp_sm_sum = disp_sm_sum + loss_disp_sm

        loss_dict = {}
        loss_dict["total_loss"] = (depth_loss_sum + flow_loss_sum + exp_loss_sum) / num_scales
        loss_dict["depth_loss"] = depth_loss_sum.detach()
        loss_dict["flow_loss"] = flow_loss_sum.detach()
        loss_dict["disp_sm_loss"] = disp_sm_sum.detach()
        loss_dict["exp_loss"] = exp_loss_sum.detach()

        # detach unused parameters
        for s in range(len(output['output_dict_r']['flows_f'])):
            output['output_dict_r']['flows_f'][s].detach_()
            output['output_dict_r']['flows_b'][s].detach_()
            output['output_dict_r']['disps_l1'][s].detach_()
            output['output_dict_r']['disps_l2'][s].detach_()
            output['output_dict_r']['masks_l1'][s].detach_()
            output['output_dict_r']['masks_l2'][s].detach_()

        return loss_dict