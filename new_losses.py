import torch
import torch.nn as nn
import torch.nn.functional as tf

from sys import exit

from utils.interpolation import interpolate2d_as
from utils.helpers import BackprojectDepth, Project3D
from utils.loss_utils import _generate_image_left, _generate_image_right, _smoothness_motion_2nd, disp_smooth_loss
from utils.loss_utils import _SSIM, _reconstruction_error, _disp2depth_kitti_K
from utils.loss_utils import _adaptive_disocc_detection, _adaptive_disocc_detection_disp
from utils.sceneflow_util import projectSceneFlow2Flow
from utils.inverse_warp import pose2flow

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.args = args

        self.flow_min_w = args.flow_min_w
        self.ssim_w = args.ssim_w
        self.flow_loss_mode = args.flow_loss_mode

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

    def depth_loss(self, disp_l, disp_r, img_l, img_r, scale):
        """ Calculate the difference between the src and tgt images 
        Inputs:
        disp_l: disparity from left to right (B, 1, H, W)
        disp_r: disparity from right to left (B, 1, H, W)
        img_l/img_r: stereo images
        """

        img_r_warp = _generate_image_left(img_r, disp_l) 
        img_diff = _reconstruction_error(img_l, img_r_warp, self.ssim_w)

        # occlusion detection
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()
        right_occ = _adaptive_disocc_detection_disp(disp_l).detach()

        # L-R Consistency loss
        proj_disp_r = _generate_image_left(disp_r, disp_l)
        proj_disp_l = _generate_image_right(disp_l, disp_r)
        lr_disp_diff_l = torch.abs(proj_disp_r - disp_l)
        lr_disp_diff_r = torch.abs(proj_disp_l - disp_r)

        # Smoothness loss
        # mean_disp = disp_l.mean(2, True).mean(3, True)
        # norm_disp = disp_l / (mean_disp + 1e-7)
        smooth_loss = disp_smooth_loss(disp_l, img_l) / (2 ** scale)

        loss_lr = lr_disp_diff_l[left_occ].mean() + lr_disp_diff_r[right_occ].mean()
        lr_disp_diff_l[~left_occ].detach_()
        lr_disp_diff_r[~right_occ].detach_()

        return img_diff, left_occ, loss_lr, smooth_loss
    
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
        ref_warp = tf.grid_sample(src, grid, padding_mode="border")
        img_diff = _reconstruction_error(tgt, ref_warp, self.ssim_w)

        return img_diff, occ_mask

    def detach_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flows_f'])):
            output_dict['flows_f'][ii].detach_()
            output_dict['flows_b'][ii].detach_()

    def forward(self, output, target):
        depth_loss_sum = 0
        flow_loss_sum = 0
        disp_lr_sum = 0
        disp_sm_sum = 0

        img_l1 = target['input_l1_aug']
        img_l2 = target['input_l2_aug']
        img_r1 = target['input_r1_aug']
        img_r2 = target['input_r2_aug']
        K_l1 = target['input_k_l1_aug']
        K_l2 = target['input_k_l2_aug']

        disps_l1 = output['disps_l1']
        disps_l2 = output['disps_l2']
        disps_r1 = output['output_dict_r']['disps_l1']
        disps_r2 = output['output_dict_r']['disps_l2']
        flows_f = output['flows_f']
        flows_b = output['flows_b']
        pose_f = output['pose_f']
        pose_b = output['pose_b']

        if self.args.use_mask:
            masks_l1 = output['mask_l1']
            masks_l2 = output['mask_l2']

        for s, _ in enumerate(disps_l1):
            flow_f = interpolate2d_as(flows_f[s], img_l1)
            flow_b = interpolate2d_as(flows_b[s], img_l1)
            disp_l1 = interpolate2d_as(disps_l1[s], img_l1)
            disp_l2 = interpolate2d_as(disps_l2[s], img_l1)
            disp_r1 = interpolate2d_as(disps_r1[s], img_l1)
            disp_r2 = interpolate2d_as(disps_r2[s], img_l1)

            if self.args.use_mask:
                mask_l1 = interpolate2d_as(masks_l1[s], img_l1)
                mask_l2 = interpolate2d_as(masks_l2[s], img_l1)

                if self.args.use_flow_mask:
                    flow_mask_l1 = 1.0 - mask_l1
                    flow_mask_l2 = 1.0 - mask_l2

            # depth diffs
            disp_diff1, left_occ1, loss_lr1, loss_disp_sm1 = self.depth_loss(disp_l1, disp_r1, img_l1, img_r1, s)
            disp_diff2, left_occ2, loss_lr2, loss_disp_sm2 = self.depth_loss(disp_l2, disp_r2, img_l2, img_r2, s)
            loss_lr = (loss_lr1 + loss_lr2) * self.disp_lr_w
            loss_disp_sm = loss_disp_sm1 + loss_disp_sm2 * self.disp_sm_w

            # pose diffs
            pose_diff1, pose_occ_b = self.flow_loss(disp_l1, img_l2, img_l1, K_l1, T=pose_f, mode='pose')
            pose_diff2, pose_occ_f = self.flow_loss(disp_l2, img_l1, img_l2, K_l2, T=pose_b, mode='pose')

            # sf diffs
            sf_diff1, sf_occ_b = self.flow_loss(disp_l1, img_l2, img_l1, K_l1, sf=flow_f, mode='sf')
            sf_diff2, sf_occ_f = self.flow_loss(disp_l2, img_l1, img_l2, K_l2, sf=flow_b, mode='sf')

            # min(pose, sf)
            flow_diffs1 = torch.cat([pose_diff1, sf_diff1], dim=1)
            flow_diffs2 = torch.cat([pose_diff2, sf_diff2], dim=1)
            min_flow_diff1, _ = flow_diffs1.min(dim=1, keepdim=True)
            min_flow_diff2, _ = flow_diffs2.min(dim=1, keepdim=True)

            mask_disp_diff1 = (disp_diff1 <= min_flow_diff1)
            mask_disp_diff2 = (disp_diff2 <= min_flow_diff2)

            depth_loss1 = disp_diff1[mask_disp_diff1 * left_occ1].mean()
            depth_loss2 = disp_diff2[mask_disp_diff2 * left_occ2].mean()
            depth_loss = depth_loss1 + depth_loss2

            if self.flow_loss_mode == 'min':
                occ_f = pose_occ_f * sf_occ_f
                occ_b = pose_occ_b * sf_occ_b
                flow_loss1 = min_flow_diff1[occ_f].mean()
                flow_loss2 = min_flow_diff2[occ_b].mean()
            elif self.flow_loss_mode == 'avg':
                occ_f = (pose_occ_f * sf_occ_f)
                occ_b = (pose_occ_b * sf_occ_b)
                flow_loss1 = flow_diffs1.mean(dim=1, keepdim=True)[occ_f].mean()
                flow_loss2 = flow_diffs2.mean(dim=1, keepdim=True)[occ_b].mean()
            elif self.flow_loss_mode == 'sep':
                occ_f = torch.cat([pose_occ_f, sf_occ_f], dim=1)
                occ_b = torch.cat([pose_occ_b, sf_occ_b], dim=1)
                flow_loss1 = flow_diffs1[occ_f].mean(dim=1, keepdim=True).mean()
                flow_loss2 = flow_diffs2[occ_b].mean(dim=1, keepdim=True).mean()

            flow_loss = flow_loss1 + flow_loss2

            depth_loss_sum = depth_loss_sum + depth_loss + loss_lr * self.disp_lr_w + loss_disp_sm * self.disp_sm_w
            disp_lr_sum = disp_lr_sum + loss_lr
            disp_sm_sum = disp_sm_sum + loss_disp_sm
            flow_loss_sum = flow_loss_sum + flow_loss

        loss_dict = {}

        loss_dict["total_loss"] = depth_loss_sum + flow_loss_sum
        loss_dict["depth_loss"] = depth_loss_sum.detach()
        loss_dict["flow_loss"] = flow_loss_sum.detach()
        loss_dict["disp_lr_loss"] = disp_lr_sum.detach()
        loss_dict["disp_sm_loss"] = disp_sm_sum.detach()

        self.detach_outputs(output['output_dict_r'])

        return loss_dict
