import torch
import torch.nn as nn
import torch.nn.functional as tf

from sys import exit
import matplotlib.pyplot as plt

from utils.helpers import BackprojectDepth, Project3D
from utils.interpolation import interpolate2d_as
from utils.inverse_warp import pose2flow, pose2sceneflow, pose_vec2mat
from utils.loss_utils import _generate_image_left, _generate_image_right, _smoothness_motion_2nd, disp_smooth_loss
from utils.loss_utils import _SSIM, _reconstruction_error, _disp2depth_kitti_K, logical_or, _elementwise_epe, robust_reconstruction_error
from utils.loss_utils import _adaptive_disocc_detection, _adaptive_disocc_detection_disp
from utils.loss_utils import _gradient_x_2nd, _gradient_y_2nd
from utils.sceneflow_util import projectSceneFlow2Flow, intrinsic_scale, reconstructPts

eps = 1e-7


class MonoDepthSFLoss(nn.Module):
    def __init__(self, args):
        super(MonoDepthSFLoss, self).__init__()

        self.args = args

        self.ssim_w = args.ssim_w
        self.use_disp_min = args.use_disp_min
        self.flow_reduce_mode = args.flow_reduce_mode

        # dis weights
        self.disp_sm_w = args.disp_sm_w

        #flow wights
        self.flow_pts_w = args.flow_pts_w
        self.flow_sm_w = args.flow_sm_w

        # mas weights
        self.mask_reg_w = args.mask_reg_w
        self.mask_sm_w = args.mask_reg_w

        # conistency weights 
        self.disp_lr_w = args.disp_lr_w
        self.static_cons_w = args.static_cons_w
        self.mask_census_w = args.mask_cons_w
        self.flow_diff_thresh = args.flow_diff_thresh

        self.use_mask = args.train_exp_mask or args.train_census_mask

        # self.scale_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.scale_weights = [4.0, 2.0, 1.0, 1.0, 1.0]


    def depth_loss(self, disp_l, disp_r, img_l, img_r, scale):
        """ Calculate the difference between the src and tgt images
        Inputs:
        disp_l: disparity from left to right (B, 1, H, W)
        disp_r: disparity from right to left (B, 1, H, W)
        img_l/img_r: stereo images
        """

        # occlusion detection
        left_occ = _adaptive_disocc_detection_disp(disp_r)

        img_r_warp = _generate_image_left(img_r, disp_l)
        img_diff = robust_reconstruction_error(img_l, img_r_warp, self.ssim_w)
        img_diff[~left_occ].detach_()

        smooth_loss = _smoothness_motion_2nd(disp_l, img_l, beta=10.0).mean() / (2**scale)

        ## L-R Consistency loss
        proj_disp_r = _generate_image_left(disp_r, disp_l)
        lr_disp_diff = torch.abs(proj_disp_r - disp_l)
        loss_lr = lr_disp_diff[left_occ].mean()
        lr_disp_diff[~left_occ].detach_()

        return img_diff, left_occ, smooth_loss, loss_lr

    
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
        occ_mask = _adaptive_disocc_detection(of).detach()

        back_proj = BackprojectDepth(b, h, w).to(device=disp.device)
        proj = Project3D(b, h, w).to(device=disp.device)

        cam_points = back_proj(depth, torch.inverse(K), mode=mode)
        grid = proj(cam_points, K, T=T, sf=sf, mode=mode)
        ref_warp = tf.grid_sample(src, grid, mode='bilinear', padding_mode="zeros")

        img_diff = robust_reconstruction_error(tgt, ref_warp, self.ssim_w)

        return img_diff, occ_mask, (cam_points, grid), occ_mask


    def structure_loss(self, pts1, pts2, grid1, grid2, sf=None, T=None, mode='pose'):
        b, _, h, w = pts1.shape

        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

        pts2_warp = reconstructPts(grid1.permute(0, 3, 1, 2), pts2)
        pts1_warp = reconstructPts(grid2.permute(0, 3, 1, 2), pts1)

        if mode == 'sf':
            pts1_tf = pts1 + sf[0]
            pts2_tf = pts2 + sf[1]
        else:
            # homogenize points
            ones = torch.ones((b, 1, h*w), requires_grad=False).cuda(device=pts1.device)
            pts1_cat = torch.cat([pts1.view(b, 3, -1), ones], dim=1)
            pts2_cat = torch.cat([pts2.view(b, 3, -1), ones], dim=1)
            pts1_tf = torch.bmm(T[0], pts1_cat).view(b, 3, h, w)
            pts2_tf = torch.bmm(T[1], pts2_cat).view(b, 3, h, w)

        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)

        return pts_diff1, pts_diff2


    def forward(self, output, target):
        depth_loss_sum = 0
        flow_loss_sum = 0
        sf_im_loss_sum = 0
        sf_pts_loss_sum = 0
        sf_sm_sum = 0
        disp_sm_sum = 0
        disp_lr_sum = 0
        flow_pts_sum = 0

        img_l1 = target['input_l1_aug']
        img_l2 = target['input_l2_aug']
        img_r1 = target['input_r1_aug']
        img_r2 = target['input_r2_aug']

        K_l1 = target['input_k_l1_aug']
        K_l2 = target['input_k_l2_aug']

        aug_size = target['aug_size']

        disps_l1 = output['disps_l1']
        disps_l2 = output['disps_l2']
        disps_r1 = output['output_dict_r']['disps_l1']
        disps_r2 = output['output_dict_r']['disps_l2']
        flows_f = output['flows_f']
        flows_b = output['flows_b']

        assert(len(disps_l1) == len(flows_f))
        assert(len(disps_l2) == len(flows_b))

        num_scales = len(disps_l1)
        for s in range(num_scales):
            flow_f = flows_f[s]
            flow_b = flows_b[s]

            disp_l1 = disps_l1[s]
            disp_l2 = disps_l2[s]
            disp_r1 = disps_r1[s]
            disp_r2 = disps_r2[s]

            img_l1 = interpolate2d_as(img_l1, disp_l1)
            img_l2 = interpolate2d_as(img_l2, disp_l2)
            img_r1 = interpolate2d_as(img_r1, disp_r1)
            img_r2 = interpolate2d_as(img_r2, disp_r2)

            # depth diffs
            disp_diff1, left_occ1, loss_disp_sm1, loss_lr1 = self.depth_loss(disp_l1, disp_r1, img_l1, img_r1, s)
            disp_diff2, left_occ2, loss_disp_sm2, loss_lr2 = self.depth_loss(disp_l2, disp_r2, img_l2, img_r2, s)
            loss_disp_sm = (loss_disp_sm1 + loss_disp_sm2) * self.disp_sm_w
            loss_lr = (loss_lr1 + loss_lr2) * self.disp_lr_w

            # denormalize disparity
            _, _, h, w = disp_l1.shape 
            disp_l1 = disp_l1 * w
            disp_l2 = disp_l2 * w

            local_scale = torch.zeros_like(aug_size)
            local_scale[:, 0] = h
            local_scale[:, 1] = w

            rel_scale = local_scale / aug_size

            K_l1_s = intrinsic_scale(K_l1, rel_scale[:, 0], rel_scale[:, 1])
            K_l2_s = intrinsic_scale(K_l2, rel_scale[:, 0], rel_scale[:, 1])

            # sf diffs
            sf_diff1, sf_occ_b, (pts1, sf_grid1), sf_occ_f = self.flow_loss(disp=disp_l1, src=img_l2, tgt=img_l1, K=K_l1_s, sf=flow_f, mode='sf')
            sf_diff2, sf_occ_f, (pts2, sf_grid2), sf_occ_b  = self.flow_loss(disp=disp_l2, src=img_l1, tgt=img_l2, K=K_l2_s, sf=flow_b, mode='sf')                

            """ SF SMOOTHNESS LOSS """
            pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
            pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
            loss_sf_sm = ( (_smoothness_motion_2nd(flow_f, img_l1, beta=10.0) / (pts_norm1 + 1e-8)).mean() + \
                (_smoothness_motion_2nd(flow_b, img_l2, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** s)
            loss_sf_sm = loss_sf_sm * self.flow_sm_w

            """ 3D STRUCTURE LOSS """
            sf_pts_diff1, sf_pts_diff2 = self.structure_loss(pts1, pts2, sf_grid1, sf_grid2, sf=[flow_f, flow_b], mode='sf')

            """ DEPTH LOSS """
            disp_mask1 = left_occ1
            disp_mask2 = left_occ2

            depth_loss1 = disp_diff1[disp_mask1].mean()
            depth_loss2 = disp_diff2[disp_mask2].mean()
            disp_diff1[~disp_mask1].detach_()
            disp_diff2[~disp_mask2].detach_()
            depth_loss = depth_loss1 + depth_loss2

            sf_occ_f = sf_occ_f * left_occ1
            sf_occ_b = sf_occ_b * left_occ2

            """ FLOW LOSS """
            flow_loss1 = (sf_diff1 * sf_occ_f.float()).sum(dim=1).mean()
            flow_loss2 = (sf_diff2 * sf_occ_b.float()).sum(dim=1).mean()
            if self.flow_pts_w > 0.0:
                flow_pts_loss1 = sf_pts_diff1.mean(dim=1, keepdim=True)[sf_occ_f].mean()
                flow_pts_loss2 = sf_pts_diff2.mean(dim=1, keepdim=True)[sf_occ_b].mean()
            else:
                flow_pts_loss1 = torch.tensor(0.0, requires_grad=False)
                flow_pts_loss2 = torch.tensor(0.0, requires_grad=False)

            # calculate losses for logging
            sf_im_loss = (sf_diff1.mean(dim=1, keepdim=True)[sf_occ_f].mean() + \
                          sf_diff2.mean(dim=1, keepdim=True)[sf_occ_b].mean()).detach()

            sf_pts_loss = (sf_pts_diff1.mean(dim=1, keepdim=True)[sf_occ_f].mean() + \
                           sf_pts_diff2.mean(dim=1, keepdim=True)[sf_occ_b].mean()).detach()

            flow_loss = flow_loss1 + flow_loss2
            pts_loss = (flow_pts_loss1 + flow_pts_loss2).detach() * self.flow_pts_w

            sf_diff1[~sf_occ_f].detach_()
            sf_diff2[~sf_occ_b].detach_()

            depth_loss_sum = depth_loss_sum + (depth_loss + loss_disp_sm + loss_lr) * self.scale_weights[s]
            flow_loss_sum = flow_loss_sum + (flow_loss + loss_sf_sm) * self.scale_weights[s] 
            sf_im_loss_sum = sf_im_loss_sum + sf_im_loss
            sf_pts_loss_sum = sf_pts_loss_sum + sf_pts_loss
            sf_sm_sum = sf_sm_sum + loss_sf_sm
            flow_pts_sum = flow_pts_sum + pts_loss
            disp_sm_sum = disp_sm_sum + loss_disp_sm
            disp_lr_sum = disp_lr_sum + loss_disp_sm

        d_loss = depth_loss_sum.detach()
        f_loss = flow_loss_sum.detach()

        m = max(d_loss, f_loss)

        d_weight = m / d_loss
        f_weight = m / f_loss

        total_loss = depth_loss_sum * d_weight + flow_loss_sum * f_weight

        loss_dict = {}
        loss_dict["total_loss"] = total_loss
        loss_dict["depth_loss"] = depth_loss_sum.detach()
        loss_dict["flow_loss"] = flow_loss_sum.detach()
        loss_dict["pts_loss"] = flow_pts_sum.detach()
        loss_dict["sf_im_loss"] = sf_im_loss_sum.detach()
        loss_dict["sf_pts_loss"] = sf_pts_loss_sum.detach()
        loss_dict["sf_smooth_loss"] = sf_sm_sum.detach()
        loss_dict["disp_sm_loss"] = disp_sm_sum.detach()

        # detach unused parameters
        for s in range(len(output['output_dict_r']['flows_f'])):
            output['output_dict_r']['flows_f'][s].detach_()
            output['output_dict_r']['flows_b'][s].detach_()
            output['output_dict_r']['disps_l1'][s].detach_()
            output['output_dict_r']['disps_l2'][s].detach_()

        return loss_dict