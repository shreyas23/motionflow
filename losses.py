import torch
import torch.nn as nn
import torch.nn.functional as tf

from sys import exit
import matplotlib.pyplot as plt

from utils.helpers import BackprojectDepth, Project3D
from utils.interpolation import interpolate2d_as
from utils.inverse_warp import pose2flow, pose_vec2mat, pose2sceneflow
from utils.loss_utils import _generate_image_left, _generate_image_right, _smoothness_motion_2nd, disp_smooth_loss
from utils.loss_utils import _SSIM, _reconstruction_error, _disp2depth_kitti_K, logical_or, _elementwise_epe
from utils.loss_utils import _adaptive_disocc_detection, _adaptive_disocc_detection_disp
from utils.loss_utils import _gradient_x_2nd, _gradient_y_2nd
from utils.sceneflow_util import projectSceneFlow2Flow, intrinsic_scale

eps = 1e-7

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

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
        self.mask_cons_w = args.mask_cons_w
        self.flow_diff_thresh = args.flow_diff_thresh

        self.use_mask = args.use_mask
        self.use_flow_mask = args.use_flow_mask
        self.use_static_mask = args.use_static_mask

        self.scale_weights = [1.0, 1.0, 1.0, 1.0, 1.0]


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
        img_diff = _reconstruction_error(img_l, img_r_warp, self.ssim_w)
        img_diff[~left_occ].detach_()

        smooth_loss = _smoothness_motion_2nd(disp_l, img_l, beta=10.0).mean() / (2**scale)

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

        img_diff = _reconstruction_error(tgt, ref_warp, self.ssim_w)

        return img_diff, occ_mask, (cam_points, grid)

    
    def mask_loss(self, mask, flow_diff, pose_err, sf_err, flow_diff_thresh=1e-3):
        reg_loss = tf.binary_cross_entropy(mask, torch.ones_like(mask))
        sm_loss = (_gradient_x_2nd(mask).abs() + _gradient_y_2nd(mask).abs()).mean()

        return reg_loss, sm_loss

    def create_census_mask(self, flow_diff, pose_err, sf_err, flow_diff_thresh=1e-3):
        # mask consensus loss
        target_mask = (pose_err <= sf_err).float().detach()
        flow_similar = (flow_diff < flow_diff_thresh).float().detach()
        census_target_mask = logical_or(target_mask, flow_similar).detach()

        return census_target_mask


    def structure_loss(self, pts1, pts2, grid1, grid2, sf=None, T=None, mode='pose', s=None):
        b, _, h, w = pts1.shape

        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)

        pts2_warp = tf.grid_sample(pts2, grid1)
        pts1_warp = tf.grid_sample(pts1, grid2)

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
        pose_im_loss_sum = 0
        pose_pts_loss_sum = 0
        sf_im_loss_sum = 0
        sf_pts_loss_sum = 0
        sf_sm_sum = 0
        disp_sm_sum = 0
        mask_loss_sum = 0
        flow_pts_sum = 0

        img_l1 = target['input_l1']
        img_l2 = target['input_l2']
        img_r1 = target['input_r1']
        img_r2 = target['input_r2']

        K_l1 = target['input_k_l1_aug']
        K_l2 = target['input_k_l2_aug']

        aug_size = target['aug_size']

        disps_l1 = output['disps_l1']
        disps_l2 = output['disps_l2']
        disps_r1 = output['output_dict_r']['disps_l1']
        disps_r2 = output['output_dict_r']['disps_l2']
        flows_f = output['flows_f']
        flows_b = output['flows_b']
        poses_f = output['pose_f']
        poses_b = output['pose_b']

        if self.args.use_mask:
            masks_l1 = output['masks_l1']
            masks_l2 = output['masks_l2']

        census_masks_l1 = []
        census_masks_l2 = []
        
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

            # flow_f = interpolate2d_as(flow_f, img_l1)
            # flow_b = interpolate2d_as(flow_b, img_l2)
            # disp_l1 = interpolate2d_as(disp_l1, img_l1)
            # disp_l2 = interpolate2d_as(disp_l2, img_l2)
            # disp_r1 = interpolate2d_as(disp_r1, img_l1)
            # disp_r2 = interpolate2d_as(disp_r2, img_l2)

            img_l1 = interpolate2d_as(img_l1, disp_l1)
            img_l2 = interpolate2d_as(img_l2, disp_l2)
            img_r1 = interpolate2d_as(img_r1, disp_r1)
            img_r2 = interpolate2d_as(img_r2, disp_r2)

            if isinstance(poses_f, list) and isinstance(poses_b, list):
                pose_b = poses_b[s]
                pose_f = poses_f[s]
            else:
                pose_f = poses_f
                pose_b = poses_b

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

            # depth diffs
            disp_diff1, left_occ1, loss_disp_sm1 = self.depth_loss(disp_l1, disp_r1, img_l1, img_r1, s)
            disp_diff2, left_occ2, loss_disp_sm2 = self.depth_loss(disp_l2, disp_r2, img_l2, img_r2, s)
            loss_disp_sm = (loss_disp_sm1 + loss_disp_sm2) * self.disp_sm_w

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

            depth_l1 = _disp2depth_kitti_K(disp_l1, K_l1_s[:, 0, 0])
            depth_l2 = _disp2depth_kitti_K(disp_l2, K_l2_s[:, 0, 0])

            # pose diffs
            pose_diff1, pose_occ_b, (_, pose_grid1) = self.flow_loss(disp=disp_l1, src=img_l2, tgt=img_l1, K=K_l1_s, T=pose_f, mode='pose')
            pose_diff2, pose_occ_f, (_, pose_grid2) = self.flow_loss(disp=disp_l2, src=img_l1, tgt=img_l2, K=K_l2_s, T=pose_b, mode='pose')

            # sf diffs
            sf_diff1, sf_occ_b, (pts1, sf_grid1)  = self.flow_loss(disp=disp_l1, src=img_l2, tgt=img_l1, K=K_l1_s, sf=flow_f, mode='sf')
            sf_diff2, sf_occ_f, (pts2, sf_grid2)  = self.flow_loss(disp=disp_l2, src=img_l1, tgt=img_l2, K=K_l2_s, sf=flow_b, mode='sf')                

            """ SF SMOOTHNESS LOSS """
            pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
            pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
            loss_sf_sm = ( (_smoothness_motion_2nd(flow_f, img_l1, beta=10.0) / (pts_norm1 + 1e-8)).mean() + \
                (_smoothness_motion_2nd(flow_b, img_l2, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** s)
            loss_sf_sm = loss_sf_sm * self.flow_sm_w

            """ 3D STRUCTURE LOSS """
            pose_pts_diff1, pose_pts_diff2 = self.structure_loss(pts1, pts2, pose_grid1, pose_grid2, T=[pose_f, pose_b], mode='pose', s=s)
            sf_pts_diff1, sf_pts_diff2 = self.structure_loss(pts1, pts2, sf_grid1, sf_grid2, sf=[flow_f, flow_b], mode='sf', s=s)

            """ DEPTH LOSS """
            disp_mask1 = left_occ1
            disp_mask2 = left_occ2
            if self.use_disp_min:
                # calculate min disparity diff across (L+1 and R)
                mask_disp_diff1 = (disp_diff1 < pose_diff1).detach()
                mask_disp_diff2 = (disp_diff2 < pose_diff2).detach()
                disp_mask1 = disp_mask1 * mask_disp_diff1
                disp_mask2 = disp_mask2 * mask_disp_diff2

            depth_loss1 = disp_diff1[disp_mask1].mean()
            depth_loss2 = disp_diff2[disp_mask2].mean()
            disp_diff1[~disp_mask1].detach_()
            disp_diff2[~disp_mask2].detach_()
            depth_loss = depth_loss1 + depth_loss2

            """ CENSUS MASK """
            # pose_sf_f = pose2sceneflow(depth_l1.squeeze(dim=1), None, torch.inverse(K_l1_s), pose_mat=pose_f)
            # pose_sf_b = pose2sceneflow(depth_l2.squeeze(dim=1), None, torch.inverse(K_l2_s), pose_mat=pose_b)
            pose_of_f = pose2flow(depth_l1.squeeze(dim=1), None, K_l1_s, torch.inverse(K_l1_s), pose_mat=pose_f)
            pose_of_b = pose2flow(depth_l2.squeeze(dim=1), None, K_l2_s, torch.inverse(K_l2_s), pose_mat=pose_b)
            of_f = projectSceneFlow2Flow(K_l1_s, flow_f, disp_l1)
            of_b = projectSceneFlow2Flow(K_l2_s, flow_b, disp_l2)
            flow_diff_f = _elementwise_epe(pose_of_f, of_f)
            flow_diff_b = _elementwise_epe(pose_of_b, of_b)
            census_mask_l1 = self.create_census_mask(flow_diff_f, pose_diff1, sf_diff1)
            census_mask_l2 = self.create_census_mask(flow_diff_b, pose_diff2, sf_diff2)
            census_masks_l1.append(census_mask_l1)
            census_masks_l2.append(census_mask_l2)

            """ MASK LOSS """
            if self.args.use_mask:
                mask_reg_loss1, mask_sm_loss1 = self.mask_loss(mask_l1, flow_diff_f, pose_diff1, sf_diff1)
                mask_loss1 = mask_reg_loss1 * self.mask_reg_w + \
                             mask_sm_loss1 * self.mask_sm_w + \
                             mask_census_loss1 * self.mask_cons_w

                mask_reg_loss2, mask_sm_loss2 = self.mask_loss(mask_l2, flow_diff_b, pose_diff2, sf_diff2)
                mask_loss2 = mask_reg_loss2 * self.mask_reg_w + \
                             mask_sm_loss2 * self.mask_sm_w + \
                             mask_census_loss2 * self.mask_cons_w

            if self.args.use_census_mask:
                mask_census_loss1 = tf.binary_cross_entropy(mask_l1, census_mask_l1)
                mask_census_loss2 = tf.binary_cross_entropy(mask_l2, census_mask_l2)

                mask_loss = mask_loss1 + mask_loss2

                pose_diff1 = pose_diff1 * mask_l1
                pose_diff2 = pose_diff2 * mask_l2
                pose_pts_diff1 = pose_pts_diff1 * mask_l1
                pose_pts_diff2 = pose_pts_diff2 * mask_l2
                if self.args.use_flow_mask:
                    flow_diff1 = flow_diff1 * flow_mask_l1
                    flow_diff2 = flow_diff2 * flow_mask_l2
                    sf_pts_diff1 = sf_pts_diff1 * mask_l1
                    sf_pts_diff2 = sf_pts_diff2 * mask_l2
            else:
                mask_loss = torch.tensor(0, requires_grad=False)

            pose_occ_f = pose_occ_f * left_occ1
            sf_occ_f = sf_occ_f * left_occ1
            pose_occ_b = pose_occ_b * left_occ2
            sf_occ_b = sf_occ_b * left_occ2


            # remove static pixels from loss calculation
            if self.args.use_static_mask:
                static_diff = _reconstruction_error(img_l1, img_l2, self.ssim_w)
                pose_static_thresh1 =  pose_diff1 < static_diff
                pose_static_thresh2 =  pose_diff2 < static_diff
                sf_static_thresh1 =  sf_diff1 < static_diff
                sf_static_thresh2 =  sf_diff2 < static_diff
                pose_occ_f = pose_occ_f * pose_static_thresh1
                sf_occ_f = sf_occ_f * sf_static_thresh1
                pose_occ_b = pose_occ_b * pose_static_thresh2
                sf_occ_b = sf_occ_b * sf_static_thresh2

            """ FLOW LOSS """
            # do not use min reduction for now... 
            if self.flow_reduce_mode == 'min':
                assert (not self.args.use_mask), "Cannot use min flow reduction while using explainability mask."
                flow_diffs1 = torch.cat([pose_diff1, sf_diff1], dim=1)
                flow_diffs2 = torch.cat([pose_diff2, sf_diff2], dim=1)
                min_flow_diff1, _ = flow_diffs1.min(dim=1, keepdim=True)
                min_flow_diff2, _ = flow_diffs2.min(dim=1, keepdim=True)
                occ_f = pose_occ_f *  sf_occ_f
                occ_b = pose_occ_b *  sf_occ_b
                flow_loss1 = min_flow_diff1[occ_f].mean()
                flow_loss2 = min_flow_diff2[occ_b].mean()
            elif self.flow_reduce_mode == 'avg':
                flow_loss1 = ((pose_diff1 * pose_occ_f.float()) + (sf_diff1 * sf_occ_f.float())).mean(dim=1).mean()
                flow_loss2 = ((pose_diff2 * pose_occ_b.float()) + (sf_diff2 * sf_occ_b.float())).mean(dim=1).mean()
                if self.flow_pts_w > 0.0:
                    flow_pts_loss1 = ((pose_pts_diff1 * pose_occ_f.float()) + (sf_pts_diff1 * sf_occ_f.float())).mean(dim=1).mean()
                    flow_pts_loss2 = ((pose_pts_diff2 * pose_occ_b.float()) + (sf_pts_diff2 * sf_occ_b.float())).mean(dim=1).mean()
                else:
                    flow_pts_loss1 = torch.tensor(0.0, requires_grad=False)
                    flow_pts_loss2 = torch.tensor(0.0, requires_grad=False)
            elif self.flow_reduce_mode == 'sum':
                flow_loss1 = ((pose_diff1 * pose_occ_f.float()) + (sf_diff1 * sf_occ_f.float())).sum(dim=1).mean()
                flow_loss2 = ((pose_diff2 * pose_occ_b.float()) + (sf_diff2 * sf_occ_b.float())).sum(dim=1).mean()
                if self.flow_pts_w > 0.0:
                    flow_pts_loss1 = pose_pts_diff1.mean(dim=1, keepdim=True)[pose_occ_f].mean() + sf_pts_diff1.mean(dim=1, keepdim=True)[sf_occ_f].mean()
                    flow_pts_loss2 = pose_pts_diff2.mean(dim=1, keepdim=True)[pose_occ_b].mean() + sf_pts_diff2.mean(dim=1, keepdim=True)[sf_occ_b].mean()
                else:
                    flow_pts_loss1 = torch.tensor(0.0, requires_grad=False)
                    flow_pts_loss2 = torch.tensor(0.0, requires_grad=False)

            # calculate losses for logging
            pose_im_loss = (pose_diff1.mean(dim=1, keepdim=True)[pose_occ_f].mean() + \
                            pose_diff2.mean(dim=1, keepdim=True)[pose_occ_b].mean()).detach()

            pose_pts_loss = (pose_pts_diff1.mean(dim=1, keepdim=True)[pose_occ_f].mean() + \
                             pose_pts_diff2.mean(dim=1, keepdim=True)[pose_occ_b].mean()).detach()

            sf_im_loss = (sf_diff1.mean(dim=1, keepdim=True)[sf_occ_f].mean() + \
                          sf_diff2.mean(dim=1, keepdim=True)[sf_occ_b].mean()).detach()

            sf_pts_loss = (sf_pts_diff1.mean(dim=1, keepdim=True)[sf_occ_f].mean() + \
                           sf_pts_diff2.mean(dim=1, keepdim=True)[sf_occ_b].mean()).detach()

            flow_loss = flow_loss1 + flow_loss2
            pts_loss = (flow_pts_loss1 + flow_pts_loss2).detach() * self.flow_pts_w

            pose_diff1[~pose_occ_f].detach_()
            sf_diff1[~sf_occ_f].detach_()
            pose_diff2[~pose_occ_b].detach_()
            sf_diff2[~sf_occ_b].detach_()

            depth_loss_sum = depth_loss_sum + (depth_loss + loss_disp_sm) * self.scale_weights[s]
            flow_loss_sum = flow_loss_sum + (flow_loss + loss_sf_sm) * self.scale_weights[s] 
            mask_loss_sum = mask_loss_sum + mask_loss * self.scale_weights[s]
            pose_im_loss_sum = pose_im_loss_sum + pose_im_loss
            pose_pts_loss_sum = pose_pts_loss_sum + pose_pts_loss
            sf_im_loss_sum = sf_im_loss_sum + sf_im_loss
            sf_pts_loss_sum = sf_pts_loss_sum + sf_pts_loss
            sf_sm_sum = sf_sm_sum + loss_sf_sm
            flow_pts_sum = flow_pts_sum + pts_loss
            disp_sm_sum = disp_sm_sum + loss_disp_sm
        
        loss_dict = {}
        loss_dict["total_loss"] = (depth_loss_sum + flow_loss_sum) / num_scales
        loss_dict["depth_loss"] = depth_loss_sum.detach()
        loss_dict["flow_loss"] = flow_loss_sum.detach()
        loss_dict["pts_loss"] = flow_pts_sum.detach()
        loss_dict["pose_im_loss"] = pose_im_loss_sum.detach()
        loss_dict["sf_im_loss"] = sf_im_loss_sum.detach()
        loss_dict["pose_pts_loss"] = pose_pts_loss_sum.detach()
        loss_dict["sf_pts_loss"] = sf_pts_loss_sum.detach()
        loss_dict["sf_smooth_loss"] = sf_sm_sum.detach()
        loss_dict["disp_sm_loss"] = disp_sm_sum.detach()
        loss_dict["mask_loss"] = mask_loss_sum.detach()

        output['census_masks_l1'] = census_masks_l1
        output['census_masks_l2'] = census_masks_l2

        # detach unused parameters
        for s in range(len(output['output_dict_r']['flows_f'])):
            output['output_dict_r']['flows_f'][s].detach_()
            output['output_dict_r']['flows_b'][s].detach_()
            output['output_dict_r']['disps_l1'][s].detach_()
            output['output_dict_r']['disps_l2'][s].detach_()
            if self.args.use_mask:
                output['output_dict_r']['masks_l1'][s].detach_()
                output['output_dict_r']['masks_l2'][s].detach_()

        return loss_dict
