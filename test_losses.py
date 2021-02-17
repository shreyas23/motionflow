import torch
import torch.nn as nn
import torch.nn.functional as tf

from utils.loss_utils import _generate_image_left, _elementwise_epe, _elementwise_l1, _smoothness_motion_2nd
from utils.loss_utils import _adaptive_disocc_detection_disp, _adaptive_disocc_detection, _SSIM, logical_or
from utils.loss_utils import _reconstruction_error, kl_div, _smoothness_1st
from models.forwardwarp_package.forward_warp import forward_warp
from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, reconstructImg, reconstructPts, projectSceneFlow2Flow, disp2depth_kitti
from utils.sceneflow_util import intrinsic_scale, reconstructFlow, reconstructMask
from models.modules_sceneflow import WarpingLayer_Flow

from utils.inverse_warp import pose2sceneflow
from models.common import GaussianSmoothing

###############################################
## Loss function
###############################################

class Loss_SceneFlow_SelfSup(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup, self).__init__()

        self.args = args
        self.weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self.ssim_w = args.ssim_w
        self.disp_smooth_w = args.disp_sm_w
        self.sf_3d_pts = args.flow_pts_w
        self.sf_3d_sm = args.flow_sm_w
        self.apply_mask = args.apply_mask
        self.apply_flow_mask = args.apply_flow_mask
        self.mask_cons_w = args.mask_cons_w
        self.mask_reg_w = args.mask_reg_w
        self.mask_sm_w = args.mask_sm_w
        self.static_cons_w = args.static_cons_w
        self.flow_diff_thresh = args.flow_diff_thresh
        self.flow_cycle_w = args.flow_cycle_w
        self.mask_cycle_w = args.mask_cycle_w
        self.feat_smooth_w = args.feat_smooth_w
        self.feat_disc_w = args.feat_disc_w

    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):
    # def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, feat_l, feat_r, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        # feat_r_warp = _generate_image_left(feat_r, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Photometric loss
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self.ssim_w) + _SSIM(img_l_aug, img_r_warp) * self.ssim_w).mean(dim=1, keepdim=True)        
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        ## Feature loss
        # feat_diff = _elementwise_l1(feat_l, feat_r_warp).mean(dim=1, keepdim=True)
        # loss_feat = (feat_diff[left_occ]).mean()
        # feat_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self.disp_smooth_w * loss_smooth, left_occ
        # return loss_img + loss_feat + self.disp_smooth_w * loss_smooth, loss_feat, left_occ


    def mask_loss(self, image, mask, census_target, scale):
        reg_loss = tf.binary_cross_entropy(mask, torch.ones_like(mask))
        sm_loss = _smoothness_motion_2nd(mask, image, beta=10.0).mean() / (2**scale)
        census_loss = tf.binary_cross_entropy(mask, census_target)

        return reg_loss, sm_loss, census_loss
    

    def create_census_mask(self, mask_flow_diff, pose_err, sf_err, smooth=False):
        if smooth:
            _, c, _, _ = pose_err.shape
            f = GaussianSmoothing(channels=c, kernel_size=3, sigma=1.0).to(device=pose_err.device)
            pose_err = f(pose_err)
            sf_err = f(sf_err)
        target_mask = (pose_err <= sf_err).float().detach()
        census_target_mask = logical_or(target_mask, mask_flow_diff).detach()
        return census_target_mask


    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, mask_l1, mask_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):
    # def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, mask_l1, mask_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, feat_l1, feat_l2, aug_size, ii):

        _, _, h_dp, w_dp = sf_f.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        ## scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp         

        pts1, k1_scale = pixel2pts_ms(k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale = pixel2pts_ms(k_l2_aug, disp_l2, local_scale / aug_size)

        _, pts1_tf, coord1 = pts2pixel_ms(k1_scale, pts1, sf_f, [h_dp, w_dp])
        _, pts2_tf, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp]) 

        pts2_warp = reconstructPts(coord1, pts2)
        pts1_warp = reconstructPts(coord2, pts1) 

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        ## Image reconstruction loss
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        ## Feature reconstruction loss
        # feat_l2_warp = reconstructImg(coord1, feat_l2)
        # feat_l1_warp = reconstructImg(coord2, feat_l1)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self.ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self.ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self.ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self.ssim_w).mean(dim=1, keepdim=True)

        loss_im1 = (img_diff1 * mask_l1)[occ_map_f].mean()
        loss_im2 = (img_diff2 * mask_l2)[occ_map_b].mean()

        img_diff1[~occ_map_f].detach_()
        img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2

        # feat_diff1 = _elementwise_l1(feat_l1, feat_l2_warp).mean(dim=1, keepdim=True) 
        # feat_diff2 = _elementwise_l1(feat_l2, feat_l1_warp).mean(dim=1, keepdim=True) 
        # loss_feat1 = (feat_diff1 * mask_l1)[occ_map_f].mean()
        # loss_feat2 = (feat_diff2 * mask_l2)[occ_map_b].mean()
        # feat_diff1[~occ_map_f].detach_()
        # feat_diff2[~occ_map_b].detach_()
        # loss_feat = loss_feat1 + loss_feat2
        
        ## Point reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        loss_pts1 = (pts_diff1 * mask_l1)[occ_map_f].mean()
        loss_pts2 = (pts_diff2 * mask_l2)[occ_map_b].mean()
        pts_diff1[~occ_map_f].detach_()
        pts_diff2[~occ_map_b].detach_()
        loss_pts = loss_pts1 + loss_pts2

        flow_b_warp = reconstructFlow(coord1, flow_b)
        flow_f_warp = reconstructFlow(coord2, flow_f)

        flow_f_cycle_diff = torch.norm(flow_f + flow_b_warp, p=1, dim=1, keepdim=True)
        flow_b_cycle_diff = torch.norm(flow_b + flow_f_warp, p=1, dim=1, keepdim=True)

        cycle_occ = occ_map_f * occ_map_b
        flow_cycle_loss = flow_f_cycle_diff[cycle_occ].mean() + flow_b_cycle_diff[cycle_occ].mean()

        mask_l2_warp = reconstructMask(coord1, torch.log(mask_l2))
        mask_l1_warp = reconstructMask(coord2, torch.log(mask_l1))
        mask_l1_cycle_diff = kl_div(mask_l2_warp, mask_l1)
        mask_l2_cycle_diff = kl_div(mask_l1_warp, mask_l2)
        mask_cycle_loss = mask_l1_cycle_diff[cycle_occ].mean() + mask_l2_cycle_diff[cycle_occ].mean()

        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        # sceneflow_loss = loss_im + loss_feat + self.sf_3d_pts * loss_pts + self.sf_3d_sm * loss_3d_s + self.flow_cycle_w * flow_cycle_loss
        sceneflow_loss = loss_im + self.sf_3d_pts * loss_pts + self.sf_3d_sm * loss_3d_s + self.flow_cycle_w * flow_cycle_loss
        
        return sceneflow_loss, loss_im, loss_pts, loss_3d_s, flow_cycle_loss, mask_cycle_loss, (img_diff1, img_diff2)
        # return sceneflow_loss, loss_im, loss_feat, loss_pts, loss_3d_s, flow_cycle_loss, mask_cycle_loss, (img_diff1, img_diff2)

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flows_f'])):
            output_dict['flows_f'][ii].detach_()
            output_dict['flows_b'][ii].detach_()
            output_dict['disps_l1'][ii].detach_()
            output_dict['disps_l2'][ii].detach_()
            output_dict['masks_l1'][ii].detach_()
            output_dict['masks_l2'][ii].detach_()
            output_dict['pose_f'][ii].detach_()
            output_dict['pose_b'][ii].detach_()
            # output_dict['feats_l1'][ii].detach_()
            # output_dict['feats_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        loss_sf_sum = 0
        loss_pose_sum = 0
        loss_mask_sum = 0
        loss_mask_reg_sum = 0
        loss_mask_sm_sum = 0
        loss_mask_census_sum = 0
        loss_mask_cycle_sum = 0
        loss_cons_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_pose_2d = 0
        loss_pose_3d = 0
        loss_sf_sm = 0
        loss_cycle_sum = 0
        loss_feat_smooth_sum = 0
        loss_sf_feat_sum = 0
        loss_pose_feat_sum = 0
        loss_disp_feat_sum = 0
        loss_feat_disc_sum = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disps_l1']
        disp_r2_dict = output_dict['output_dict_r']['disps_l2']

        poses_f = output_dict['pose_f']
        poses_b = output_dict['pose_b']
        masks_l1 = output_dict['masks_l1']
        masks_l2 = output_dict['masks_l2']

        # feats_l1 = output_dict['feats_l1']
        # feats_l2 = output_dict['feats_l2']

        # feats_r1 = output_dict['output_dict_r']['feats_l1']
        # feats_r2 = output_dict['output_dict_r']['feats_l2']

        census_masks_l1 = []
        census_masks_l2 = []

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2, pose_f, pose_b, mask_l1, mask_l2) in enumerate(zip(output_dict['flows_f'], 
        # for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2, pose_f, pose_b, mask_l1, mask_l2, feat_l1, feat_l2, feat_r1, feat_r2) in enumerate(zip(output_dict['flows_f'], 
                                                                                                  output_dict['flows_b'], 
                                                                                                  output_dict['disps_l1'], 
                                                                                                  output_dict['disps_l2'], 
                                                                                                  disp_r1_dict, disp_r2_dict, 
                                                                                                  poses_f, poses_b,
                                                                                                  masks_l1, masks_l2,
                                                                                                #   feats_l1, feats_l2,
                                                                                                #   feats_r1, feats_r2
                                                                                                  )):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            ## feature smoothness loss
            # loss_feat_smooth_l1 = _smoothness_motion_2nd(feat_l1, img_l1_aug, beta=0).mean(dim=1, keepdim=True).mean()
            # loss_feat_smooth_l2 = _smoothness_motion_2nd(feat_l2, img_l2_aug, beta=0).mean(dim=1, keepdim=True).mean()
            # loss_feat_smooth_sum = loss_feat_smooth_sum + (loss_feat_smooth_l1 + loss_feat_smooth_l2) * self.weights[ii]

            ## feature distinctio
            # loss_feat_disc_l1 = _smoothness_1st(feat_l1, img_l1_aug).mean(dim=1, keepdim=True).mean()
            # loss_feat_disc_l2 = _smoothness_1st(feat_l2, img_l2_aug).mean(dim=1, keepdim=True).mean()
            # loss_feat_disc_sum = loss_feat_disc_sum + (loss_feat_disc_l1 + loss_feat_disc_l2) * self.weights[ii]

            ## Disp Loss
            loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            # loss_disp_l1, loss_disp_feat_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, feat_l1, feat_r1, ii)
            # loss_disp_l2, loss_disp_feat_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, feat_l2, feat_r2, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self.weights[ii]
            # loss_disp_feat_sum = loss_disp_feat_sum + loss_disp_feat_l1 + loss_disp_feat_l2

            ## Sceneflow Loss           
            if self.apply_flow_mask:
                flow_mask_l1 = 1.0 - mask_l1
                flow_mask_l2 = 1.0 - mask_l2
            else:
                flow_mask_l1 = torch.ones_like(mask_l1, requires_grad=False)
                flow_mask_l2 = torch.ones_like(mask_l2, requires_grad=False)

            loss_sceneflow, loss_im, loss_pts, loss_3d_s, sf_cycle, _, (sf_diff_f, sf_diff_b) = self.sceneflow_loss(sf_f, sf_b, 
            # loss_sceneflow, loss_im, loss_sf_feat, loss_pts, loss_3d_s, sf_cycle, _, (sf_diff_f, sf_diff_b) = self.sceneflow_loss(sf_f, sf_b, 
                                                                                                disp_l1, disp_l2,
                                                                                                disp_occ_l1, disp_occ_l2,
                                                                                                flow_mask_l1, flow_mask_l2,
                                                                                                k_l1_aug, k_l2_aug,
                                                                                                img_l1_aug, img_l2_aug, 
                                                                                                # feat_l1, feat_l2,
                                                                                                aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self.weights[ii]            
            loss_sf_2d = loss_sf_2d + loss_im            
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s
            loss_cycle_sum = loss_cycle_sum + sf_cycle

            _, _, h, w = disp_l1.shape
            disp_l1_s = disp_l1 * w
            disp_l2_s = disp_l2 * w

            ## scale
            local_scale = torch.zeros_like(aug_size)
            local_scale[:, 0] = h
            local_scale[:, 1] = w         

            rel_scale = local_scale / aug_size

            K_l1_s = intrinsic_scale(k_l1_aug, rel_scale[:, 0], rel_scale[:, 1])
            K_l2_s = intrinsic_scale(k_l2_aug, rel_scale[:, 0], rel_scale[:, 1])

            depth_l1 = disp2depth_kitti(disp_l1_s, K_l1_s[:, 0, 0])
            depth_l2 = disp2depth_kitti(disp_l2_s, K_l2_s[:, 0, 0])
            pose_sf_f = pose2sceneflow(depth_l1, None, K_l1_s, pose_mat=pose_f)
            pose_sf_b = pose2sceneflow(depth_l2, None, K_l2_s, pose_mat=pose_b)

            loss_pose, loss_pose_im, loss_pose_pts, _, _, mask_cycle_loss, (pose_diff_f, pose_diff_b) = self.sceneflow_loss(pose_sf_f, pose_sf_b, 
            # loss_pose, loss_pose_im, loss_pose_feat, loss_pose_pts, _, _, mask_cycle_loss, (pose_diff_f, pose_diff_b) = self.sceneflow_loss(pose_sf_f, pose_sf_b, 
                                                                                        disp_l1, disp_l2,
                                                                                        disp_occ_l1, disp_occ_l2,
                                                                                        mask_l1, mask_l2,
                                                                                        k_l1_aug, k_l2_aug,
                                                                                        img_l1_aug, img_l2_aug, 
                                                                                        # feat_l1, feat_l2,
                                                                                        aug_size, ii)

            loss_pose_sum = loss_pose_sum + loss_pose * self.weights[ii]            
            loss_pose_2d = loss_pose_2d + loss_pose_im
            loss_pose_3d = loss_pose_3d + loss_pose_pts

            # loss_sf_feat_sum = loss_sf_feat_sum + loss_sf_feat
            # loss_pose_feat_sum = loss_pose_feat_sum + loss_pose_feat

            # mask_flow_diff_f = ((pose_sf_f - sf_f).abs() <= self.flow_diff_thresh).prod(dim=1, keepdim=True).float()
            # mask_flow_diff_b = ((pose_sf_b - sf_b).abs() <= self.flow_diff_thresh).prod(dim=1, keepdim=True).float()
            mask_flow_diff_f = (_elementwise_epe(pose_sf_f, sf_f) <= self.flow_diff_thresh).float()
            mask_flow_diff_b = (_elementwise_epe(pose_sf_b, sf_b) <= self.flow_diff_thresh).float()
            census_tgt_l1 = self.create_census_mask(mask_flow_diff_f, pose_diff_f, sf_diff_f)
            census_tgt_l2 = self.create_census_mask(mask_flow_diff_b, pose_diff_b, sf_diff_b)
            mask_reg_loss_l1, mask_sm_loss_l1, mask_census_loss_l1 = self.mask_loss(img_l1_aug, mask_l1, census_tgt_l1, ii)
            mask_reg_loss_l2, mask_sm_loss_l2, mask_census_loss_l2 = self.mask_loss(img_l2_aug, mask_l2, census_tgt_l2, ii)

            # store census masks
            census_masks_l1.append(census_tgt_l1)
            census_masks_l2.append(census_tgt_l2)

            mask_reg_loss = mask_reg_loss_l1 + mask_reg_loss_l2
            mask_sm_loss = mask_sm_loss_l1 + mask_sm_loss_l2
            mask_census_loss = mask_census_loss_l1 + mask_census_loss_l2

            loss_mask_sum = loss_mask_sum + (mask_reg_loss * self.mask_reg_w + \
                                            mask_sm_loss * self.mask_sm_w + \
                                            mask_census_loss * self.mask_cons_w) * self.weights[ii]

            loss_mask_reg_sum = loss_mask_reg_sum + mask_reg_loss
            loss_mask_sm_sum = loss_mask_sm_sum + mask_sm_loss
            loss_mask_census_sum = loss_mask_census_sum + mask_census_loss
            loss_mask_cycle_sum = loss_mask_cycle_sum + mask_cycle_loss

            # if self.args.apply_mask:
            #     static_mask_l1 = mask_l1 #* census_tgt_l1.detach()
            #     static_mask_l2 = mask_l2 #* census_tgt_l2.detach()
            # else:
            #     static_mask_l1 = torch.ones_like(mask_l1, requires_grad=False)
            #     static_mask_l2 = torch.ones_like(mask_l2, requires_grad=False)

            flow_diff_f = _elementwise_epe(sf_f, pose_sf_f)
            flow_diff_b = _elementwise_epe(sf_b, pose_sf_b)
            cons_loss_f = (flow_diff_f * mask_l1).mean()
            cons_loss_b = (flow_diff_b * mask_l2).mean()
            cons_loss = (cons_loss_f + cons_loss_b)

            loss_cons_sum = loss_cons_sum + cons_loss

        # finding weight
        f_loss = max(loss_sf_sum.detach(), loss_pose_sum.detach())
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        if self.args.train_exp_mask:
            loss_mask_sum = loss_mask_sum * f_weight

        total_loss = loss_sf_sum * f_weight + \
                     loss_dp_sum * d_weight + \
                     loss_pose_sum * f_weight + \
                     loss_mask_sum + \
                     loss_cons_sum * self.static_cons_w #+ \
                    #  loss_feat_smooth_sum * self.feat_smooth_w + \
                    #  loss_feat_disc_sum * self.feat_disc_w

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum.detach()
        # loss_dict['disp_feat'] = loss_disp_feat_sum.detach()
        loss_dict["sf"] = loss_sf_sum.detach()
        loss_dict["s_2"] = loss_sf_2d.detach()
        loss_dict["s_3"] = loss_sf_3d.detach()
        loss_dict["s_3s"] = loss_sf_sm.detach()
        # loss_dict['sf_feat'] = loss_sf_feat_sum.detach()
        loss_dict["pose"] = loss_pose_sum.detach()
        loss_dict["pose_2"] = loss_pose_2d.detach()
        loss_dict["pose_3"] = loss_pose_3d.detach()
        # loss_dict['pose_feat'] = loss_pose_feat_sum.detach()
        loss_dict["mask"] = loss_mask_sum.detach()
        loss_dict["mask_sm"] = loss_mask_sm_sum.detach()
        loss_dict["mask_census"] = loss_mask_census_sum.detach()
        loss_dict["mask_cycle"] = loss_mask_cycle_sum.detach()
        loss_dict["mask_reg"] = loss_mask_reg_sum.detach()
        loss_dict["static_cons"] = loss_cons_sum.detach()
        loss_dict["cycle"] = loss_cycle_sum.detach()
        # loss_dict['feat_smooth'] = loss_feat_smooth_sum.detach()
        # loss_dict['feat_disc'] = loss_feat_disc_sum.detach()
        loss_dict["total_loss"] = total_loss


        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        output_dict['census_masks_l1'] = census_masks_l1
        output_dict['census_masks_l2'] = census_masks_l2

        return loss_dict