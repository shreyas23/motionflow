import torch
import torch.nn as nn
import torch.nn.functional as tf

from models.forwardwarp_package.forward_warp import forward_warp
from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, reconstructImg, reconstructPts, projectSceneFlow2Flow
from utils.sceneflow_util import pts2pixel_pose_ms
from utils.monodepth_eval import compute_errors, compute_d1_all
from models.modules_sceneflow import WarpingLayer_Flow

from utils.inverse_warp import inverse_warp, pose_vec2mat, flow_warp, pose2flow, pose2sceneflow
from utils.sceneflow_util import pixel2pts_ms_depth

import matplotlib.pyplot as plt

eps = 1e-8

###############################################
## Basic Module 
###############################################

def logical_or(x, y):
    return 1 - (1-x)*(1-y)

def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)

def _elementwise_l1(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=1, dim=1, keepdim=True)

def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)

def _SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    SSIM_img = torch.clamp((1 - SSIM) / 2, 0, 1)

    return tf.pad(SSIM_img, pad=(1, 1, 1, 1), mode='constant', value=0)

def _apply_disparity(img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = tf.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

    return output

def _apply_disparity_ret(img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = tf.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='zeros')

    return output, flow_field

def _generate_flow_left(flow, r2l):
    warp_flow = None
    return warp_flow
def _generate_flow_right(flow, l2r):
    warp_flow = None
    return warp_flow
def _generate_pts_left(pts, r2l):
    warp_pts = None
    return warp_pts
def _generate_pts_right(pts, l2r):
    warp_pts = None
    return warp_pts

def _generate_image_left(img, disp):
    return _apply_disparity(img, -disp)

def _generate_image_right(img, disp):
    return _apply_disparity(img, disp)

def _adaptive_disocc_detection(flow):

    # init mask
    b, _, h, w, = flow.size()
    mask = torch.ones(b, 1, h, w, dtype=flow.dtype, device=flow.device).float().requires_grad_(False)    
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1) 
    disocc_map = (disocc > 0.5)

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(b, 1, h, w, dtype=torch.bool, device=flow.device).requires_grad_(False)
        
    return disocc_map

def _adaptive_disocc_detection_disp(disp):

    # # init
    b, _, h, w, = disp.size()
    mask = torch.ones(b, 1, h, w, dtype=disp.dtype, device=disp.device).float().requires_grad_(False)
    flow = torch.zeros(b, 2, h, w, dtype=disp.dtype, device=disp.device).float().requires_grad_(False)
    flow[:, 0:1, :, : ] = disp * w
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1) 
    disocc_map = (disocc > 0.5)

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(b, 1, h, w, dtype=torch.bool, device=disp.device).requires_grad_(False)
        
    return disocc_map

def _gradient_x(img):
    img = tf.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx

def _gradient_y(img):
    img = tf.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy

def _gradient_x_2nd(img):
    img_l = tf.pad(img, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]
    img_r = tf.pad(img, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]
    gx = img_l + img_r - 2 * img
    return gx

def _gradient_y_2nd(img):
    img_t = tf.pad(img, (0, 0, 1, 0), mode="replicate")[:, :, :-1, :]
    img_b = tf.pad(img, (0, 0, 0, 1), mode="replicate")[:, :, 1:, :]
    gy = img_t + img_b - 2 * img
    return gy

def _smoothness_motion_2nd(sf, img, beta=1):
    sf_grad_x = _gradient_x_2nd(sf)
    sf_grad_y = _gradient_y_2nd(sf)

    img_grad_x = _gradient_x(img) 
    img_grad_y = _gradient_y(img) 
    weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x), 1, keepdim=True) * beta)
    weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y), 1, keepdim=True) * beta)

    smoothness_x = sf_grad_x * weights_x
    smoothness_y = sf_grad_y * weights_y

    return (smoothness_x.abs() + smoothness_y.abs())

def _disp2depth_kitti_K(disp, k_value): 

    mask = (disp > 0).float()
    depth = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (disp + (1.0 - mask))

    return depth

def _depth2disp_kitti_K(depth, k_value):

    disp = k_value.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth

    return disp


###############################################
## Loss function
###############################################


class Loss_SceneFlow_SelfSup_JointStereo(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_JointStereo, self).__init__()
        self._args = args
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]

        self._ssim_w = 0.85

        # disp weights
        self._disp_smooth_w = 0.1

        #sf weights
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200

        # pose weights
        self._pose_smooth_w = args.pose_sm_w

        # mask weights
        self._mask_reg_w = args.mask_reg_w
        self._mask_sm_w = args.mask_sm_w
        self._mask_cons_w = args.mask_cons_w
        self._flow_diff_thresh = args.flow_diff_thresh

        # consistency weights 
        self._mask_lr_w = args.mask_lr_w
        self._sf_lr_w = args.sf_lr_w
        self._pose_lr_w = args.pose_lr_w
        self._disp_lr_w = args.disp_lr_w
        self._static_cons_w = args.static_cons_w


    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()
        right_occ = _adaptive_disocc_detection_disp(disp_l).detach()

        ## Photometric loss
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        ## L-R Consistency loss
        proj_disp_r = _generate_image_left(disp_r, disp_l)
        proj_disp_l = _generate_image_right(disp_l, disp_r)
        lr_disp_diff_l = torch.abs(proj_disp_r - disp_l)
        lr_disp_diff_r = torch.abs(proj_disp_l - disp_r)

        loss_lr = lr_disp_diff_l[left_occ].mean() + lr_disp_diff_r[right_occ].mean()
        lr_disp_diff_l[~left_occ].detach_()
        lr_disp_diff_r[~right_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        loss = loss_img + self._disp_smooth_w * loss_smooth + self._disp_lr_w * loss_lr

        return loss, loss_lr, left_occ, right_occ

    def mask_reg_loss(self, mask):
        loss = tf.binary_cross_entropy(mask, torch.ones_like(mask))
        return loss * self._mask_reg_w

    def static_cons_loss(self, mask, sf, pose, disp, disp_occ, intrinsics, aug_size):
        # convert pose params to induced static scene flow
        _, _, h_dp, w_dp = disp.size()
        disp = disp * w_dp
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp         
        _, intrinsics_scaled, depth = pixel2pts_ms_depth(intrinsics, disp, local_scale / aug_size)

        pose_flow = pose2flow(depth.squeeze(dim=1), None, intrinsics, torch.inverse(intrinsics_scaled), pose_mat=pose)
        flow = projectSceneFlow2Flow(intrinsics_scaled, sf, disp)

        flow_diff = (pose_flow - flow).abs().mean(dim=1, keepdim=True)

        # static consistency loss
        flow_loss = _elementwise_epe(pose_flow, flow).mean(dim=1, keepdim=True)
        loss = (flow_loss * mask)[disp_occ].mean()
        flow_loss[~disp_occ].detach_()

        return loss * self._static_cons_w, flow_diff.detach()

    def mask_consensus_loss(self, mask, flow_diff, pose_err, sf_err):
        # mask consensus loss
        target_mask = (pose_err <= sf_err).float()
        flow_similar = (flow_diff < self._flow_diff_thresh).float()
        census_target_mask = logical_or(target_mask, flow_similar).detach()
        loss = tf.binary_cross_entropy(mask, census_target_mask)
        return loss * self._mask_cons_w, census_target_mask

    def mask_lr_loss(self, mask_l, mask_r, disp_l, disp_r, left_occ, right_occ):
        mask_warp_l = _generate_image_left(mask_r, disp_l)
        mask_warp_r = _generate_image_right(mask_l, disp_r)
        lr_mask_diff_l = torch.abs(mask_warp_r - mask_l)
        lr_mask_diff_r = torch.abs(mask_warp_l - mask_r)

        loss_lr = lr_mask_diff_l[left_occ].mean() + lr_mask_diff_r[right_occ].mean()
        lr_mask_diff_l[~left_occ].detach_()
        lr_mask_diff_r[~right_occ].detach_()

        return loss_lr * self._mask_lr_w

    def flow_lr_loss(self, flow_l, flow_r, cam_l2r, cam_r2l, left_occ, right_occ):
        flow_warp_l = _generate_flow_left(flow_r, cam_r2l)
        flow_warp_r = _generate_flow_right(flow_l, cam_l2r)

        diff_l = _elementwise_epe(flow_warp_r, flow_l) .mean(dim=1, keepdim=True)
        diff_r = _elementwise_epe(flow_warp_l, flow_r) .mean(dim=1, keepdim=True)

        loss_lr = diff_l[left_occ].mean() + diff_r[right_occ].mean()
        diff_l[~left_occ].detach_()
        diff_r[~right_occ].detach_()
        
        return loss_lr

    def pts_lr_loss(self, disp_l, disp_r, cam_l2r, cam_r2l, k_l_aug, k_r_aug, left_occ, right_occ, aug_size):

        b, _, h_dp, w_dp = disp_l.size()
        disp_l = disp_l * w_dp
        disp_r = disp_r * w_dp

        # scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp         

        pts_l, _ = pixel2pts_ms(k_l_aug, disp_l, local_scale / aug_size)

        # transform points into right cam coord. system
        pts_l_flat = torch.cat([pts_l.reshape((b, 3, -1)), torch.ones(b, 1, h_dp*w_dp)]) # B, 4, H*W
        pts_l_tform = torch.bmm(cam_l2r, pts_l_flat).reshape((b, 3, h_dp, w_dp))  # B, 3, H, W
        proj_pts_l = _generate_image_right(pts_l_tform, disp_r)

        pts_r, _ = pixel2pts_ms(k_r_aug, disp_r, local_scale / aug_size)

        # transform points into left cam coord. system
        pts_r_flat = torch.cat([pts_r.reshape((b, 3, -1)), torch.ones(b, 1, h_dp*w_dp)]) # B, 4, H*W
        pts_r_tform = torch.bmm(cam_r2l, pts_r_flat).reshape((b, 3, h_dp, w_dp))  # B, 3, H, W
        proj_pts_r = _generate_image_left(pts_r_tform, disp_l)

        pts_norm_l = torch.norm(pts_l, p=2, dim=1, keepdim=True)
        pts_norm_r = torch.norm(pts_r, p=2, dim=1, keepdim=True)

        diff_l = _elementwise_epe(proj_pts_r, pts_l).mean(dim=1, keepdim=True) / (pts_norm_l + 1e-8)
        diff_r = _elementwise_epe(proj_pts_l, pts_r).mean(dim=1, keepdim=True) / (pts_norm_r + 1e-8)

        loss_lr = diff_l[left_occ].mean() + diff_r[right_occ].mean()
        diff_l[~left_occ].detach_()
        diff_r[~right_occ].detach_()

        return loss_lr

    def mask_loss(self, mask):
        reg_loss = tf.binary_cross_entropy(mask, torch.ones_like(mask))
        sm_loss = (_gradient_x_2nd(mask).abs() + _gradient_y_2nd(mask).abs()).mean()

        loss = reg_loss * self._mask_reg_w + sm_loss * self._mask_sm_w

        return loss, reg_loss, sm_loss

    def old_pose_loss(self, pose, disp, disp_occ, intrinsics, ref_img, tgt_img, aug_size, mask=None):

        _, _, h_dp, w_dp = disp.size()
        disp = disp * w_dp

        ## scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp         

        pts, intrinsics_scaled, depth = pixel2pts_ms_depth(intrinsics, disp, local_scale / aug_size)
        _, coord = pts2pixel_pose_ms(intrinsics_scaled, pts, None, [h_dp, w_dp], pose_mat=pose)

        ref_warped = reconstructImg(coord, ref_img)
        valid_pixels = (ref_warped.sum(dim=1, keepdim=True) !=0).detach()
        valid_pixels = valid_pixels * disp_occ.detach()

        img_diff = (_elementwise_l1(tgt_img, ref_warped) * (1.0 - self._ssim_w) + _SSIM(tgt_img, ref_warped) * self._ssim_w).mean(dim=1, keepdim=True)
        if mask is not None:
            loss_img = (img_diff * mask)[valid_pixels].mean()
        else:
            loss_img = img_diff[valid_pixels].mean()
            
        img_diff[~valid_pixels].detach_()

        ## 3D motion smoothness loss
        pts_norm = torch.norm(pts, p=2, dim=1, keepdim=True)
        flow = pose2sceneflow(depth.squeeze(dim=1), None, intrinsics_scaled, torch.inverse(intrinsics_scaled), pose_mat=pose)
        
        loss_smooth = ( _smoothness_motion_2nd(flow, tgt_img, beta=10.0) / (pts_norm + 1e-8)).mean()
        loss = loss_img + loss_smooth * self._pose_smooth_w

        return loss, loss_img, loss_smooth, img_diff

    def pose_loss(self, 
                  pose_f, pose_b, 
                  pose_fr, pose_br,
                  disp_l1, disp_l2,
                  disp_r1, disp_r2,
                  disp_occ_l1, disp_occ_l2,
                  disp_occ_r1, disp_occ_r2,
                  k_l1_aug, k_l2_aug,
                  k_r1_aug, k_r2_aug,
                  img_l1_aug, img_l2_aug,
                  aug_size, ii,
                  mask_f=None, mask_b=None):

        _, _, h_dp, w_dp = disp_l1.size()
        disp_l1 = disp_l1 * w_dp
        disp_l2 = disp_l2 * w_dp

        ## scale
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp         

        pts1, k1_scale, depth_l1 = pixel2pts_ms_depth(k_l1_aug, disp_l1, local_scale / aug_size)
        pts2, k2_scale, depth_l2 = pixel2pts_ms_depth(k_l2_aug, disp_l2, local_scale / aug_size)

        pts1_tf, coord1 = pts2pixel_pose_ms(k1_scale, pts1, None, [h_dp, w_dp], pose_f)
        pts2_tf, coord2 = pts2pixel_pose_ms(k2_scale, pts2, None, [h_dp, w_dp], pose_b) 

        pts2_warp = reconstructPts(coord1, pts2)
        pts1_warp = reconstructPts(coord2, pts1) 

        flow_f = pose2flow(depth_l1.squeeze(dim=1), None, k1_scale, torch.inverse(k1_scale), pose_mat=pose_f)
        flow_b = pose2flow(depth_l2.squeeze(dim=1), None, k2_scale, torch.inverse(k2_scale), pose_mat=pose_b)
        sf_f = pose2sceneflow(depth_l1.squeeze(dim=1), None, torch.inverse(k1_scale), pose_mat=pose_f)
        sf_b = pose2sceneflow(depth_l2.squeeze(dim=1), None, torch.inverse(k2_scale), pose_mat=pose_b)
        occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        # flow lr loss and pts lr loss
        # pts_r1, k_r1_scale, depth_r1 = pixel2pts_ms_depth(k_r1_aug, disp_r1, local_scale / aug_size)
        # pts_r2, k_r2_scale, depth_r2 = pixel2pts_ms_depth(k_r2_aug, disp_r2, local_scale / aug_size)
        # sf_fr = pose2sceneflow(depth_r1.squeeze(dim=1), None, torch.inverse(k_r1_scale), pose_mat=pose_fr)
        # sf_br = pose2sceneflow(depth_r2.squeeze(dim=1), None, torch.inverse(k_r2_scale), pose_mat=pose_br)
        # loss_lr_flow1 = self.pts_lr_loss(sf_f, sf_fr, disp_l1, disp_r1, disp_occ_l1, disp_occ_r1) 
        # loss_lr_flow2 = self.pts_lr_loss(sf_b, sf_br, disp_l2, disp_r2, disp_occ_l2, disp_occ_r2)
        # loss_lr_flow = (loss_lr_flow1 + loss_lr_flow2)

        ## Image reconstruction loss
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        valid_l2_warp = (img_l2_warp.sum(dim=1, keepdim=True) !=0).detach()
        valid_l1_warp = (img_l1_warp.sum(dim=1, keepdim=True) !=0).detach()
        occ_map_f *= valid_l2_warp
        occ_map_b *= valid_l1_warp

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        if mask_f is not None:
            img_diff1 = img_diff1 * mask_f
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        if mask_b is not None:
            img_diff2 = img_diff2 * mask_b
        loss_im1 = img_diff1[occ_map_f].mean()
        loss_im2 = img_diff2[occ_map_b].mean()
        img_diff1[~occ_map_f].detach_()
        img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2

        ## Point reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        if mask_f is not None:
            pts_diff1 = pts_diff1 * mask_f
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        if mask_b is not None:
            pts_diff2 = pts_diff2 * mask_b
        loss_pts1 = pts_diff1[occ_map_f].mean()
        loss_pts2 = pts_diff2[occ_map_b].mean()
        pts_diff1[~occ_map_f].detach_()
        pts_diff2[~occ_map_b].detach_()
        loss_pts = loss_pts1 + loss_pts2

        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s #+ loss_lr * self._pose_lr_w
        
        # return sceneflow_loss, loss_im, loss_pts, loss_3d_s, loss_lr, [img_diff1, img_diff2]
        return sceneflow_loss, loss_im, loss_pts, loss_3d_s, [img_diff1, img_diff2]


    def sceneflow_loss(self, 
                       sf_f, sf_b, 
                       sf_fr, sf_br, 
                       disp_l1, disp_l2,
                       disp_r1, disp_r2,
                       disp_occ_l1, disp_occ_l2,
                       disp_occ_r1, disp_occ_r2,
                       k_l1_aug, k_l2_aug,
                       k_r1_aug, k_r2_aug,
                       img_l1_aug, img_l2_aug,
                       aug_size, ii,
                       mask_f=None, mask_b=None):

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

        # loss_lr1 = self.flow_lr_loss(sf_f, sf_fr, cam_l2r, cam_r2l, disp_occ_l1, disp_occ_r1) 
        # loss_lr2 = self.flow_lr_loss(sf_b, sf_br, cam_l2r, cam_r2l, disp_occ_l2, disp_occ_r2)
        # loss_lr = (loss_lr1 + loss_lr2)

        ## Image reconstruction loss
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        if mask_f is not None:
            img_diff1 = img_diff1 * mask_f
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        if mask_b is not None:
            img_diff2 = img_diff2 * mask_b
        loss_im1 = img_diff1[occ_map_f].mean()
        loss_im2 = img_diff2[occ_map_b].mean()
        img_diff1[~occ_map_f].detach_()
        img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2

        ## Point reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        if mask_f is not None:
            pts_diff1 = pts_diff1 * mask_f
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        if mask_b is not None:
            pts_diff2 = pts_diff2 * mask_b
        loss_pts1 = pts_diff1[occ_map_f].mean()
        loss_pts2 = pts_diff2[occ_map_b].mean()
        pts_diff1[~occ_map_f].detach_()
        pts_diff2[~occ_map_b].detach_()
        loss_pts = loss_pts1 + loss_pts2

        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s #+ loss_lr * self._sf_lr_w
        
        # return sceneflow_loss, loss_im, loss_pts, loss_3d_s, loss_lr, [img_diff1, img_diff2]
        return sceneflow_loss, loss_im, loss_pts, loss_3d_s, [img_diff1, img_diff2]

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            if self._sf_lr_w == 0.0:
                output_dict['flow_f'][ii].detach_()
                output_dict['flow_b'][ii].detach_()
            if self._pose_lr_w == 0.0:
                output_dict['pose_f'][ii].detach_()
                output_dict['pose_b'][ii].detach_()

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_pose_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0
        loss_pose_im_sum = 0
        loss_pose_pts_sum = 0
        loss_pose_sm_sum = 0
        loss_mask_reg_sum = 0
        loss_mask_sm_sum = 0
        loss_mask_consensus_sum = 0
        loss_static_cons_sum = 0
        loss_lr_sf_sum = 0
        loss_lr_pose_sum = 0
        loss_lr_mask_sum = 0
        loss_lr_disp_sum = 0
        loss_mask_sum = 0
        
        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        k_r1_aug = target_dict['input_k_r1_aug']
        k_r2_aug = target_dict['input_k_r2_aug']
        aug_size = target_dict['aug_size']

        if 'mask_l1' in output_dict:
            masks_l1 = output_dict['mask_l1']
            masks_l2 = output_dict['mask_l2']
        else:
            masks_l1 = [None] * len(output_dict['flow_f'])
            masks_l2 = [None] * len(output_dict['flow_f'])
        
        out_masks_l2 = []
        out_masks_l1 = []


        sf_fr = output_dict['output_dict_r']['flow_f']
        sf_br = output_dict['output_dict_r']['flow_b']
        pose_fr = output_dict['output_dict_r']['pose_f']
        pose_br = output_dict['output_dict_r']['pose_b']
        disps_r1 = output_dict['output_dict_r']['disp_l1']
        disps_r2 = output_dict['output_dict_r']['disp_l2']
        masks_r1 = output_dict['output_dict_r']['mask_l1']
        masks_r2 = output_dict['output_dict_r']['mask_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2, pose_f, pose_b, mask_l1, mask_l2, mask_r1, mask_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], 
                                                                                                                                      output_dict['disp_l1'], output_dict['disp_l2'], 
                                                                                                                                      disps_r1, disps_r2,
                                                                                                                                      output_dict['pose_f'], output_dict['pose_b'],
                                                                                                                                      masks_l1, masks_l2,
                                                                                                                                      masks_r1, masks_r2)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            ## Disp Loss
            loss_disp_l1, loss_lr_cons1, disp_occ_l1, disp_occ_r1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, loss_lr_cons2, disp_occ_l2, disp_occ_r2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]
            loss_lr_disp_sum += loss_lr_cons1 + loss_lr_cons2

            flow_mask_l1 = None
            flow_mask_l2 = None

            ## Sceneflow Loss
            loss_sceneflow, loss_im, loss_pts, loss_3d_s, sf_diffs = self.sceneflow_loss(sf_f, sf_b,
            # loss_sceneflow, loss_im, loss_pts, loss_3d_s, loss_sf_lr, sf_diffs = self.sceneflow_loss(sf_f, sf_b,
                                                                                                     sf_fr[ii], sf_br[ii],
                                                                                                     disp_l1, disp_l2,
                                                                                                     disp_r1, disp_r2,
                                                                                                     disp_occ_l1, disp_occ_l2,
                                                                                                     disp_occ_r1, disp_occ_r2,
                                                                                                     k_l1_aug, k_l2_aug,
                                                                                                     k_r1_aug, k_r2_aug,
                                                                                                     img_l1_aug, img_l2_aug, 
                                                                                                     aug_size, ii,
                                                                                                     flow_mask_l1, flow_mask_l2)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]
            loss_sf_2d = loss_sf_2d + loss_im
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s
            # loss_lr_sf_sum += loss_sf_lr
            
            loss_pose, loss_pose_im, loss_pose_pts, loss_pose_3d_s, pose_diffs = self.pose_loss(pose_f, pose_b,
            # loss_pose, loss_pose_im, loss_pose_pts, loss_pose_3d_s, loss_pose_lr, pose_diffs = self.pose_loss(pose_f, pose_b,
                                                                                                              pose_fr[ii], pose_br[ii],
                                                                                                              disp_l1, disp_l2,
                                                                                                              disp_r1, disp_r2,
                                                                                                              disp_occ_l1, disp_occ_l2,
                                                                                                              disp_occ_r1, disp_occ_r2,
                                                                                                              k_l1_aug, k_l2_aug,
                                                                                                              k_r1_aug, k_r2_aug,
                                                                                                              img_l1_aug, img_l2_aug, 
                                                                                                              aug_size, ii,
                                                                                                              mask_l1, mask_l2)

            loss_pose_sum += loss_pose * self._weights[ii]
            loss_pose_im_sum += loss_pose_im
            loss_pose_pts_sum += loss_pose_pts
            loss_pose_sm_sum += loss_pose_3d_s
            # loss_lr_pose_sum += loss_pose_lr

            # mask loss
            loss_mask_b, loss_mask_reg_b, loss_mask_sm_b = self.mask_loss(mask_l2)
            loss_mask_f, loss_mask_reg_f, loss_mask_sm_f = self.mask_loss(mask_l1)

            loss_mask_sum += (loss_mask_b + loss_mask_f) * self._weights[ii]
            loss_mask_reg_sum += (loss_mask_reg_b + loss_mask_reg_f)
            loss_mask_sm_sum += (loss_mask_sm_b + loss_mask_sm_f)

            # mask lr consensus loss
            disp_occ_r1 = _adaptive_disocc_detection_disp(disp_l1)
            disp_occ_r2 = _adaptive_disocc_detection_disp(disp_l2)
            loss_mask_lr1 = self.mask_lr_loss(mask_l1, mask_r1, disp_l1, disp_r1, disp_occ_l1, disp_occ_r1)
            loss_mask_lr2 = self.mask_lr_loss(mask_l2, mask_r2, disp_l2, disp_r2, disp_occ_l2, disp_occ_r2)
            loss_lr_mask_sum += (loss_mask_lr1 + loss_mask_lr2) * self._weights[ii]

            # static consistency sum
            loss_static_cons_b, flow_diff_b = self.static_cons_loss(mask_l2, sf_b, pose_b, disp_l2, disp_occ_l2, k_l2_aug, aug_size)
            loss_static_cons_f, flow_diff_f = self.static_cons_loss(mask_l1, sf_f, pose_f, disp_l1, disp_occ_l1, k_l1_aug, aug_size)
            loss_static_cons_sum += (loss_static_cons_b + loss_static_cons_f) * self._weights[ii]

            # mask consensus sum
            loss_mask_consensus_l2, census_mask_l2 = self.mask_consensus_loss(mask_l2, flow_diff_b, pose_diffs[1], sf_diffs[1])
            loss_mask_consensus_l1, census_mask_l1 = self.mask_consensus_loss(mask_l1, flow_diff_f, pose_diffs[0], sf_diffs[0])
            loss_mask_consensus_sum += (loss_mask_consensus_l2 + loss_mask_consensus_l1) * self._weights[ii]

            out_masks_l2.append(census_mask_l2)
            out_masks_l1.append(census_mask_l1)

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        p_loss = loss_pose_sum.detach()

        max_val = torch.max(torch.max(f_loss, d_loss), p_loss)

        f_weight = max_val / f_loss
        d_weight = max_val / d_loss
        p_weight = max_val / p_loss

        total_loss = loss_sf_sum * f_weight + \
                     loss_dp_sum * d_weight + \
                     loss_pose_sum * p_weight + \
                     loss_mask_sum * p_weight + \
                     loss_mask_consensus_sum + loss_lr_mask_sum + loss_static_cons_sum

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["pose"] = loss_pose_sum
        loss_dict["pose_im"] = loss_pose_im_sum
        loss_dict["pose_pts"] = loss_pose_pts_sum
        loss_dict["pose_smooth"] = loss_pose_sm_sum
        loss_dict["mask"] = loss_mask_sum
        loss_dict["mask_reg"] = loss_mask_reg_sum
        loss_dict["mask_smooth"] = loss_mask_sm_sum
        loss_dict["mask_consensus"] = loss_mask_consensus_sum
        # loss_dict["sf_lr"] = loss_lr_sf_sum
        # loss_dict["pose_lr"] = loss_lr_pose_sum
        loss_dict["mask_lr"] = loss_lr_mask_sum
        loss_dict["static_cons"] = loss_static_cons_sum
        loss_dict["disp_lr"] = loss_lr_disp_sum
        loss_dict["total_loss"] = total_loss

        output_dict["census_masks_l2"] = out_masks_l2
        output_dict["census_masks_l1"] = out_masks_l1

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict