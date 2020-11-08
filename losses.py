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
from utils.sceneflow_util import pixel2pts_ms_depth, intrinsic_scale, disp2depth_kitti

from pprint import pprint

eps = 1e-8

###############################################
## Basic Module 
###############################################

def logical_or(a, b):
    return 1 - (1 - a)*(1 - b)

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
## Consistency functions
###############################################

def static_cons_loss(mask, sf, pose, disp, flow_occ, pose_occ, disp_occ, intrinsics, aug_size, pose_mat):
    # convert pose params to induced static scene flow
    _, _, h_dp, w_dp = disp.size()
    disp = disp * w_dp
    local_scale = torch.zeros_like(aug_size)
    local_scale[:, 0] = h_dp
    local_scale[:, 1] = w_dp         

    _, kscale, depth = pixel2pts_ms_depth(intrinsics, disp, local_scale / aug_size)

    pose_flow = pose2flow(depth.squeeze(dim=1), pose, kscale, torch.inverse(kscale), pose_mat=pose_mat)
    flow = projectSceneFlow2Flow(kscale, sf, disp)
    flow_diff = (pose_flow - flow).abs().mean(dim=1, keepdim=True)

    occ_map = flow_occ * pose_occ * disp_occ

    # static consistency loss
    flow_loss = _elementwise_epe(pose_flow, flow).mean(dim=1, keepdim=True)
    loss = (flow_loss * mask)[occ_map].mean()
    flow_loss[~occ_map].detach_()

    return loss, flow_diff.detach()

def mask_consensus_loss(mask, flow_diff, pose_err, sf_err, flow_diff_thresh):
    # mask consensus loss
    target_mask = (pose_err <= sf_err).float().detach()
    flow_similar = (flow_diff < flow_diff_thresh).float().detach()
    census_target_mask = logical_or(target_mask, flow_similar).detach()
    loss = tf.binary_cross_entropy(mask, census_target_mask)
    return loss, census_target_mask

def mask_lr_loss(mask_l, mask_r, disp_l, disp_r, left_occ, right_occ):
    mask_warp_l = _generate_image_left(mask_r, disp_l)
    mask_warp_r = _generate_image_right(mask_l, disp_r)
    lr_mask_diff_l = torch.abs(mask_warp_r - mask_l)
    lr_mask_diff_r = torch.abs(mask_warp_l - mask_r)

    loss_lr = lr_mask_diff_l[left_occ].mean() + lr_mask_diff_r[right_occ].mean()
    lr_mask_diff_l[~left_occ].detach_()
    lr_mask_diff_r[~right_occ].detach_()

    return loss_lr

def flow_lr_loss(flow_l, flow_r, disp_l, disp_r, left_occ, right_occ):
    flow_warp_l = _generate_image_left(flow_r, disp_l)
    flow_warp_r = _generate_image_right(flow_l, disp_r)

    diff_l = _elementwise_epe(flow_warp_r, flow_l).mean(dim=1, keepdim=True)
    diff_r = _elementwise_epe(flow_warp_l, flow_r).mean(dim=1, keepdim=True)

    loss_lr = diff_l[left_occ].mean() + diff_r[right_occ].mean()
    diff_l[~left_occ].detach_()
    diff_r[~right_occ].detach_()
    
    return loss_lr

###############################################
## Loss function
###############################################

class Loss_SceneFlow_SelfSup_JointIter(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_JointIter, self).__init__()
        self._args = args
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]

        self._ssim_w = 0.85

        # disp weights
        self._disp_sm_w = args.disp_sm_w

        #sf weights
        self._sf_3d_pts = args.sf_pts_w
        self._sf_3d_sm = args.sf_sm_w

        # pose weights
        self._pose_pts_w = args.pose_pts_w
        self._pose_sm_w = args.pose_sm_w

        # mask weights
        self._mask_reg_w = args.mask_reg_w
        self._mask_sm_w = args.mask_sm_w
        self._mask_cons_w = args.mask_cons_w
        self._flow_diff_thresh = args.flow_diff_thresh

        # consistency weights 
        self._fb_w = args.fb_w
        self._mask_lr_w = args.mask_lr_w
        self._disp_lr_w = args.disp_lr_w
        self._static_cons_w = args.static_cons_w

        self._use_flow_mask = args.use_flow_mask


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

        loss = loss_img + self._disp_sm_w * loss_smooth + self._disp_lr_w * loss_lr * self._disp_lr_w

        return loss, loss_img, loss_lr, left_occ, right_occ


    def mask_loss(self, mask):
        reg_loss = tf.binary_cross_entropy(mask, torch.ones_like(mask))
        sm_loss = (_gradient_x_2nd(mask).abs() + _gradient_y_2nd(mask).abs()).mean()
        loss = reg_loss * self._mask_reg_w + sm_loss * self._mask_sm_w

        return loss, reg_loss, sm_loss


    def pose_loss(self, 
                  pose_f, pose_b, 
                  disp_l1, disp_l2,
                  disp_occ_l1, disp_occ_l2,
                  k_l1_aug, k_l2_aug,
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
        pose_loss = loss_im + self._pose_pts_w * loss_pts + self._pose_sm_w * loss_3d_s
        
        return pose_loss, loss_im, loss_pts, loss_3d_s, [img_diff1, img_diff2], occ_map_f, occ_map_b


    def sceneflow_loss(self, 
                       sf_f, sf_b, 
                       disp_l1, disp_l2,
                       disp_occ_l1, disp_occ_l2,
                       k_l1_aug, k_l2_aug,
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
        sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
        
        return sceneflow_loss, loss_im, loss_pts, loss_3d_s, [img_diff1, img_diff2], occ_map_f, occ_map_b

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['pose_f'][ii].detach_()
            output_dict['pose_b'][ii].detach_()

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_disp_im_sum = 0
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
        loss_lr_mask_sum = 0
        loss_lr_disp_sum = 0
        loss_mask_sum = 0
        loss_pose_fb_sum = 0
        loss_sf_fb_sum = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']

        aug_size = target_dict['aug_size']

        if 'mask_l1' in output_dict:
            masks_l1 = output_dict['mask_l1']
            masks_l2 = output_dict['mask_l2']
        else:
            masks_l1 = [None] * len(output_dict['flow_f'])
            masks_l2 = [None] * len(output_dict['flow_f'])
        
        out_masks_l2 = []
        out_masks_l1 = []

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
            loss_disp_l1, loss_disp_im1, loss_lr_cons1, disp_occ_l1, disp_occ_r1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)

            loss_disp_l2, loss_disp_im2, loss_lr_cons2, disp_occ_l2, disp_occ_r2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]
            loss_disp_im_sum = loss_disp_im_sum + loss_disp_im1 + loss_disp_im2
            loss_lr_disp_sum = loss_lr_disp_sum + loss_lr_cons1 + loss_lr_cons2

            if self._use_flow_mask:
                flow_mask_l1 = (1.0 - mask_l1)
                flow_mask_l2 = (1.0 - mask_l2)
            else:             
                flow_mask_l1 = None
                flow_mask_l2 = None

            ## Sceneflow Loss
            loss_sceneflow, loss_im, loss_pts, loss_3d_s, sf_diffs, flow_occ_f, flow_occ_b = self.sceneflow_loss(sf_f, sf_b,
                                                                                                                disp_l1, disp_l2,
                                                                                                                disp_occ_l1, disp_occ_l2,
                                                                                                                k_l1_aug, k_l2_aug,
                                                                                                                img_l1_aug, img_l2_aug, 
                                                                                                                aug_size, ii,
                                                                                                                flow_mask_l1, flow_mask_l2)

            if self._fb_w > 0.0:
                loss_sceneflow_fb, _, _, _, _, _, _ = self.sceneflow_loss(-sf_b, -sf_f,
                                                                        disp_l1, disp_l2,
                                                                        disp_occ_l1, disp_occ_l2,
                                                                        k_l1_aug, k_l2_aug,
                                                                        img_l1_aug, img_l2_aug, 
                                                                        aug_size, ii,
                                                                        flow_mask_l1, flow_mask_l2)
            else:
                loss_sceneflow_fb = torch.tensor(0)

            loss_sf_sum = loss_sf_sum + (loss_sceneflow + loss_sceneflow_fb * self._fb_w) * self._weights[ii]
            loss_sf_2d = loss_sf_2d + loss_im
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s
            loss_sf_fb_sum = loss_sf_fb_sum + loss_sceneflow_fb
            
            loss_pose, loss_pose_im, loss_pose_pts, loss_pose_3d_s, pose_diffs, pose_occ_f, pose_occ_b = self.pose_loss(pose_f, pose_b,
                                                                                                                disp_l1, disp_l2,
                                                                                                                disp_occ_l1, disp_occ_l2,
                                                                                                                k_l1_aug, k_l2_aug,
                                                                                                                img_l1_aug, img_l2_aug, 
                                                                                                                aug_size, ii,
                                                                                                                mask_l1, mask_l2)

            # forward-backward consistency
            if self._fb_w > 0.0:
                inv_pose_f = torch.cat([pose_f[:, :3, :3].transpose(1, 2), torch.bmm(pose_f[:, :3, :3].transpose(1, 2), -pose_f[:, :, -1:])], dim=-1)
                inv_pose_b = torch.cat([pose_b[:, :3, :3].transpose(1, 2), torch.bmm(pose_b[:, :3, :3].transpose(1, 2), -pose_b[:, :, -1:])], dim=-1)

                loss_pose_fb, _, _, _, _, _, _= self.pose_loss(inv_pose_b, inv_pose_f,
                                                            disp_l1, disp_l2,
                                                            disp_occ_l1, disp_occ_l2,
                                                            k_l1_aug, k_l2_aug,
                                                            img_l1_aug, img_l2_aug, 
                                                            aug_size, ii,
                                                            mask_l1, mask_l2)
            else:
                loss_pose_fb = torch.tensor(0)

            loss_pose_sum = loss_pose_sum + (loss_pose + loss_pose_fb * self._fb_w) * self._weights[ii]
            loss_pose_im_sum = loss_pose_im_sum + loss_pose_im
            loss_pose_pts_sum = loss_pose_pts_sum + loss_pose_pts
            loss_pose_sm_sum = loss_pose_sm_sum + loss_pose_3d_s
            loss_pose_fb_sum = loss_pose_fb_sum + loss_pose_fb

            # mask loss
            if self._mask_reg_w > 0.0:
                loss_mask_b, loss_mask_reg_b, loss_mask_sm_b = self.mask_loss(mask_l2)
                loss_mask_f, loss_mask_reg_f, loss_mask_sm_f = self.mask_loss(mask_l1)
            else:
                loss_mask_b, loss_mask_reg_b, loss_mask_sm_b = torch.tensor(0), torch.tensor(0), torch.tensor(0)
                loss_mask_f, loss_mask_reg_f, loss_mask_sm_f = torch.tensor(0), torch.tensor(0), torch.tensor(0)

            loss_mask_sum = loss_mask_sum + (loss_mask_b + loss_mask_f) * self._weights[ii]
            loss_mask_reg_sum = loss_mask_reg_sum + (loss_mask_reg_b + loss_mask_reg_f)
            loss_mask_sm_sum = loss_mask_sm_sum + (loss_mask_sm_b + loss_mask_sm_f)

            # mask lr consensus loss
            if self._mask_lr_w > 0.0:
                loss_mask_lr1 = mask_lr_loss(mask_l1, mask_r1, disp_l1, disp_r1, disp_occ_l1, disp_occ_r1)
                loss_mask_lr2 = mask_lr_loss(mask_l2, mask_r2, disp_l2, disp_r2, disp_occ_l2, disp_occ_r2)
            else:
                loss_mask_lr1 = torch.tensor(0)
                loss_mask_lr2 = torch.tensor(0)

            loss_lr_mask_sum = loss_lr_mask_sum + (loss_mask_lr1 + loss_mask_lr2) * self._weights[ii]

            if ii == 0:
                output_dict['flow_occ_f'] = flow_occ_f
                output_dict['pose_occ_f'] = pose_occ_f
                output_dict['flow_occ_b'] = flow_occ_b
                output_dict['pose_occ_b'] = pose_occ_b
                output_dict['sf_diffs_b'] = sf_diffs[1]
                output_dict['pose_diffs_b'] = pose_diffs[1]
                output_dict['sf_diffs_f'] = sf_diffs[0]
                output_dict['pose_diffs_f'] = pose_diffs[0]

            # static consistency sum
            if self._static_cons_w > 0.0 or self._mask_cons_w > 0.0:
                loss_static_cons_f, flow_diff_f = static_cons_loss(mask_l1, sf_f, None, disp_l1, flow_occ_f, pose_occ_f, disp_occ_l1, k_l1_aug, aug_size, pose_mat=pose_f)
                loss_static_cons_b, flow_diff_b = static_cons_loss(mask_l2, sf_b, None, disp_l2, flow_occ_b, pose_occ_b, disp_occ_l2, k_l2_aug, aug_size, pose_mat=pose_b)
            else:
                loss_static_cons_f, flow_diff_f = torch.tensor(0), None
                loss_static_cons_b, flow_diff_b = torch.tensor(0), None

            loss_static_cons_sum = loss_static_cons_sum + (loss_static_cons_b + loss_static_cons_f) * self._weights[ii]

            # mask consensus sum
            if self._mask_cons_w > 0.0:
                loss_mask_consensus_l2, census_mask_l2 = mask_consensus_loss(mask_l2, flow_diff_b, pose_diffs[1], sf_diffs[1], self._flow_diff_thresh)
                loss_mask_consensus_l1, census_mask_l1 = mask_consensus_loss(mask_l1, flow_diff_f, pose_diffs[0], sf_diffs[0], self._flow_diff_thresh)
            else:
                loss_mask_consensus_l2, census_mask_l2 = torch.tensor(0), mask_l2
                loss_mask_consensus_l1, census_mask_l1 = torch.tensor(0), mask_l1

            loss_mask_consensus_sum = loss_mask_consensus_sum + (loss_mask_consensus_l2 + loss_mask_consensus_l1) * self._weights[ii]

            out_masks_l2.append(census_mask_l2)
            out_masks_l1.append(census_mask_l1)

        # finding weight
        f_loss = loss_sf_sum.detach()
        p_loss = loss_pose_sum.detach()
        d_loss = loss_dp_sum.detach()

        max_val = torch.max(f_loss, torch.max(p_loss, d_loss))

        f_weight = max_val / f_loss
        d_weight = max_val / d_loss
        p_weight = max_val / p_loss

        total_loss = loss_sf_sum * f_weight + \
                     loss_dp_sum * d_weight + \
                     loss_pose_sum * p_weight + \
                     loss_mask_sum * p_weight + \
                     loss_mask_consensus_sum * p_weight * self._mask_cons_w + \
                     loss_lr_mask_sum * self._mask_lr_w + loss_static_cons_sum * self._static_cons_w

        loss_dict = {}
        loss_dict["disp"] = loss_dp_sum
        loss_dict["disp_im"] = loss_disp_im_sum
        loss_dict["disp_lr"] = loss_lr_disp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["sf_fb"] = loss_sf_fb_sum
        loss_dict["pose"] = loss_pose_sum
        loss_dict["pose_im"] = loss_pose_im_sum
        loss_dict["pose_pts"] = loss_pose_pts_sum
        loss_dict["pose_fb"] = loss_pose_fb_sum
        loss_dict["pose_smooth"] = loss_pose_sm_sum
        loss_dict["mask"] = loss_mask_sum
        loss_dict["mask_reg"] = loss_mask_reg_sum
        loss_dict["mask_smooth"] = loss_mask_sm_sum
        loss_dict["mask_consensus"] = loss_mask_consensus_sum
        # loss_dict["pose_lr"] = loss_lr_pose_sum
        loss_dict["mask_lr"] = loss_lr_mask_sum
        loss_dict["static_cons"] = loss_static_cons_sum
        loss_dict["total_loss"] = total_loss

        output_dict["census_masks_l2"] = out_masks_l2
        output_dict["census_masks_l1"] = out_masks_l1

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict

class Loss_SceneFlow_SelfSup_Joint(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_Joint, self).__init__()
        self._args = args
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]

        self._ssim_w = 0.85

        # disp weights
        self._disp_sm_w = args.disp_sm_w

        #sf weights
        self._sf_3d_pts = args.sf_pts_w
        self._sf_3d_sm = args.sf_sm_w

        # pose weights
        self._pose_pts_w = args.pose_pts_w
        self._pose_sm_w = args.pose_sm_w

        # mask weights
        self._mask_reg_w = args.mask_reg_w
        self._mask_sm_w = args.mask_sm_w
        self._mask_cons_w = args.mask_cons_w
        self._flow_diff_thresh = args.flow_diff_thresh

        # consistency weights 
        self._mask_lr_w = args.mask_lr_w
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

        loss = loss_img + self._disp_sm_w * loss_smooth + self._disp_lr_w * loss_lr * self._disp_lr_w

        return loss, loss_img, loss_lr, left_occ, right_occ

    def mask_loss(self, mask):
        reg_loss = tf.binary_cross_entropy(mask, torch.ones_like(mask))
        sm_loss = (_gradient_x_2nd(mask).abs() + _gradient_y_2nd(mask).abs()).mean()
        loss = reg_loss * self._mask_reg_w + sm_loss * self._mask_sm_w

        return loss, reg_loss, sm_loss

    def pose_loss(self, 
                  pose_f, pose_b, 
                  disp_l1, disp_l2,
                  disp_occ_l1, disp_occ_l2,
                  k_l1_aug, k_l2_aug,
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

        pts1_tf, coord1 = pts2pixel_pose_ms(k1_scale, pts1, pose_f, [h_dp, w_dp])
        pts2_tf, coord2 = pts2pixel_pose_ms(k2_scale, pts2, pose_b, [h_dp, w_dp])

        pts2_warp = reconstructPts(coord1, pts2)
        pts1_warp = reconstructPts(coord2, pts1) 

        flow_f = pose2flow(depth_l1.squeeze(dim=1), pose_f, k1_scale, torch.inverse(k1_scale))
        flow_b = pose2flow(depth_l2.squeeze(dim=1), pose_b, k2_scale, torch.inverse(k2_scale))
        sf_f = pose2sceneflow(depth_l1.squeeze(dim=1), pose_f, torch.inverse(k1_scale))
        sf_b = pose2sceneflow(depth_l2.squeeze(dim=1), pose_b, torch.inverse(k2_scale))
        occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

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
        pose_loss = loss_im + self._pose_pts_w * loss_pts + self._pose_sm_w * loss_3d_s
        
        return pose_loss, loss_im, loss_pts, loss_3d_s, [img_diff1, img_diff2], occ_map_f, occ_map_b


    def sceneflow_loss(self, 
                       sf_f, sf_b, 
                       disp_l1, disp_l2,
                       disp_occ_l1, disp_occ_l2,
                       k_l1_aug, k_l2_aug,
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
        sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
        
        return sceneflow_loss, loss_im, loss_pts, loss_3d_s, [img_diff1, img_diff2], occ_map_f, occ_map_b

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['pose_f'][ii].detach_()
            output_dict['pose_b'][ii].detach_()

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_disp_im_sum = 0
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
        loss_lr_mask_sum = 0
        loss_lr_disp_sum = 0
        loss_mask_sum = 0
        
        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']

        aug_size = target_dict['aug_size']

        if 'mask_l1' in output_dict:
            masks_l1 = output_dict['mask_l1']
            masks_l2 = output_dict['mask_l2']
        else:
            masks_l1 = [None] * len(output_dict['flow_f'])
            masks_l2 = [None] * len(output_dict['flow_f'])
        
        out_masks_l2 = []
        out_masks_l1 = []

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
            loss_disp_l1, loss_disp_im1, loss_lr_cons1, disp_occ_l1, disp_occ_r1 = self.depth_loss_left_img(disp_l1, disp_r1, 
                                                                                                           img_l1_aug, img_r1_aug, 
                                                                                                           ii)

            loss_disp_l2, loss_disp_im2, loss_lr_cons2, disp_occ_l2, disp_occ_r2 = self.depth_loss_left_img(disp_l2, disp_r2, 
                                                                                                           img_l2_aug, img_r2_aug, 
                                                                                                           ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]
            loss_disp_im_sum = loss_disp_im_sum + (loss_disp_im1 + loss_disp_im2)
            loss_lr_disp_sum = loss_lr_disp_sum + (loss_lr_cons1 + loss_lr_cons2)

            flow_mask_l1 = None
            flow_mask_l2 = None

            ## Sceneflow Loss
            loss_sceneflow, loss_im, loss_pts, loss_3d_s, sf_diffs, flow_occ_f, flow_occ_b = self.sceneflow_loss(sf_f, sf_b,
                                                                                                                disp_l1, disp_l2,
                                                                                                                disp_occ_l1, disp_occ_l2,
                                                                                                                k_l1_aug, k_l2_aug,
                                                                                                                img_l1_aug, img_l2_aug, 
                                                                                                                aug_size, ii,
                                                                                                                flow_mask_l1, flow_mask_l2)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]
            loss_sf_2d = loss_sf_2d + loss_im
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s
            
            loss_pose, loss_pose_im, loss_pose_pts, loss_pose_3d_s, pose_diffs, pose_occ_f, pose_occ_b = self.pose_loss(pose_f, pose_b,
                                                                                                                disp_l1, disp_l2,
                                                                                                                disp_occ_l1, disp_occ_l2,
                                                                                                                k_l1_aug, k_l2_aug,
                                                                                                                img_l1_aug, img_l2_aug, 
                                                                                                                aug_size, ii,
                                                                                                                mask_l1, mask_l2)

            loss_pose_sum = loss_pose_sum + loss_pose * self._weights[ii]
            loss_pose_im_sum = loss_pose_im_sum + loss_pose_im
            loss_pose_pts_sum = loss_pose_pts_sum + loss_pose_pts
            loss_pose_sm_sum = loss_pose_sm_sum + loss_pose_3d_s

            # mask loss
            loss_mask_b, loss_mask_reg_b, loss_mask_sm_b = self.mask_loss(mask_l2)
            loss_mask_f, loss_mask_reg_f, loss_mask_sm_f = self.mask_loss(mask_l1)
            loss_mask_sum = loss_mask_sum + (loss_mask_b + loss_mask_f) * self._weights[ii]
            loss_mask_reg_sum = loss_mask_reg_sum + (loss_mask_reg_b + loss_mask_reg_f)
            loss_mask_sm_sum = loss_mask_sm_sum + (loss_mask_sm_b + loss_mask_sm_f)

            # mask lr consensus loss
            loss_mask_lr1 = mask_lr_loss(mask_l1, mask_r1, disp_l1, disp_r1, disp_occ_l1, disp_occ_r1)
            loss_mask_lr2 = mask_lr_loss(mask_l2, mask_r2, disp_l2, disp_r2, disp_occ_l2, disp_occ_r2)
            loss_lr_mask_sum = loss_lr_mask_sum + (loss_mask_lr1 + loss_mask_lr2) * self._weights[ii]

            # static consistency sum
            loss_static_cons_f, flow_diff_f = static_cons_loss(mask_l1, sf_f, pose_f, disp_l1, flow_occ_f, pose_occ_f, disp_occ_l1, k_l1_aug, aug_size, None)
            loss_static_cons_b, flow_diff_b = static_cons_loss(mask_l2, sf_b, pose_b, disp_l2, flow_occ_b, pose_occ_b, disp_occ_l2, k_l2_aug, aug_size, None)
            loss_static_cons_sum = loss_static_cons_sum + (loss_static_cons_b + loss_static_cons_f) * self._weights[ii]

            # mask consensus sum
            loss_mask_consensus_l2, census_mask_l2 = mask_consensus_loss(mask_l2, flow_diff_b, pose_diffs[1], sf_diffs[1], self._flow_diff_thresh)
            loss_mask_consensus_l1, census_mask_l1 = mask_consensus_loss(mask_l1, flow_diff_f, pose_diffs[0], sf_diffs[0], self._flow_diff_thresh)
            loss_mask_consensus_sum = loss_mask_consensus_sum + (loss_mask_consensus_l2 + loss_mask_consensus_l1) * self._weights[ii]

            out_masks_l2.append(census_mask_l2)
            out_masks_l1.append(census_mask_l1)

        # finding weight
        f_loss = loss_sf_sum.detach()
        p_loss = loss_pose_sum.detach()
        d_loss = loss_dp_sum.detach()

        max_val = torch.max(torch.max(f_loss, d_loss), p_loss)

        f_weight = max_val / f_loss
        d_weight = max_val / d_loss
        p_weight = max_val / p_loss

        total_loss = loss_sf_sum * f_weight + \
                     loss_dp_sum * d_weight + \
                     loss_pose_sum * p_weight + \
                     loss_mask_sum * p_weight + \
                     loss_mask_consensus_sum * self._mask_cons_w * p_weight + \
                     loss_lr_mask_sum * self._mask_lr_w + loss_static_cons_sum * self._static_cons_w 

        loss_dict = {}
        loss_dict["disp"] = loss_dp_sum
        loss_dict["disp_im"] = loss_disp_im_sum
        loss_dict["disp_lr"] = loss_lr_disp_sum
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
        # loss_dict["pose_lr"] = loss_lr_pose_sum
        loss_dict["mask_lr"] = loss_lr_mask_sum
        loss_dict["static_cons"] = loss_static_cons_sum
        loss_dict["total_loss"] = total_loss

        output_dict["census_masks_l2"] = out_masks_l2
        output_dict["census_masks_l1"] = out_masks_l1

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict

class Loss_SceneFlow_SelfSup(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup, self).__init__()
                
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_sm_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200

    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Photometric loss
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_sm_w * loss_smooth, left_occ


    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

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

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1[occ_map_f].mean()
        loss_im2 = img_diff2[occ_map_b].mean()
        img_diff1[~occ_map_f].detach_()
        img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2
        
        ## Point reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        loss_pts1 = pts_diff1[occ_map_f].mean()
        loss_pts2 = pts_diff2[occ_map_b].mean()
        pts_diff1[~occ_map_f].detach_()
        pts_diff2[~occ_map_b].detach_()
        loss_pts = loss_pts1 + loss_pts2

        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
        
        return sceneflow_loss, loss_im, loss_pts, loss_3d_s

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0
        
        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            ## Disp Loss
            loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

            ## Sceneflow Loss           
            loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
                                                                            disp_l1, disp_l2,
                                                                            disp_occ_l1, disp_occ_l2,
                                                                            k_l1_aug, k_l2_aug,
                                                                            img_l1_aug, img_l2_aug, 
                                                                            aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
            loss_sf_2d = loss_sf_2d + loss_im            
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict


class Loss_SceneFlow_SemiSupFinetune(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SemiSupFinetune, self).__init__()        

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._unsup_loss = Loss_SceneFlow_SelfSup(args)


    def forward(self, output_dict, target_dict):

        loss_dict = {}

        unsup_loss_dict = self._unsup_loss(output_dict, target_dict)
        unsup_loss = unsup_loss_dict['total_loss']

        ## Ground Truth
        gt_disp1 = target_dict['target_disp']
        gt_disp1_mask = (target_dict['target_disp_mask']==1).float()   
        gt_disp2 = target_dict['target_disp2_occ']
        gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()   
        gt_flow = target_dict['target_flow']
        gt_flow_mask = (target_dict['target_flow_mask']==1).float()

        b, _, h_dp, w_dp = gt_disp1.size()     

        disp_loss = 0
        flow_loss = 0

        for ii, sf_f in enumerate(output_dict['flow_f_pp']):

            ## disp1
            disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][ii], gt_disp1, mode="bilinear") * w_dp
            valid_abs_rel = torch.abs(gt_disp1 - disp_l1) * gt_disp1_mask
            valid_abs_rel[gt_disp1_mask == 0].detach_()
            disp_l1_loss = valid_abs_rel[gt_disp1_mask != 0].mean()

            ## Flow Loss
            sf_f_up = interpolate2d_as(sf_f, gt_flow, mode="bilinear")
            out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], sf_f_up, disp_l1)
            valid_epe = _elementwise_robust_epe_char(out_flow, gt_flow) * gt_flow_mask
                
            valid_epe[gt_flow_mask == 0].detach_()
            flow_l1_loss = valid_epe[gt_flow_mask != 0].mean()

            ## disp1_next
            out_depth_l1 = _disp2depth_kitti_K(disp_l1, target_dict['input_k_l1'][:, 0, 0])
            out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
            out_depth_l1_next = out_depth_l1 + sf_f_up[:, 2:3, :, :]
            disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, target_dict['input_k_l1'][:, 0, 0])

            valid_abs_rel = torch.abs(gt_disp2 - disp_l1_next) * gt_disp2_mask
            valid_abs_rel[gt_disp2_mask == 0].detach_()
            disp_l2_loss = valid_abs_rel[gt_disp2_mask != 0].mean()
             
            disp_loss = disp_loss + (disp_l1_loss + disp_l2_loss) * self._weights[ii]
            flow_loss = flow_loss + flow_l1_loss * self._weights[ii]

        # finding weight
        u_loss = unsup_loss.detach()
        d_loss = disp_loss.detach()
        f_loss = flow_loss.detach()

        max_val = torch.max(torch.max(f_loss, d_loss), u_loss)

        u_weight = max_val / u_loss
        d_weight = max_val / d_loss 
        f_weight = max_val / f_loss 

        total_loss = unsup_loss * u_weight + disp_loss * d_weight + flow_loss * f_weight
        loss_dict["unsup_loss"] = unsup_loss
        loss_dict["dp_loss"] = disp_loss
        loss_dict["fl_loss"] = flow_loss
        loss_dict["total_loss"] = total_loss

        return loss_dict


class Loss_SceneFlow_SelfSup_NoOcc(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_NoOcc, self).__init__()
                
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_sm_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200


    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        # left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Photometric loss: 
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = img_diff.mean()
        # loss_img = (img_diff[left_occ]).mean()
        # img_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_sm_w * loss_smooth#, left_occ


    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

        ## Depth2Pts
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
        # occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        # occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        ## Image reconstruction loss
        # img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
        # img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1.mean()
        loss_im2 = img_diff2.mean()
        # loss_im1 = img_diff1[occ_map_f].mean()
        # loss_im2 = img_diff2[occ_map_b].mean()
        # img_diff1[~occ_map_f].detach_()
        # img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2
        
        ## Point Reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        loss_pts1 = pts_diff1.mean()
        loss_pts2 = pts_diff2.mean()
        # loss_pts1 = pts_diff1[occ_map_f].mean()
        # loss_pts2 = pts_diff2[occ_map_b].mean()
        # pts_diff1[~occ_map_f].detach_()
        # pts_diff2[~occ_map_b].detach_()
        loss_pts = loss_pts1 + loss_pts2

        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_pts * loss_pts + self._sf_3d_sm * loss_3d_s
        
        return sceneflow_loss, loss_im, loss_pts, loss_3d_s

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ## SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        loss_sf_3d = 0
        loss_sf_sm = 0
        
        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            ## Depth Loss
            loss_disp_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


            ## Sceneflow Loss           
            loss_sceneflow, loss_im, loss_pts, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
                                                                            disp_l1, disp_l2,
                                                                            k_l1_aug, k_l2_aug,
                                                                            img_l1_aug, img_l2_aug, 
                                                                            aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
            loss_sf_2d = loss_sf_2d + loss_im            
            loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict

class Loss_SceneFlow_SelfSup_NoPts(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_NoPts, self).__init__()
                
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_sm_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200


    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Photometric loss: 
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_sm_w * loss_smooth, left_occ


    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, disp_occ_l1, disp_occ_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

        ## Depth2Pts
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

        # pts2_warp = reconstructPts(coord1, pts2)
        # pts1_warp = reconstructPts(coord2, pts1) 

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        ## Image reconstruction loss
        # img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
        # img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1[occ_map_f].mean()
        loss_im2 = img_diff2[occ_map_b].mean()
        img_diff1[~occ_map_f].detach_()
        img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2
        
        # ## Point Reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        # pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        # pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        # loss_pts1 = pts_diff1[occ_map_f].mean()
        # loss_pts2 = pts_diff2[occ_map_b].mean()
        # pts_diff1[~occ_map_f].detach_()
        # pts_diff2[~occ_map_b].detach_()
        # loss_pts = loss_pts1 + loss_pts2

        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_sm * loss_3d_s# + self._sf_3d_pts * loss_pts
        
        return sceneflow_loss, loss_im, loss_3d_s#, loss_pts

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ## SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        # loss_sf_3d = 0
        loss_sf_sm = 0
        
        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            ## Depth Loss
            loss_disp_l1, disp_occ_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, disp_occ_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


            ## Sceneflow Loss           
            loss_sceneflow, loss_im, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
                                                                            disp_l1, disp_l2,
                                                                            disp_occ_l1, disp_occ_l2,
                                                                            k_l1_aug, k_l2_aug,
                                                                            img_l1_aug, img_l2_aug, 
                                                                            aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
            loss_sf_2d = loss_sf_2d + loss_im            
            # loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        # loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict

class Loss_SceneFlow_SelfSup_NoPtsNoOcc(nn.Module):
    def __init__(self, args):
        super(Loss_SceneFlow_SelfSup_NoPtsNoOcc, self).__init__()
                
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_sm_w = 0.1
        self._sf_3d_pts = 0.2
        self._sf_3d_sm = 200


    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        # left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Photometric loss: 
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = img_diff.mean()
        # loss_img = (img_diff[left_occ]).mean()
        # img_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_sm_w * loss_smooth#, left_occ


    def sceneflow_loss(self, sf_f, sf_b, disp_l1, disp_l2, k_l1_aug, k_l2_aug, img_l1_aug, img_l2_aug, aug_size, ii):

        ## Depth2Pts
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

        # pts2_warp = reconstructPts(coord1, pts2)
        # pts1_warp = reconstructPts(coord2, pts1) 

        flow_f = projectSceneFlow2Flow(k1_scale, sf_f, disp_l1)
        flow_b = projectSceneFlow2Flow(k2_scale, sf_b, disp_l2)
        # occ_map_b = _adaptive_disocc_detection(flow_f).detach() * disp_occ_l2
        # occ_map_f = _adaptive_disocc_detection(flow_b).detach() * disp_occ_l1

        ## Image reconstruction loss
        # img_l2_warp = self.warping_layer_aug(img_l2, flow_f, aug_scale, coords)
        # img_l1_warp = self.warping_layer_aug(img_l1, flow_b, aug_scale, coords)
        img_l2_warp = reconstructImg(coord1, img_l2_aug)
        img_l1_warp = reconstructImg(coord2, img_l1_aug)

        img_diff1 = (_elementwise_l1(img_l1_aug, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1_aug, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        img_diff2 = (_elementwise_l1(img_l2_aug, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2_aug, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
        loss_im1 = img_diff1.mean()
        loss_im2 = img_diff2.mean()
        # loss_im1 = img_diff1[occ_map_f].mean()
        # loss_im2 = img_diff2[occ_map_b].mean()
        # img_diff1[~occ_map_f].detach_()
        # img_diff2[~occ_map_b].detach_()
        loss_im = loss_im1 + loss_im2
        
        ## Point Reconstruction Loss
        pts_norm1 = torch.norm(pts1, p=2, dim=1, keepdim=True)
        pts_norm2 = torch.norm(pts2, p=2, dim=1, keepdim=True)
        # pts_diff1 = _elementwise_epe(pts1_tf, pts2_warp).mean(dim=1, keepdim=True) / (pts_norm1 + 1e-8)
        # pts_diff2 = _elementwise_epe(pts2_tf, pts1_warp).mean(dim=1, keepdim=True) / (pts_norm2 + 1e-8)
        # loss_pts1 = pts_diff1.mean()
        # loss_pts2 = pts_diff2.mean()
        # loss_pts1 = pts_diff1[occ_map_f].mean()
        # loss_pts2 = pts_diff2[occ_map_b].mean()
        # pts_diff1[~occ_map_f].detach_()
        # pts_diff2[~occ_map_b].detach_()
        # loss_pts = loss_pts1 + loss_pts2

        ## 3D motion smoothness loss
        loss_3d_s = ( (_smoothness_motion_2nd(sf_f, img_l1_aug, beta=10.0) / (pts_norm1 + 1e-8)).mean() + (_smoothness_motion_2nd(sf_b, img_l2_aug, beta=10.0) / (pts_norm2 + 1e-8)).mean() ) / (2 ** ii)

        ## Loss Summnation
        sceneflow_loss = loss_im + self._sf_3d_sm * loss_3d_s # + self._sf_3d_pts * loss_pts
        
        return sceneflow_loss, loss_im, loss_3d_s # , loss_pts

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['flow_f'])):
            output_dict['flow_f'][ii].detach_()
            output_dict['flow_b'][ii].detach_()
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ## SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_sf_sum = 0
        loss_dp_sum = 0
        loss_sf_2d = 0
        # loss_sf_3d = 0
        loss_sf_sm = 0
        
        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (sf_f, sf_b, disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'], output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert(sf_f.size()[2:4] == sf_b.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l1.size()[2:4])
            assert(sf_f.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], sf_b)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], sf_f)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], sf_b)

            ## Depth Loss
            loss_disp_l1 = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2 = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]


            ## Sceneflow Loss           
            loss_sceneflow, loss_im, loss_3d_s = self.sceneflow_loss(sf_f, sf_b, 
                                                                            disp_l1, disp_l2,
                                                                            k_l1_aug, k_l2_aug,
                                                                            img_l1_aug, img_l2_aug, 
                                                                            aug_size, ii)

            loss_sf_sum = loss_sf_sum + loss_sceneflow * self._weights[ii]            
            loss_sf_2d = loss_sf_2d + loss_im            
            # loss_sf_3d = loss_sf_3d + loss_pts
            loss_sf_sm = loss_sf_sm + loss_3d_s

        # finding weight
        f_loss = loss_sf_sum.detach()
        d_loss = loss_dp_sum.detach()
        max_val = torch.max(f_loss, d_loss)
        f_weight = max_val / f_loss
        d_weight = max_val / d_loss

        total_loss = loss_sf_sum * f_weight + loss_dp_sum * d_weight

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["sf"] = loss_sf_sum
        loss_dict["s_2"] = loss_sf_2d
        # loss_dict["s_3"] = loss_sf_3d
        loss_dict["s_3s"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict


###############################################
## Ablation - Separate Decoder
###############################################

class Loss_Flow_Only(nn.Module):
    def __init__(self):
        super(Loss_Flow_Only, self).__init__()

        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._warping_layer = WarpingLayer_Flow()

    def forward(self, output_dict, target_dict):

        ## Loss
        total_loss = 0
        loss_sf_2d = 0
        loss_sf_sm = 0

        for ii, (sf_f, sf_b) in enumerate(zip(output_dict['flow_f'], output_dict['flow_b'])):

            ## Depth2Pts            
            img_l1 = interpolate2d_as(target_dict["input_l1_aug"], sf_f)
            img_l2 = interpolate2d_as(target_dict["input_l2_aug"], sf_b)

            img_l2_warp = self._warping_layer(img_l2, sf_f)
            img_l1_warp = self._warping_layer(img_l1, sf_b)
            occ_map_f = _adaptive_disocc_detection(sf_b).detach()
            occ_map_b = _adaptive_disocc_detection(sf_f).detach()

            img_diff1 = (_elementwise_l1(img_l1, img_l2_warp) * (1.0 - self._ssim_w) + _SSIM(img_l1, img_l2_warp) * self._ssim_w).mean(dim=1, keepdim=True)
            img_diff2 = (_elementwise_l1(img_l2, img_l1_warp) * (1.0 - self._ssim_w) + _SSIM(img_l2, img_l1_warp) * self._ssim_w).mean(dim=1, keepdim=True)
            loss_im1 = img_diff1[occ_map_f].mean()
            loss_im2 = img_diff2[occ_map_b].mean()
            img_diff1[~occ_map_f].detach_()
            img_diff2[~occ_map_b].detach_()
            loss_im = loss_im1 + loss_im2

            loss_smooth = _smoothness_motion_2nd(sf_f / 20.0, img_l1, beta=10.0).mean() + _smoothness_motion_2nd(sf_b / 20.0, img_l2, beta=10.0).mean()
            
            total_loss = total_loss + (loss_im + 10.0 * loss_smooth) * self._weights[ii]
            
            loss_sf_2d = loss_sf_2d + loss_im 
            loss_sf_sm = loss_sf_sm + loss_smooth

        loss_dict = {}
        loss_dict["ofd2"] = loss_sf_2d
        loss_dict["ofs2"] = loss_sf_sm
        loss_dict["total_loss"] = total_loss

        return loss_dict

class Eval_Flow_Only(nn.Module):
    def __init__(self):
        super(Eval_Flow_Only, self).__init__()
    

    def upsample_flow_as(self, flow, output_as):
        size_inputs = flow.size()[2:4]
        size_targets = output_as.size()[2:4]
        resized_flow = tf.interpolate(flow, size=size_targets, mode="bilinear", align_corners=True)
        # correct scaling of flow
        u, v = resized_flow.chunk(2, dim=1)
        u *= float(size_targets[1] / size_inputs[1])
        v *= float(size_targets[0] / size_inputs[0])
        return torch.cat([u, v], dim=1)


    def forward(self, output_dict, target_dict):

        loss_dict = {}

        im_l1 = target_dict['input_l1']
        batch_size, _, _, _ = im_l1.size()

        gt_flow = target_dict['target_flow']
        gt_flow_mask = target_dict['target_flow_mask']

        ## Flow EPE
        out_flow = self.upsample_flow_as(output_dict['flow_f'][0], gt_flow)
        valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask.float()
        loss_dict["epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
        
        flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
        outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
        loss_dict["f1"] = (outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68

        output_dict["out_flow_pp"] = out_flow

        return loss_dict


class Loss_Disp_Only(nn.Module):
    def __init__(self, args):
        super(Loss_Disp_Only, self).__init__()
                
        self._weights = [4.0, 2.0, 1.0, 1.0, 1.0]
        self._ssim_w = 0.85
        self._disp_sm_w = 0.1


    def depth_loss_left_img(self, disp_l, disp_r, img_l_aug, img_r_aug, ii):

        img_r_warp = _generate_image_left(img_r_aug, disp_l)
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()

        ## Image loss: 
        img_diff = (_elementwise_l1(img_l_aug, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l_aug, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        
        loss_img = (img_diff[left_occ]).mean()
        img_diff[~left_occ].detach_()

        ## Disparities smoothness
        loss_smooth = _smoothness_motion_2nd(disp_l, img_l_aug, beta=10.0).mean() / (2 ** ii)

        return loss_img + self._disp_sm_w * loss_smooth, left_occ

    def detaching_grad_of_outputs(self, output_dict):
        
        for ii in range(0, len(output_dict['disp_l1'])):
            output_dict['disp_l1'][ii].detach_()
            output_dict['disp_l2'][ii].detach_()

        return None

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ## SceneFlow Loss
        batch_size = target_dict['input_l1'].size(0)
        loss_dp_sum = 0

        k_l1_aug = target_dict['input_k_l1_aug']
        k_l2_aug = target_dict['input_k_l2_aug']
        aug_size = target_dict['aug_size']

        disp_r1_dict = output_dict['output_dict_r']['disp_l1']
        disp_r2_dict = output_dict['output_dict_r']['disp_l2']

        for ii, (disp_l1, disp_l2, disp_r1, disp_r2) in enumerate(zip(output_dict['disp_l1'], output_dict['disp_l2'], disp_r1_dict, disp_r2_dict)):

            assert(disp_l1.size()[2:4] == disp_l2.size()[2:4])
            
            ## For image reconstruction loss
            img_l1_aug = interpolate2d_as(target_dict["input_l1_aug"], disp_l1)
            img_l2_aug = interpolate2d_as(target_dict["input_l2_aug"], disp_l2)
            img_r1_aug = interpolate2d_as(target_dict["input_r1_aug"], disp_l1)
            img_r2_aug = interpolate2d_as(target_dict["input_r2_aug"], disp_l2)

            ## Depth Loss
            loss_disp_l1, _ = self.depth_loss_left_img(disp_l1, disp_r1, img_l1_aug, img_r1_aug, ii)
            loss_disp_l2, _ = self.depth_loss_left_img(disp_l2, disp_r2, img_l2_aug, img_r2_aug, ii)
            loss_dp_sum = loss_dp_sum + (loss_disp_l1 + loss_disp_l2) * self._weights[ii]

        total_loss = loss_dp_sum

        loss_dict = {}
        loss_dict["dp"] = loss_dp_sum
        loss_dict["total_loss"] = total_loss

        self.detaching_grad_of_outputs(output_dict['output_dict_r'])

        return loss_dict

class Eval_Disp_Only(nn.Module):
    def __init__(self):
        super(Eval_Disp_Only, self).__init__()


    def forward(self, output_dict, target_dict):
        

        loss_dict = {}

        ## Depth Eval
        gt_disp = target_dict['target_disp']
        gt_disp_mask = (target_dict['target_disp_mask']==1)
        intrinsics = target_dict['input_k_l1']

        out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * gt_disp.size(3)
        out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
        gt_depth_pp = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

        output_dict_displ = eval_module_disp_depth(gt_disp, gt_disp_mask, out_disp_l1, gt_depth_pp, out_depth_l1)

        output_dict["out_disp_l_pp"] = out_disp_l1
        output_dict["out_depth_l_pp"] = out_depth_l1

        loss_dict["d1"] = output_dict_displ['otl']

        loss_dict["ab"] = output_dict_displ['abs_rel']
        loss_dict["sq"] = output_dict_displ['sq_rel']
        loss_dict["rms"] = output_dict_displ['rms']
        loss_dict["lrms"] = output_dict_displ['log_rms']
        loss_dict["a1"] = output_dict_displ['a1']
        loss_dict["a2"] = output_dict_displ['a2']
        loss_dict["a3"] = output_dict_displ['a3']


        return loss_dict


###############################################
## MonoDepth Experiment
###############################################

class Basis_MonoDepthLoss(nn.Module):
    def __init__(self):
        super(Basis_MonoDepthLoss, self).__init__()
        self.ssim_w = 0.85
        self.disp_gradient_w = 0.1
        self.lr_w = 1.0
        self.n = 4

    def scale_pyramid(self, img_input, depths):
        scaled_imgs = []
        for _, depth in enumerate(depths):
            scaled_imgs.append(interpolate2d_as(img_input, depth))
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = tf.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = tf.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy

    def apply_disparity(self, img, disp):
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

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
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

        return tf.pad(SSIM_img, pad=(1,1,1,1), mode='constant', value=0)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i]) for i in range(self.n)]

    def forward(self, disp_l, disp_r, img_l, img_r):

        self.n = len(disp_l)

        ## Image pyramid
        img_l_pyramid = self.scale_pyramid(img_l, disp_l)
        img_r_pyramid = self.scale_pyramid(img_r, disp_r)

        ## Disocc map
        right_occ = [_adaptive_disocc_detection_disp(-disp_l[i]) for i in range(self.n)]
        left_occ  = [_adaptive_disocc_detection_disp(disp_r[i]) for i in range(self.n)]


        ## Image reconstruction loss
        left_est = [self.generate_image_left(img_r_pyramid[i], disp_l[i]) for i in range(self.n)]
        right_est = [self.generate_image_right(img_l_pyramid[i], disp_r[i]) for i in range(self.n)]

        # L1
        l1_left = [torch.mean((torch.abs(left_est[i] - img_l_pyramid[i])).mean(dim=1, keepdim=True)[left_occ[i]]) for i in range(self.n)]
        l1_right = [torch.mean((torch.abs(right_est[i] - img_r_pyramid[i])).mean(dim=1, keepdim=True)[right_occ[i]]) for i in range(self.n)]

        # SSIM
        ssim_left = [torch.mean((self.SSIM(left_est[i], img_l_pyramid[i])).mean(dim=1, keepdim=True)[left_occ[i]]) for i in range(self.n)]
        ssim_right = [torch.mean((self.SSIM(right_est[i], img_r_pyramid[i])).mean(dim=1, keepdim=True)[right_occ[i]]) for i in range(self.n)]

        image_loss_left = [self.ssim_w * ssim_left[i] + (1 - self.ssim_w) * l1_left[i] for i in range(self.n)]
        image_loss_right = [self.ssim_w * ssim_right[i] + (1 - self.ssim_w) * l1_right[i] for i in range(self.n)]
        image_loss = sum(image_loss_left + image_loss_right)


        ## L-R Consistency loss
        right_left_disp = [self.generate_image_left(disp_r[i], disp_l[i]) for i in range(self.n)]
        left_right_disp = [self.generate_image_right(disp_l[i], disp_r[i]) for i in range(self.n)]

        lr_left_loss = [torch.mean((torch.abs(right_left_disp[i] - disp_l[i]))[left_occ[i]]) for i in range(self.n)]
        lr_right_loss = [torch.mean((torch.abs(left_right_disp[i] - disp_r[i]))[right_occ[i]]) for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)


        ## Disparities smoothness
        disp_left_smoothness = self.disp_smoothness(disp_l, img_l_pyramid)
        disp_right_smoothness = self.disp_smoothness(disp_r, img_r_pyramid)

        disp_left_loss = [torch.mean(torch.abs(disp_left_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_right_loss = [torch.mean(torch.abs(disp_right_smoothness[i])) / 2 ** i for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)


        ## Loss sum
        loss = image_loss + self.disp_gradient_w * disp_gradient_loss + self.lr_w * lr_loss

        return loss

class Loss_MonoDepth(nn.Module):
    def __init__(self):

        super(Loss_MonoDepth, self).__init__()
        self._depth_loss = Basis_MonoDepthLoss()

    def forward(self, output_dict, target_dict):

        loss_dict = {}
        depth_loss = self._depth_loss(output_dict['disp_l1'], output_dict['disp_r1'], target_dict['input_l1'], target_dict['input_r1'])
        loss_dict['total_loss'] = depth_loss

        return loss_dict


class Loss_PoseDepth(nn.Module):
    def __init__(self):

        super(Loss_PoseDepth, self).__init__()
        self._depth_loss = Basis_MonoDepthLoss()
        self._ssim_w = 0.85


    def pose_loss(self, pose, disp, disp_occ, intrinsics, ref_imgs, tgt_imgs):
        _, _, _, w_dp = disp.size()
        disp = disp * w_dp

        depth = _disp2depth_kitti_K(disp, intrinsics[:, 0, 0])
        flow = pose2flow(depth.squeeze(dim=1), pose.squeeze(dim=1), intrinsics, torch.inverse(intrinsics))
        ref_warped= flow_warp(ref_imgs, flow)
        img_diff = (_elementwise_l1(tgt_imgs, ref_warped) * (1.0 - self._ssim_w) + _SSIM(tgt_imgs, ref_warped) * self._ssim_w).mean(dim=1, keepdim=True)
        if disp_occ:
          loss = img_diff[disp_occ.detach()].mean()
        else:
          loss = img_diff.mean()

        return loss


    def forward(self, output_dict, target_dict):

        loss_dict = {}
        depth_loss = self._depth_loss(output_dict['disp_l2'], output_dict['disp_r2'], target_dict['input_l2_aug'], target_dict['input_r2_aug'])
        pose_loss = self.pose_loss(output_dict['pose'], output_dict['disp_l2'][0], None, target_dict['input_k_l2_aug'], target_dict['input_l1_aug'], target_dict['input_l2_aug'])

        d_loss = depth_loss.detach()
        p_loss = pose_loss.detach()

        max_val = torch.max(d_loss, p_loss)

        d_weight = max_val / d_loss
        p_weight = max_val / p_loss

        total_loss = depth_loss * d_weight + pose_loss * p_weight

        loss_dict['total_loss'] = total_loss
        loss_dict['pose'] = pose_loss
        loss_dict['dp'] = depth_loss

        return loss_dict


class Eval_MonoDepth(nn.Module):
    def __init__(self):
        super(Eval_MonoDepth, self).__init__()

    def forward(self, output_dict, target_dict):
        
        loss_dict = {}

        ## Depth Eval
        gt_disp = target_dict['target_disp']
        gt_disp_mask = (target_dict['target_disp_mask']==1)
        intrinsics = target_dict['input_k_l1_orig']

        out_disp_l_pp = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * gt_disp.size(3)
        out_depth_l_pp = _disp2depth_kitti_K(out_disp_l_pp, intrinsics[:, 0, 0])
        out_depth_l_pp = torch.clamp(out_depth_l_pp, 1e-3, 80)
        gt_depth_pp = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

        output_dict_displ = eval_module_disp_depth(gt_disp, gt_disp_mask, out_disp_l_pp, gt_depth_pp, out_depth_l_pp)

        output_dict["out_disp_l_pp"] = out_disp_l_pp
        output_dict["out_depth_l_pp"] = out_depth_l_pp
        loss_dict["ab_r"] = output_dict_displ['abs_rel']
        loss_dict["sq_r"] = output_dict_displ['sq_rel']

        return loss_dict

###############################################

