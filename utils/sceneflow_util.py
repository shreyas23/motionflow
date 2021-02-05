from __future__ import absolute_import, division, print_function

import torch
from torch import nn
import torch.nn.functional as tf

from .loss_utils import _disp2depth_kitti_K, _adaptive_disocc_detection, _reconstruction_error, logical_or
from .inverse_warp import pose_vec2mat, pose2flow, pose2sceneflow
from .interpolation import interpolate2d_as

eps = 1e-8


def reconstruction_err(disp, src, tgt, K, sf=None, T=None, mode='pose', ssim_w=0.85):
    """ Calculate the difference between the src and tgt images 
    Inputs:
    disp: disparity from left to right (B, 1, H, W)
    src/tgt: consecutive images from left camera
    flow: scene flow from tgt to src (B, 3, H, W)
    pose: pose transform from tgt to src (B, 3, 3)
    """

    from .helpers import BackprojectDepth, Project3D

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

    img_diff = _reconstruction_error(tgt, ref_warp, ssim_w)

    return img_diff, occ_mask, (cam_points, grid), occ_mask


def pose_process_flow(src_img, tgt_img, pose, sf, disp, mask, K, aug_size, mask_thresh=0.3, flow_diff_thresh=0.1):
    # denormalize disparity
    _, _, h, w = disp.shape 
    disp = disp * w

    local_scale = torch.zeros_like(aug_size)
    local_scale[:, 0] = h
    local_scale[:, 1] = w

    rel_scale = local_scale / aug_size

    K_s = intrinsic_scale(K, rel_scale[:, 0], rel_scale[:, 1])

    depth = _disp2depth_kitti_K(disp, K_s[:, 0, 0])
    pose_sf = pose2sceneflow(depth, None, K_s, torch.inverse(K_s), pose_mat=pose)

    rigidity_mask = (mask >= mask_thresh).float()
    mask_flow_diff = ((pose_sf - sf).abs() < flow_diff_thresh).prod(dim=1, keepdim=True).float()
    rigidity_mask_comb = logical_or(rigidity_mask, mask_flow_diff)
    rigid_mask = (rigidity_mask_comb > 0).float()
    non_rigid_mask = (rigidity_mask_comb == 0).float()

    processed_flow = rigid_mask.expand_as(pose_sf) + non_rigid_mask.expand_as(sf) * sf

    return processed_flow, rigidity_mask_comb


def post_processing(l_disp, r_disp):
    
    b, _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    grid_l = torch.linspace(0.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w).float().requires_grad_(False).cuda()
    l_mask = 1.0 - torch.clamp(20 * (grid_l - 0.05), 0, 1)
    r_mask = torch.flip(l_mask, [3])
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def flow_horizontal_flip(flow_input):

    flow_flip = torch.flip(flow_input, [3])
    flow_flip[:, 0:1, :, :] *= -1

    return flow_flip.contiguous()


def disp2depth_kitti(pred_disp, fx, min_depth=1e-3, max_depth=80):

    pred_depth = fx.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (pred_disp + eps) 
    pred_depth = torch.clamp(pred_depth, min_depth, max_depth)

    return pred_depth


def get_pixelgrid(b, h, w):
    grid_h = torch.linspace(0.0, w - 1, w).view(1, 1, 1, w).expand(b, 1, h, w)
    grid_v = torch.linspace(0.0, h - 1, h).view(1, 1, h, 1).expand(b, 1, h, w)

    ones = torch.ones_like(grid_h)
    pixelgrid = torch.cat((grid_h, grid_v, ones), dim=1).float().requires_grad_(False).cuda()

    return pixelgrid


def pixel2pts(intrinsics, depth):
    b, _, h, w = depth.size()

    pixelgrid = get_pixelgrid(b, h, w)

    depth_mat = depth.view(b, 1, -1)
    pixel_mat = pixelgrid.view(b, 3, -1)
    pts_mat = torch.matmul(torch.inverse(intrinsics.cpu()).cuda(), pixel_mat) * depth_mat
    pts = pts_mat.view(b, -1, h, w)

    return pts, pixelgrid


def pts2pixel(pts, intrinsics):
    b, _, h, w = pts.size()
    proj_pts = torch.matmul(intrinsics, pts.view(b, 3, -1))
    pixels_mat = proj_pts.div(proj_pts[:, 2:3, :] + 1e-8)[:, 0:2, :]

    return pixels_mat.view(b, 2, h, w)


def intrinsic_scale(intrinsic, scale_y, scale_x):
    b, h, w = intrinsic.size()
    fx = intrinsic[:, 0, 0] * scale_x
    fy = intrinsic[:, 1, 1] * scale_y
    cx = intrinsic[:, 0, 2] * scale_x
    cy = intrinsic[:, 1, 2] * scale_y

    zeros = torch.zeros_like(fx)
    r1 = torch.stack([fx, zeros, cx], dim=1)
    r2 = torch.stack([zeros, fy, cy], dim=1)
    r3 = torch.tensor([0., 0., 1.], requires_grad=False).cuda().unsqueeze(0).expand(b, -1)
    intrinsic_s = torch.stack([r1, r2, r3], dim=1)

    return intrinsic_s


def pixel2pts_ms(intrinsic, output_disp, rel_scale):
    # pixel2pts
    intrinsic_dp_s = intrinsic_scale(intrinsic, rel_scale[:,0], rel_scale[:,1])
    output_depth = disp2depth_kitti(output_disp, intrinsic_dp_s[:, 0, 0])
    pts, _ = pixel2pts(intrinsic_dp_s, output_depth)

    return pts, intrinsic_dp_s


def pixel2pts_ms_depth(intrinsic, output_disp, rel_scale):
    # pixel2pts
    intrinsic_dp_s = intrinsic_scale(intrinsic, rel_scale[:,0], rel_scale[:,1])
    output_depth = disp2depth_kitti(output_disp, intrinsic_dp_s[:, 0, 0])
    pts, _ = pixel2pts(intrinsic_dp_s, output_depth)

    return pts, intrinsic_dp_s, output_depth


def pts2pixel_ms(intrinsic, pts, output_sf, disp_size):

    # +sceneflow and reprojection
    sf_s = tf.interpolate(output_sf, disp_size, mode="bilinear", align_corners=True)
    pts_tform = pts + sf_s
    coord = pts2pixel(pts_tform, intrinsic)
    norm_coord_w = coord[:, 0:1, :, :] / (disp_size[1] - 1) * 2 - 1
    norm_coord_h = coord[:, 1:2, :, :] / (disp_size[0] - 1) * 2 - 1
    norm_coord = torch.cat((norm_coord_w, norm_coord_h), dim=1)

    return sf_s, pts_tform, norm_coord


def pts2pixel_pose_ms(intrinsic, pts, pose, disp_size, pose_mat=None):
    b, _, h, w = pts.size() 

    if pose is not None and pose_mat is None:
        pose_mat = pose_vec2mat(pose)

    ones = torch.ones((b, 1, h*w), requires_grad=False).cuda(device=pts.device)
    pts_tform = torch.bmm(pose_mat, torch.cat([pts.view((b, 3, -1)), ones], dim=1)) # Bx3x4
    pts_tform = pts_tform.view((b, 3, h, w))
    coord = pts2pixel(pts_tform, intrinsic)
    norm_coord_w = coord[:, 0:1, :, :] / (disp_size[1] - 1) * 2 - 1
    norm_coord_h = coord[:, 1:2, :, :] / (disp_size[0] - 1) * 2 - 1
    norm_coord = torch.cat((norm_coord_w, norm_coord_h), dim=1)

    return pts_tform, norm_coord


def reconstructImg(coord, img):
    grid = coord.transpose(1, 2).transpose(2, 3)
    img_warp = tf.grid_sample(img, grid)

    mask = torch.ones_like(img, requires_grad=False)
    mask = tf.grid_sample(mask, grid)
    mask = (mask >= 1.0).float()
    return img_warp * mask


def reconstructPts(coord, pts):
    grid = coord.transpose(1, 2).transpose(2, 3)
    pts_warp = tf.grid_sample(pts, grid)

    mask = torch.ones_like(pts, requires_grad=False)
    mask = tf.grid_sample(mask, grid)
    mask = (mask >= 1.0).float()
    return pts_warp * mask


def reconstructFlow(coord, flow):
    grid = coord.transpose(1, 2).transpose(2, 3)
    flow_warp = tf.grid_sample(flow, grid)

    mask = torch.ones_like(flow, requires_grad=False)
    mask = tf.grid_sample(mask, grid)
    mask = (mask >= 1.0).float()
    return flow_warp * mask


def projectSceneFlow2Flow(intrinsic, sceneflow, disp):

    _, _, h, w = disp.size()

    output_depth = disp2depth_kitti(disp, intrinsic[:, 0, 0])
    pts, pixelgrid = pixel2pts(intrinsic, output_depth)

    sf_s = tf.interpolate(sceneflow, [h, w], mode="bilinear", align_corners=True)
    pts_tform = pts + sf_s
    coord = pts2pixel(pts_tform, intrinsic)
    flow = coord - pixelgrid[:, 0:2, :, :]

    return flow
