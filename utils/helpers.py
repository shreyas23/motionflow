import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np

from .loss_utils import _disp2depth_kitti_K, _adaptive_disocc_detection, _generate_image_left, _reconstruction_error
from .inverse_warp import pose_vec2mat, pose2flow
from .interpolation import interpolate2d_as
from .sceneflow_util import projectSceneFlow2Flow, intrinsic_scale

### Helper functions ###

def add_pose(pose_mat, pose_res):
    b, _, _ = pose_mat.shape
    pose_mat_res = pose_vec2mat(pose_res)
    pose_mat_full = torch.cat([pose_mat, torch.zeros(b, 1, 4).to(device=pose_mat.device)], dim=1)
    pose_mat_full[:, -1, -1] = 1
    return torch.bmm(pose_mat_res, pose_mat_full)


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return tf.interpolate(x, scale_factor=2, mode="nearest")


def invert_pose(pose):
    pose_mat = pose_vec2mat(pose)
    R = pose_mat[:, :3, :3].transpose(1, 2)
    t = pose_mat[:, :3, -1:] * -1

    return pose_mat, torch.cat([R, torch.matmul(R, t)], dim=-1)


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K, mode='pose'):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        if mode == 'pose':
            cam_points = torch.cat([cam_points, self.ones], 1)
        else:
            cam_points = cam_points.view(self.batch_size, 3, self.height, self.width)

        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T=None, sf=None, mode='pose'):

        if mode == 'pose':
            assert (T is not None), "T cannot be None when mode is pose..."
            P = torch.matmul(K, T)[:, :3, :]
            cam_points = torch.matmul(P, points)

        elif mode == 'sf':
            assert (sf is not None), "flow cannot be None when mode is sf..."
            b, _, h, w = sf.shape
            points = points + sf
            points = points.view(b, 3, -1)
            cam_points = torch.matmul(K, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        return pix_coords
    

class Warp_SceneFlow(nn.Module):
    def __init__(self):
        super(Warp_SceneFlow, self).__init__()
 
    def forward(self, x, sceneflow, disp, k1, input_size):
        
        _, _, _, disp_w = disp.size()
        disp = interpolate2d_as(disp, x) * disp_w

        b, _, h_x, w_x = x.shape
        local_scale = torch.zeros_like(input_size)
        local_scale[:, 0] = h_x
        local_scale[:, 1] = w_x
        rel_scale = local_scale / input_size
        k1_s = intrinsic_scale(k1, rel_scale[:, 0], rel_scale[:, 1])

        backproject = BackprojectDepth(b, h_x, w_x).to(device=x.device)
        project = Project3D(b, h_x, w_x).to(device=x.device)
        depth = _disp2depth_kitti_K(disp, k1_s[:, 0, 0])

        cam_points = backproject(depth, torch.inverse(k1_s), mode='sf')
        grid = project(cam_points, k1_s, sf=sceneflow, mode='sf')
        x_warp = tf.grid_sample(x, grid, padding_mode="zeros")

        return x_warp


class Warp_Pose(nn.Module):
    def __init__(self):
        super(Warp_Pose, self).__init__()
 
    def forward(self, x, pose, disp, k1, input_size):
        
        _, _, _, disp_w = disp.size()
        disp = interpolate2d_as(disp, x) * disp_w

        b, _, h_x, w_x = x.shape
        local_scale = torch.zeros_like(input_size)
        local_scale[:, 0] = h_x
        local_scale[:, 1] = w_x
        rel_scale = local_scale / input_size
        k1_s = intrinsic_scale(k1, rel_scale[:, 0], rel_scale[:, 1])

        backproject = BackprojectDepth(b, h_x, w_x).to(device=x.device)
        project = Project3D(b, h_x, w_x).to(device=x.device)
        depth = _disp2depth_kitti_K(disp, k1_s[:, 0, 0])

        cam_points = backproject(depth, torch.inverse(k1_s), mode='pose')
        grid = project(cam_points, k1_s, T=pose, mode='pose')
        x_warp = tf.grid_sample(x, grid, padding_mode="zeros")

        return x_warp
