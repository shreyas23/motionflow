import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np

from .loss_utils import _disp2depth_kitti_K, _adaptive_disocc_detection, _generate_image_left, _reconstruction_error
from .inverse_warp import pose_vec2mat, pose2flow
from .interpolation import interpolate2d_as
from .sceneflow_util import projectSceneFlow2Flow

### Helper functions ###

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return tf.interpolate(x, scale_factor=2, mode="nearest")


def sample_grid(x, grid):
    """ Samples grid with border padding (uses border values when value is outside range [-1, 1])
    """
    tf.grid_sample(x, grid, padding_mode='border')


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
    

def visualize_output(args, input_dict, output_dict, epoch, writer):

    assert (writer is not None), "tensorboard writer not provided"

    img_l1 = input_dict['input_l1_aug'].detach()
    img_l2 = input_dict['input_l2_aug'].detach()
    img_r2 = input_dict['input_r2_aug'].detach()
    K = input_dict['input_k_l2_aug'].detach()
    disp_l1 = interpolate2d_as(output_dict['disps_l1'][0].detach(), img_l1)
    disp_l2 = interpolate2d_as(output_dict['disps_l2'][0].detach(), img_l1)
    flow_b = interpolate2d_as(output_dict['flows_b'][0].detach(), img_l1)
    pose_b = output_dict['pose_b'].detach()

    # input
    writer.add_images('input_l1', img_l1, epoch)
    writer.add_images('input_l2', img_l2, epoch)
    writer.add_images('input_r2', img_r2, epoch)

    # create (back)proj classes
    b, _, h, w = img_l1.shape
    back_proj = BackprojectDepth(b, h, w).to(device=img_l1.device)
    proj = Project3D(b, h, w).to(device=img_l1.device)

    # depth
    disp_warp = _generate_image_left(img_r2, disp_l2) 
    writer.add_images('disp', disp_l2, epoch)
    writer.add_images('disp_warp', disp_warp, epoch)

    b, _, h, w = disp_l1.shape
    disp_l1 = disp_l1 * w
    disp_l2 = disp_l2 * w

    # visualize depth
    depth = _disp2depth_kitti_K(disp_l2, K[:, 0, 0])
    writer.add_images('depth', depth, epoch)

    # pose warp
    cam_points = back_proj(depth, torch.inverse(K), mode='pose')
    grid = proj(cam_points, K, T=pose_b, sf=None, mode='pose')
    ref_warp = tf.grid_sample(img_l1, grid, mode="bilinear", padding_mode="border")
    writer.add_images('pose_warp', ref_warp, epoch)

    # pose occ map
    depth_l1 = _disp2depth_kitti_K(disp_l2, K[:, 0, 0])
    pose_f = output_dict['pose_f'].detach()
    pose_flow = pose2flow(depth_l1.squeeze(dim=1), None, K, torch.inverse(K), pose_mat=pose_f)
    pose_occ_b = _adaptive_disocc_detection(pose_flow)
    writer.add_images('pose_occ', pose_occ_b, epoch)

    # pose err
    pose_diff = _reconstruction_error(img_l2, ref_warp, 0.85)
    writer.add_images('pose_diff', pose_diff, epoch)

    # sf warp
    cam_points = back_proj(depth, torch.inverse(K), mode='sf')
    grid = proj(cam_points, K, T=None, sf=flow_b, mode='sf')
    ref_warp = tf.grid_sample(img_l1, grid, mode="bilinear", padding_mode="border")
    writer.add_images('sf_warp', ref_warp, epoch)

    # sf err
    sf_diff = _reconstruction_error(img_l2, ref_warp, 0.85)
    writer.add_images('sf_diff', sf_diff, epoch)

    # sf occ map
    flow_f = projectSceneFlow2Flow(K, output_dict['flows_f'][0].detach(), disp_l1)
    sf_occ_b = _adaptive_disocc_detection(flow_f)
    writer.add_images('sf_occ', sf_occ_b, epoch)

    return