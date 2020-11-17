import torch
import torch.nn as nn
import torch.nn.functional as tf

from .inverse_warp import pose_vec2mat

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
    if pose.shape[-1] == 6:
        pose_mat = pose_vec2mat(pose.squeeze(dim=1))
    else:
        pose_mat = pose

    R = pose_mat[:, :3, :3].transpose(1, 2)
    t = pose_mat[:, :3, 1:] * -1

    return torch.cat([R, torch.matmul(R, t)], dim=-1)