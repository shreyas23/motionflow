# based on github.com/ClementPinard/SfMLearner-Pytorch

from __future__ import division
import torch
from torch.autograd import Variable

pixel_coords = None


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w)).type_as(depth)  # [1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w)).type_as(depth)  # [1, H, W]
    ones = Variable(torch.ones(1,h,w)).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) != h or pixel_coords.size(3) != w:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).contiguous().view(b, 3, -1)  # [B, 3, H*W]
    cam_coords = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]
    if padding_mode == 'zeros':
        X_mask = ((X_norm > 1)+(X_norm < -1)).detach()
        X_norm[X_mask] = 2  # make sure that no point in warped image is a combinaison of im and gray
        Y_mask = ((Y_norm > 1)+(Y_norm < -1)).detach()
        Y_norm[Y_mask] = 2

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(b,h,w,2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).view(B, 3, 3)

    rotMat = xmat.bmm(ymat).bmm(zmat)
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

def flow_warp(img, flow, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        flow: flow map of the target image -- [B, 2, H, W]
    Returns:
        Source image warped to the target image plane
    """
    check_sizes(img, 'img', 'BCHW')
    check_sizes(flow, 'flow', 'B2HW')

    bs, _, h, w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(v).expand_as(v)  # [bs, H, W]

    X = grid_x + u
    Y = grid_y + v

    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)
    grid_tf = torch.stack((X,Y), dim=3)
    img_tf = torch.nn.functional.grid_sample(img, grid_tf, padding_mode=padding_mode)

    return img_tf


def pose2flow(depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode=None, pose_mat=None):
    """
    Converts pose parameters to rigid optical flow
    """
    check_sizes(depth, 'depth', 'BHW')
    # check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')
    assert(intrinsics_inv.size() == intrinsics.size())

    bs, h, w = depth.size()

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]

    if pose is not None and pose_mat is None:
        pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]
    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]

    X = (w-1)*(src_pixel_coords[:,:,:,0]/2.0 + 0.5) - grid_x
    Y = (h-1)*(src_pixel_coords[:,:,:,1]/2.0 + 0.5) - grid_y

    return torch.stack((X,Y), dim=1)

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

def pose2sceneflow(depth, pose, intrinsics, rotation_mode='euler', padding_mode=None, pose_mat=None):
    """
    Converts pose parameters to rigid scene flow
    """
    check_sizes(depth, 'depth', 'B1HW')
    # check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    _, _, h, w = depth.size()

    # projection matrix from target to points
    # cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]
    cam_coords, _ = pixel2pts(intrinsics, depth)  # [B,3,H,W]

    # tform matrix
    if pose is not None and pose_mat is None:
        pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]
    R = pose_mat[:, :, :-1]
    t = pose_mat[:, :, -1:]

    # Get projection matrix for tgt camera frame to source camera frame
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if R is not None:
        pcoords = R.bmm(cam_coords_flat)
    else:
        pcoords = cam_coords_flat

    if t is not None:
        pcoords = pcoords + t  # [B, 3, H*W]

    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)
    cam_coords_tform = torch.stack((X, Y, Z), dim=1).view(b, 3, h, w)  # [B,3,H,W]

    flow = cam_coords_tform - cam_coords

    return flow


def flow2oob(flow):
    check_sizes(flow, 'flow', 'B2HW')

    bs, _, h, w = flow.size()
    u = flow[:,0,:,:]
    v = flow[:,1,:,:]

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(v).expand_as(v)  # [bs, H, W]

    X = grid_x + u
    Y = grid_y + v

    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)
    oob = (X.abs()>1).add(Y.abs()>1)>0
    return oob

def occlusion_mask(grid, depth):
    check_sizes(img, 'grid', 'BHW2')
    check_sizes(depth, 'depth', 'BHW')

    mask = grid

    return mask


def inverse_warp(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros', pose_mat=None):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
        intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
        Source image warped to the target image plane
    """
    # check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
    # check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')

    assert(intrinsics_inv.size() == intrinsics.size())

    batch_size, _, h, w = img.size()

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]

    if pose is not None and pose_mat is None:
        pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    proj_cam_coords = torch.bmm(pose_mat, torch.cat([cam_coords.view(batch_size, 3, h*w), 
                                            torch.ones(batch_size, 1, h*w).to(device=cam_coords.device, dtype=cam_coords.dtype)], dim=1))
    proj_cam_coords = proj_cam_coords.view(batch_size, 3, h, w)

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]
    projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)
    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, src_pixel_coords, cam_coords, proj_cam_coords, valid_points
