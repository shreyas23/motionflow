import torch
import torch.nn as nn
import torch.nn.functional as tf
from models.forwardwarp_package.forward_warp import forward_warp

###############################################
# Loss Utils
###############################################

eps = 1e-8

def kl_div(p, q):
    """
    Calculate the KL Divergence denoted: KL(p||q) = (1/h*w) * p(x) * log(p(x) / q(x))
    """
    div = torch.exp(p) * (p - torch.log(q))
    return torch.abs(div.mean(dim=1, keepdim=True))

def logical_or(a, b):
    return 1 - ((1 - a) * (1 -b))

def _reconstruction_error(tgt_img, ref_img_warped, ssim_w):
    diff = (_elementwise_l1(tgt_img, ref_img_warped) * (1.0 - ssim_w) +
            _SSIM(tgt_img, ref_img_warped) * ssim_w).mean(dim=1, keepdim=True)
    return diff

def robust_reconstruction_error(tgt_img, ref_img_warped, ssim_w):
    residual = tgt_img - ref_img_warped
    diff = (robust_l1_per_pix(residual) * (1.0 - ssim_w) +
            _SSIM(tgt_img, ref_img_warped) * ssim_w).mean(dim=1, keepdim=True)
    return diff

def _elementwise_epe(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=2, dim=1, keepdim=True)

def _elementwise_l1(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.norm(residual, p=1, dim=1, keepdim=True)

def _elementwise_robust_epe_char(input_flow, target_flow):
    residual = target_flow - input_flow
    return torch.pow(torch.norm(residual, p=2, dim=1, keepdim=True) + 0.01, 0.4)

def robust_l1_per_pix(x, q=0.5, eps=1e-2):
    x = torch.pow((x.pow(2) + eps), q)
    return x

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
    x_base = torch.linspace(0, 1, width).repeat(
        batch_size, height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(
        batch_size, width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    # Disparity is passed in NCHW format with 1 channel
    x_shifts = disp[:, 0, :, :]
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = tf.grid_sample(img, 2 * flow_field - 1,
                            mode='bilinear', padding_mode='zeros')

    return output


def _generate_image_left(img, disp):
    return _apply_disparity(img, -disp)


def _generate_image_right(img, disp):
    return _apply_disparity(img, disp)


def _adaptive_disocc_detection(flow):

    # init mask
    b, _, h, w, = flow.size()
    mask = torch.ones(b, 1, h, w, dtype=flow.dtype,
                      device=flow.device).float().requires_grad_(False)
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1)
    disocc_map = (disocc > 0.5)

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(
            b, 1, h, w, dtype=torch.bool, device=flow.device).requires_grad_(False)

    return disocc_map


def _adaptive_disocc_detection_disp(disp):

    # # init
    b, _, h, w, = disp.size()
    mask = torch.ones(b, 1, h, w, dtype=disp.dtype,
                      device=disp.device).float().requires_grad_(False)
    flow = torch.zeros(b, 2, h, w, dtype=disp.dtype,
                       device=disp.device).float().requires_grad_(False)
    flow[:, 0:1, :, :] = disp * w
    flow = flow.transpose(1, 2).transpose(2, 3)

    disocc = torch.clamp(forward_warp()(mask, flow), 0, 1)
    disocc_map = (disocc > 0.5)

    if disocc_map.float().sum() < (b * h * w / 2):
        disocc_map = torch.ones(
            b, 1, h, w, dtype=torch.bool, device=disp.device).requires_grad_(False)

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

def _smoothness_1st(feat, img, beta=1):
    feat_grad_x = _gradient_x(feat)
    feat_grad_y = _gradient_y(feat)

    weights_x = torch.exp(-torch.mean(torch.abs(feat_grad_x),
                                      1, keepdim=True) * beta)
    weights_y = torch.exp(-torch.mean(torch.abs(feat_grad_y),
                                      1, keepdim=True) * beta)

    smoothness_x = feat_grad_x * weights_x
    smoothness_y = feat_grad_y * weights_y

    return -(smoothness_x.abs() + smoothness_y.abs())

def _smoothness_motion_2nd(sf, img, beta=1):
    sf_grad_x = _gradient_x_2nd(sf)
    sf_grad_y = _gradient_y_2nd(sf)

    img_grad_x = _gradient_x(img)
    img_grad_y = _gradient_y(img)
    weights_x = torch.exp(-torch.mean(torch.abs(img_grad_x),
                                      1, keepdim=True) * beta)
    weights_y = torch.exp(-torch.mean(torch.abs(img_grad_y),
                                      1, keepdim=True) * beta)

    smoothness_x = sf_grad_x * weights_x
    smoothness_y = sf_grad_y * weights_y

    return (smoothness_x.abs() + smoothness_y.abs())


def _disp2depth_kitti_K(disp, fx, min_depth=1e-3, max_depth=80):

    depth = fx.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / (disp + eps)
    depth = torch.clamp(depth, min_depth, max_depth)

    return depth


def _depth2disp_kitti_K(depth, fx):

    disp = fx.unsqueeze(1).unsqueeze(1).unsqueeze(1) * 0.54 / depth

    return disp

# from monodepth v2
def disp_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
       The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()
