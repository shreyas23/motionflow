import torch
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as tf
from torch.utils.data import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from augmentations import Augmentation_SceneFlow, Augmentation_Resize_Only
from datasets.kitti_raw_monosf import KITTI_Raw_KittiSplit_Train, KITTI_Raw_KittiSplit_Valid
from datasets.kitti_raw_monosf import KITTI_Raw_EigenSplit_Train, KITTI_Raw_EigenSplit_Valid
from models.JointModel import JointModel
from models.Model import Model
from losses import Loss

from .loss_utils import _disp2depth_kitti_K, _adaptive_disocc_detection, _generate_image_left, _reconstruction_error
from .loss_utils import _adaptive_disocc_detection_disp
from .inverse_warp import pose_vec2mat, pose2flow
from .interpolation import interpolate2d_as
from .sceneflow_util import projectSceneFlow2Flow, intrinsic_scale
from .helpers import BackprojectDepth, Project3D

import numpy as np

def get_model(args):
    if args.model_name == 'joint':
        model = JointModel(args)
    else:
        model = Model(args)

    return model


def get_loss(args, gpu):

    loss = Loss(args).cuda(device=gpu)

    return loss

def get_augmentations(args):
    train_augmentations = Augmentation_SceneFlow(args)
    val_augmentations = Augmentation_Resize_Only(args)

    return train_augmentations, val_augmentations


def get_dataset(args, gpu):
    DATASET_NAME = args.dataset_name
    DATA_ROOT = args.data_root

    if DATASET_NAME == 'KITTI':
        train_dataset = KITTI_Raw_KittiSplit_Train(args, DATA_ROOT, num_examples=args.num_examples)
        if args.validate and gpu == 0:
            val_dataset = KITTI_Raw_KittiSplit_Valid(args, DATA_ROOT)
        else:
            val_dataset = None

    elif DATASET_NAME == 'KITTI_EIGEN':
        train_dataset = KITTI_Raw_EigenSplit_Train(args, DATA_ROOT, num_examples=args.num_examples)
        if args.validate and gpu == 0:
            val_dataset = KITTI_Raw_EigenSplit_Valid(args, DATA_ROOT)
        else:
            val_dataset = None
    else:
        raise NotImplementedError

    return train_dataset, val_dataset


def step(args, data_dict, model, loss, augmentations, optimizer, gpu):
    # Get input and target tensor keys
    input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
    target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
    tensor_keys = input_keys + target_keys

    # transfer to cuda
    if args.cuda:
        for k, v in data_dict.items():
            if k in tensor_keys:
                data_dict[k] = v.cuda(device=gpu, non_blocking=True)

    if augmentations is not None:
        with torch.no_grad():
            data_dict = augmentations(data_dict)

    for k, t in data_dict.items():
        if k in input_keys:
            data_dict[k] = t.requires_grad_(True)
        if k in target_keys:
            data_dict[k] = t.requires_grad_(False)

    output_dict = model(data_dict)
    loss_dict = loss(output_dict, data_dict)

    return loss_dict, output_dict


def train_one_epoch(args, model, loss, dataloader, optimizer, augmentations, lr_scheduler, gpu):

    loss_dict_avg = None

    if gpu == 0:
        dataloader_iter = tqdm(dataloader)
    else:
        dataloader_iter = dataloader

    for i, data in enumerate(dataloader_iter):
        loss_dict, output_dict = step(
            args, data, model, loss, augmentations, optimizer, gpu)
        
        if loss_dict_avg is None:
            loss_dict_avg = {k:0 for k in loss_dict.keys()}

        # calculate gradients and then do Adam step
        optimizer.zero_grad()
        total_loss = loss_dict['total_loss']
        assert (not torch.isnan(total_loss)), f"training loss is nan, exiting...{loss_dict}"
        total_loss.backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        if args.grad_clip_value > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)
        optimizer.step()

        for key in loss_dict.keys():
            loss_dict_avg[key] += loss_dict[key].detach()

    return loss_dict_avg, output_dict, data


def evaluate(args, model, loss, dataloader, augmentations, gpu):
    model.eval()
    assert (model.training == False)

    loss_dict_avg = None

    with torch.no_grad():
        if gpu == 0:
            dataloader_iter = tqdm(dataloader)
        else:
            dataloader_iter = dataloader

        for data_dict in dataloader_iter:
            # Get input and target tensor keys
            input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
            target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
            tensor_keys = input_keys + target_keys

            # Possibly transfer to Cuda
            if args.cuda:
                for k, v in data_dict.items():
                    if k in tensor_keys:
                        data_dict[k] = v.cuda(non_blocking=True)

            data_dict = augmentations(data_dict)
            output_dict = model(data_dict)
            loss_dict = loss(output_dict, data_dict)

            if loss_dict_avg is None:
                loss_dict_avg = {k:0 for k in loss_dict.keys()}

            for key in loss_dict.keys():
                loss_dict_avg[key] += loss_dict[key]

    return loss_dict_avg, output_dict, data_dict


def evaluate_pose(args, model, loss, dataloader, augmentations, gpu):

    def compute_ate(gtruth_xyz, pred_xyz):
        alignment_error = pred_xyz - gtruth_xyz
        rmse = np.sqrt(sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
        return rmse

    model.eval()
    assert (model.training == False)

    loss_dict_sum = None

    with torch.no_grad():
        if gpu == 0:
            dataloader_iter = tqdm(dataloader)
        else:
            dataloader_iter = dataloader

        ates = []

        for data_dict in dataloader_iter:
            # Get input and target tensor keys
            input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
            target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
            tensor_keys = input_keys + target_keys

            # Possibly transfer to Cuda
            if args.cuda:
                for k, v in data_dict.items():
                    if k in tensor_keys:
                        data_dict[k] = v.cuda(non_blocking=True)

            data_dict = augmentations(data_dict)
            output_dict = model(data_dict)

            pose_t = output_dict['pose_b'][0][:, :3, 3].double().cpu()
            gt_t = data_dict['target_pose'][:, :3, 3].double().cpu()
            
            ate = compute_ate(gt_t, pose_t)
            ates.append(np.array(ate))

        ates = torch.tensor(ates).cuda(device=gpu)

    return ates


def visualize_output(args, input_dict, output_dict, epoch, writer, prefix):

    use_mask = args.train_exp_mask or args.train_census_mask

    assert (writer is not None), "tensorboard writer not provided"
    assert prefix in {'train', 'val', 'test'}
    prefix += '/'

    img_l1 = input_dict['input_l1'].detach()
    img_l2 = input_dict['input_l2'].detach()
    img_r2 = input_dict['input_r2'].detach()
    K = input_dict['input_k_l2_aug'].detach()

    disp_r2 = interpolate2d_as(output_dict['output_dict_r']['disps_l2'][0].detach(), img_l1)

    if prefix in {'val', 'test'}:
        disp_l1 = interpolate2d_as(output_dict['disps_l1_pp'][0].detach(), img_l1)
        disp_l2 = interpolate2d_as(output_dict['disps_l2_pp'][0].detach(), img_l1)
        sf_b = interpolate2d_as(output_dict['flows_b_pp'][0].detach(), img_l1)
        if use_mask:
            mask_l1 = interpolate2d_as(output_dict['masks_l1_pp'][0].detach(), img_l1)
            mask_l2 = interpolate2d_as(output_dict['masks_l2_pp'][0].detach(), img_l1)
    else:
        disp_l1 = interpolate2d_as(output_dict['disps_l1'][0].detach(), img_l1)
        disp_l2 = interpolate2d_as(output_dict['disps_l2'][0].detach(), img_l1)
        sf_b = interpolate2d_as(output_dict['flows_b'][0].detach(), img_l1)
        if use_mask:
            mask_l1 = interpolate2d_as(output_dict['masks_l1'][0].detach(), img_l1)
            mask_l2 = interpolate2d_as(output_dict['masks_l2'][0].detach(), img_l1)

    if 'pose_b' in output_dict:
        poses_b = output_dict['pose_b']
        poses_f = output_dict['pose_f']
        if isinstance(poses_b, list) and isinstance(poses_f, list):
            pose_f = poses_f[0].detach()
            pose_b = poses_b[0].detach()
        else:
            pose_f = poses_f.detach()
            pose_b = poses_b.detach()

        writer.add_text(prefix + 'pose_f', str(pose_f.cpu().detach().numpy()), epoch)
        writer.add_text(prefix + 'pose_b', str(pose_b.cpu().detach().numpy()), epoch)

    # input
    writer.add_images(prefix + 'input_l1', img_l1, epoch)
    writer.add_images(prefix + 'input_l2', img_l2, epoch)
    writer.add_images(prefix + 'input_r2', img_r2, epoch)

    # create (back)proj classes
    b, _, h, w = img_l1.shape
    back_proj = BackprojectDepth(b, h, w).to(device=img_l1.device)
    proj = Project3D(b, h, w).to(device=img_l1.device)

    # depth
    disp_warp = _generate_image_left(img_r2, disp_l2) 
    writer.add_images(prefix + 'disp', disp_l2, epoch)
    writer.add_images(prefix + 'disp_warp', disp_warp, epoch)
    disp_diff = _reconstruction_error(img_l2, disp_warp, args.ssim_w)
    writer.add_images(prefix + 'disp_diff', disp_diff, epoch)
    disp_occ = _adaptive_disocc_detection_disp(disp_r2)
    writer.add_images(prefix + 'disp_occ', disp_occ, epoch)

    b, _, h, w = disp_l1.shape
    disp_l1 = disp_l1 * w
    disp_l2 = disp_l2 * w

    # visualize depth
    depth = _disp2depth_kitti_K(disp_l2, K[:, 0, 0])
    writer.add_images(prefix + 'depth', depth, epoch)

    # static err map
    static_diff = _reconstruction_error(img_l2, img_l1, args.ssim_w)
    writer.add_images(prefix + 'static_diff', static_diff, epoch)

    # pose warp
    if 'pose_b' in output_dict:
        cam_points = back_proj(depth, torch.inverse(K), mode='pose')
        grid = proj(cam_points, K, T=pose_b, sf=None, mode='pose')
        ref_warp = tf.grid_sample(img_l1, grid, mode="bilinear", padding_mode="zeros")
        writer.add_images(prefix + 'pose_warp', ref_warp, epoch)

        # pose occ map
        depth_l1 = _disp2depth_kitti_K(disp_l1, K[:, 0, 0])
        pose_flow = pose2flow(depth_l1.squeeze(dim=1), None, K, torch.inverse(K), pose_mat=pose_f)
        pose_occ_b = _adaptive_disocc_detection(pose_flow)
        writer.add_images(prefix + 'pose_occ', pose_occ_b, epoch)

        # pose err
        pose_diff = _reconstruction_error(img_l2, ref_warp, args.ssim_w)
        writer.add_images(prefix + 'pose_diff', pose_diff, epoch)

    # sf warp
    cam_points = back_proj(depth, torch.inverse(K), mode='sf')
    grid = proj(cam_points, K, T=None, sf=sf_b, mode='sf')
    ref_warp = tf.grid_sample(img_l1, grid, mode="bilinear", padding_mode="zeros")
    writer.add_images(prefix + 'sf_warp', ref_warp, epoch)

    # sf err
    sf_diff = _reconstruction_error(img_l2, ref_warp, args.ssim_w)
    writer.add_images(prefix + 'sf_diff', sf_diff, epoch)

    # sf occ map
    flow_f = projectSceneFlow2Flow(K, output_dict['flows_f'][0].detach(), disp_l1)
    flow_b = projectSceneFlow2Flow(K, sf_b, disp_l2)
    sf_occ_b = _adaptive_disocc_detection(flow_f)
    sf_occ_f = _adaptive_disocc_detection(flow_b)
    writer.add_images(prefix + 'sf_occ_f', sf_occ_f, epoch)
    writer.add_images(prefix + 'sf_occ_b', sf_occ_b, epoch)

    # pts = cam_points.permute(0, 2, 3, 1).reshape(b, h*w, 3)
    # colors = img_l2.permute(0, 2, 3, 1).reshape(b, h*w, 3)
    # writer.add_mesh(tag='pc_l2', vertices=pts, colors=colors)

    # motion mask
    if use_mask:
        mask_l1_thresh = (mask_l1 > args.mask_thresh).float()
        mask_l2_thresh = (mask_l2 > args.mask_thresh).float()
        writer.add_images(prefix + 'pre_thresh_mask_l1', mask_l1, epoch)
        writer.add_images(prefix + 'thresh_mask_l1', mask_l1_thresh, epoch)
        writer.add_images(prefix + 'pre_thresh_mask_l2', mask_l2, epoch)
        writer.add_images(prefix + 'thresh_mask_l2', mask_l2_thresh, epoch)
    if 'census_masks_l2' in output_dict:
        census_mask_l2 = interpolate2d_as(output_dict['census_masks_l2'][0].detach(), img_l1)
        writer.add_images(prefix + 'target_census_mask', census_mask_l2, epoch)
    if 'rigidity_masks_l2_pp' in output_dict:
        rigid_mask = interpolate2d_as(output_dict['rigidity_masks_l2_pp'][0].detach(), img_l1)
        writer.add_images(prefix + 'rigidity_masks_l2_pp', rigid_mask, epoch)

    return