import os
import argparse
import datetime
import numpy as np
from sys import exit
from time import time
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from augmentations import Augmentation_SceneFlow, Augmentation_Resize_Only

from datasets.kitti_raw_monosf import KITTI_Raw_KittiSplit_Train, KITTI_Raw_KittiSplit_Valid

from models.SceneNet import SceneNet
from models.SceneNetStereo import SceneNetStereo
from models.model_monosceneflow import MonoSceneFlow
from models.PoseDepthNet import PoseDispNet

from utils.inverse_warp import flow_warp, pose2flow, inverse_warp, pose_vec2mat
from utils.sceneflow_util import projectSceneFlow2Flow, disp2depth_kitti, reconstructImg
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, pts2pixel_pose_ms, pixel2pts_ms_depth

from losses import Loss_SceneFlow_SelfSup, Loss_SceneFlow_SelfSup_Pose, Loss_SceneFlow_SelfSup_PoseStereo
from losses import _generate_image_left, _adaptive_disocc_detection
from losses import Loss_PoseDepth


parser = argparse.ArgumentParser(description="Self Supervised Joint Learning of Scene Flow, Disparity, Rigid Camera Motion, and Motion Segmentation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# runtime params
parser.add_argument('--data_root', help='path to dataset', required=True)
parser.add_argument('--epochs', type=int, required=True,
                    help='number of epochs to run')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='resume from checkpoint (using experiment name)')
parser.add_argument('--cuda', type=bool, default=True, help='use gpu?')
parser.add_argument('--no_logging', type=bool, default=False,
                    help="are you logging this experiment?")
parser.add_argument('--log_dir', type=str, default="/external/cnet/checkpoints",
                    help="are you logging this experiment?")
parser.add_argument('--log_freq', type=int, default=1, help='how often to log statistics')
parser.add_argument('--exp_dir', type=str, default='test',
                    help='name of experiment, chkpts stored in checkpoints/experiment')
parser.add_argument('--exp_name', type=str, default='test',
                    help='name of experiment, chkpts stored in checkpoints/exp_dir/exp_name')
parser.add_argument('--validate', type=bool, default=False,
                    help='set to true if validating model')
parser.add_argument('--ckpt', type=str, default="",
                    help="path to model checkpoint if using one")
parser.add_argument('--use_pretrained', type=bool,
                    default=False, help="use pretrained model from authors")

# module params
parser.add_argument('--model_name', type=str,
                    default="scenenet", help="name of model")
parser.add_argument('--encoder_name', type=str, default="pwc",
                    help="which encoder to use for Scene Net")

# dataset params
parser.add_argument('--dataset_name', default='KITTI', help='KITTI or Carla')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_views', type=int, default=2,
                    help="number of views present in training data")
parser.add_argument('--num_examples', type=int, default=-1,
                    help="number of examples to train on per epoch")
parser.add_argument('--num_workers', type=int, default=8,
                    help="number of workers for the dataloader")
parser.add_argument('--shuffle_dataset', type=bool,
                    default=False, help='shuffle the dataset?')
parser.add_argument('--resize_only', type=bool, default=False,
                    help='only do resize augmentation on input data')

# weight params
parser.add_argument('--pose_sm_w', type=float, default=200, help='mask consensus weight')
parser.add_argument('--disp_lr_w', type=float, default=1.0, help='mask consensus weight')
parser.add_argument('--disp_smooth_w', type=float, default=0.1, help='mask consensus weight')
parser.add_argument('--mask_reg_w', type=float, default=0.2, help='mask consensus weight')
parser.add_argument('--mask_sm_w', type=float, default=0.1, help='mask consensus weight')
parser.add_argument('--static_cons_w', type=float, default=0.0, help='mask consensus weight')
parser.add_argument('--mask_cons_w', type=float, default=0.0, help='mask consensus weight')
parser.add_argument('--flow_diff_thresh', type=float, default=1e-3, help='mask consensus weight')

# learning params
parser.add_argument('--lr', type=float, default=2e-4,
                    help='initial learning rate')
parser.add_argument('--lr_sched_type', type=str, default="none",
                    help="path to model checkpoint if using one")
parser.add_argument('--lr_gamma', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum for sgd or alpha param for adam')
parser.add_argument('--beta', type=float, default=0.999,
                    help='beta param for adam')
parser.add_argument('--weight_decay', type=float,
                    default=0.0, help='weight decay')
parser.add_argument('--dropout', type=bool, default=False,
                    help='dropout for regularization', choices=[True, False])
parser.add_argument('--grad_clip', type=float, default=0,
                    help='gradient clipping threshold')

# model params
parser.add_argument('--train_consistency', type=bool, default=False,
                    help="whether to use consistency losses in training procedure")
parser.add_argument('--use_bn', type=bool, default=False,
                    help="whether to use batch-norm in training procedure")
parser.add_argument('--use_mask', type=bool, default=False,
                    help="whether to use consensus mask in training procedure")

# etc.
parser.add_argument('--multi_gpu', type=bool,
                    default=False, help='use multiple gpus')
parser.add_argument('--debugging', type=bool,
                    default=False, help='are you debugging?')
parser.add_argument('--finetuning', type=bool, default=False,
                    help='finetuning on supervised data')
parser.add_argument('--evaluation', type=bool,
                    default=False, help='evaluating on data')
parser.add_argument('--torch_seed', default=123768,
                    help='random seed for reproducibility')
parser.add_argument('--cuda_seed', default=543987,
                    help='random seed for reproducibility')

args = parser.parse_args()


def main():
  for arg in vars(args):
    print(arg, ":", getattr(args, arg))
  if args.batch_size == 1 and args.use_bn is True:
    raise Exception

  torch.autograd.set_detect_anomaly(True)
  torch.manual_seed(args.torch_seed)
  torch.cuda.manual_seed(args.cuda_seed)

  DATASET_NAME = args.dataset_name
  DATA_ROOT = args.data_root

  # load the dataset/dataloader
  print("Loading dataset and dataloaders...")
  if DATASET_NAME == 'KITTI':
    if args.model_name == 'scenenet':
      model = SceneNet(args)
      loss = Loss_SceneFlow_SelfSup_Pose(args)
    elif args.model_name == 'scenenet_stereo':
      model = SceneNetStereo(args)
      loss = Loss_SceneFlow_SelfSup_PoseStereo(args)
    elif args.model_name == 'monoflow':
      model = MonoSceneFlow(args)
      loss = Loss_SceneFlow_SelfSup(args)
    elif args.model_name == 'posedepth':
      model = PoseDispNet(args)
      loss = Loss_PoseDepth()
    else:
      raise NotImplementedError

    # define dataset
    train_dataset = KITTI_Raw_KittiSplit_Train(
        args, DATA_ROOT, num_examples=args.num_examples, flip_augmentations=False, preprocessing_crop=True)
    train_dataloader = DataLoader(train_dataset, args.batch_size,
                                  shuffle=args.shuffle_dataset, num_workers=args.num_workers, pin_memory=True)
    val_dataset = KITTI_Raw_KittiSplit_Valid(
        args, DATA_ROOT, num_examples=args.num_examples)
    val_dataset=None

    val_dataloader = DataLoader(val_dataset, 4, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_dataset else None

    # define augmentations
    if args.resize_only:
      print("Augmentations: Augmentation_Resize_Only")
      augmentations = Augmentation_Resize_Only(args, photometric=False)
    else:
      print("Augmentations: Augmentation_SceneFlow")
      augmentations = Augmentation_SceneFlow(args)
  else:
    raise NotImplementedError

  # load the model
  print("Loding model and augmentations and placing on gpu...")

  if args.cuda:
    if augmentations is not None:
        augmentations = augmentations.cuda()
    model = model.cuda()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"The model has {num_params} learnable parameters")

  # load optimizer and lr scheduler
  optimizer = Adam(model.parameters(), lr=args.lr, betas=[
                   args.momentum, args.beta], weight_decay=args.weight_decay)

  if args.lr_sched_type == 'plateau':
    print("Using plateau lr schedule")
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_gamma, verbose=True, mode='min', patience=10)
  elif args.lr_sched_type == 'step':
    print("Using step lr schedule")
    milestones = [30]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.lr_gamma)
  elif args.lr_sched_type == 'none':
    lr_scheduler = None

  # set up logging
  if not args.no_logging:
    if not os.path.isdir(args.log_dir):
      os.mkdir(args.log_dir)
    log_dir = os.path.join(args.log_dir, args.exp_dir)
    if not os.path.isdir(log_dir):
      os.mkdir(log_dir)
    if args.exp_name == "":
      exp_name = datetime.datetime.now().strftime("%H%M%S-%Y%m%d")
    else:
      exp_name = args.exp_name
    log_dir = os.path.join(log_dir, exp_name)
    writer = SummaryWriter(log_dir)

  if args.ckpt != "" and args.use_pretrained:
    state_dict = torch.load(args.ckpt)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
      name = k[7:]
      new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
  elif args.start_epoch > 0:
    load_epoch = args.start_epoch - 1
    ckpt_fp = os.path.join(log_dir, f"{load_epoch}.ckpt")

    print(f"Loading model from {ckpt_fp}...")

    ckpt = torch.load(ckpt_fp)
    assert (ckpt['epoch'] ==
            load_epoch), "epoch from state dict does not match with args"
    model.load_state_dict(ckpt)

  model = model.train()

  # run training loop
  for epoch in range(args.start_epoch, args.epochs + 1):
    print(f"Training epoch: {epoch}...")
    train_loss_avg_dict, output_dict, input_dict = train_one_epoch(
        args, model, loss, train_dataloader, optimizer, augmentations, lr_scheduler)
    print(f"\t Epoch {epoch} train loss avg:")
    pprint(train_loss_avg_dict)

    if val_dataset is not None:
      print(f"Validation epoch: {epoch}...")
      val_loss_avg = eval(args, model, loss, val_dataloader, augmentations)
      print(f"\t Epoch {epoch} val loss avg: {val_loss_avg}")

    if not args.no_logging:
      for k, v in train_loss_avg_dict.items():
        writer.add_scalar(f'loss/train/{k}', train_loss_avg_dict[k], epoch)
      if epoch % args.log_freq == 0:
        visualize_output(args, input_dict, output_dict, epoch, writer)

    # assert (not torch.isnan(train_loss_avg_dict['total_loss'])), "avg training loss is nan"

    if args.lr_sched_type == 'plateau':
      lr_scheduler.step(train_loss_avg_dict['total_loss'])
    elif args.lr_sched_type == 'step':
      lr_scheduler.step(epoch)

    # save model
    if not args.no_logging:
      if epoch % 1000 == 0 or epoch == args.epochs:
        fp = os.path.join(log_dir, f"{epoch}.ckpt")
        torch.save(model.state_dict(), fp)

  if not args.no_logging:
    writer.flush()

  return


def step(args, data_dict, model, loss, augmentations, optimizer):
  # Get input and target tensor keys
  input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
  target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
  tensor_keys = input_keys + target_keys

  # Possibly transfer to Cuda
  if args.cuda:
    for k, v in data_dict.items():
      if k in tensor_keys:
        data_dict[k] = v.cuda(non_blocking=True)

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

#   training_loss = loss_dict['total_loss'].detach()
#   assert (not torch.isnan(training_loss)), f"training_loss is NaN: {loss_dict}"

  return loss_dict, output_dict


def train_one_epoch(args, model, loss, dataloader, optimizer, augmentations, lr_scheduler):

  loss_dict_avg = None

  for data in tqdm(dataloader):
    loss_dict, output_dict = step(
        args, data, model, loss, augmentations, optimizer)
    
    if loss_dict_avg is None:
        loss_dict_avg = {k:0 for k in loss_dict.keys()}

    # calculate gradients and then do Adam step
    optimizer.zero_grad()
    total_loss = loss_dict['total_loss']
    total_loss.backward()
    if args.grad_clip > 0:
      torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
    optimizer.step()

    for key in loss_dict.keys():
      loss_dict_avg[key] += loss_dict[key].detach()

  n = len(dataloader)
  for key in loss_dict_avg.keys():
    loss_dict_avg[key] /= n

  return loss_dict_avg, output_dict, data


def eval(args, model, loss, dataloader, augmentations):
  val_loss_sum = 0.

  for data_dict in tqdm(dataloader):
    with torch.no_grad():
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
      val_loss_sum += loss_dict['total_loss']

  val_loss_avg = val_loss_sum / len(dataloader)
  return val_loss_avg


def visualize_output(args, input_dict, output_dict, epoch, writer):

    assert (writer is not None), "tensorboard writer not provided"

    img_l1_aug = input_dict['input_l1_aug'].detach()
    img_l2_aug = input_dict['input_l2_aug'].detach()
    img_r2_aug = input_dict['input_r2_aug'].detach()
    k_l2_aug = input_dict['input_k_l2_aug'].detach()
    aug_size = input_dict['aug_size']

    # input
    writer.add_images('input_l1', img_l1_aug, epoch)
    writer.add_images('input_l2', img_l2_aug, epoch)
    writer.add_images('input_r2', img_r2_aug, epoch)

    if args.model_name in ['scenenet', 'scenenet_stereo']:
        sf_b = output_dict['flow_b'][0].detach()
        pose = output_dict['pose_b'].detach()
        print(f"Transformation matrices for epoch: {epoch}, {pose}")
        if args.use_mask:
            mask = output_dict['masks_b'][0].detach()
            census_mask = output_dict['census_masks_b'][0].detach()
            writer.add_images('mask', mask, epoch)
            writer.add_images('census mask', census_mask, epoch)

        # disparity
        disp_l2 = output_dict['disp_l2'][0].detach()
        writer.add_images('img_r2_disp_warp', _generate_image_left(img_r2_aug, disp_l2), epoch)
        _, _, h_dp, w_dp = sf_b.size()
        disp_l2 = disp_l2 * w_dp
        writer.add_images('disp_l2', disp_l2, epoch)

        # scene flow
        local_scale = torch.zeros_like(aug_size)
        local_scale[:, 0] = h_dp
        local_scale[:, 1] = w_dp

        pts2, k2_scale, depth = pixel2pts_ms_depth(
            k_l2_aug, disp_l2, local_scale / aug_size)

        _, _, coord2 = pts2pixel_ms(k2_scale, pts2, sf_b, [h_dp, w_dp])
        img_l1_warp = reconstructImg(coord2, img_l1_aug)
        writer.add_images('img_l1_warp', img_l1_warp, epoch)

        # camera pose
        _, coord2 = pts2pixel_pose_ms(k2_scale, pts2, pose, [h_dp, w_dp])
        img_l1_warp_cam = reconstructImg(coord2, img_l1_aug)
        writer.add_images('img_l1_warp_cam', img_l1_warp_cam, epoch)

    if args.model_name in ['posedepth']:
        pose = output_dict['pose_b'].detach()
        _, _, h_dp, w_dp = disp_l2.size()
        disp_l2 = disp_l2 * w_dp

        # camera pose
        depth = disp2depth_kitti(disp_l2, k_l2_aug[:, 0, 0])
        img_l1_warp_cam = inverse_warp(img_l1_aug, depth.squeeze(
            dim=1), pose.squeeze(dim=1), k_l2_aug, torch.inverse(k_l2_aug))

        writer.add_images('img_l1_warp_cam', img_l1_warp_cam, epoch)


if __name__ == '__main__':
  main()
