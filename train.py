import os
import gc
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
import torch.distributed as dist

from augmentations import Augmentation_SceneFlow, Augmentation_Resize_Only

from datasets.kitti_raw_monosf import KITTI_Raw_KittiSplit_Train, KITTI_Raw_KittiSplit_Valid, KITTI_Raw_EigenSplit_Train, KITTI_Raw_EigenSplit_Valid

from utils.helpers import visualize_output
from utils.inverse_warp import flow_warp, pose2flow, inverse_warp, pose_vec2mat
from utils.sceneflow_util import projectSceneFlow2Flow, disp2depth_kitti, reconstructImg
from utils.sceneflow_util import pixel2pts_ms, pts2pixel_ms, pts2pixel_pose_ms, pixel2pts_ms_depth

from new_losses import Loss
from losses import Loss_SceneFlow_SelfSup_Separate
from models.Model import Model

from losses import _generate_image_left, _adaptive_disocc_detection


parser = argparse.ArgumentParser(description="Self Supervised Joint Learning of Scene Flow, Disparity, Rigid Camera Motion, and Motion Segmentation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# distributed params
parser.add_argument("--local_rank", type=int, default=0)

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
parser.add_argument('--save_freq', type=int, default=1, help='how often to save model state dict')
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
parser.add_argument('--decoder_type', type=str, default="full",
                    help="which decoder to use for Scene Net")

# dataset params
parser.add_argument('--dataset_name', default='KITTI', help='KITTI or Carla')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--num_views', type=int, default=2,
                    help="number of views present in training data")
parser.add_argument('--num_examples', type=int, default=-1,
                    help="number of examples to train on per epoch")
parser.add_argument('--num_workers', type=int, default=8,
                    help="number of workers for the dataloader")
parser.add_argument('--shuffle', type=bool,
                    default=False, help='shuffle the dataset?')
parser.add_argument('--resize_only', type=bool, default=False,
                    help='only do resize augmentation on input data')

# weight params
parser.add_argument('--ssim_w', type=float, default=0.85, help='mask consensus weight')
parser.add_argument('--flow_min_w', type=float, default=0.5, help='mask consensus weight')
parser.add_argument('--sf_pts_w', type=float, default=0.2, help='mask consensus weight')
parser.add_argument('--sf_sm_w', type=float, default=200, help='mask consensus weight')
parser.add_argument('--pose_sm_w', type=float, default=200, help='mask consensus weight')
parser.add_argument('--pose_pts_w', type=float, default=0.2, help='mask consensus weight')
parser.add_argument('--pose_lr_w', type=float, default=1.0, help='mask consensus weight')
parser.add_argument('--mask_lr_w', type=float, default=1.0, help='mask consensus weight')
parser.add_argument('--disp_lr_w', type=float, default=1.0, help='mask consensus weight')
parser.add_argument('--disp_sm_w', type=float, default=0.1, help='mask consensus weight')
parser.add_argument('--mask_reg_w', type=float, default=0.2, help='mask consensus weight')
parser.add_argument('--mask_sm_w', type=float, default=0.1, help='mask consensus weight')
parser.add_argument('--static_cons_w', type=float, default=0.0, help='mask consensus weight')
parser.add_argument('--mask_cons_w', type=float, default=0.0, help='mask consensus weight')
parser.add_argument('--flow_diff_thresh', type=float, default=1e-3, help='mask consensus weight')

parser.add_argument('--flow_reduce_mode', type=str, default="avg",
                    help='only do resize augmentation on input data')
parser.add_argument('--pt_encoder', type=bool, default=True,
                    help='only do resize augmentation on input data')
parser.add_argument('--use_flow_mask', type=bool, default=False,
                    help='only do resize augmentation on input data')

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
parser.add_argument('--grad_clip_norm', type=float, default=0,
                    help='gradient clipping threshold')
parser.add_argument('--grad_clip_value', type=float, default=0,
                    help='gradient clipping threshold')

# model params
parser.add_argument('--train_consistency', type=bool, default=False,
                    help="whether to use consistency losses in training procedure")
parser.add_argument('--use_bn', type=bool, default=False,
                    help="whether to use batch-norm in training procedure")
parser.add_argument('--use_mask', type=bool, default=False,
                    help="whether to use consensus mask in training procedure")
parser.add_argument('--use_ppm', type=bool, default=False,
                    help="whether to use ppm")
parser.add_argument('--num_scales', type=int,
                    default=4, help='number of gpus used for training')

# etc.
parser.add_argument('--num_gpus', type=int,
                    default=1, help='number of gpus used for training')
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
    print(f"{arg}: {getattr(args, arg)}")

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
      train_dataset = KITTI_Raw_KittiSplit_Train(args, DATA_ROOT, num_examples=args.num_examples)
      train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=True)
      if args.validate:
          val_dataset = KITTI_Raw_KittiSplit_Valid(args, DATA_ROOT)
          val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_dataset else None
      else:
          val_dataset = None
          val_dataloader = None

  elif DATASET_NAME == 'KITTI_EIGEN':
      train_dataset = KITTI_Raw_EigenSplit_Train(args, DATA_ROOT, num_examples=args.num_examples)
      train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, pin_memory=True)
      if args.validate:
          val_dataset = KITTI_Raw_EigenSplit_Valid(args, DATA_ROOT)
          val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_dataset else None
      else:
          val_dataset = None
          val_dataloader = None
  else:
      raise NotImplementedError

  model = Model(args).cuda()
  loss = Loss(args).cuda()
#   loss = Loss_SceneFlow_SelfSup_Separate(args).cuda()

  # define augmentations
  if args.resize_only:
      print("Augmentations: Augmentation_Resize_Only")
      augmentations = Augmentation_Resize_Only(args, photometric=False)
  else:
      print("Augmentations: Augmentation_SceneFlow")
      augmentations = Augmentation_SceneFlow(args)

  # load the model
  print("Loding model and augmentations and placing on gpu...")

  if args.cuda:
    loss = loss.cuda()
    if augmentations is not None:
        augmentations = augmentations.cuda()
    
    device = torch.device("cuda:0")
    model = model.to(device=device)

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
    milestones = []
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

    model.train()

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

    if args.lr_sched_type == 'plateau':
      lr_scheduler.step(train_loss_avg_dict['total_loss'])
    elif args.lr_sched_type == 'step':
      lr_scheduler.step(epoch)

    # save model
    if not args.no_logging:
      for k, v in train_loss_avg_dict.items():
        writer.add_scalar(f'loss/train/{k}', train_loss_avg_dict[k], epoch)
      if epoch % args.log_freq == 0:
        visualize_output(args, input_dict, output_dict, epoch, writer)

      fp = os.path.join(log_dir, f"{epoch}.ckpt")

      if args.save_freq > 0:
        if epoch % args.save_freq == 0:
          torch.save(model.state_dict(), fp)
      elif epoch == args.epochs:
        torch.save(model.state_dict(), fp)

  if not args.no_logging:
    writer.flush()


def step(args, data_dict, model, loss, augmentations, optimizer):
  # Get input and target tensor keys
  input_keys = list(filter(lambda x: "input" in x, data_dict.keys()))
  target_keys = list(filter(lambda x: "target" in x, data_dict.keys()))
  tensor_keys = input_keys + target_keys

  # transfer to cuda
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
    if args.grad_clip_norm > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
    if args.grad_clip_value > 0:
      torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)
    optimizer.step()

    for key in loss_dict.keys():
      loss_dict_avg[key] += loss_dict[key].detach().item()

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

if __name__ == '__main__':
  main()