import gc
import os
import json
import argparse
import datetime
from tqdm import tqdm
from pprint import pprint
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from augmentations import Augmentation_SceneFlow, Augmentation_Resize_Only
from datasets.kitti_raw_monosf import KITTI_Raw_KittiSplit_Train, KITTI_Raw_KittiSplit_Valid
from datasets.kitti_raw_monosf import KITTI_Raw_EigenSplit_Train, KITTI_Raw_EigenSplit_Valid
from models.JointModel import JointModel
from models.Model import Model
from losses import Loss

from params import Params
from utils.train_utils import step, evaluate, train_one_epoch, visualize_output

args = Params().args

def main():
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    train(args)

    print("--- Training complete ---")

def train(args):

    if args.batch_size == 1 and args.use_bn is True:
        raise Exception

    # set some torch params
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed(args.cuda_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DATASET_NAME = args.dataset_name
    DATA_ROOT = args.data_root

    if args.model_name == 'joint':
        model = JointModel(args).cuda()
        from old_losses import Loss_SceneFlow_SelfSup_JointIter
        loss = Loss_SceneFlow_SelfSup_JointIter(args).cuda()
    else:
        model = Model(args).cuda() 
        loss = Loss(args).cuda()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} learnable parameters")

    print("Loading dataset and dataloaders")
    if DATASET_NAME == 'KITTI':
        train_dataset = KITTI_Raw_KittiSplit_Train(args, DATA_ROOT, num_examples=args.num_examples)
        train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True)
        if args.validate:
            val_dataset = KITTI_Raw_KittiSplit_Valid(args, DATA_ROOT)
            val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_dataset else None
        else:
            val_dataset = None
            val_dataloader = None

    elif DATASET_NAME == 'KITTI_EIGEN':
        train_dataset = KITTI_Raw_EigenSplit_Train(args, DATA_ROOT, num_examples=args.num_examples)
        train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True)
        if args.validate:
            val_dataset = KITTI_Raw_EigenSplit_Valid(args, DATA_ROOT)
            val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_dataset else None
        else:
            val_dataset = None
            val_dataloader = None
    else:
        raise NotImplementedError

    # define augmentations
    train_augmentations = Augmentation_SceneFlow(args)
    val_augmentations = Augmentation_Resize_Only(args)
    if args.cuda:
        train_augmentations = train_augmentations.cuda()
        val_augmentations = val_augmentations.cuda()

    # load optimizer and lr scheduler
    optimizer = Adam(model.parameters(), 
                     lr=args.lr, 
                     betas=[args.momentum, args.beta], 
                     weight_decay=args.weight_decay)

    if args.lr_sched_type == 'plateau':
        print("Using plateau lr schedule")
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.lr_gamma, verbose=True, mode='min', patience=1)
    elif args.lr_sched_type == 'step':
        print("Using step lr schedule")
        milestones = [10, 15]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=args.lr_gamma)
    elif args.lr_sched_type == 'none':
        lr_scheduler = None

    # set up logging
    if not args.no_logging:
        log_dir = os.path.join(args.log_root, args.exp_dir)
        os.makedirs(log_dir, exist_ok=True)

        if args.exp_name == "":
            exp_name = datetime.datetime.now().strftime("%d%m%y-%I%M%S")
        else:
            exp_name = args.exp_name

        log_dir = os.path.join(log_dir, exp_name)

        print(f"All logs will be stored at: {log_dir}")

        writer = SummaryWriter(log_dir)
    
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    curr_epoch = args.start_epoch

    if args.ckpt != "":
        print(f"Loading model from {args.ckpt}")
        model.load_state_dict(
            torch.load(args.ckpt)['model'])
        optimizer.load_state_dict(
            torch.load(args.ckpt)['optimizer'])
    elif args.start_epoch > 1:
        load_epoch = args.start_epoch - 1
        ckpt_fp = os.path.join(log_dir, f"{load_epoch}.ckpt")
        model.load_state_dict(
            torch.load(ckpt_fp)['model'])
        optimizer.load_state_dict(
            torch.load(ckpt_fp)['optimizer'])

    model.train()

    # run training loop
    for epoch in range(curr_epoch, curr_epoch + args.epochs):

        print(f"Training epoch: {epoch}...\n")

        train_loss_avg_dict, output_dict, input_dict = train_one_epoch(
            args, model, loss, train_dataloader, optimizer, train_augmentations, lr_scheduler, 0)

        print(f"\t Epoch {epoch} train loss avg:")
        pprint(train_loss_avg_dict)
        print("\n")

        if args.validate:
            print(f"Validation epoch: {epoch}...\n")
            val_loss_avg_dict, val_output_dict, val_input_dict = evaluate(args, model, loss, val_dataloader, val_augmentations)
            print(f"\t Epoch {epoch} val loss avg:")
            pprint(val_loss_avg_dict)
            print("\n")
        else:
            val_output_dict, val_input_dict = None, None

        if args.lr_sched_type == 'plateau':
            lr_scheduler.step(train_loss_avg_dict['total_loss'])
        elif args.lr_sched_type == 'step':
            lr_scheduler.step(epoch)

        if not args.no_logging:
            fp = os.path.join(log_dir, f"{epoch}.ckpt")
            for k, v in train_loss_avg_dict.items():
                writer.add_scalar(f'train/{k}', v.item(), epoch)
            if args.validate:
                for k, v in val_loss_avg_dict.items():
                    writer.add_scalar(f'val/{k}', v.item(), epoch)

            if args.save_freq > 0:
                if epoch % args.save_freq == 0 or epoch == args.epochs:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, fp)

            if epoch % args.log_freq == 0:
                visualize_output(args, input_dict, output_dict, epoch, writer)
                del input_dict
                del output_dict

                if args.validate:
                    visualize_output(args, val_input_dict, val_output_dict, epoch, writer)
                    del val_input_dict
                    del val_output_dict

                writer.flush()

            if args.save_freq > 0:
                if epoch % args.save_freq == 0:
                    model.load_state_dict(torch.load(fp)['model'])
                    optimizer.load_state_dict(torch.load(fp)['optimizer'])

        gc.collect()


if __name__ == '__main__':
  main()
