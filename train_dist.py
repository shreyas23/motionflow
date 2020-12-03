import gc
import os
import json
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from pprint import pprint
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
import torch.multiprocessing as mp
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

from params import Params
from utils.train_utils import step, evaluate, train_one_epoch, visualize_output

args = Params().args

def main():
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    print(f"Using {torch.cuda.device_count()} GPUs...")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8889'

    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.num_gpus * args.num_nodes

    # set up logging
    log_dir = os.path.join(args.log_root, args.exp_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.exp_name == "":
        exp_name = datetime.datetime.now().strftime("%I:%M:%S-%m-%d-%y")
    else:
        exp_name = args.exp_name
    args.log_dir = os.path.join(log_dir, exp_name)
    print(f"All logs will be stored at: {args.log_dir}")

    mp.spawn(train, nprocs=args.num_gpus, args=(args,))


def cleanup_env():
    dist.destroy_process_group()


def train(gpu, args):

    rank = args.nr * args.num_gpus + gpu
    dist.init_process_group(backend="nccl", world_size=args.world_size, rank=rank)

    # set device for process
    torch.cuda.set_device(gpu)

    if args.batch_size == 1 and args.use_bn is True:
        raise Exception

    # set some torch params
    torch.manual_seed(args.torch_seed)
    torch.cuda.manual_seed(args.cuda_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    DATASET_NAME = args.dataset_name
    DATA_ROOT = args.data_root

    print(f"Loading model onto gpu: {gpu}")

    if args.model_name == 'joint':
        model = JointModel(args).cuda(device=gpu)
    else:
        model = Model(args).cuda(device=gpu)

    loss = Loss(args).cuda(device=gpu)

    if args.use_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} learnable parameters")

    print(f"Loading dataset and dataloaders onto gpu: {gpu}...")
    if DATASET_NAME == 'KITTI':
        train_dataset = KITTI_Raw_KittiSplit_Train(args, DATA_ROOT, num_examples=args.num_examples)
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
        train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
        if args.validate and gpu ==0:
            val_dataset = KITTI_Raw_KittiSplit_Valid(args, DATA_ROOT)
            val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_dataset else None
        else:
            val_dataset = None
            val_dataloader = None

    elif DATASET_NAME == 'KITTI_EIGEN':
        train_dataset = KITTI_Raw_EigenSplit_Train(args, DATA_ROOT, num_examples=args.num_examples)
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
        train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
        if args.validate and gpu ==0:
            val_dataset = KITTI_Raw_EigenSplit_Valid(args, DATA_ROOT)
            val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_dataset else None
        else:
            val_dataset = None
            val_dataloader = None
    else:
        raise NotImplementedError

    # train_dataset, val_dataset = get_dataset(args, gpu)

    # if args.num_gpus > 1:
    #     train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
    # else:
    #     train_sampler = None
    # train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, shuffle=(train_sampler is None))
    # val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_dataset else None

    # define augmentations
    train_augmentations = Augmentation_SceneFlow(args)
    val_augmentations = Augmentation_Resize_Only(args)

    if args.cuda:
        train_augmentations = train_augmentations.cuda(device=gpu)
        val_augmentations = val_augmentations.cuda(device=gpu)

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

        writer = SummaryWriter(args.log_dir) if gpu == 0 else None
    
        if gpu == 0:
            with open(os.path.join(args.log_dir, 'args.txt'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)

    curr_epoch = args.start_epoch

    if args.ckpt != "":
        print(f"Loading model from {args.ckpt} onto gpu: {gpu}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(
            torch.load(args.ckpt, map_location=map_location)['model'])
        optimizer.load_state_dict(
            torch.load(args.ckpt, map_location=map_location)['optimizer'])
    elif args.start_epoch > 1:
        load_epoch = args.start_epoch - 1
        ckpt_fp = os.path.join(args.log_dir, f"{load_epoch}.ckpt")
        print(f"Loading model from {ckpt_fp} onto gpu: {gpu}")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        model.load_state_dict(
            torch.load(ckpt_fp, map_location=map_location)['model'])
        optimizer.load_state_dict(
            torch.load(ckpt_fp, map_location=map_location)['optimizer'])

    model.train()

    # run training loop
    for epoch in range(curr_epoch, curr_epoch + args.epochs):

        # need to set epoch in order to shuffle indices
        train_sampler.set_epoch(epoch)

        if gpu == 0:
            print(f"Training epoch: {epoch}...\n")

        train_loss_avg_dict, output_dict, input_dict = train_one_epoch(
            args, model, loss, train_dataloader, optimizer, train_augmentations, lr_scheduler, gpu)

        if gpu == 0:
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
            fp = os.path.join(args.log_dir, f"{epoch}.ckpt")
            if gpu == 0:
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
                    
                    dist.barrier()

                    # configure map_location properly
                    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
                    model.load_state_dict(
                        torch.load(fp, map_location=map_location)['model'])
                    optimizer.load_state_dict(
                        torch.load(fp, map_location=map_location)['optimizer'])

                    print(f"Loaded the saved model onto gpu: {gpu}")

        gc.collect()

    cleanup_env()


if __name__ == '__main__':
  main()
