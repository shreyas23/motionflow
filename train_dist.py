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
from datasets.kitti_2015_train import KITTI_2015_MonoSceneFlow
from models.JointModel import JointModel
from models.Model import Model
from models.ResModel import ResModel
from losses import Loss
from losses_eval import Eval_SceneFlow_KITTI_Train
from old_losses import Loss_SceneFlow_SelfSup_JointIter

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
        exp_name = datetime.datetime.now().strftime("%m-%d-%y-%I-%M-%S")
    else:
        exp_name = args.exp_name
    args.log_dir = os.path.join(log_dir, exp_name)
    print(f"All logs will be stored at: {args.log_dir}")

    mp.spawn(train, nprocs=args.num_gpus, args=(args,))

    return


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
    TEST_DATA_ROOT = args.test_data_root

    print(f"Loading model onto gpu: {gpu}")

    if args.model_name == 'joint':
        model = JointModel(args).cuda(device=gpu)
    elif args.model_name == 'split':
        model = Model(args).cuda(device=gpu)
    elif args.model_name == 'residual':
        model = ResModel(args).cuda(device=gpu)
    else:
        raise NotImplementedError

    loss = Loss(args).cuda(device=gpu)
    test_loss = Eval_SceneFlow_KITTI_Train(args).cuda(device=gpu)

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
            val_dataset = KITTI_Raw_KittiSplit_Valid(args, DATA_ROOT, num_examples=args.num_examples)
            val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            val_dataset = None
            val_dataloader = None

    elif DATASET_NAME == 'KITTI_EIGEN':
        train_dataset = KITTI_Raw_EigenSplit_Train(args, DATA_ROOT, num_examples=args.num_examples)
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
        train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
        if args.validate:
            val_dataset = KITTI_Raw_EigenSplit_Valid(args, DATA_ROOT, num_examples=args.num_examples)
            val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=rank)
            val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)
        else:
            val_dataset = None
            val_dataloader = None
    else:
        raise NotImplementedError

    test_dataset = KITTI_2015_MonoSceneFlow(args, data_root=TEST_DATA_ROOT)
    test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=rank)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=test_sampler)

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
        assert (args.milestones is not None), "Need to specify miletones in params"
        print("Using step lr schedule")
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.lr_gamma)
    elif args.lr_sched_type == 'none':
        lr_scheduler = None

    # set up logging
    if not args.no_logging:
        if gpu == 0:
            writer = SummaryWriter(args.log_dir)
            suffix = 'args.txt'
            args_fp = os.path.join(args.log_dir, suffix)
            if os.path.isfile(args_fp):
                suffix = 'args_census.txt' 
                args_fp = os.path.join(args.log_dir, suffix)
            with open(args_fp, 'w') as f:
                json.dump(args.__dict__, f, indent=2)
        else:
            writer = None

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

    # run training loop
    for epoch in range(curr_epoch, curr_epoch + args.epochs):

        # need to set epoch in order to shuffle indices
        train_sampler.set_epoch(epoch)

        if gpu == 0:
            print(f"Training epoch: {epoch}...\n")

        model.train()

        train_loss_avg_dict, output_dict, input_dict, n = train_one_epoch(
            args, model, loss, train_dataloader, optimizer, train_augmentations, lr_scheduler, gpu)

        with torch.no_grad():
            n = torch.tensor(n, requires_grad=False).cuda(device=gpu)
            dist.reduce(n, dst=0, op=dist.ReduceOp.SUM)

            train_loss_avg_dict['total_loss'].detach_()

            if gpu == 0:

                loss_names = []
                all_losses = []
                for k in sorted(train_loss_avg_dict.keys()):
                    loss_names.append(k)
                    all_losses.append(train_loss_avg_dict[k].cuda(device=gpu).float())

                all_losses = torch.stack(all_losses, dim=0)

                all_losses /= (n * args.batch_size)
                train_reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}

                print(f"\t Epoch {epoch} train loss avg:")
                pprint(train_reduced_losses)
                print("\n")

            if args.validate:
                val_loss_avg_dict, val_output_dict, val_input_dict, n = evaluate(args, model, loss, val_dataloader, val_augmentations, gpu)

                val_loss_avg_dict['total_loss'].detach_()

                with torch.no_grad():
                    n = torch.tensor(n, requires_grad=False).cuda(device=gpu)
                    dist.reduce(n, dst=0, op=dist.ReduceOp.SUM)

                    if gpu == 0:
                        loss_names = []
                        all_losses = []
                        for k in sorted(val_loss_avg_dict.keys()):
                            loss_names.append(k)
                            all_losses.append(val_loss_avg_dict[k].cuda(device=gpu).float())

                        all_losses = torch.stack(all_losses, dim=0)

                        all_losses /= n
                        val_reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}

                        print(f"Validation epoch: {epoch}...\n")
                        print(f"\t Epoch {epoch} val loss avg:")
                        pprint(val_reduced_losses)
                        print("\n")

                test_loss_avg_dict, test_output_dict, test_input_dict, n = evaluate(args, model, test_loss, test_dataloader, val_augmentations, gpu)

                test_loss_avg_dict['total_loss'].detach_()

                with torch.no_grad():
                    n = torch.tensor(n, requires_grad=False).cuda(device=gpu)
                    dist.reduce(n, dst=0, op=dist.ReduceOp.SUM)

                    if gpu == 0:
                        loss_names = []
                        all_losses = []
                        for k in sorted(test_loss_avg_dict.keys()):
                            loss_names.append(k)
                            all_losses.append(test_loss_avg_dict[k].cuda(device=gpu).float())

                        all_losses = torch.stack(all_losses, dim=0)

                        all_losses /= n
                        test_reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}

                        print(f"Test epoch: {epoch}...\n")
                        print(f"\t Epoch {epoch} test loss avg:")
                        pprint(test_reduced_losses)
                        print("\n")
            else:
                val_output_dict, val_input_dict = None, None
                test_output_dict, test_input_dict = None, None

        if args.lr_sched_type == 'plateau':
            lr_scheduler.step(train_loss_avg_dict['total_loss'])
        elif args.lr_sched_type == 'step':
            lr_scheduler.step(epoch)

        if not args.no_logging:
            fp = os.path.join(args.log_dir, f"{epoch}.ckpt")
            if gpu == 0:
                for k, v in train_reduced_losses.items():
                    writer.add_scalar(f'train/{k}', v.item(), epoch)
                if args.validate:
                    for k, v in val_reduced_losses.items():
                        writer.add_scalar(f'val/{k}', v.item(), epoch)

                if args.save_freq > 0:
                    if epoch % args.save_freq == 0 or epoch == args.epochs:
                        print(f"Saving model for epoch: {epoch} to: {fp}")
                        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, fp)

                if epoch % args.log_freq == 0:
                    visualize_output(args, input_dict, output_dict, epoch, writer, prefix='train')
                    del input_dict
                    del output_dict
                    del train_loss_avg_dict
                    del train_reduced_losses

                    if args.validate:
                        visualize_output(args, val_input_dict, val_output_dict, epoch, writer, prefix='val')
                        del val_input_dict
                        del val_output_dict
                        del val_loss_avg_dict
                        del val_reduced_losses

                        visualize_output(args, test_input_dict, test_output_dict, epoch, writer, prefix='test')

                        del test_input_dict
                        del test_output_dict
                        del test_loss_avg_dict
                        del test_reduced_losses

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

        gc.collect()

    cleanup_env()


if __name__ == '__main__':
  main()
