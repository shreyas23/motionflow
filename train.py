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
from datasets.kitti_raw_monosf import KITTI_Odom_Test
from datasets.kitti_2015_train import KITTI_2015_MonoSceneFlow
from losses_eval import Eval_SceneFlow_KITTI_Train, Eval_Odom_KITTI_Raw
from models.JointModel import JointModel
from models.Model import Model
from models.ResModel import ResModel
from models.MonoDepthSFModel import MonoDepthSFModel
from models.MonoSF import MonoSceneFlow, MonoSFLoss
from losses import Loss
from monodepth_losses import MonoDepthSFLoss
from test_losses import Loss_SceneFlow_SelfSup
from models.SplitModel import SplitModel, SplitLoss

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
    TEST_DATA_ROOT = args.test_data_root

    loss = Loss(args).cuda()
    test_loss = Eval_SceneFlow_KITTI_Train(args)
    odom_loss = Eval_Odom_KITTI_Raw(args)

    if args.model_name == 'joint':
        print("Using joint scene flow model")
        model = JointModel(args).cuda()
        loss = Loss_SceneFlow_SelfSup(args).cuda()
    elif args.model_name == 'split':
        print("Using split scene flow model")
        model = SplitModel(args).cuda()
        loss = SplitLoss(args).cuda()
    elif args.model_name == 'monodepth':
        print("Using monodepth scene flow model")
        model = MonoDepthSFModel(args).cuda()
        loss = MonoDepthSFLoss(args).cuda()
    elif args.model_name == 'monosf':
        model = MonoSceneFlow(args).cuda()
        loss = MonoSFLoss(args).cuda()
    else:
        raise NotImplementedError

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {num_params} learnable parameters")

    print("Loading dataset and dataloaders")
    if DATASET_NAME == 'KITTI':
        train_dataset = KITTI_Raw_KittiSplit_Train(args, DATA_ROOT, num_examples=args.num_examples)
        train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True)
        if args.validate:
            val_dataset = KITTI_Raw_KittiSplit_Valid(args, DATA_ROOT, num_examples=args.num_examples)
            val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            val_dataset = None
            val_dataloader = None

    elif DATASET_NAME == 'KITTI_EIGEN':
        print(DATA_ROOT)
        train_dataset = KITTI_Raw_EigenSplit_Train(args, DATA_ROOT, num_examples=args.num_examples)
        train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, pin_memory=True)
        if args.validate:
            val_dataset = KITTI_Raw_EigenSplit_Valid(args, DATA_ROOT, num_examples=args.num_examples)
            val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            val_dataset = None
            val_dataloader = None
    else:
        raise NotImplementedError

    test_dataset = KITTI_2015_MonoSceneFlow(args, data_root=TEST_DATA_ROOT)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    odom_09_dataset = KITTI_Odom_Test(args, root=DATA_ROOT, seq="09")
    odom_09_dataloader = DataLoader(odom_09_dataset, shuffle=False, batch_size=1, pin_memory=True)
    odom_10_dataset = KITTI_Odom_Test(args, root=DATA_ROOT, seq="10")
    odom_10_dataloader = DataLoader(odom_10_dataset, shuffle=False, batch_size=1, pin_memory=True)

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
        milestones = []
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=args.lr_gamma)
    elif args.lr_sched_type == 'cyclic':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=5e-5, max_lr=3e-4,step_size_up=5,mode="triangular2", cycle_momentum=False)
    elif args.lr_sched_type == 'none':
        lr_scheduler = None

    # set up logging
    if args.exp_name == "":
        exp_name = datetime.datetime.now().strftime("%d%m%y-%I%M%S")
    else:
        exp_name = args.exp_name

    log_dir = os.path.join(args.log_root, exp_name)
    print(f"All logs will be stored at: {log_dir}")

    if not args.no_logging:
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

    model = model.train()

    # run training loop
    for epoch in range(curr_epoch, curr_epoch + args.epochs):

        print(f"Training epoch: {epoch}...\n")

        train_loss_avg_dict, output_dict, input_dict = train_one_epoch(
            args, model, loss, train_dataloader, optimizer, train_augmentations, lr_scheduler, 0)

        for k, v in train_loss_avg_dict.items():
            train_loss_avg_dict[k] = v / len(train_dataset)

        print(f"\t Epoch {epoch} train loss avg:")
        pprint(train_loss_avg_dict)
        print("\n")
        if epoch % args.log_freq == 0 and not args.no_logging:
            visualize_output(args, input_dict, output_dict, epoch, writer, prefix='train')
            writer.flush()

        del input_dict
        del output_dict
        del train_loss_avg_dict

        if args.validate:
            print(f"Validation epoch: {epoch}...\n")
            val_loss_avg_dict, val_output_dict, val_input_dict = evaluate(args, model, loss, val_dataloader, val_augmentations, gpu=0)

            for k, v in val_loss_avg_dict.items():
                val_loss_avg_dict[k] = v / len(val_dataset)

            print(f"\t Epoch {epoch} val loss avg:")
            pprint(val_loss_avg_dict)
            print("\n")
            if epoch % args.log_freq == 0 and not args.no_logging:
                visualize_output(args, val_input_dict, val_output_dict, epoch, writer, prefix='val')
                writer.flush()
            del val_input_dict
            del val_output_dict
            del val_loss_avg_dict
            
            odom_09_loss_avg_dict, odom_09_input_dict, odom_09_output_dict = evaluate(args, model, odom_loss, odom_09_dataloader, val_augmentations, gpu=0)
            odom_10_loss_avg_dict, odom_10_input_dict, odom_10_output_dict = evaluate(args, model, odom_loss, odom_10_dataloader, val_augmentations, gpu=0)
            del odom_09_input_dict
            del odom_09_output_dict
            del odom_10_input_dict
            del odom_10_output_dict

            for k, v in odom_09_loss_avg_dict.items():
                odom_09_loss_avg_dict[k] = v / len(odom_09_loss_avg_dict)

            for k, v in odom_10_loss_avg_dict.items():
                odom_10_loss_avg_dict[k] = v / len(odom_10_loss_avg_dict)

            test_loss_avg_dict, test_output_dict, test_input_dict = evaluate(args, model, test_loss, test_dataloader, val_augmentations, gpu=0)

            for k, v in test_loss_avg_dict.items():
                test_loss_avg_dict[k] = v / len(test_dataset)

            test_loss_avg_dict.update(odom_09_loss_avg_dict)
            test_loss_avg_dict.update(odom_10_loss_avg_dict)

            print(f"\t Epoch {epoch} test loss avg:")
            pprint(test_loss_avg_dict)
            print("\n")
            if epoch % args.log_freq == 0 and not args.no_logging:
                visualize_output(args, test_input_dict, test_output_dict, epoch, writer, prefix='test')
                writer.flush()
            del test_input_dict
            del test_output_dict
            del test_loss_avg_dict
        else:
            val_output_dict, val_input_dict = None, None
            test_output_dict, test_input_dict = None, None

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
                for k, v in test_loss_avg_dict.items():
                    writer.add_scalar(f'test/{k}', v.item(), epoch)

            if args.save_freq > 0:
                if epoch % args.save_freq == 0 or epoch == args.epochs:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, fp)

            if epoch % args.log_freq == 0:
                visualize_output(args, input_dict, output_dict, epoch, writer, prefix='train')
                writer.flush()
                del input_dict
                del output_dict
                del train_loss_avg_dict

                if args.validate:
                    visualize_output(args, val_input_dict, val_output_dict, epoch, writer, prefix='val')
                    writer.flush()
                    del val_input_dict
                    del val_output_dict
                    del val_loss_avg_dict

                    visualize_output(args, test_input_dict, test_output_dict, epoch, writer, prefix='test')
                    writer.flush()
                    del test_input_dict
                    del test_output_dict
                    del test_loss_avg_dict


            if args.save_freq > 0:
                if epoch % args.save_freq == 0:
                    model.load_state_dict(torch.load(fp)['model'])
                    optimizer.load_state_dict(torch.load(fp)['optimizer'])

        gc.collect()


if __name__ == '__main__':
  main()
