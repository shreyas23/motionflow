import torch
import torch.nn as nn
from collections import OrderedDict

from datasets.kitti_raw_monosf import KITTI_Raw_EigenSplit_Train, KITTI_Raw_EigenSplit_Valid
from datasets.kitti_2015_train import KITTI_2015_MonoSceneFlow_Full

from models.Model import Model
from models.JointModel import JointModel
from losses import Loss

from augmentations import Augmentation_Resize_Only
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from pprint import pprint

class Args:
    cuda = True
    use_bn = False
    momentum = 0.9
    beta = 0.999
    weight_decay=0.0
    use_mask = False
    use_flow_mask = False
    flow_min_w = 0.5
    flow_reduce_mode='sum'
    ssim_w = 0.85
    sf_lr_w = 0.0
    pose_lr_w = 0.0
    mask_lr_w = 1.0
    disp_lr_w = 1.0
    disp_pts_w = 0.0
    sf_pts_w = 0.2
    sf_sm_w = 200
    fb_w = 0.0
    pose_sm_w = 200
    pose_pts_w = 0.2
    disp_sm_w = 0.2
    disp_smooth_w = 0.1
    mask_reg_w = 0.2
    encoder_name="resnet"
    model_name='joint'
    static_cons_w = 1.0
    mask_cons_w = 0.2
    mask_sm_w = 0.1
    flow_diff_thresh=1e-3
    evaluation=True
    num_scales = 4
    pt_encoder=True
    do_pose_c2f=False
    use_disp_min=False
    flow_pts_w=0.2
    flow_sm_w=200
    use_static_mask=False
    batch_size=2

args = Args()

model = JointModel(args).cuda()

state_dict = torch.load('pretrained/49.ckpt')['model']
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model = model.eval()

del state_dict
del new_state_dict

augmentation = Augmentation_Resize_Only(args).cuda()
loss = Loss(args).cuda()

val_dataset = KITTI_2015_MonoSceneFlow_Full(args, root='/external/datasets/kitti2015/')
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=2, pin_memory=True)

num_examples = 1

outs = []

for i, data in enumerate(val_loader):
    with torch.no_grad():
        # Get input and target tensor keys
        input_keys = list(filter(lambda x: "input" in x, data.keys()))
        target_keys = list(filter(lambda x: "target" in x, data.keys()))
        tensor_keys = input_keys + target_keys

        # Possibly transfer to Cuda
        for k, v in data.items():
            if k in tensor_keys:
                data[k] = v.cuda(non_blocking=True)

        aug_data = augmentation(data)
        out = model(aug_data)

        ds = {
            'disps_l2_pp': out['disps_l2_pp'][0].cpu(),
            'pose_b': out['pose_b'][0].cpu(),
            'flows_b_pp': out['flows_b_pp'][0].cpu(),
            'K': aug_data['input_k_l1_aug'].cpu(),
            'img_l2': aug_data['input_l2'].cpu()
            }

        outs.append(ds)

        if i == num_examples:
            break

import numpy as np
import open3d as o3d
from open3d import JVisualizer
from utils.loss_utils import _disp2depth_kitti_K
from utils.helpers import BackprojectDepth

disp = outs[0]['disps_l2_pp']
img = outs[0]['img_l2']
K = outs[0]['K']
inv_K = torch.inverse(K)

b, _, h, w = disp.shape

bp = BackprojectDepth(b, h, w)

disp *= w
depth = _disp2depth_kitti_K(disp, K[:, 0, 0])

pts = bp(depth, inv_K)[0].squeeze()
pts = pts[:-1, :].T.cpu().numpy()

img = img[0].squeeze().permute(1, 2, 0)
img = img.reshape(-1, 3).cpu().numpy()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.colors = o3d.utility.Vector3dVector(img)

o3d.visualization.draw_geometries([pcd])
