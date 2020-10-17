# adapted from https://github.com/anuragranj/cc/blob/master/test_pose.py
import torch
import argparse
import numpy as np
from path import Path
from tqdm import tqdm

from models.SceneNetStereoJoint import SceneNetStereoJoint

from utils.inverse_warp import pose_vec2mat

from kitti_eval.pose_evaluation_utils import test_framework_KITTI as test_framework

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='scenenet_joint')

parser.add_argument("--eval_sf", action='store_true')
parser.add_argument("--eval_flow", action='store_true')
parser.add_argument("--eval_pose", action='store_true')
parser.add_argument("--eval_mask", action='store_true')

parser.add_argument("--state_dict_path", default='pretrained/model.ckpt')

parser.add_argument("--dataset_dir", default="/ceph/datasets/")
parser.add_argument("--output_dir", default="./outputs")

args = parser.parse_args()

def compute_flow_error(gt, pred):
    return

def compute_mask_error(gt, pred):
    return

def compute_sf_error(gt, pred:
    return

def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length

def main():

    state_dict = torch.load(args.state_dict_path)

    if args.model_name == 'scenenet_joint':
        model = SceneNetStereoJoint(args)
    else:
        raise NotImplementedError

    model.load_state_dict(state_dict)

    eval_data = None

    for i, data in enumerate(eval_data):
        pose_metrics = compute_pose_error(data)
        flow_metrics = compute_flow_error(data)
        mask_metrics = compute_mask_error(data)
        sf_metrics = compute_sf_error(data)

    return

if __name__ == "__main__":
    main()
