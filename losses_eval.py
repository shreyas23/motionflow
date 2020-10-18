import torch
import torch.nn as nn
from utils.interpolation import interpolate2d_as
from utils.sceneflow_util import projectSceneFlow2Flow
from utils.loss_utils import _depth2disp_kitti_K, _disp2depth_kitti_K, _elementwise_epe

from utils.monodepth_eval import compute_d1_all, compute_errors


def eval_module_disp_depth(gt_disp, gt_disp_mask, output_disp, gt_depth, output_depth):
    
    loss_dict = {}
    batch_size = gt_disp.size(0)
    gt_disp_mask_f = gt_disp_mask.float()

    ## KITTI disparity metric
    d_valid_epe = _elementwise_epe(output_disp, gt_disp) * gt_disp_mask_f
    d_outlier_epe = (d_valid_epe > 3).float() * ((d_valid_epe / gt_disp) > 0.05).float() * gt_disp_mask_f
    loss_dict["otl"] = (d_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
    loss_dict["otl_img"] = d_outlier_epe

    ## MonoDepth metric
    abs_rel, sq_rel, rms, log_rms, a1, a2, a3 = compute_errors(gt_depth[gt_disp_mask], output_depth[gt_disp_mask])        
    loss_dict["abs_rel"] = abs_rel
    loss_dict["sq_rel"] = sq_rel
    loss_dict["rms"] = rms
    loss_dict["log_rms"] = log_rms
    loss_dict["a1"] = a1
    loss_dict["a2"] = a2
    loss_dict["a3"] = a3

    return loss_dict


class Eval_SceneFlow_KITTI_Train(nn.Module):
    def __init__(self, args):
        super(Eval_SceneFlow_KITTI_Train, self).__init__()


    def forward(self, output_dict, target_dict):

        loss_dict = {}

        gt_flow = target_dict['target_flow']
        gt_flow_mask = (target_dict['target_flow_mask']==1).float()

        gt_disp = target_dict['target_disp']
        gt_disp_mask = (target_dict['target_disp_mask']==1).float()

        gt_disp2_occ = target_dict['target_disp2_occ']
        gt_disp2_mask = (target_dict['target_disp2_mask_occ']==1).float()

        gt_sf_mask = gt_flow_mask * gt_disp_mask * gt_disp2_mask

        intrinsics = target_dict['input_k_l1']                

        ##################################################
        ## Depth 1
        ##################################################

        batch_size, _, _, width = gt_disp.size()

        out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], gt_disp, mode="bilinear") * width
        out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
        gt_depth_l1 = _disp2depth_kitti_K(gt_disp, intrinsics[:, 0, 0])

        dict_disp0_occ = eval_module_disp_depth(gt_disp, gt_disp_mask.bool(), out_disp_l1, gt_depth_l1, out_depth_l1)
        
        output_dict["out_disp_l_pp"] = out_disp_l1
        output_dict["out_depth_l_pp"] = out_depth_l1

        d0_outlier_image = dict_disp0_occ['otl_img']
        loss_dict["d_abs"] = dict_disp0_occ['abs_rel']
        loss_dict["d_sq"] = dict_disp0_occ['sq_rel']
        loss_dict["d1"] = dict_disp0_occ['otl']

        ##################################################
        ## Optical Flow Eval
        ##################################################
        
        out_sceneflow = interpolate2d_as(output_dict['flow_f_pp'][0], gt_flow, mode="bilinear")
        out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])

        ## Flow Eval
        valid_epe = _elementwise_epe(out_flow, gt_flow) * gt_flow_mask
        loss_dict["f_epe"] = (valid_epe.view(batch_size, -1).sum(1)).mean() / 91875.68
        output_dict["out_flow_pp"] = out_flow

        flow_gt_mag = torch.norm(target_dict["target_flow"], p=2, dim=1, keepdim=True) + 1e-8
        flow_outlier_epe = (valid_epe > 3).float() * ((valid_epe / flow_gt_mag) > 0.05).float() * gt_flow_mask
        loss_dict["f1"] = (flow_outlier_epe.view(batch_size, -1).sum(1)).mean() / 91875.68


        ##################################################
        ## Depth 2
        ##################################################

        out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
        out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
        gt_depth_l1_next = _disp2depth_kitti_K(gt_disp2_occ, intrinsics[:, 0, 0])

        dict_disp1_occ = eval_module_disp_depth(gt_disp2_occ, gt_disp2_mask.bool(), out_disp_l1_next, gt_depth_l1_next, out_depth_l1_next)
        
        output_dict["out_disp_l_pp_next"] = out_disp_l1_next
        output_dict["out_depth_l_pp_next"] = out_depth_l1_next

        d1_outlier_image = dict_disp1_occ['otl_img']
        loss_dict["d2"] = dict_disp1_occ['otl']


        ##################################################
        ## Scene Flow Eval
        ##################################################

        outlier_sf = (flow_outlier_epe.bool() + d0_outlier_image.bool() + d1_outlier_image.bool()).float() * gt_sf_mask
        loss_dict["sf"] = (outlier_sf.view(batch_size, -1).sum(1)).mean() / 91873.4

        return loss_dict


class Eval_SceneFlow_KITTI_Test(nn.Module):
    def __init__(self):
        super(Eval_SceneFlow_KITTI_Test, self).__init__()

    def forward(self, output_dict, target_dict):

        loss_dict = {}

        ##################################################
        ## Depth 1
        ##################################################
        input_l1 = target_dict['input_l1']
        intrinsics = target_dict['input_k_l1']

        out_disp_l1 = interpolate2d_as(output_dict["disp_l1_pp"][0], input_l1, mode="bilinear") * input_l1.size(3)
        out_depth_l1 = _disp2depth_kitti_K(out_disp_l1, intrinsics[:, 0, 0])
        out_depth_l1 = torch.clamp(out_depth_l1, 1e-3, 80)
        output_dict["out_disp_l_pp"] = out_disp_l1

        ##################################################
        ## Optical Flow Eval
        ##################################################
        out_sceneflow = interpolate2d_as(output_dict['flow_f_pp'][0], input_l1, mode="bilinear")
        out_flow = projectSceneFlow2Flow(target_dict['input_k_l1'], out_sceneflow, output_dict["out_disp_l_pp"])        
        output_dict["out_flow_pp"] = out_flow

        ##################################################
        ## Depth 2
        ##################################################
        out_depth_l1_next = out_depth_l1 + out_sceneflow[:, 2:3, :, :]
        out_disp_l1_next = _depth2disp_kitti_K(out_depth_l1_next, intrinsics[:, 0, 0])
        output_dict["out_disp_l_pp_next"] = out_disp_l1_next        

        loss_dict['sf'] = (out_disp_l1_next * 0).sum()

        return loss_dict
