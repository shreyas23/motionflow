import torch
import torch.nn as nn
import torch.nn.functional as tf

from utils.loss_utils import *
from utils.interpolation import interpolate2d_as


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.args = args

        self.ssim_w = args.ssim_w

        # dis weights
        self.disp_sm_w = args.disp_sm_w

        #sf wights
        self.sf_3d_pts = args.sf_pts_w
        self.sf_3d_sm = args.sf_sm_w

        # pos weights
        self.pose_pts_w = args.pose_pts_w
        self.pose_sm_w = args.pose_sm_w

        # mas weights
        self.mask_reg_w = args.mask_reg_w
        self.mask_sm_w = args.mask_sm_w
        self.mask_cons_w = args.mask_cons_w
        self.flow_diff_thresh = args.flow_diff_thresh

        # conistency weights 
        self.fb_w = args.fb_w
        self.mask_lr_w = args.mask_lr_w
        self.disp_lr_w = args.disp_lr_w
        self.static_cons_w = args.static_cons_w

        self.use_flow_mask = args.use_flow_mask



    def depth_loss(self, disp_l, disp_r, img_l, img_r):
        img_r_warp = _generate_image_left(img_r, disp_l) 
        img_diff = (_elementwise_l1(img_l, img_r_warp) * (1.0 - self._ssim_w) + _SSIM(img_l, img_r_warp) * self._ssim_w).mean(dim=1, keepdim=True)        

        # occlusion detection
        left_occ = _adaptive_disocc_detection_disp(disp_r).detach()
        right_occ = _adaptive_disocc_detection_disp(disp_l).detach()

        ## L-R Consistency loss
        proj_disp_r = _generate_image_left(disp_r, disp_l)
        proj_disp_l = _generate_image_right(disp_l, disp_r)
        lr_disp_diff_l = torch.abs(proj_disp_r - disp_l)
        lr_disp_diff_r = torch.abs(proj_disp_l - disp_r)

        loss_lr = lr_disp_diff_l[left_occ].mean() + lr_disp_diff_r[right_occ].mean()
        lr_disp_diff_l[~left_occ].detach_()
        lr_disp_diff_r[~right_occ].detach_()

        return img_diff, left_occ, loss_lr
    
    def motion_loss(self, ):
        return

    def forward(self, target, output):
        depth_loss_sum = 0
        sf_loss_sum = 0
        pose_loss_sum = 0
        mask_loss_sum = 0
        cons_loss_sum = 0

        img_l1 = target['input_l1_aug']
        img_l2 = target['input_l2_aug']
        img_r1 = target['input_r1_aug']
        img_r2 = target['input_r2_aug']
        K_l1 = target['input_k_l1_aug']
        K_l2 = target['input_k_l2_aug']

        disps_l1 = output['disps_l1']
        disps_l2 = output['disps_l2']
        disps_r1 = output['output_dict_r']['disps_l1']
        disps_r2 = output['output_dict_r']['disps_l2']
        flows_f = output['flows_f']
        flows_b = output['flows_b']
        pose_f = output['pose_f']
        pose_b = output['pose_b']

        if self.args.use_mask:
            masks_l1 = output['mask_l1']
            masks_l2 = output['mask_l2']

        for s in enumerate(len(disps_l1)):
            flow_f = interpolate2d_as(flows_f[s], img_l1)
            flow_b = interpolate2d_as(flows_b[s], img_l1)
            disp_l1 = interpolate2d_as(disps_l1[s], img_l1)
            disp_l2 = interpolate2d_as(disps_l2[s], img_l1)
            disp_r1 = interpolate2d_as(disps_r1[s], img_l1)
            disp_r2 = interpolate2d_as(disps_r2[s], img_l1)

            if self.args.use_mask:
                mask_l1 = interpolate2d_as(masks_l1[s], img_l1)
                mask_l2 = interpolate2d_as(masks_l2[s], img_l1)

            # depth loss
            depth_diff1, left_occ1, loss_lr1 = self.depth_loss(disp_l1, disp_r1, img_l1, img_r1)
            depth_diff2, left_occ2, loss_lr2 = self.depth_loss(disp_l2, disp_r2, img_l2, img_r2)
            loss_lr_sum = (loss_lr1 + loss_lr2) * self.disp_lr_w

            #pose diffs
            pose_diff1, pose_occ_b = self.pose_loss(pose_f, disp_l1, img_l1, img_l2, K_l1)
            pose_diff2, pose_occ_f = self.pose_loss(pose_b, disp_l2, img_l2, img_l1, K_l2)

            # sf diffs
            sf_diff1, sf_occ_b = self.sf_loss(flow_f, disp_l1, img_l1, img_l2, K_l1)
            sf_diff2, sf_occ_f = self.sf_loss(flow_b, disp_l2, img_l2, img_l1, K_l2)



            depth_loss1 = torch.min()
            depth_loss_sum = depth_loss_sum + depth_loss


        loss_dict = {}

        loss_dict["total_loss"] = depth_loss_sum + \
                                  sf_loss_sum + \
                                  pose_loss_sum + \
                                  mask_loss_sum + \
                                  cons_loss_sum
                                  

        return loss_dict