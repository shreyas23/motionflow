from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn

from .modules_monodepth import Resnet18_MonoDepth_Single, Resnet18_MonoDepth_Single_CamConv
from utils.monodepth_eval import disp_post_processing

from .decoders import PoseNet


class PoseDispNet(nn.Module):
    def __init__(self, args):
        super(PoseDispNet, self).__init__()

        self._args = args
        self._mono_resnet18 = Resnet18_MonoDepth_Single()
        self.pose_net = PoseNet()

    def forward(self, input_dict):

        output_dict = {}
        disp_l1_1, disp_l1_2, disp_l1_3, disp_l1_4 = self._mono_resnet18(input_dict['input_l2_aug'])
        disp_r1_1, disp_r1_2, disp_r1_3, disp_r1_4 = self._mono_resnet18(torch.flip(input_dict['input_r2_aug'], [3]))

        disp_r1_1 = torch.flip(disp_r1_1, [3])
        disp_r1_2 = torch.flip(disp_r1_2, [3])
        disp_r1_3 = torch.flip(disp_r1_3, [3])
        disp_r1_4 = torch.flip(disp_r1_4, [3])

        output_dict['disp_l2'] = [disp_l1_1, disp_l1_2, disp_l1_3, disp_l1_4]
        output_dict['disp_r2'] = [disp_r1_1, disp_r1_2, disp_r1_3, disp_r1_4]

        x = torch.cat([input_dict['input_l2_aug'], input_dict['input_l1_aug']], dim=1)
        output_dict['pose'] = self.pose_net(x)

        return output_dict


class MonoDepth_CamConv(nn.Module):
    def __init__(self, args):
        super(MonoDepth_CamConv, self).__init__()

        self._args = args
        self._mono_resnet18 = Resnet18_MonoDepth_Single_CamConv()

    def forward(self, input_dict):

        output_dict = {}

        if not self._args.evaluation:
            disp_l1_1, disp_l1_2, disp_l1_3, disp_l1_4 = self._mono_resnet18(input_dict['input_l1'], input_dict['input_k_l1'])
            disp_r1_1, disp_r1_2, disp_r1_3, disp_r1_4 = self._mono_resnet18(torch.flip(input_dict['input_r1'], [3]), input_dict['input_k_r1_flip'])

            disp_r1_1 = torch.flip(disp_r1_1, [3])
            disp_r1_2 = torch.flip(disp_r1_2, [3])
            disp_r1_3 = torch.flip(disp_r1_3, [3])
            disp_r1_4 = torch.flip(disp_r1_4, [3])

            output_dict['disp_l1'] = [disp_l1_1, disp_l1_2, disp_l1_3, disp_l1_4]
            output_dict['disp_r1'] = [disp_r1_1, disp_r1_2, disp_r1_3, disp_r1_4]

        else:
            input_img = torch.cat((input_dict['input_l1'], torch.flip(input_dict['input_l1'], [3])), dim=0)
            intrinsic = torch.cat((input_dict['input_k_l1'], input_dict['input_k_l1_flip']), dim=0)
            disp_l1_1, disp_l1_2, disp_l1_3, disp_l1_4 = self._mono_resnet18(input_img, intrinsic)
            out_disp_1_pp = disp_post_processing(disp_l1_1)
            output_dict['disp_l1_pp'] = [out_disp_1_pp]

        return output_dict
