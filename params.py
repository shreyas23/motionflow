import argparse

class Params:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Self Supervised Joint Learning of Scene Flow, Disparity, Rigid Camera Motion, and Motion Segmentation",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # distributed params
        self.parser.add_argument("--local_rank", type=int, default=0)
        self.parser.add_argument("--num_nodes", type=int, default=1)
        self.parser.add_argument("--nr", type=int, default=0)

        # runtime params
        self.parser.add_argument('--data_root', help='path to dataset', default="/external/datasets/kitti_data_jpg")
        self.parser.add_argument('--test_data_root', help='path to 2015 dataset', default="/external/datasets/kitti2015")
        self.parser.add_argument('--epochs', type=int, default=20, help='number of epochs to run')
        self.parser.add_argument('--start_epoch', type=int, default=1,
                            help='resume from checkpoint (using experiment name)')
        self.parser.add_argument('--cuda', type=bool, default=True, help='use gpu?')
        self.parser.add_argument('--no_logging', type=bool, default=False,
                            help="are you logging this experiment?")
        self.parser.add_argument('--log_root', type=str, default="/external/cnet/checkpoints",
                            help="are you logging this experiment?")
        self.parser.add_argument('--log_freq', type=int, default=1, help='how often to log statistics')
        self.parser.add_argument('--save_freq', type=int, default=1, help='how often to save model state dict')
        self.parser.add_argument('--exp_dir', type=str, default='test',
                            help='name of experiment, chkpts stored in checkpoints/experiment')
        self.parser.add_argument('--exp_name', type=str, default='test',
                            help='name of experiment, chkpts stored in checkpoints/exp_dir/exp_name')
        self.parser.add_argument('--validate', type=bool, default=False,
                            help='set to true if validating model')
        self.parser.add_argument('--ckpt', type=str, default="",
                            help="path to model checkpoint if using one")

        # model params
        self.parser.add_argument('--model_name', type=str,
                            default="split", help="name of model")
        self.parser.add_argument('--encoder_name', type=str, default="pwc",
                            help="which encoder to use for Scene Net")
        self.parser.add_argument('--decoder_type', type=str, default="full",
                            help="which decoder to use for Scene Net")
        self.parser.add_argument('--pt_encoder', type=bool, default=False, help='only do resize augmentation on input data')
        self.parser.add_argument('--use_bn', type=bool, default=False, help="whether to use batch-norm in training procedure")
        self.parser.add_argument('--use_ppm', type=bool, default=False, help="whether to use consensus mask in training procedure")
        self.parser.add_argument('--num_scales', type=int, default=4, help="whether to use consensus mask in training procedure")
        self.parser.add_argument('--do_pose_c2f', type=bool, default=False, help="whether to use consensus mask in training procedure")
        self.parser.add_argument('--use_bottleneck', type=bool, default=False, help="whether to use consensus mask in training procedure")

        # dataset params
        self.parser.add_argument('--dataset_name', default='KITTI', help='KITTI or Eigen')
        self.parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        self.parser.add_argument('--num_views', type=int, default=2,
                            help="number of views present in training data")
        self.parser.add_argument('--num_examples', type=int, default=-1,
                            help="number of examples to train on per epoch")
        self.parser.add_argument('--num_workers', type=int, default=8,
                            help="number of workers for the dataloader")
        self.parser.add_argument('--shuffle', type=bool,
                            default=False, help='shuffle the dataset?')
        self.parser.add_argument('--resize_only', type=bool, default=False,
                            help='only do resize augmentation on input data')
        self.parser.add_argument('--no_flip_augs', type=bool, default=False,
                            help='only do resize augmentation on input data')

        # weight params
        self.parser.add_argument('--ssim_w', type=float, default=0.85, help='mask consensus weight')
        self.parser.add_argument('--flow_pts_w', type=float, default=0.0, help='mask consensus weight')
        self.parser.add_argument('--flow_cycle_w', type=float, default=0.0, help='mask consensus weight')
        self.parser.add_argument('--flow_sm_w', type=float, default=200, help='mask consensus weight')
        self.parser.add_argument('--disp_sm_w', type=float, default=0.2, help='mask consensus weight')
        self.parser.add_argument('--disp_lr_w', type=float, default=0.0, help='mask consensus weight')
        self.parser.add_argument('--mask_sm_w', type=float, default=0.1, help='mask consensus weight')
        self.parser.add_argument('--mask_reg_w', type=float, default=0.3, help='mask consensus weight')
        self.parser.add_argument('--static_cons_w', type=float, default=0.0, help='mask consensus weight')
        self.parser.add_argument('--mask_cons_w', type=float, default=0.2, help='mask consensus weight')
        self.parser.add_argument('--flow_diff_thresh', type=float, default=0.1, help='mask consensus weight')
        self.parser.add_argument('--mask_thresh', type=float, default=0.3, help='mask consensus weight')

        # learning params
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        self.parser.add_argument('--lr_sched_type', type=str, default="none", help="path to model checkpoint if using one")
        self.parser.add_argument('--milestones', nargs='+', type=int)
        self.parser.add_argument('--lr_gamma', type=float, default=0.1, help='initial learning rate')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd or alpha param for adam')
        self.parser.add_argument('--beta', type=float, default=0.999, help='beta param for adam')
        self.parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
        self.parser.add_argument('--grad_clip_norm', type=float, default=0, help='gradient clipping threshold')
        self.parser.add_argument('--grad_clip_value', type=float, default=0, help='gradient clipping threshold')

        # loss params
        self.parser.add_argument('--flow_reduce_mode', type=str, default="min", help='only do resize augmentation on input data')
        self.parser.add_argument('--use_disp_min', type=bool, default=False, help='only do resize augmentation on input data')
        self.parser.add_argument('--train_exp_mask', type=bool, default=False, help="whether to use consensus mask in training procedure")
        self.parser.add_argument('--train_census_mask', type=bool, default=False, help="whether to use consensus mask in training procedure")
        self.parser.add_argument('--apply_mask', type=bool, default=False, help="whether to use consensus mask in training procedure")
        self.parser.add_argument('--apply_flow_mask', type=bool, default=False, help="whether to use consensus mask in training procedure")
        self.parser.add_argument('--use_static_mask', type=bool, default=False, help="whether to use consensus mask in training procedure")

        # etc.
        self.parser.add_argument('--debugging', type=bool, default=False, help='are you debugging?')
        self.parser.add_argument('--finetuning', type=bool, default=False, help='finetuning on supervised data')
        self.parser.add_argument('--evaluation', type=bool, default=False, help='evaluating on data')
        self.parser.add_argument('--torch_seed', default=123768, help='random seed for reproducibility')
        self.parser.add_argument('--cuda_seed', default=543987, help='random seed for reproducibility')

        self.args = self.parser.parse_args()

        if self.args.train_exp_mask or self.args.train_census_mask:
            assert (self.args.train_exp_mask ^ self.args.train_census_mask), "Can only either train exp mask or census mask"
        
        if self.args.pt_encoder:
            assert (self.args.batch_size > 1), "Batch size must be greater than one if training using pre-trained resnet encoder"

        # if self.args.encoder_name == 'resnet':
        #     assert (self.args.use_bn and self.args.batch_size > 1), "If using resnet encoder, must use batch norm with batch size greater than one."