python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                 --exp_dir="resnet_test" \
                 --exp_name="test1" \
                 --dataset_name="KITTI_EIGEN" \
                 --num_examples=-1 \
                 --model_name="scenenet_joint_mono_iter" \
                 --encoder_name="resnet" \
                 --start_epoch=1 \
                 --epochs=25 \
                 --batch_size=4 \
                 --log_dir="/ceph/checkpoints/" \
                 --log_freq=1 \
                 --save_freq=1 \
                 --num_workers=20 \
                 --lr_sched_type='step' \
                 --lr=2e-4 \
                 --use_mask=True \
                 --lr_gamma=0.5 \
                 --use_bn=True \
                 --sf_pts_w=0.2 \
                 --sf_sm_w=200 \
                 --pose_pts_w=0.2 \
                 --pose_sm_w=200 \
                 --disp_lr_w=1.0 \
                 --disp_sm_w=0.1 \
                 --mask_reg_w=0.1 \
                 --mask_lr_w=0.0 \
                 --mask_cons_w=0.3 \
                 --static_cons_w=0.01 \
                 --flow_diff_thresh=1e-3
                #   &&
# python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
#                  --exp_dir="weights_test" \
#                  --exp_name="test1" \
#                  --dataset_name="KITTI_EIGEN" \
#                  --num_examples=-1 \
#                  --model_name="scenenet_joint_mono_iter" \
#                  --start_epoch=11 \
#                  --epochs=15 \
#                  --batch_size=4 \
#                  --log_dir="/ceph/checkpoints/" \
#                  --log_freq=1 \
#                  --save_freq=1 \
#                  --num_workers=20 \
#                  --lr_sched_type='step' \
#                  --lr=1e-4 \
#                  --use_mask=True \
#                  --lr_gamma=0.5 \
#                  --use_bn=True \
#                  --sf_pts_w=0.2 \
#                  --sf_sm_w=200 \
#                  --pose_pts_w=0.2 \
#                  --pose_sm_w=200 \
#                  --disp_lr_w=1.0 \
#                  --disp_sm_w=0.1 \
#                  --mask_reg_w=0.0 \
#                  --mask_lr_w=0.0 \
#                  --mask_cons_w=0.3 \
#                  --static_cons_w=0.05 \
#                  --flow_diff_thresh=1e-3