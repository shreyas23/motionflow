python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                 --exp_dir="jointnet_stereo" \
                 --exp_name="v1" \
                 --num_examples=-1 \
                 --model_name="scenenet_joint_stereo" \
                 --start_epoch=1 \
                 --epochs=5 \
                 --batch_size=4 \
                 --log_dir="/ceph/checkpoints/" \
                 --log_freq=1 \
                 --save_freq=1 \
                 --num_workers=32 \
                 --lr_sched_type='none' \
                 --lr=1e-4 \
                 --use_mask=True \
                 --lr_gamma=0.5 \
                 --use_bn=True \
                 --pose_sm_w=200 \
                 --pts_lr_w=1.0 \
                 --mask_lr_w=1.0 \
                 --disp_lr_w=1.0 \
                 --mask_reg_w=0.3 \
                 --mask_sm_w=0.1 \
                 --static_cons_w=0.0 \
                 --mask_cons_w=0.0 \
                 --flow_diff_thresh=1e-3 &&
python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                 --exp_dir="jointnet_stereo" \
                 --exp_name="v1" \
                 --num_examples=-1 \
                 --model_name="scenenet_joint_stereo" \
                 --start_epoch=6 \
                 --epochs=15 \
                 --batch_size=4 \
                 --log_dir="/ceph/checkpoints/" \
                 --log_freq=1 \
                 --save_freq=1 \
                 --num_workers=32 \
                 --lr_sched_type='step' \
                 --lr=1e-4 \
                 --use_mask=True \
                 --lr_gamma=0.5 \
                 --use_bn=True \
                 --pose_sm_w=200 \
                 --pts_lr_w=1.0 \
                 --mask_lr_w=1.0 \
                 --disp_lr_w=1.0 \
                 --mask_reg_w=0.1 \
                 --mask_sm_w=0.1 \
                 --static_cons_w=0.1 \
                 --mask_cons_w=0.3 \
                 --flow_diff_thresh=1e-3