python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                 --exp_dir="jointnet_stereo_eigen" \
                 --exp_name="" \
                 --dataset_name="KITTI_EIGEN" \
                 --num_examples=-1 \
                 --model_name="scenenet_joint_stereo" \
                 --start_epoch=1 \
                 --epochs=10 \
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
                 --sf_pts_w=0.2 \
                 --sf_sm_w=200 \
                 --pose_pts_w=0.2 \
                 --pose_sm_w=200 \
                 --disp_lr_w=1.0 \
                 --disp_sm_w=0.1 \
                 --mask_reg_w=0.2 \
                 --mask_lr_w=1.0 \
                 --mask_cons_w=0.0 \
                 --static_cons_w=0.0 \
                 --flow_diff_thresh=1e-3 &&
python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                 --exp_dir="jointnet_stereo_eigen" \
                 --exp_name="" \
                 --dataset_name="KITTI_EIGEN" \
                 --num_examples=-1 \
                 --model_name="scenenet_joint_stereo" \
                 --start_epoch=11 \
                 --epochs=20 \
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
                 --sf_pts_w=0.2 \
                 --sf_sm_w=200 \
                 --pose_pts_w=0.2 \
                 --pose_sm_w=200 \
                 --disp_lr_w=1.0 \
                 --disp_sm_w=0.1 \
                 --mask_reg_w=0.01 \
                 --mask_lr_w=1.0 \
                 --mask_cons_w=0.3 \
                 --static_cons_w=0.05 \
                 --flow_diff_thresh=1e-3