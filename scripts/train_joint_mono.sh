python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                 --exp_dir="joint_mono" \
                 --exp_name="v1" \
                 --num_examples=-1 \
                 --model_name="scenenet_joint_mono" \
                 --epochs=25 \
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
                 --mask_reg_w=0.2 \
                 --mask_sm_w=0.1 \
                 --static_cons_w=0.0 \
                 --mask_cons_w=0.0 \
                 --flow_diff_thresh=1e-3
