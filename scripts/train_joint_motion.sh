python3 train_dist.py --data_root="/ceph/kitti_jpg/" \
                 --exp_dir="joint_motion" \
                 --exp_name="" \
                 --num_examples=100 \
                 --model_name="scenenet_joint" \
                 --epochs=50 \
                 --batch_size=4 \
                 --log_dir="/ceph/checkpoints/" \
                 --log_freq=1 \
                 --save_freq=1 \
                 --num_workers=16 \
                 --lr_sched_type='step' \
                 --lr=2e-4 \
                 --use_mask=True \
                 --lr_gamma=0.5 \
                 --use_bn=True \
                 --pose_sm_w=200 \
                 --mask_lr_w=1.0 \
                 --disp_lr_w=1.0 \
                 --mask_reg_w=0.2 \
                 --mask_sm_w=0.1 \
                 --static_cons_w=0.0 \
                 --mask_cons_w=0.3 \
                 --flow_diff_thresh=1e-3