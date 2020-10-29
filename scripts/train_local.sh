python3 train.py --data_root="/external/datasets/kitti_data_jpg/" \
                 --exp_dir="scenenet_joint_iter" \
                 --exp_name="" \
                 --num_examples=10 \
                 --resize_only=True \
                 --model_name="scenenet_joint_iter" \
                 --encoder_name="pwc" \
                 --epochs=100 \
                 --num_gpus=1 \
                 --batch_size=2 \
                 --log_dir="/external/checkpoints/" \
                 --log_freq=1 \
                 --save_freq=0 \
                 --num_workers=8 \
                 --lr_sched_type='step' \
                 --lr=1e-4 \
                 --use_mask=True \
                 --lr_gamma=0.5 \
                 --use_bn=True \
                 --shuffle=True \
                 --sf_pts_w=0.2 \
                 --sf_sm_w=200 \
                 --pose_sm_w=200 \
                 --pose_pts_w=0.2 \
                 --pose_lr_w=0.0 \
                 --disp_pts_w=0.0 \
                 --disp_sm_w=0.1 \
                 --disp_lr_w=1.0 \
                 --mask_lr_w=1.0 \
                 --mask_sm_w=0.0 \
                 --mask_reg_w=0.2 \
                 --static_cons_w=0.01 \
                 --mask_cons_w=0.3 \
                 --flow_diff_thresh=1e-3
