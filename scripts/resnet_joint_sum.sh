python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                 --log_root="/ceph/checkpoints/" \
                 --exp_dir="resnet_joint_sum" \
                 --exp_name="" \
                 --log_freq=1 \
                 --save_freq=1 \
                 --dataset_name="KITTI_EIGEN" \
                 --validate=True \
                 --model_name='joint' \
                 --encoder_name='resnet' \
                 --num_examples=-1 \
                 --num_workers=16 \
                 --start_epoch=1 \
                 --epochs=50 \
                 --batch_size=2 \
                 --lr=2e-4 \
                 --lr_gamma=0.5 \
                 --lr_sched_type='step' \
                 --milestones 5 15 25 40 \
                 --flow_reduce_mode='sum' \
                 --disp_sm_w=0.1 \
                 --flow_sm_w=200 \
                 --mask_sm_w=0.0 \
                 --mask_reg_w=0.0 \
                 --flow_pts_w=0.2 \
                 --mask_cons_w=0.0 \
                 --flow_diff_thresh=1e-2