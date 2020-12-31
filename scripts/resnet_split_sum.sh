python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                 --log_root="/ceph/checkpoints/" \
                 --exp_dir="resnet_split_sum" \
                 --exp_name="" \
                 --log_freq=1 \
                 --save_freq=1 \
                 --dataset_name="KITTI_EIGEN" \
                 --model_name="split" \
                 --num_examples=-1 \
                 --num_workers=16 \
                 --start_epoch=1 \
                 --epochs=30 \
                 --batch_size=2 \
                 --lr=1e-4 \
                 --lr_gamma=0.5 \
                 --lr_sched_type='step' \
                 --milestones 15 25 \
                 --flow_reduce_mode='sum' \
                 --disp_sm_w=0.1 \
                 --flow_sm_w=200 \
                 --mask_sm_w=0.0 \
                 --mask_reg_w=0.0 \
                 --flow_pts_w=0.0 \
                 --mask_cons_w=0.0
                 #--use_disp_min=True \