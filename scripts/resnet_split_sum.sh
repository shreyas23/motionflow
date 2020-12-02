python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                 --log_dir="/ceph/checkpoints/" \
                 --exp_dir="resnet_split_sum" \
                 --exp_name="021220-121955" \
                 --dataset_name="KITTI_EIGEN" \
                 --num_examples=-1 \
                 --validate=True \
                 --lr=1e-4 \
                 --lr_gamma=0.5 \
                 --lr_sched_type='step' \
                 --flow_reduce_mode='sum' \
                 --use_mask=True \
                 --use_disp_min=True \
                 --epochs=24 \
                 --start_epoch=2 \
                 --batch_size=2 \
                 --log_freq=1 \
                 --save_freq=1 \
                 --num_workers=16 \
                 --disp_sm_w=0.2 \
                 --flow_sm_w=200 \
                 --flow_pts_w=0.0 \
                 --mask_sm_w=0.1 \
                 --mask_reg_w=0.2