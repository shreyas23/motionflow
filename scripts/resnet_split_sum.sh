python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                 --log_dir="/ceph/checkpoints/" \
                 --exp_dir="resnet_split_sum" \
                 --exp_name="" \
                 --log_freq=1 \
                 --save_freq=1 \
                 --dataset_name="KITTI_EIGEN" \
                 --num_examples=-1 \
                 --start_epoch=1 \
                 --epochs=25 \
                 --batch_size=2 \
                 --validate=True \
                 --lr=1e-4 \
                 --lr_gamma=0.5 \
                 --lr_sched_type='step' \
                 --flow_reduce_mode='sum' \
                 --use_mask=True \
                 --use_disp_min=True \
                 --num_workers=16 \
                 --disp_sm_w=0.2 \
                 --flow_sm_w=200 \
                 --mask_sm_w=0.0 \
                 --mask_reg_w=0.2 \
                 --flow_pts_w=0.0 \
                 --mask_cons_w=0.0