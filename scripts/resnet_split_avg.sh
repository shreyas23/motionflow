python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                 --exp_dir="resnet_split_avg" \
                 --exp_name="" \
                 --dataset_name="KITTI_EIGEN" \
                 --num_examples=-1 \
                 --lr=4e-5 \
                 --lr_gamma=0.5 \
                 --lr_sched_type='none' \
                 --flow_reduce_mode='avg' \
                 --epochs=15 \
                 --batch_size=2 \
                 --log_dir="/ceph/checkpoints/" \
                 --log_freq=1 \
                 --save_freq=1 \
                 --num_workers=16 \
                 --disp_sm_w=1e-3 \
                 --disp_lr_w=0.0
