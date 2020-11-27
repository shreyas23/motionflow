python3 train.py --data_root="/external/datasets/kitti_data_jpg/" \
                 --exp_dir="debug_unused" \
                 --exp_name="" \
                 --resize_only=True \
                 --dataset_name="KITTI_EIGEN" \
                 --num_examples=6 \
                 --shuffle=True \
                 --lr=1e-4 \
                 --lr_gamma=0.5 \
                 --flow_reduce_mode='avg' \
                 --epochs=20 \
                 --batch_size=2 \
                 --log_dir="/external/checkpoints" \
                 --log_freq=1 \
                 --save_freq=0 \
                 --num_workers=8 \
                 --disp_sm_w=1.0 \
                 --disp_lr_w=0.0 #\
                #  --no_logging=True