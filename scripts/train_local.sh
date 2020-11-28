python3 train.py --data_root="/external/datasets/kitti_data_jpg/" \
                 --log_dir="/external/checkpoints" \
                 --exp_dir="local" \
                 --exp_name="" \
                 --resize_only=True \
                 --dataset_name="KITTI_EIGEN" \
                 --num_examples=6 \
                 --shuffle=True \
                 --lr=1e-4 \
                 --lr_gamma=0.5 \
                 --flow_reduce_mode='avg' \
                 --epochs=2 \
                 --batch_size=1 \
                 --log_freq=1 \
                 --save_freq=1 \
                 --num_workers=8 \
                 --disp_sm_w=1.0 \
                 --use_mask=True \
                 --use_disp_min=True \
                 --mask_reg_w=0.2