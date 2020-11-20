python3 train.py --data_root="/external/datasets/kitti_data_jpg/" \
                 --exp_dir="test_new_model" \
                 --exp_name="" \
                 --resize_only=True \
                 --dataset_name="KITTI_EIGEN" \
                 --num_examples=2 \
                 --shuffle=True \
                 --lr=1e-4 \
                 --lr_gamma=0.5 \
                 --flow_loss_mode='min' \
                 --epochs=10 \
                 --batch_size=1 \
                 --log_dir="/external/checkpoints/" \
                 --log_freq=1 \
                 --save_freq=0 \
                 --num_workers=8 \
                 --disp_sm_w=1e-3 \
                 --disp_lr_w=1.0 \
                 --no_logging=True
