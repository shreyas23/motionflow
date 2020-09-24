python3 train.py --data_root=/ceph/kitti_jpg/ --exp_name="overfit" --num_examples=1000 --resize_only=True --model_name="scenenet" --epochs=10 --batch_size=4 --log_dir="/ceph/checkpoints/" --num_workers=16 --lr_sched_type='step' --lr=3e-4 --use_mask=True
