python3 train.py --data_root=/ceph/kitti_jpg/ --exp_name="test_cons" --num_examples=500 --resize_only=True --model_name="scenenet" --epochs=50 --batch_size=4 --log_dir="/ceph/checkpoints/" --num_workers=8 --lr_sched_type='step' --lr=1e-4 --use_mask=True --lr_gamma=0.1