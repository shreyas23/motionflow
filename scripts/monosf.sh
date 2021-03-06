python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                      --test_data_root="/mnt/data/kitti_jpg/kitti2015/" \
                      --log_root="/ceph/checkpoints/" \
                      --exp_dir="monosf" \
                      --exp_name="v1" \
                      --log_freq=1 \
                      --save_freq=1 \
                      --dataset_name="KITTI_EIGEN" \
                      --model_name='monosf' \
                      --encoder_name='pwc' \
                      --num_examples=-1 \
                      --num_workers=16 \
                      --start_epoch=1 \
                      --epochs=62 \
                      --batch_size=4 \
                      --lr=2e-4 \
                      --lr_gamma=0.5 \
                      --lr_sched_type='step' \
                      --milestones 20 40 50 \
                      --disp_sm_w=0.1 \
                      --flow_sm_w=200 \
                      --flow_pts_w=0.0 \
                      --validate=True