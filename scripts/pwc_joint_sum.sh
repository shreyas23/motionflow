python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                      --test_data_root="/mnt/data/kitti_jpg/kitti2015/" \
                      --log_root="/ceph/checkpoints/" \
                      --exp_dir="pwc_joint_sum" \
                      --exp_name="v1_no_bw/" \
                      --log_freq=1 \
                      --save_freq=1 \
                      --dataset_name="KITTI_EIGEN" \
                      --model_name='joint' \
                      --encoder_name='pwc' \
                      --num_examples=-1 \
                      --num_workers=16 \
                      --start_epoch=16 \
                      --epochs=5 \
                      --batch_size=4 \
                      --lr=1.5e-4 \
                      --lr_gamma=0.5 \
                      --lr_sched_type='none' \
                      --flow_reduce_mode='sum' \
                      --disp_sm_w=0.1 \
                      --flow_sm_w=200 \
                      --mask_sm_w=0.1 \
                      --mask_reg_w=0.3 \
                      --flow_pts_w=0.0 \
                      --flow_cycle_w=0.0 \
                      --mask_cons_w=0.2 \
                      --static_cons_w=0.0 \
                      --flow_diff_thresh=0.15 \
                      --mask_thresh=0.5 \
                      --train_census_mask=True \
                      --validate=True
python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                      --test_data_root="/mnt/data/kitti_jpg/kitti2015/" \
                      --log_root="/ceph/checkpoints/" \
                      --exp_dir="pwc_joint_sum" \
                      --exp_name="v1_no_bw/" \
                      --log_freq=1 \
                      --save_freq=1 \
                      --dataset_name="KITTI_EIGEN" \
                      --model_name='joint' \
                      --encoder_name='pwc' \
                      --num_examples=-1 \
                      --num_workers=16 \
                      --start_epoch=21 \
                      --epochs=45 \
                      --batch_size=4 \
                      --lr=1e-4 \
                      --lr_gamma=0.5 \
                      --lr_sched_type='step' \
                      --milestones 39 47 54 \
                      --flow_reduce_mode='sum' \
                      --disp_sm_w=0.1 \
                      --disp_lr_w=0.0 \
                      --flow_sm_w=200 \
                      --mask_sm_w=0.1 \
                      --mask_reg_w=0.3 \
                      --flow_pts_w=0.0 \
                      --flow_cycle_w=0.0 \
                      --mask_cons_w=0.2 \
                      --static_cons_w=0.2 \
                      --flow_diff_thresh=0.15 \
                      --mask_thresh=0.5 \
                      --train_census_mask=True \
                      --apply_mask=True \
                      --apply_flow_mask=True \
                      --validate=True