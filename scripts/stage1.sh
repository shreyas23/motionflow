python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
                      --test_data_root="/mnt/data/kitti_jpg/kitti2015/" \
                      --log_root="/ceph/checkpoints/" \
                      --exp_dir="finale" \
                      --exp_name="stage1_small" \
                      --log_freq=1 \
                      --save_freq=1 \
                      --dataset_name="KITTI_EIGEN" \
                      --model_name='joint' \
                      --encoder_name='pwc' \
                      --train_census_mask=True \
                      --apply_mask=True \
                      --validate=True \
                      --num_examples=-1 \
                      --num_workers=16 \
                      --start_epoch=31 \
                      --epochs=10 \
                      --batch_size=4 \
                      --lr=1e-4 \
                      --lr_gamma=0.5 \
                      --lr_sched_type='step' \
                      --milestones 30 \
                      --flow_reduce_mode='sum' \
                      --disp_sm_w=0.05 \
                      --flow_sm_w=10 \
                      --flow_pts_w=0.2 \
                      --flow_cycle_w=0.0 \
                      --flow_diff_thresh=0.05 \
                      --mask_sm_w=0.1 \
                      --mask_reg_w=0.3 \
                      --mask_cons_w=0.15 \
                      --mask_thresh=0.5 \
                      --static_cons_w=0.0

# python3 train_dist.py --data_root="/mnt/data/kitti_jpg/" \
#                       --test_data_root="/mnt/data/kitti_jpg/kitti2015/" \
#                       --log_root="/ceph/checkpoints/" \
#                       --exp_dir="pwc_joint_sum_lr" \
#                       --exp_name="pwc_2e4_pose_corr" \
#                       --log_freq=1 \
#                       --save_freq=1 \
#                       --dataset_name="KITTI_EIGEN" \
#                       --model_name='joint' \
#                       --encoder_name='pwc' \
#                       --use_pose_corr=True \
#                       --train_census_mask=True \
#                       --apply_mask=True \
#                       --apply_flow_mask=True \
#                       --validate=True \
#                       --num_examples=-1 \
#                       --num_workers=16 \
#                       --start_epoch=41 \
#                       --epochs=20 \
#                       --batch_size=4 \
#                       --lr=5e-5 \
#                       --lr_gamma=0.5 \
#                       --lr_sched_type='step' \
#                       --milestones 55 \
#                       --flow_reduce_mode='sum' \
#                       --disp_sm_w=0.1 \
#                       --disp_lr_w=0.0 \
#                       --flow_sm_w=200 \
#                       --flow_pts_w=0.2 \
#                       --flow_cycle_w=0.0 \
#                       --flow_diff_thresh=0.05 \
#                       --mask_sm_w=0.1 \
#                       --mask_reg_w=0.01 \
#                       --mask_cons_w=0.3 \
#                       --mask_cycle_w=0.0 \
#                       --mask_thresh=0.5 \
#                       --static_cons_w=0.0