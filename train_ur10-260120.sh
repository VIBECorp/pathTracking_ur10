#!/bin/bash

# Train UR10 스크립트
# launch.json의 "Train UR10" 설정을 기반으로 작성됨

python tracking/train.py \
    --logdir tracking_training_ur10 \
    --name ur10_random_260120 \
    --robot_scene 8 \
    --online_trajectory_time_step 0.1 \
    --hidden_layer_activation swish \
    --online_trajectory_duration 16.0 \
    --obstacle_scene 0 \
    --target_link_offset "[0, 0, 0]" \
    --last_layer_activation tanh \
    --no_log_std_activation \
    --use_controller_target_velocities \
    --spline_dir ur10/random_260113-2/train \
    --spline_u_arc_start_range "[0.0, 0.8]" \
    --spline_u_arc_diff_min 0.2 \
    --spline_normalize_duration \
    --spline_termination_max_deviation 0.1 \
    --obs_spline_n_next 7 \
    --obs_spline_add_length \
    --obs_spline_add_distance_per_knot \
    --spline_distance_max_reward 3.0 \
    --spline_deviation_max_threshold 0.1 \
    --punish_spline_max_deviation \
    --spline_max_deviation_max_punishment 2.0 \
    --punish_spline_mean_deviation \
    --spline_mean_deviation_max_punishment 2.0 \
    --spline_deviation_weighting_factors "[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]" \
    --batch_size_factor 12.0 \
    --spline_braking_extra_time_steps 0 \
    --terminate_on_robot_stop \
    --solver_iterations 50 \
    --iterations_per_checkpoint 20 \
    --time 500 \
    --num_workers 20 \
    --num_gpus 1 \
    # --checkpoint /home/gpu/ray_results/tracking_training_ur10/ur10_random_260113/PPO_TrackingEnvSpline_388aa_00000_0_2026-01-13_21-05-58/checkpoint_001280/checkpoint-1280
