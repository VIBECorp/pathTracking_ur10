#!/bin/bash

# Train KUKA script based on launch.json configuration
# This script runs the training with the same parameters as the VS Code launch configuration

cd "$(dirname "$0")"

python tracking/train.py \
    --logdir tracking_training_kuka \
    --name kuka_random \
    --robot_scene 0 \
    --online_trajectory_time_step 0.1 \
    --hidden_layer_activation swish \
    --online_trajectory_duration 16.0 \
    --obstacle_scene 0 \
    --target_link_offset "[0, 0, 0.126]" \
    --last_layer_activation tanh \
    --no_log_std_activation \
    --use_controller_target_velocities \
    --spline_dir industrial/random/train \
    --spline_u_arc_start_range "[0.0, 0.8]" \
    --spline_u_arc_diff_min 0.2 \
    --spline_normalize_duration \
    --spline_termination_max_deviation 0.25 \
    --obs_spline_n_next 7 \
    --obs_spline_add_length \
    --obs_spline_add_distance_per_knot \
    --spline_distance_max_reward 2.0 \
    --spline_deviation_max_threshold 0.25 \
    --punish_spline_max_deviation \
    --spline_max_deviation_max_punishment 0.9 \
    --punish_spline_mean_deviation \
    --spline_mean_deviation_max_punishment 0.9 \
    --spline_deviation_weighting_factors "[1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6]" \
    --batch_size_factor 1.0 \
    --spline_braking_extra_time_steps 0 \
    --terminate_on_robot_stop \
    --solver_iterations 50 \
    --iterations_per_checkpoint 20 \
    --time 500 \
    --num_workers 10 \
    --num_gpus 1
