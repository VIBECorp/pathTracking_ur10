#!/bin/bash

# Run random_agent with UR10 (6 DOF) to generate random trajectories
# launch.json의 "Random Agent" 설정을 기반으로 작성됨
python tracking/random_agent.py \
    --robot_scene 8 \
    --obstacle_scene 0 \
    --online_trajectory_time_step 0.1 \
    --online_trajectory_duration 16.0 \
    --store_trajectory \
    --acc_limit_factor 0.4 \
    --jerk_limit_factor 1.0 \
    --torque_limit_factor 1.0 \
    --pos_limit_factor 1.0 \
    --filter_torque_violations \
    --logging_level INFO \
    --episodes 1000 \
    --no_log_torque_violations