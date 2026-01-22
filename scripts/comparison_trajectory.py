#!/usr/bin/env python
"""
실제 로봇 trajectory와 시뮬레이션 trajectory를 비교하는 스크립트

두 가지 형식의 파일을 비교:
1. trajectory_from_real 형식: type, joint, time
2. episode JSON 형식: trajectory_setpoints (positions가 joint position)
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def load_real_robot_trajectory(filepath):
    """
    실제 로봇 trajectory 파일 로드
    
    Args:
        filepath: JSON 파일 경로
    
    Returns:
        joint: 관절 위치 배열 (N x num_joints)
        time: 시간 배열 (나노초 단위)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    joint = np.array(data['joint'])
    time_ns = np.array(data['time'], dtype=np.float64)
    
    # 나노초를 초로 변환
    time_s = time_ns * 1e-9
    
    # continuous 구간만 추출 (type이 True인 부분)
    type_array = np.array(data['type'])
    continuous_mask = type_array == True
    
    if np.any(continuous_mask):
        joint_continuous = joint[continuous_mask]
        time_continuous = time_s[continuous_mask]
        
        # 첫 시간을 0으로 만들기
        if len(time_continuous) > 0:
            time_continuous = time_continuous - time_continuous[0]
    else:
        joint_continuous = joint
        time_continuous = time_s - time_s[0] if len(time_s) > 0 else time_s
    
    return joint_continuous, time_continuous


def load_simulation_trajectory(filepath, freq=240.0):
    """
    시뮬레이션 trajectory 파일 로드
    
    Args:
        filepath: JSON 파일 경로
        freq: 샘플링 주파수 (Hz)
    
    Returns:
        joint: 관절 위치 배열 (N x num_joints)
        time: 시간 배열 (초)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'trajectory_setpoints' not in data:
        raise ValueError("trajectory_setpoints not found in file")
    
    positions = data['trajectory_setpoints']['positions']
    joint = np.array(positions)
    
    # 시간 배열 생성 (freq 기반)
    num_points = len(joint)
    time_step = 1.0 / freq
    time = np.arange(num_points) * time_step
    
    return joint, time


def plot_comparison(real_joint, real_time, sim_joint, sim_time, output_dir=None):
    """
    두 trajectory를 비교하는 그래프 생성
    
    Args:
        real_joint: 실제 로봇 관절 위치 (N x num_joints)
        real_time: 실제 로봇 시간 (초)
        sim_joint: 시뮬레이션 관절 위치 (M x num_joints)
        sim_time: 시뮬레이션 시간 (초)
        output_dir: 출력 디렉토리 (None이면 화면에 표시)
    """
    num_joints = min(real_joint.shape[1], sim_joint.shape[1])
    
    # Figure 1: Normalized time (0~1)
    fig1, axes1 = plt.subplots(num_joints, 1, figsize=(12, 2.5 * num_joints))
    if num_joints == 1:
        axes1 = [axes1]
    
    fig1.suptitle('Trajectory Comparison (Normalized Time)', fontsize=16)
    
    for i in range(num_joints):
        ax = axes1[i]
        
        # Normalize time to 0~1
        real_time_norm = (real_time - real_time[0]) / (real_time[-1] - real_time[0]) if len(real_time) > 1 and real_time[-1] > real_time[0] else np.linspace(0, 1, len(real_time))
        sim_time_norm = (sim_time - sim_time[0]) / (sim_time[-1] - sim_time[0]) if len(sim_time) > 1 and sim_time[-1] > sim_time[0] else np.linspace(0, 1, len(sim_time))
        
        ax.plot(real_time_norm, real_joint[:, i], 'b-', label='Real Robot', linewidth=1.5, alpha=0.7)
        ax.plot(sim_time_norm, sim_joint[:, i], 'r--', label='Simulation', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Normalized Time (0~1)', fontsize=10)
        ax.set_ylabel(f'Joint {i} Position (rad)', fontsize=10)
        ax.set_title(f'Joint {i}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path1 = os.path.join(output_dir, 'trajectory_comparison_normalized.png')
        plt.savefig(output_path1, dpi=150, bbox_inches='tight')
        print(f"Normalized time graph saved to {output_path1}")
    else:
        plt.show()
    
    plt.close(fig1)
    
    # Figure 2: Actual time
    fig2, axes2 = plt.subplots(num_joints, 1, figsize=(12, 2.5 * num_joints))
    if num_joints == 1:
        axes2 = [axes2]
    
    fig2.suptitle('Trajectory Comparison (Actual Time)', fontsize=16)
    
    for i in range(num_joints):
        ax = axes2[i]
        
        ax.plot(real_time, real_joint[:, i], 'b-', label='Real Robot', linewidth=1.5, alpha=0.7)
        ax.plot(sim_time, sim_joint[:, i], 'r--', label='Simulation', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel(f'Joint {i} Position (rad)', fontsize=10)
        ax.set_title(f'Joint {i}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path2 = os.path.join(output_dir, 'trajectory_comparison_actual_time.png')
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"Actual time graph saved to {output_path2}")
    else:
        plt.show()
    
    plt.close(fig2)
    
    # Figure 3: Shifted simulation trajectory (normalized time)
    # 시뮬레이션 초기 위치와 실제 로봇 초기 위치의 차이 계산
    real_initial_position = real_joint[0] if len(real_joint) > 0 else np.zeros(num_joints)
    sim_initial_position = sim_joint[0] if len(sim_joint) > 0 else np.zeros(num_joints)
    position_offset = real_initial_position - sim_initial_position
    
    # 시뮬레이션 joint position에 offset 적용
    sim_joint_shifted = sim_joint + position_offset
    
    print(f"\nInitial Position Offset: {position_offset}")
    print(f"  Real Robot Initial: {real_initial_position}")
    print(f"  Simulation Initial: {sim_initial_position}")
    
    fig3, axes3 = plt.subplots(num_joints, 1, figsize=(12, 2.5 * num_joints))
    if num_joints == 1:
        axes3 = [axes3]
    
    fig3.suptitle('Trajectory Comparison with Shifted Simulation (Normalized Time)', fontsize=16)
    
    for i in range(num_joints):
        ax = axes3[i]
        
        # Normalize time to 0~1
        real_time_norm = (real_time - real_time[0]) / (real_time[-1] - real_time[0]) if len(real_time) > 1 and real_time[-1] > real_time[0] else np.linspace(0, 1, len(real_time))
        sim_time_norm = (sim_time - sim_time[0]) / (sim_time[-1] - sim_time[0]) if len(sim_time) > 1 and sim_time[-1] > sim_time[0] else np.linspace(0, 1, len(sim_time))
        
        ax.plot(real_time_norm, real_joint[:, i], 'b-', label='Real Robot', linewidth=1.5, alpha=0.7)
        ax.plot(sim_time_norm, sim_joint[:, i], 'r--', label='Simulation (Original)', linewidth=1.5, alpha=0.5)
        ax.plot(sim_time_norm, sim_joint_shifted[:, i], 'g--', label='Simulation (Shifted)', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Normalized Time (0~1)', fontsize=10)
        ax.set_ylabel(f'Joint {i} Position (rad)', fontsize=10)
        ax.set_title(f'Joint {i} (Offset: {position_offset[i]:.6f} rad)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path3 = os.path.join(output_dir, 'trajectory_comparison_shifted_normalized.png')
        plt.savefig(output_path3, dpi=150, bbox_inches='tight')
        print(f"Shifted simulation graph (normalized time) saved to {output_path3}")
    else:
        plt.show()
    
    plt.close(fig3)
    
    # 통계 정보 출력
    print("\n" + "="*60)
    print("Trajectory Comparison Statistics")
    print("="*60)
    print(f"Real Robot:")
    print(f"  - Number of points: {len(real_joint)}")
    print(f"  - Duration: {real_time[-1] - real_time[0]:.3f} seconds" if len(real_time) > 1 else f"  - Duration: 0 seconds")
    print(f"  - Number of joints: {real_joint.shape[1]}")
    print(f"\nSimulation:")
    print(f"  - Number of points: {len(sim_joint)}")
    print(f"  - Duration: {sim_time[-1] - sim_time[0]:.3f} seconds" if len(sim_time) > 1 else f"  - Duration: 0 seconds")
    print(f"  - Number of joints: {sim_joint.shape[1]}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='실제 로봇 trajectory와 시뮬레이션 trajectory 비교',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python scripts/comparison_trajectory.py \\
      --real trajectory_from_real/trajectory_20260116_070425.json \\
      --simulation trajectory_from_real/splines/trajectory_20260116_070425/episode_1_2904343.json \\
      --freq 240 \\
      --output_dir trajectory_from_real/comparison
        """
    )
    
    parser.add_argument('--real', type=str, required=True,
                        help='실제 로봇 trajectory 파일 경로 (trajectory_from_real 형식)')
    parser.add_argument('--simulation', type=str, required=True,
                        help='시뮬레이션 trajectory 파일 경로 (episode JSON 형식)')
    parser.add_argument('--freq', type=float, default=240.0,
                        help='시뮬레이션 샘플링 주파수 (Hz, 기본값: 240)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 디렉토리 (지정하지 않으면 화면에 표시)')
    
    args = parser.parse_args()
    
    # 파일 로드
    print(f"Loading real robot trajectory from {args.real}...")
    try:
        real_joint, real_time = load_real_robot_trajectory(args.real)
        print(f"  Loaded {len(real_joint)} points")
    except Exception as e:
        print(f"Error loading real robot trajectory: {e}")
        sys.exit(1)
    
    print(f"Loading simulation trajectory from {args.simulation}...")
    print(f"  Using sampling frequency: {args.freq} Hz (time step: {1.0/args.freq:.6f} seconds)")
    try:
        sim_joint, sim_time = load_simulation_trajectory(args.simulation, args.freq)
        print(f"  Loaded {len(sim_joint)} points")
    except Exception as e:
        print(f"Error loading simulation trajectory: {e}")
        sys.exit(1)
    
    # 그래프 생성
    print("\nGenerating comparison graphs...")
    plot_comparison(real_joint, real_time, sim_joint, sim_time, args.output_dir)
    
    print("\nComparison complete!")


if __name__ == '__main__':
    main()
