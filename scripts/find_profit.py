#!/usr/bin/env python
"""
실제 로봇 경로와 최적 경로들을 비교하여 최적의 경로를 선택하는 스크립트

입력:
- 실제 로봇 경로 파일 (trajectory_from_real 형식)
- 최적 경로 폴더 (episode_*.json 파일들)
- deviation 관측 초기 비율 (예: 0.1)

평가 조건:
1. trajectory duration이 짧을수록 높은 점수
2. 최종 위치 차이가 작을수록 높은 점수
3. 초기 구간의 deviation이 작을수록 높은 점수 (normalized time 기준)
"""

import argparse
import json
import numpy as np
import os
import glob
from pathlib import Path


def load_real_robot_trajectory(filepath):
    """
    실제 로봇 trajectory 파일 로드
    
    Args:
        filepath: JSON 파일 경로
    
    Returns:
        joint: 관절 위치 배열 (N x num_joints)
        time: 시간 배열 (초)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    joint = np.array(data['joint'])
    time_ns = np.array(data['time'], dtype=np.float64)
    
    # 나노초를 초로 변환
    time_s = time_ns * 1e-9
    
    # type 배열 저장 (원본 유지)
    type_array = np.array(data['type']) if 'type' in data else None
    
    # continuous 구간만 추출 (type이 True인 부분)
    if type_array is not None:
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
    else:
        joint_continuous = joint
        time_continuous = time_s - time_s[0] if len(time_s) > 0 else time_s
    
    return joint_continuous, time_continuous, type_array


def load_optimized_trajectory(filepath, trajectory_time_step=0.1):
    """
    최적 경로 파일 로드
    
    Args:
        filepath: JSON 파일 경로
        trajectory_time_step: 각 action 간 시간 간격 (초, 기본값 0.1)
    
    Returns:
        joint: 관절 위치 배열 (N x num_joints)
        time: 시간 배열 (초)
        duration: trajectory duration (초)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'trajectory_setpoints' not in data:
        raise ValueError("trajectory_setpoints not found in file")
    
    positions = data['trajectory_setpoints']['positions']
    joint = np.array(positions)
    
    # duration 계산: actions 개수 * trajectory_time_step
    num_actions = len(data.get('actions', []))
    duration = num_actions * trajectory_time_step
    
    # 시간 배열 생성 (control points에 대한 시간)
    # 각 action마다 여러 control points가 생성되므로
    # control_time_step을 사용하여 시간 배열 생성
    num_points = len(joint)
    
    # control_time_step 계산: duration을 num_points로 나눔
    if num_points > 0:
        control_time_step = duration / num_points
        time = np.arange(num_points) * control_time_step
    else:
        time = np.array([])
    
    return joint, time, duration


def shift_and_save_trajectory(traj_file, real_initial_position, trajectory_time_step=0.1):
    """
    최적 경로의 초기 위치를 실제 로봇 초기 위치에 맞춰 shift하고 파일에 저장
    
    Args:
        traj_file: 최적 경로 파일 경로
        real_initial_position: 실제 로봇 초기 위치 (num_joints,) - selected_idx가 없을 때 사용
        trajectory_time_step: 각 action 간 시간 간격 (초, 기본값 0.1)
    
    Returns:
        shifted: shift가 적용되었는지 여부
        offset: 적용된 offset
    """
    with open(traj_file, 'r') as f:
        data = json.load(f)
    
    if 'trajectory_setpoints' not in data:
        return False, None
    
    positions = np.array(data['trajectory_setpoints']['positions'])
    
    # 최적 경로의 초기 위치
    opt_initial_position = positions[0] if len(positions) > 0 else None
    
    if opt_initial_position is None:
        return False, None
    
    # 원본 초기 위치 결정: selected_idx와 joint_origin이 있으면 selected_idx의 첫 번째 인덱스 사용
    if 'selected_idx' in data and 'joint_origin' in data:
        selected_idx = sorted(data['selected_idx'])
        joint_origin = np.array(data['joint_origin'])
        if len(selected_idx) > 0 and len(joint_origin) > 0:
            first_selected_idx = selected_idx[0]
            if first_selected_idx < len(joint_origin):
                # selected_idx의 첫 번째에 해당하는 원본 위치 사용
                reference_initial_position = np.array(joint_origin[first_selected_idx])
                print(f"  {os.path.basename(traj_file)}: selected_idx[{first_selected_idx}]의 원본 위치 사용")
            else:
                reference_initial_position = real_initial_position
        else:
            reference_initial_position = real_initial_position
    else:
        # selected_idx가 없으면 기존처럼 real_initial_position 사용
        reference_initial_position = real_initial_position
    
    # 관절 수 확인 및 맞추기
    num_real_joints = len(reference_initial_position)
    num_opt_joints = len(opt_initial_position)
    num_joints = min(num_real_joints, num_opt_joints)
    
    # offset 계산
    position_offset = reference_initial_position[:num_joints] - opt_initial_position[:num_joints]
    
    # offset이 거의 없으면 (1e-6 이하) skip
    if np.linalg.norm(position_offset) < 1e-6:
        return False, position_offset
    
    # 전체 trajectory에 offset 적용 (원본 데이터는 보존)
    positions_shifted = positions.copy()
    positions_shifted[:, :num_joints] = positions[:, :num_joints] + position_offset
    
    # 최적 경로 데이터만 업데이트 (원본 데이터 _origin 필드는 보존)
    data['trajectory_setpoints']['positions'] = positions_shifted.tolist()
    
    # _origin 필드들은 절대 수정하지 않음 (원본 데이터 보존)
    # joint_origin, time_origin, type_origin, tool_origin, task_origin 등은 그대로 유지됨
    
    with open(traj_file, 'w') as f:
        json.dump(data, f)
    
    return True, position_offset


def calculate_final_position_deviation(real_joint, opt_joint):
    """
    최종 위치 차이 계산
    
    Args:
        real_joint: 실제 로봇 관절 위치 (N x num_joints)
        opt_joint: 최적 경로 관절 위치 (M x num_joints)
    
    Returns:
        deviation: 최종 위치 차이 (L2 norm)
    """
    real_final = real_joint[-1]
    opt_final = opt_joint[-1]
    
    # 관절 수가 다를 수 있으므로 최소값 사용
    num_joints = min(len(real_final), len(opt_final))
    deviation = np.linalg.norm(real_final[:num_joints] - opt_final[:num_joints])
    
    return deviation


def calculate_initial_deviation(real_joint, real_time, opt_joint, opt_time, observation_ratio, time_step=0.001, real_type_array=None):
    """
    초기 구간의 deviation 계산 (normalized time 기준)
    
    Args:
        real_joint: 실제 로봇 관절 위치 (N x num_joints)
        real_time: 실제 로봇 시간 (초)
        opt_joint: 최적 경로 관절 위치 (M x num_joints)
        opt_time: 최적 경로 시간 (초)
        observation_ratio: 관측 초기 비율 (예: 0.1)
        time_step: 시간 간격 (초, 기본값 0.001)
        real_type_array: 실제 로봇의 type 배열 (discrete/continuous 구분용, optional)
    
    Returns:
        sum_deviation: 초기 구간의 deviation 합
        is_all_discrete: 원본 데이터가 모두 discrete인지 여부
    """
    # 원본 데이터가 모두 discrete인 경우 확인
    is_all_discrete = False
    if real_type_array is not None:
        # type 배열을 numpy 배열로 변환하고 boolean으로 변환
        type_array = np.array(real_type_array, dtype=bool)
        # continuous 구간이 있는지 확인 (type이 True인 부분)
        continuous_mask = type_array == True
        continuous_count = np.sum(continuous_mask)
        if continuous_count == 0:
            # 모두 discrete인 경우 플래그 설정
            is_all_discrete = True
            print("  원본 데이터가 모두 discrete이므로 초기 deviation 점수를 0으로 설정")
    else:
        # type 배열이 없으면 모두 discrete로 간주하지 않음 (기존 동작 유지)
        is_all_discrete = False
    
    # normalized time으로 변환
    if len(real_time) > 1 and real_time[-1] > real_time[0]:
        real_time_norm = (real_time - real_time[0]) / (real_time[-1] - real_time[0])
    else:
        real_time_norm = np.linspace(0, 1, len(real_time))
    
    if len(opt_time) > 1 and opt_time[-1] > opt_time[0]:
        opt_time_norm = (opt_time - opt_time[0]) / (opt_time[-1] - opt_time[0])
    else:
        opt_time_norm = np.linspace(0, 1, len(opt_time))
    
    # 관측 구간: 0 ~ observation_ratio
    observation_end = observation_ratio
    
    # normalized time에서 샘플링할 시간 포인트 생성
    num_samples = int(observation_end / time_step)
    sample_times_norm = np.linspace(0, observation_end, num_samples)
    
    # 실제 로봇과 최적 경로에서 각 시간 포인트에 해당하는 관절 위치 보간
    num_joints = min(real_joint.shape[1], opt_joint.shape[1])
    deviations = []
    
    for t_norm in sample_times_norm:
        # 실제 로봇에서 보간
        real_idx = np.searchsorted(real_time_norm, t_norm)
        if real_idx == 0:
            real_pos = real_joint[0, :num_joints]
        elif real_idx >= len(real_joint):
            real_pos = real_joint[-1, :num_joints]
        else:
            # 선형 보간
            t1, t2 = real_time_norm[real_idx-1], real_time_norm[real_idx]
            if t2 > t1:
                alpha = (t_norm - t1) / (t2 - t1)
                real_pos = (1 - alpha) * real_joint[real_idx-1, :num_joints] + alpha * real_joint[real_idx, :num_joints]
            else:
                real_pos = real_joint[real_idx-1, :num_joints]
        
        # 최적 경로에서 보간
        opt_idx = np.searchsorted(opt_time_norm, t_norm)
        if opt_idx == 0:
            opt_pos = opt_joint[0, :num_joints]
        elif opt_idx >= len(opt_joint):
            opt_pos = opt_joint[-1, :num_joints]
        else:
            # 선형 보간
            t1, t2 = opt_time_norm[opt_idx-1], opt_time_norm[opt_idx]
            if t2 > t1:
                alpha = (t_norm - t1) / (t2 - t1)
                opt_pos = (1 - alpha) * opt_joint[opt_idx-1, :num_joints] + alpha * opt_joint[opt_idx, :num_joints]
            else:
                opt_pos = opt_joint[opt_idx-1, :num_joints]
        
        # deviation 계산 (L2 norm)
        deviation = np.linalg.norm(real_pos - opt_pos)
        deviations.append(deviation)
    
    sum_deviation = np.sum(deviations)
    
    return sum_deviation, is_all_discrete


def calculate_score(duration, final_deviation, initial_deviation_sum, 
                   duration_weight=1.0, final_weight=1.0, initial_weight=1.0, is_all_discrete=False):
    """
    종합 점수 계산
    
    Args:
        duration: trajectory duration (초)
        final_deviation: 최종 위치 차이
        initial_deviation_sum: 초기 구간 deviation 합
        duration_weight: duration 점수 가중치
        final_weight: 최종 위치 차이 점수 가중치
        initial_weight: 초기 deviation 점수 가중치
        is_all_discrete: 원본 데이터가 모두 discrete인지 여부
    
    Returns:
        score: 종합 점수 (높을수록 좋음)
    """
    # 각 점수를 정규화하기 위해 역수 사용 (작을수록 좋은 값이므로)
    # epsilon을 추가하여 0으로 나누는 것을 방지
    epsilon = 1e-6
    
    # duration 점수: 짧을수록 높은 점수
    duration_score = 1.0 / (duration + epsilon)
    
    # 최종 위치 차이 점수: 작을수록 높은 점수
    final_score = 5.0 / (final_deviation + epsilon)
    
    # 초기 deviation 점수: 작을수록 높은 점수
    # 원본 데이터가 모두 discrete인 경우 점수를 계산하지 않고 가중치도 0으로 설정
    if is_all_discrete:
        initial_score = 0.0
        # 모두 discrete인 경우 종합 점수 계산에서 제외
        effective_initial_weight = 0.0
    else:
        initial_score = 3.0 / (initial_deviation_sum + epsilon)
        effective_initial_weight = initial_weight
    
    # 가중 평균 (모두 discrete인 경우 initial_score는 제외)
    total_score = (duration_weight * duration_score + 
                  final_weight * final_score + 
                  effective_initial_weight * initial_score)
    
    return total_score, {
        'duration_score': duration_score,
        'final_score': final_score,
        'initial_score': initial_score
    }


def find_best_trajectory(real_trajectory_file, optimized_trajectories_dir, observation_ratio, time_step=0.001, trajectory_time_step=0.1):
    """
    최적의 trajectory 찾기
    
    Args:
        real_trajectory_file: 실제 로봇 trajectory 파일 경로
        optimized_trajectories_dir: 최적 경로 파일들이 있는 디렉토리
        observation_ratio: 관측 초기 비율 (예: 0.1)
        time_step: 시간 간격 (초, 기본값 0.001)
        trajectory_time_step: 각 action 간 시간 간격 (초, 기본값 0.1) - env_config.json에서 읽은 값이 우선
    
    Returns:
        best_file: 최적 경로 파일 경로
        best_score: 최고 점수
        best_info: 최적 경로의 상세 정보
    """
    # env_config.json에서 online_trajectory_time_step 읽기
    env_config_path = os.path.join(optimized_trajectories_dir, "env_config.json")
    if os.path.exists(env_config_path):
        try:
            with open(env_config_path, 'r') as f:
                env_config = json.load(f)
            if 'online_trajectory_time_step' in env_config:
                trajectory_time_step = env_config['online_trajectory_time_step']
                print(f"env_config.json에서 online_trajectory_time_step 읽음: {trajectory_time_step}")
        except Exception as e:
            print(f"경고: env_config.json 읽기 실패 ({e}), 기본값 사용: {trajectory_time_step}")
    else:
        print(f"env_config.json을 찾을 수 없음 ({env_config_path}), 기본값 사용: {trajectory_time_step}")
    
    # 실제 로봇 경로 로드
    print(f"실제 로봇 경로 로드: {real_trajectory_file}")
    real_joint, real_time, real_type_array = load_real_robot_trajectory(real_trajectory_file)
    print(f"  - 관절 수: {real_joint.shape[1]}, 포인트 수: {len(real_joint)}")
    print(f"  - Duration: {real_time[-1]:.3f}s")
    if real_type_array is not None:
        continuous_count = np.sum(np.array(real_type_array) == True)
        print(f"  - type 배열: {len(real_type_array)}개, continuous: {continuous_count}개")
        if continuous_count == 0:
            print(f"  - 모두 discrete입니다 (초기 deviation 점수는 0으로 설정됨)")
    else:
        print(f"  - type 배열 없음")
    
    # 실제 로봇 초기 위치
    real_initial_position = real_joint[0] if len(real_joint) > 0 else np.zeros(real_joint.shape[1])
    print(f"  - 초기 위치: {real_initial_position}")
    
    # 최적 경로 파일들 찾기
    trajectory_files = glob.glob(os.path.join(optimized_trajectories_dir, "trajectory_data/episode_*.json"))
    print(f"\n최적 경로 파일 {len(trajectory_files)}개 발견")
    
    if len(trajectory_files) == 0:
        raise ValueError(f"최적 경로 파일을 찾을 수 없습니다: {optimized_trajectories_dir}")
    
    # 각 최적 경로의 초기 위치를 실제 로봇 초기 위치에 맞춰 shift하고 저장
    print("\n최적 경로 초기 위치 조정 중...")
    shifted_count = 0
    for traj_file in trajectory_files:
        shifted, offset = shift_and_save_trajectory(traj_file, real_initial_position, trajectory_time_step)
        if shifted:
            shifted_count += 1
            print(f"  {os.path.basename(traj_file)}: offset 적용됨 (offset norm: {np.linalg.norm(offset):.6f})")
    print(f"총 {shifted_count}개 파일에 offset 적용 완료")
    
    # 각 경로에 대해 점수 계산
    scores = []
    file_infos = []
    
    for traj_file in trajectory_files:
        try:
            # 최적 경로 로드
            opt_joint, opt_time, duration = load_optimized_trajectory(traj_file, trajectory_time_step)
            
            # 최종 위치 차이 계산
            final_deviation = calculate_final_position_deviation(real_joint, opt_joint)
            
            # 초기 구간 deviation 계산
            initial_deviation_sum, is_all_discrete = calculate_initial_deviation(
                real_joint, real_time, opt_joint, opt_time, observation_ratio, time_step, real_type_array
            )
            
            # 점수 계산
            score, score_details = calculate_score(duration, final_deviation, initial_deviation_sum, 
                                                  is_all_discrete=is_all_discrete)
            
            scores.append(score)
            file_infos.append({
                'file': traj_file,
                'duration': duration,
                'final_deviation': final_deviation,
                'initial_deviation_sum': initial_deviation_sum,
                'score': score,
                'score_details': score_details,
                'is_all_discrete': is_all_discrete
            })
            
            print(f"\n{os.path.basename(traj_file)}:")
            print(f"  Duration: {duration:.3f}s")
            print(f"  최종 위치 차이: {final_deviation:.6f}")
            if is_all_discrete:
                print(f"  초기 deviation 합: {initial_deviation_sum:.6f} (모두 discrete이므로 점수 계산에서 제외)")
            else:
                print(f"  초기 deviation 합: {initial_deviation_sum:.6f}")
                print(f"  초기 deviation 점수: {score_details['initial_score']:.6f}")
            print(f"  종합 점수: {score:.6f}")
            
        except Exception as e:
            print(f"경고: {traj_file} 처리 중 오류 발생: {e}")
            continue
    
    if len(scores) == 0:
        raise ValueError("유효한 최적 경로를 찾을 수 없습니다.")
    
    # 최고 점수 경로 찾기
    best_idx = np.argmax(scores)
    best_info = file_infos[best_idx]
    
    return best_info['file'], best_info['score'], best_info


def main():
    parser = argparse.ArgumentParser(
        description='실제 로봇 경로와 최적 경로들을 비교하여 최적의 경로를 선택'
    )
    parser.add_argument('--real_trajectory', type=str, required=True,
                        help='실제 로봇 경로 파일 경로')
    parser.add_argument('--optimized_dir', type=str, required=True,
                        help='최적 경로 파일들이 있는 디렉토리')
    parser.add_argument('--observation_ratio', type=float, default=0.1,
                        help='deviation 관측 초기 비율 (기본값: 0.1)')
    parser.add_argument('--time_step', type=float, default=0.001,
                        help='시간 간격 (초, 기본값: 0.001)')
    parser.add_argument('--trajectory_time_step', type=float, default=0.1,
                        help='각 action 간 시간 간격 (초, 기본값: 0.1). env_config.json의 online_trajectory_time_step이 있으면 우선 사용')
    
    args = parser.parse_args()
    
    # 최적 경로 찾기
    best_file, best_score, best_info = find_best_trajectory(
        args.real_trajectory,
        args.optimized_dir,
        args.observation_ratio,
        args.time_step,
        args.trajectory_time_step
    )
    
    # 결과 출력
    print("\n" + "="*80)
    print("최적 경로 선택 결과")
    print("="*80)
    print(f"선택된 파일: {best_file}")
    print(f"종합 점수: {best_score:.6f}")
    print(f"\n상세 정보:")
    print(f"  Duration: {best_info['duration']:.3f}s")
    print(f"  최종 위치 차이: {best_info['final_deviation']:.6f}")
    print(f"  초기 deviation 합: {best_info['initial_deviation_sum']:.6f}")
    print(f"\n점수 세부사항:")
    print(f"  Duration 점수: {best_info['score_details']['duration_score']:.6f}")
    print(f"  최종 위치 차이 점수: {best_info['score_details']['final_score']:.6f}")
    if best_info.get('is_all_discrete', False):
        print(f"  초기 deviation 점수: 계산 제외 (모두 discrete)")
    else:
        print(f"  초기 deviation 점수: {best_info['score_details']['initial_score']:.6f}")
    print("="*80)


if __name__ == '__main__':
    main()
