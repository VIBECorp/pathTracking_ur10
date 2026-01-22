#!/usr/bin/env python
"""
최적 경로 파일을 실제 로봇 포맷으로 변환하는 스크립트

입력:
- 최적 경로 파일 (episode JSON 형식)
- 로봇 제어 주기 (Hz, 기본값: 125Hz)

출력:
- 실제 로봇 포맷 JSON 파일 (type, joint, time)
"""

import argparse
import json
import numpy as np
import os
from pathlib import Path


def convert_numpy_types(obj):
    """
    numpy 타입을 Python 기본 타입으로 변환하는 재귀 함수
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def convert_optimized_to_real_robot_format(optimized_file, output_file, control_freq=125.0,
                                          last_timestamp=None, use_final_position=False):
    """
    최적 경로 파일을 실제 로봇 포맷으로 변환
    
    Args:
        optimized_file: 최적 경로 파일 경로 (episode JSON 형식)
        output_file: 출력 파일 경로 (실제 로봇 포맷)
        control_freq: 로봇 제어 주기 (Hz, 기본값: 125Hz)
    
    Returns:
        output_file: 출력 파일 경로
    """
    # 최적 경로 파일 로드
    with open(optimized_file, 'r') as f:
        data = json.load(f)
    
    if 'trajectory_setpoints' not in data:
        raise ValueError("trajectory_setpoints not found in file")
    
    positions = data['trajectory_setpoints']['positions']
    optimized_joint = np.array(positions)
    
    # _origin 데이터와 selected_idx가 있으면 원본 데이터를 기본으로 사용
    has_origin = 'joint_origin' in data and 'time_origin' in data and 'type_origin' in data
    has_selected_idx = 'selected_idx' in data and len(data['selected_idx']) > 0
    
    if has_origin and has_selected_idx:
        # 원본 데이터 로드
        joint_origin = np.array(data['joint_origin'])
        time_origin = np.array(data['time_origin'])
        type_origin = np.array(data['type_origin'])
        tool_origin = np.array(data['tool_origin']) if 'tool_origin' in data else None
        
        selected_idx = sorted(data['selected_idx'])
        selected_idx_set = set(selected_idx)
        
        # selected_idx의 첫 번째 인덱스 찾기 (연속 그룹의 시작)
        first_selected_idx = selected_idx[0] if len(selected_idx) > 0 else None
        
        # 새로운 배열 생성: selected_idx 이전 원본 + 최적화된 연속 경로 + selected_idx 이후 원본
        joint_list = []
        time_list = []
        type_list = []
        tool_list = []
        
        # selected_idx의 최소값과 최대값
        min_selected = min(selected_idx) if selected_idx else -1
        max_selected = max(selected_idx) if selected_idx else -1
        
        # 1. selected_idx 이전의 원본 데이터 추가 (selected_idx에 포함되지 않은 것만)
        for orig_idx in range(min_selected):
            if orig_idx not in selected_idx_set:
                joint_list.append(joint_origin[orig_idx])
                time_list.append(time_origin[orig_idx])
                type_list.append(bool(type_origin[orig_idx]))  # numpy bool_를 Python bool로 변환
                if tool_origin is not None:
                    tool_list.append(tool_origin[orig_idx])
        
        # 2. 최적화된 연속 경로 전체 추가 (type: true)
        num_optimized = len(optimized_joint)
        if num_optimized > 0:
            # 최적화된 경로의 time 계산 (연속적으로)
            # 이전 원본 데이터의 마지막 time부터 시작
            if len(time_list) > 0:
                start_time = time_list[-1]
            else:
                start_time = 0
            
            # time step 계산 (초 단위)
            time_step = 1.0 / control_freq  # 예: 125Hz -> 0.008초
            
            # 최적화된 경로의 time 생성 (연속적으로)
            optimized_times = start_time + np.arange(1, num_optimized + 1) * time_step * 1e9
            
            # 최적화된 경로 추가
            for i in range(num_optimized):
                joint_list.append(optimized_joint[i])
                time_list.append(optimized_times[i])
                type_list.append(True)  # 연속 경로는 type: true
                if tool_origin is not None and first_selected_idx is not None:
                    # tool은 selected_idx 첫 번째의 값을 사용
                    tool_list.append(tool_origin[first_selected_idx])
        
        # 3. selected_idx 이후의 원본 데이터 추가 (selected_idx에 포함되지 않은 것만)
        # 최적화된 경로의 마지막 자세 저장
        optimized_final_position = optimized_joint[-1] if num_optimized > 0 else None
        
        first_after_selected = True  # selected_idx 이후의 첫 번째 원본 데이터인지 추적
        for orig_idx in range(max_selected + 1, len(joint_origin)):
            if orig_idx not in selected_idx_set:
                # 첫 번째 원본 데이터는 최적화된 경로의 마지막 자세로 업데이트
                if first_after_selected and optimized_final_position is not None:
                    joint_list.append(optimized_final_position)
                    first_after_selected = False
                else:
                    joint_list.append(joint_origin[orig_idx])
                
                time_list.append(time_origin[orig_idx])
                type_list.append(bool(type_origin[orig_idx]))  # numpy bool_를 Python bool로 변환
                if tool_origin is not None:
                    tool_list.append(tool_origin[orig_idx])
        
        joint = np.array(joint_list)
        time_ns = np.array(time_list)
        type_array = type_list
        num_points = len(joint)
        
        # tool 데이터 처리
        if tool_origin is not None:
            tool_array = tool_list
        else:
            tool_array = None
        
    else:
        # 기존 로직: 최적화된 trajectory만 사용
        joint = optimized_joint
        num_points = len(joint)
        tool_array = None
        
        if num_points == 0:
            raise ValueError("positions 배열이 비어있습니다.")
        
        # time step 계산 (초 단위)
        time_step = 1.0 / control_freq  # 예: 125Hz -> 0.008초
        
        # time 배열 생성 (나노초 단위)
        # 0부터 시작하여 time_step만큼 증가
        time_ns = np.arange(num_points) * time_step * 1e9  # 나노초로 변환
        
        # type 배열 생성 (모두 true)
        type_array = [True] * len(time_ns)
    
    # use_final_position 옵션 처리
    if use_final_position:
        if last_timestamp is None:
            raise ValueError("use_final_position 옵션을 사용하려면 last_timestamp(초 단위)를 지정해야 합니다.")
        if 'final_position' not in data:
            raise ValueError("optimized_file에 final_position이 없습니다.")
        final_pos = np.array(data['final_position'])
        last_ts_ns = int(last_timestamp * 1e9)
        # 기존 time보다 작거나 같으면 1ns 뒤로 보정
        if len(time_ns) > 0 and last_ts_ns <= time_ns[-1]:
            last_ts_ns = int(time_ns[-1]) + 1
        time_ns = np.append(time_ns, last_ts_ns)
        joint = np.vstack([joint, final_pos])
        type_array.append(True)
    
    time_ns_int = time_ns.astype(np.int64)  # int로 변환
    
    # type_array를 Python bool 타입으로 변환 (numpy bool_ 제거)
    type_array_python = [bool(t) for t in type_array]
    
    # 실제 로봇 포맷으로 변환
    real_robot_data = {
        'type': type_array_python,
        'joint': joint.tolist(),
        'time': time_ns_int.tolist()
    }
    
    # tool 데이터 추가 (있는 경우)
    if 'tool_origin' in data:
        if has_origin and has_selected_idx and tool_array is not None:
            real_robot_data['tool'] = tool_array
        elif not (has_origin and has_selected_idx):
            # _origin이 없거나 selected_idx가 없는 경우 tool_origin을 그대로 사용
            real_robot_data['tool'] = data['tool_origin']
    
    # _origin 필드 유지 (원본 데이터 보존)
    origin_fields = ['task_origin', 'time_origin', 'joint_origin', 'tool_origin', 'type_origin']
    for field in origin_fields:
        if field in data:
            real_robot_data[field] = data[field]
    
    # selected_idx 유지
    if 'selected_idx' in data:
        real_robot_data['selected_idx'] = data['selected_idx']
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # numpy 타입을 Python 기본 타입으로 변환
    real_robot_data = convert_numpy_types(real_robot_data)
    
    # 파일 저장
    with open(output_file, 'w') as f:
        json.dump(real_robot_data, f)
    
    print(f"변환 완료: {optimized_file} -> {output_file}")
    print(f"  - 포인트 수: {num_points}")
    print(f"  - 관절 수: {joint.shape[1] if len(joint.shape) > 1 else 1}")
    print(f"  - 제어 주기: {control_freq}Hz ({time_step*1000:.3f}ms)")
    print(f"  - Duration: {time_ns[-1] * 1e-9:.3f}s")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='최적 경로 파일을 실제 로봇 포맷으로 변환'
    )
    parser.add_argument('--optimized_file', type=str, required=True,
                        help='최적 경로 파일 경로 (episode JSON 형식)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='출력 파일 경로 (실제 로봇 포맷)')
    parser.add_argument('--control_freq', type=float, default=125.0,
                        help='로봇 제어 주기 (Hz, 기본값: 125Hz)')
    parser.add_argument('--last_timestamp', type=float, default=None,
                        help='마지막 timestamp (초 단위). use_final_position과 함께 사용')
    parser.add_argument('--use_final_position', action='store_true', default=False,
                        help='final_position을 joint 마지막에 추가하고 last_timestamp를 time 마지막에 추가')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not os.path.exists(args.optimized_file):
        raise FileNotFoundError(f"최적 경로 파일을 찾을 수 없습니다: {args.optimized_file}")
    
    # 변환 실행
    convert_optimized_to_real_robot_format(
        args.optimized_file,
        args.output_file,
        args.control_freq,
        args.last_timestamp,
        args.use_final_position
    )
    
    print(f"\n변환 완료: {args.output_file}")


if __name__ == '__main__':
    main()
