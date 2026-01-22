#!/usr/bin/env python
"""
실제 로봇 데이터를 Spline 형식으로 변환하는 스크립트

이 스크립트는 실제 로봇의 움직임을 기록한 JSON 파일을 읽어서
evaluate.py에서 사용할 수 있는 spline 형식으로 변환합니다.

JSON 파일 형식:
- type: discrete(false) 또는 continuous(true) 모드
- task: 작업 공간(cartesian space) 위치 및 자세
- joint: 관절 위치
- time: 나노초 단위 시간
"""

import argparse
import datetime
import glob
import json
import logging
import numpy as np
import os
import sys

# Add parent directory to path to import tracking modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tracking.utils.spline_utils import Spline


def extract_continuous_segments(data):
    """
    데이터에서 continuous 구간을 추출합니다.
    
    Returns:
        continuous_segments: [(start_idx, end_idx, time_offset), ...] 리스트
    """
    type_array = np.array(data['type'])
    time_array = np.array(data['time'], dtype=np.float64)
    
    continuous_segments = []
    in_continuous = False
    segment_start = None
    time_offset = None
    
    for i in range(len(type_array)):
        if type_array[i]:  # continuous
            if not in_continuous:
                # continuous 구간 시작
                in_continuous = True
                segment_start = i
                time_offset = time_array[i]  # 첫 continuous 시간을 offset으로 사용
        else:  # discrete
            if in_continuous:
                # continuous 구간 종료
                continuous_segments.append((segment_start, i, time_offset))
                in_continuous = False
                segment_start = None
                time_offset = None
    
    # 마지막이 continuous로 끝나는 경우
    if in_continuous:
        continuous_segments.append((segment_start, len(type_array), time_offset))
    
    return continuous_segments


def process_trajectory_data(data, mode='continuous_only', 
                           resampling_distance=0.1,
                           length_correction_step_size=0.00002,
                           use_normalized_length_correction_step_size=False,
                           discrete_indices=None):
    """
    실제 로봇 데이터를 spline으로 변환합니다.
    
    Args:
        data: JSON에서 로드한 딕셔너리
        mode: 'continuous_only' 또는 'all'
        resampling_distance: 스플라인 리샘플링 거리
        length_correction_step_size: 길이 보정 스텝 크기
        use_normalized_length_correction_step_size: 정규화된 길이 보정 스텝 크기 사용 여부
        discrete_indices: discrete pose 인덱스 리스트 (None이면 모든 discrete pose 선택)
    
    Returns:
        splines: [(spline, segment_info), ...] 리스트
    """
    type_array = np.array(data['type'])
    joint_data = np.array(data['joint'])
    time_array = np.array(data['time'], dtype=np.float64)
    
    num_points = len(type_array)
    num_joints = len(joint_data[0]) if num_points > 0 else 0
    
    logging.info(f"총 {num_points}개의 데이터 포인트, {num_joints}개 관절")
    
    splines = []
    
    if mode == 'continuous_only':
        # continuous 구간만 추출
        continuous_segments = extract_continuous_segments(data)
        logging.info(f"총 {len(continuous_segments)}개의 continuous 구간 발견")
        
        for seg_idx, (start_idx, end_idx, time_offset) in enumerate(continuous_segments):
            segment_joint_data = joint_data[start_idx:end_idx]
            segment_time_data = time_array[start_idx:end_idx]
            
            # 시간이 0이 아닌 경우 시간 기반 정규화 (첫 시간을 0으로)
            if len(segment_time_data) > 1:
                segment_time_normalized = (segment_time_data - segment_time_data[0]) * 1e-9  # ns -> s
                segment_duration = segment_time_normalized[-1]
            else:
                segment_duration = 0.0
            
            logging.info(f"구간 {seg_idx + 1}: 인덱스 {start_idx}~{end_idx-1}, "
                        f"{len(segment_joint_data)}개 포인트, "
                        f"지속 시간: {segment_duration:.6f}초")
            
            if len(segment_joint_data) < 2:
                logging.warning(f"구간 {seg_idx + 1}: 포인트가 2개 미만이어서 스킵합니다")
                continue
            
            # joint 데이터를 spline 형식으로 변환 (joints x waypoints)
            curve_data = segment_joint_data.T
            
            # Spline 생성
            try:
                spline = Spline(
                    curve_data=curve_data,
                    curve_data_slicing_step=1,
                    length_correction_step_size=length_correction_step_size,
                    use_normalized_length_correction_step_size=use_normalized_length_correction_step_size,
                    method="auto",
                    curvature_at_ends=0
                )
                
                # 리샘플링
                spline_resampled = spline.copy_with_resampling(
                    resampling_distance=resampling_distance,
                    use_normalized_resampling_distance=False,
                    use_curvature_for_resampling=False,
                    length_correction_step_size=length_correction_step_size,
                    use_normalized_length_correction_step_size=use_normalized_length_correction_step_size
                )
                
                # continuous 구간의 인덱스 리스트 생성
                selected_idx = list(range(start_idx, end_idx))
                
                splines.append((spline_resampled, {
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'segment_duration': segment_duration,
                    'num_points': len(segment_joint_data),
                    'selected_idx': selected_idx  # continuous 인덱스 자동 할당
                }))
                
                logging.info(f"구간 {seg_idx + 1}: Spline 생성 완료, "
                            f"길이: {spline_resampled.get_length():.4f}, "
                            f"최대 knot 간 거리: {spline_resampled.max_dis_between_knots:.6f}")
            except Exception as e:
                logging.error(f"구간 {seg_idx + 1}: Spline 생성 실패 - {e}")
                continue
    
    else:  # mode == 'all'
        # 전체 데이터를 하나의 spline으로 변환
        # discrete 구간은 다음/이전 포인트와 선형 보간으로 처리
        
        # discrete pose 인덱스 찾기
        all_discrete_indices = np.where(~type_array)[0]  # type이 False인 인덱스
        continuous_indices = np.where(type_array)[0]  # type이 True인 인덱스
        
        logging.info(f"Discrete pose: {len(all_discrete_indices)}개, Continuous pose: {len(continuous_indices)}개")
        
        # discrete pose 선택
        if discrete_indices is not None:
            # 사용자가 지정한 인덱스 사용
            discrete_indices_set = set(discrete_indices)
            # 유효한 인덱스만 필터링 (원본 데이터 범위 내)
            valid_discrete_indices = [idx for idx in discrete_indices if 0 <= idx < len(type_array)]
            invalid_indices = discrete_indices_set - set(valid_discrete_indices)
            if invalid_indices:
                logging.warning(f"유효하지 않은 discrete 인덱스 무시: {sorted(invalid_indices)}")
            # discrete인지 확인
            selected_discrete_indices = [idx for idx in valid_discrete_indices if idx in all_discrete_indices]
            non_discrete_indices = set(valid_discrete_indices) - set(selected_discrete_indices)
            if non_discrete_indices:
                logging.warning(f"지정한 인덱스 중 discrete가 아닌 것 무시: {sorted(non_discrete_indices)}")
            logging.info(f"사용자 지정 discrete 인덱스: {len(selected_discrete_indices)}개 선택됨")
        else:
            # 모든 discrete pose 포함
            selected_discrete_indices = all_discrete_indices.tolist()
            logging.info(f"모든 discrete pose 포함: {len(selected_discrete_indices)}개")
        
        # 선택된 인덱스: discrete + continuous 모두 포함 (continuous가 없어도 됨)
        selected_indices = sorted(set(selected_discrete_indices + continuous_indices.tolist()))
        selected_idx = selected_indices  # 원본 데이터에서 선택된 인덱스
        
        if len(continuous_indices) == 0:
            logging.info(f"Continuous pose가 없습니다. Discrete pose만 사용하여 spline 생성합니다.")
        
        logging.info(f"선택된 인덱스: {len(selected_idx)}개 (discrete: {len(selected_discrete_indices)}개, continuous: {len(continuous_indices)}개)")
        
        # 선택된 포인트가 2개 미만이면 에러
        if len(selected_indices) < 2:
            logging.error(f"선택된 포인트가 2개 미만입니다 (선택된 포인트: {len(selected_indices)}개)")
            return splines
        
        # 선택된 인덱스에 해당하는 joint 데이터만 사용
        selected_joint_data = joint_data[selected_indices]
        
        # joint 데이터를 spline 형식으로 변환
        curve_data = selected_joint_data.T
        
        try:
            spline = Spline(
                curve_data=curve_data,
                curve_data_slicing_step=1,
                length_correction_step_size=length_correction_step_size,
                use_normalized_length_correction_step_size=use_normalized_length_correction_step_size,
                method="auto",
                curvature_at_ends=0
            )
            
            # 리샘플링
            spline_resampled = spline.copy_with_resampling(
                resampling_distance=resampling_distance,
                use_normalized_resampling_distance=False,
                use_curvature_for_resampling=False,
                length_correction_step_size=length_correction_step_size,
                use_normalized_length_correction_step_size=use_normalized_length_correction_step_size
            )
            
            # 전체 시간 계산 (continuous가 있으면 continuous 시간 사용, 없으면 전체 시간 사용)
            total_duration = 0.0
            if len(time_array) > 1:
                if len(continuous_indices) > 1:
                    # continuous 시간 사용
                    continuous_times = time_array[continuous_indices]
                    total_duration = (continuous_times[-1] - continuous_times[0]) * 1e-9
                elif len(selected_indices) > 1:
                    # continuous가 없으면 선택된 인덱스의 시간 범위 사용
                    selected_times = time_array[selected_indices]
                    total_duration = (selected_times[-1] - selected_times[0]) * 1e-9
            
            splines.append((spline_resampled, {
                'start_idx': 0,
                'end_idx': len(joint_data),
                'segment_duration': total_duration,
                'num_points': len(selected_joint_data),
                'includes_discrete': True,
                'selected_idx': selected_idx  # 원본 데이터에서 선택된 인덱스
            }))
            
            logging.info(f"Spline 생성 완료, 길이: {spline_resampled.get_length():.4f}, "
                        f"최대 knot 간 거리: {spline_resampled.max_dis_between_knots:.6f}")
        except Exception as e:
            logging.error(f"Spline 생성 실패 - {e}")
    
    return splines


def save_splines(splines, output_dir, base_filename, segment_info_list, original_data=None):
    """
    생성된 spline들을 JSON 파일로 저장합니다.
    
    Args:
        splines: Spline 객체 리스트
        output_dir: 출력 디렉토리 (최상위 디렉토리)
        base_filename: 기본 파일명 (확장자 제외)
        segment_info_list: 각 segment의 정보 리스트
        original_data: 원본 데이터 딕셔너리 (task, time, joint, tool, type 포함)
    
    Returns:
        saved_files: 저장된 파일 경로 리스트
        spline_dir: spline 파일들이 저장된 디렉토리 경로
    """
    # {output_dir}/{base_filename}/spline/ 구조로 저장
    json_name_dir = os.path.join(output_dir, base_filename)
    spline_dir = os.path.join(json_name_dir, 'spline')
    os.makedirs(spline_dir, exist_ok=True)
    
    saved_files = []
    
    for idx, (spline, segment_info) in enumerate(zip(splines, segment_info_list)):
        # evaluate.py는 episode_*.json 패턴으로 파일을 찾습니다
        if len(splines) == 1:
            filename = f"episode_{base_filename}.json"
        else:
            filename = f"episode_{base_filename}_segment_{idx + 1:03d}.json"
        
        output_path = os.path.join(spline_dir, filename)
        
        # Spline 저장
        spline.save_to_json(output_path, make_dir=False)
        
        # segment 정보 추가 (execution_time 등)
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        
        saved_data['execution_time'] = segment_info.get('segment_duration', 0.0)
        saved_data['segment_info'] = segment_info
        
        # selected_idx 추가 (discrete pose 선택 정보)
        if 'selected_idx' in segment_info:
            saved_data['selected_idx'] = segment_info['selected_idx']
        
        # 원본 데이터 추가 (_origin 접미사)
        if original_data is not None:
            if 'task' in original_data:
                saved_data['task_origin'] = original_data['task']
            if 'time' in original_data:
                saved_data['time_origin'] = original_data['time']
            if 'joint' in original_data:
                saved_data['joint_origin'] = original_data['joint']
            if 'tool' in original_data:
                saved_data['tool_origin'] = original_data['tool']
            if 'type' in original_data:
                saved_data['type_origin'] = original_data['type']
        
        with open(output_path, 'w') as f:
            f.write(json.dumps(saved_data, sort_keys=True))
            f.flush()
        
        saved_files.append(output_path)
        logging.info(f"Spline 저장 완료: {output_path}")
    
    return saved_files, spline_dir


def convert_real_trajectory_to_spline(input_file, output_dir, 
                                     mode='continuous_only',
                                     resampling_distance=0.1,
                                     length_correction_step_size=0.00002,
                                     use_normalized_length_correction_step_size=False,
                                     discrete_indices=None):
    """
    실제 로봇 궤적 파일을 spline으로 변환합니다.
    
    Args:
        input_file: 입력 JSON 파일 경로
        output_dir: 출력 디렉토리
        mode: 'continuous_only' 또는 'all'
        resampling_distance: 스플라인 리샘플링 거리
        length_correction_step_size: 길이 보정 스텝 크기
        use_normalized_length_correction_step_size: 정규화된 길이 보정 스텝 크기 사용 여부
        discrete_indices: discrete pose 인덱스 리스트 (None이면 모든 discrete pose 선택)
    
    Returns:
        saved_files: 저장된 파일 경로 리스트
    """
    # 입력 파일 로드
    logging.info(f"입력 파일 로드: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 필수 필드 확인
    required_fields = ['type', 'joint', 'time']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"필수 필드 '{field}'가 없습니다")
    
    # 데이터 변환
    splines = process_trajectory_data(
        data=data,
        mode=mode,
        resampling_distance=resampling_distance,
        length_correction_step_size=length_correction_step_size,
        use_normalized_length_correction_step_size=use_normalized_length_correction_step_size,
        discrete_indices=discrete_indices
    )
    
    if not splines:
        raise ValueError("생성된 spline이 없습니다")
    
    # 파일 저장
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    segment_info_list = [info for _, info in splines]
    spline_objects = [spline for spline, _ in splines]
    
    saved_files, spline_dir = save_splines(
        splines=spline_objects,
        output_dir=output_dir,
        base_filename=base_filename,
        segment_info_list=segment_info_list,
        original_data=data
    )
    
    # {output_dir}/{base_filename}/spline_config.json 생성 (evaluate.py에서 사용)
    json_name_dir = os.path.join(output_dir, base_filename)
    os.makedirs(json_name_dir, exist_ok=True)
    
    max_distance_between_knots = max(s.max_dis_between_knots for s in spline_objects)
    
    # 실제 로봇 데이터의 첫 번째 joint position 가져오기
    initial_joint_position = None
    final_joint_position = None
    if len(data['joint']) > 0:
        initial_joint_position = list(data['joint'][0])
        logging.info(f"실제 로봇 초기 위치: {initial_joint_position}")
        final_joint_position = list(data['joint'][-1])
        logging.info(f"실제 로봇 최종 위치: {final_joint_position}")

    spline_config = {
        'max_dis_between_knots': float(max_distance_between_knots),
        'num_train': len(spline_objects),
        'num_test': 0,
        'resampling_distance': resampling_distance,
        'curvature_sampling_distance': None,
        'length_correction_step_size': float(length_correction_step_size * 10),
        'use_normalized_length_correction_step_size': use_normalized_length_correction_step_size
    }
    
    # 실제 로봇 데이터의 초기 위치가 있으면 저장
    if initial_joint_position is not None:
        spline_config['initial_position'] = initial_joint_position
    if final_joint_position is not None:
        spline_config['final_position'] = final_joint_position
    
    config_path = os.path.join(json_name_dir, 'spline_config.json')
    with open(config_path, 'w') as f:
        f.write(json.dumps(spline_config))
        f.flush()
    
    logging.info(f"spline_config.json 저장 완료: {config_path}")
    logging.info(f"총 {len(saved_files)}개의 spline 파일이 저장되었습니다")
    
    return saved_files


def get_input_files(input_path):
    """
    입력 경로가 파일인지 폴더인지 확인하고, 처리할 파일 목록을 반환합니다.
    
    Args:
        input_path: 입력 파일 또는 폴더 경로
    
    Returns:
        input_files: 처리할 JSON 파일 경로 리스트
    """
    if os.path.isfile(input_path):
        # 단일 파일인 경우
        return [input_path]
    elif os.path.isdir(input_path):
        # 폴더인 경우 모든 JSON 파일 찾기
        json_files = sorted(glob.glob(os.path.join(input_path, "*.json")))
        if not json_files:
            raise ValueError(f"폴더 '{input_path}'에 JSON 파일이 없습니다")
        return json_files
    else:
        raise ValueError(f"입력 경로 '{input_path}'가 파일도 폴더도 아닙니다")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='실제 로봇 데이터를 Spline 형식으로 변환',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 단일 파일 변환
  python scripts/convert_real_trajectory_to_spline.py \\
      --input trajectory_from_real/trajectory_20260116_070043.json \\
      --output_dir splines/real_robot \\
      --mode continuous_only
  
  # 폴더 내 모든 JSON 파일 변환
  python scripts/convert_real_trajectory_to_spline.py \\
      --input trajectory_from_real/ \\
      --output_dir splines/real_robot \\
      --mode continuous_only
  
  # discrete와 continuous 모두 변환
  python scripts/convert_real_trajectory_to_spline.py \\
      --input trajectory_from_real/trajectory_20260116_070043.json \\
      --output_dir splines/real_robot \\
      --mode all
  
  # discrete pose 인덱스 지정하여 변환 (--mode all 자동 적용)
  python scripts/convert_real_trajectory_to_spline.py \\
      --input trajectory_from_real/trajectory_20260116_070043.json \\
      --output_dir splines/real_robot \\
      --discrete_indices "[0, 5, 10, 15]"
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='입력 JSON 파일 경로 또는 폴더 경로')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='출력 디렉토리')
    parser.add_argument('--mode', type=str, default='continuous_only',
                        choices=['continuous_only', 'all'],
                        help='변환 모드: continuous_only (continuous만) 또는 all (전체)')
    parser.add_argument('--resampling_distance', type=float, default=0.1,
                        help='스플라인 리샘플링 거리 (기본값: 0.1)')
    parser.add_argument('--length_correction_step_size', type=float, default=0.00002,
                        help='길이 보정 스텝 크기 (기본값: 0.00002)')
    parser.add_argument('--use_normalized_length_correction_step_size', action='store_true',
                        help='정규화된 길이 보정 스텝 크기 사용')
    parser.add_argument('--discrete_indices', type=str, default=None,
                        help='discrete pose 인덱스 리스트 (JSON 형식, 예: "[0, 5, 10]"). 지정하지 않으면 모든 discrete pose 포함')
    parser.add_argument('--logging_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='로깅 레벨')
    
    args = parser.parse_args()
    
    # discrete_indices 파싱
    discrete_indices = None
    if args.discrete_indices is not None:
        try:
            discrete_indices = json.loads(args.discrete_indices)
            if not isinstance(discrete_indices, list):
                raise ValueError("discrete_indices는 리스트 형식이어야 합니다")
            # 정수로 변환
            discrete_indices = [int(idx) for idx in discrete_indices]
        except json.JSONDecodeError as e:
            raise ValueError(f"discrete_indices 파싱 실패: {e}")
        except (ValueError, TypeError) as e:
            raise ValueError(f"discrete_indices 형식 오류: {e}")
        
        # discrete_indices가 지정되면 mode를 'all'로 자동 설정
        if args.mode != 'all':
            logging.warning(f"--discrete_indices가 지정되어 --mode를 'all'로 자동 변경합니다 (원래 값: {args.mode})")
            args.mode = 'all'
    
    # 로깅 설정
    logging.basicConfig(
        level=getattr(logging, args.logging_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 입력 파일 목록 가져오기
    try:
        input_files = get_input_files(args.input)
        logging.info(f"처리할 파일 {len(input_files)}개 발견")
        for i, f in enumerate(input_files, 1):
            logging.info(f"  {i}. {f}")
    except Exception as e:
        logging.error(f"입력 파일 확인 실패: {e}")
        sys.exit(1)
    
    # 각 파일 변환 실행
    all_saved_files = []
    all_spline_dirs = []
    
    for file_idx, input_file in enumerate(input_files, 1):
        try:
            logging.info(f"\n{'='*60}")
            logging.info(f"파일 {file_idx}/{len(input_files)} 처리: {os.path.basename(input_file)}")
            logging.info(f"{'='*60}")
            
            saved_files = convert_real_trajectory_to_spline(
                input_file=input_file,
                output_dir=args.output_dir,
                mode=args.mode,
                resampling_distance=args.resampling_distance,
                length_correction_step_size=args.length_correction_step_size,
                use_normalized_length_correction_step_size=args.use_normalized_length_correction_step_size,
                discrete_indices=discrete_indices
            )
            
            all_saved_files.extend(saved_files)
            
            # spline 디렉토리 경로 저장
            json_name = os.path.splitext(os.path.basename(input_file))[0]
            spline_dir_path = os.path.join(args.output_dir, json_name, 'spline')
            all_spline_dirs.append(spline_dir_path)
            
        except Exception as e:
            logging.error(f"파일 '{input_file}' 변환 실패: {e}", exc_info=True)
            logging.warning(f"다음 파일 계속 처리...")
            continue
    
    # 결과 출력
    print("\n" + "="*60)
    print("변환 완료!")
    print("="*60)
    print(f"총 {len(all_saved_files)}개의 spline 파일이 저장되었습니다")
    
    if len(all_saved_files) > 0:
        print(f"\n저장된 spline 디렉토리:")
        for spline_dir in all_spline_dirs:
            print(f"  - {spline_dir}")
        
        print(f"\nevaluate.py에서 사용하려면:")
        if len(all_spline_dirs) == 1:
            print(f"  python tracking/evaluate.py --spline_dir {all_spline_dirs[0]} ...")
        else:
            print(f"  # 각 spline 디렉토리를 개별적으로 사용하거나,")
            print(f"  # 상위 디렉토리를 지정하여 모든 spline을 사용할 수 있습니다:")
            print(f"  python tracking/evaluate.py --spline_dir {args.output_dir} ...")
    
    if len(all_saved_files) == 0:
        logging.error("변환된 파일이 없습니다")
        sys.exit(1)
