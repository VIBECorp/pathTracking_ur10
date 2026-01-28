#!/usr/bin/env python
"""
로봇 기록 경로 → 최적 경로 자동화 파이프라인

입력: 로봇 기록 경로 파일 또는 폴더
- 파일: 해당 파일 1개에 대한 최적 경로 1개 출력
- 폴더: 폴더 내 각 JSON 파일에 대해 최적 경로 1개씩 출력 (10개 파일 → 10개 최적 경로)

파이프라인 단계:
1. Recorded to Spline (convert_real_trajectory_to_spline.py)
2. Get Optimized Trajectory (get_optimized_trajectory.py, ray 사용)
3. Find Best Trajectory (find_profit.py)
4. Make Trajectory File (make_trajectory_file.py)

실행 시 tracking 용 conda 환경(예: track)에서 실행하세요. (Step 2에서 ray 필요)
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# launch.json 기본값
DEFAULT_CHECKPOINT = (
    "/home/gpu/ray_results/tracking_training_ur10/ur10_random_260113/"
    "PPO_TrackingEnvSpline_190a2_00000_0_2026-01-15_16-23-33/checkpoint_000380/checkpoint-380"
)
DEFAULT_SPLINES_DIR = "trajectory_from_real/splines"
DEFAULT_OUTPUT_BASE = "path_optimization"
DEFAULT_CONVERT_MODE = "continuous_only"
DEFAULT_TARGET_DURATION = 2.0
DEFAULT_TARGET_COUNT = 10
DEFAULT_OBSERVATION_RATIO = 0.1
DEFAULT_TIME_STEP = 0.001
DEFAULT_TRAJECTORY_TIME_STEP = 0.1
DEFAULT_CONTROL_FREQ = 125.0


def get_input_files(input_path: str) -> list[Path]:
    """입력이 파일이면 [파일], 폴더면 폴더 내 *.json 목록 반환."""
    p = Path(input_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"입력 경로가 없습니다: {input_path}")
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            raise ValueError(f"폴더에 JSON 파일이 없습니다: {input_path}")
        return files
    raise ValueError(f"파일/폴더가 아닙니다: {input_path}")


def run_convert_to_spline(
    repo_root: Path,
    input_file: Path,
    output_dir: Path,
    mode: str = DEFAULT_CONVERT_MODE,
    env: dict | None = None,
) -> Path:
    """Step 1: Recorded to Spline. spline 디렉토리 경로 반환."""
    base = input_file.stem
    out = Path(output_dir) / base / "spline"
    # 이미 존재하면 스킵 (재실행 시)
    if out.exists() and any(out.glob("episode_*.json")):
        logging.info("Step 1 [Recorded to Spline] 스킵 (이미 존재): %s", out)
        return out

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "convert_real_trajectory_to_spline.py"),
        "--input", str(input_file),
        "--output_dir", str(output_dir),
        "--mode", mode,
    ]
    env = env or os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    return out


def run_get_optimized_trajectory(
    repo_root: Path,
    checkpoint: str,
    spline_dir: Path,
    output_dir: Path,
    name: str,
    *,
    target_duration: float = DEFAULT_TARGET_DURATION,
    target_count: int = DEFAULT_TARGET_COUNT,
    spline_termination_max_deviation: float = 0.3,
    vel_limit_factor: float = 2.5,
    acc_limit_factor: float = 2.5,
    online_trajectory_time_step: float = 0.01,
    num_workers: int = 0,
    obstacle_scene: int = 0,
    env: dict | None = None,
) -> None:
    """Step 2: Get Optimized Trajectory."""
    cmd = [
        sys.executable,
        str(repo_root / "tracking" / "get_optimized_trajectory.py"),
        "--name", name,
        "--checkpoint", checkpoint,
        "--spline_dir", str(spline_dir),
        "--output_dir", str(output_dir),
        "--num_workers", str(num_workers),
        "--store_metrics",
        "--obstacle_scene", str(obstacle_scene),
        "--logging_level", "INFO",
        "--store_trajectory",
        "--store_actions",
        "--target_duration", str(target_duration),
        "--target_count", str(target_count),
        "--spline_termination_max_deviation", str(spline_termination_max_deviation),
        "--vel_limit_factor", str(vel_limit_factor),
        "--acc_limit_factor", str(acc_limit_factor),
        "--online_trajectory_time_step", str(online_trajectory_time_step),
    ]
    env = env or os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


def run_find_best_trajectory(
    real_trajectory: Path,
    optimized_dir: Path,
    observation_ratio: float = DEFAULT_OBSERVATION_RATIO,
    time_step: float = DEFAULT_TIME_STEP,
    trajectory_time_step: float = DEFAULT_TRAJECTORY_TIME_STEP,
) -> Path:
    """Step 3: Find Best Trajectory. 최적 episode 파일 경로 반환."""
    from find_profit import find_best_trajectory as _find

    best_file, _, _ = _find(
        str(real_trajectory),
        str(optimized_dir),
        observation_ratio,
        time_step,
        trajectory_time_step,
    )
    return Path(best_file)


def run_make_trajectory_file(
    repo_root: Path,
    optimized_file: Path,
    output_file: Path,
    control_freq: float = DEFAULT_CONTROL_FREQ,
) -> None:
    """Step 4: Make Trajectory File."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "make_trajectory_file.py"),
        "--optimized_file", str(optimized_file),
        "--output_file", str(output_file),
        "--control_freq", str(control_freq),
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def process_one(
    repo_root: Path,
    input_file: Path,
    splines_dir: Path,
    output_base: Path,
    checkpoint: str,
    *,
    convert_mode: str = DEFAULT_CONVERT_MODE,
    target_duration: float = DEFAULT_TARGET_DURATION,
    target_count: int = DEFAULT_TARGET_COUNT,
    observation_ratio: float = DEFAULT_OBSERVATION_RATIO,
    time_step: float = DEFAULT_TIME_STEP,
    trajectory_time_step: float = DEFAULT_TRAJECTORY_TIME_STEP,
    control_freq: float = DEFAULT_CONTROL_FREQ,
    spline_termination_max_deviation: float = 0.3,
    vel_limit_factor: float = 2.5,
    acc_limit_factor: float = 2.5,
    online_trajectory_time_step: float = 0.01,
) -> Path:
    """단일 입력 파일에 대해 1~4단계 실행. 최종 _opt.json 경로 반환."""
    base = input_file.stem
    spline_dir = run_convert_to_spline(repo_root, input_file, splines_dir, mode=convert_mode)
    optim_dir = output_base / base
    name = f"path_opt_{base}"

    run_get_optimized_trajectory(
        repo_root,
        checkpoint,
        spline_dir,
        optim_dir,
        name,
        target_duration=target_duration,
        target_count=target_count,
        spline_termination_max_deviation=spline_termination_max_deviation,
        vel_limit_factor=vel_limit_factor,
        acc_limit_factor=acc_limit_factor,
        online_trajectory_time_step=online_trajectory_time_step,
    )

    best_file = run_find_best_trajectory(
        input_file,
        optim_dir,
        observation_ratio=observation_ratio,
        time_step=time_step,
        trajectory_time_step=trajectory_time_step,
    )

    out_file = optim_dir / "trajectory_data" / f"{base}_opt.json"
    run_make_trajectory_file(repo_root, best_file, out_file, control_freq=control_freq)
    return out_file


def main() -> None:
    ap = argparse.ArgumentParser(
        description="로봇 기록 경로 → 최적 경로 파이프라인 (파일/폴더 입력)"
    )
    ap.add_argument(
        "input",
        type=str,
        help="로봇 기록 경로 JSON 파일 또는 JSON이 있는 폴더",
    )
    ap.add_argument(
        "--output_base",
        type=str,
        default=DEFAULT_OUTPUT_BASE,
        help=f"최적화 결과 기본 디렉터리 (기본: {DEFAULT_OUTPUT_BASE})",
    )
    ap.add_argument(
        "--splines_dir",
        type=str,
        default=DEFAULT_SPLINES_DIR,
        help=f"Spline 중간 결과 디렉터리 (기본: {DEFAULT_SPLINES_DIR})",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="RL 체크포인트 경로",
    )
    ap.add_argument(
        "--convert_mode",
        "--mode",
        dest="convert_mode",
        type=str,
        default=DEFAULT_CONVERT_MODE,
        choices=("continuous_only", "all"),
        help="Recorded to Spline 모드: continuous_only(기본, continuous 구간만), all(discrete+continuous) (기본: continuous_only)",
    )
    ap.add_argument(
        "--target_duration",
        type=float,
        default=DEFAULT_TARGET_DURATION,
        help="목표 trajectory duration (초). 이 값 미만일 때만 파일 유지 (기본: 2.0)",
    )
    ap.add_argument(
        "--target_count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="목표 저장 횟수. 이 횟수만큼 저장되면 Get Optimized 종료 (기본: 10)",
    )
    ap.add_argument(
        "--spline_termination_max_deviation",
        type=float,
        default=0.3,
        help="spline deviation 종료 임계값 (기본: 0.3)",
    )
    ap.add_argument(
        "--vel_limit_factor",
        type=float,
        default=2.5,
        help="속도 제한 계수 (기본: 2.5)",
    )
    ap.add_argument(
        "--acc_limit_factor",
        type=float,
        default=2.5,
        help="가속도 제한 계수 (기본: 2.5)",
    )
    ap.add_argument(
        "--online_trajectory_time_step",
        type=float,
        default=0.01,
        help="online trajectory 시간 간격 (초) (기본: 0.01)",
    )
    ap.add_argument("--observation_ratio", type=float, default=DEFAULT_OBSERVATION_RATIO)
    ap.add_argument("--time_step", type=float, default=DEFAULT_TIME_STEP)
    ap.add_argument("--trajectory_time_step", type=float, default=DEFAULT_TRAJECTORY_TIME_STEP)
    ap.add_argument("--control_freq", type=float, default=DEFAULT_CONTROL_FREQ)
    ap.add_argument("--skip_existing", action="store_true", help="이미 _opt.json 있으면 해당 파일 스킵")
    ap.add_argument("--logging_level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.logging_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    repo_root = REPO_ROOT
    splines_dir = Path(args.splines_dir)
    output_base = Path(args.output_base)

    try:
        files = get_input_files(args.input)
    except (FileNotFoundError, ValueError) as e:
        logging.error("%s", e)
        sys.exit(1)

    logging.info("입력 %s개: %s", len(files), args.input)
    results: list[Path] = []
    failed: list[tuple[Path, str]] = []  # (파일, 에러 메시지)

    for i, f in enumerate(files, 1):
        base = f.stem
        opt_path = output_base / base / "trajectory_data" / f"{base}_opt.json"
        if args.skip_existing and opt_path.exists():
            logging.info("[%d/%d] 스킵 (이미 존재): %s", i, len(files), opt_path)
            results.append(opt_path)
            continue

        logging.info("[%d/%d] 처리 중: %s", i, len(files), f.name)
        try:
            out = process_one(
                repo_root,
                f,
                splines_dir,
                output_base,
                args.checkpoint,
                convert_mode=args.convert_mode,
                target_duration=args.target_duration,
                target_count=args.target_count,
                observation_ratio=args.observation_ratio,
                time_step=args.time_step,
                trajectory_time_step=args.trajectory_time_step,
                control_freq=args.control_freq,
                spline_termination_max_deviation=args.spline_termination_max_deviation,
                vel_limit_factor=args.vel_limit_factor,
                acc_limit_factor=args.acc_limit_factor,
                online_trajectory_time_step=args.online_trajectory_time_step,
            )
            results.append(out)
            logging.info("  → %s", out)
        except subprocess.CalledProcessError as e:
            msg = f"subprocess 실패 (exit {e.returncode})"
            logging.warning("  [건너뜀] %s: %s", f.name, msg)
            failed.append((f, msg))
        except Exception as e:
            msg = str(e)
            logging.warning("  [건너뜀] %s: %s", f.name, msg, exc_info=False)
            failed.append((f, msg))

    print("\n" + "=" * 60)
    print("파이프라인 완료")
    print("=" * 60)
    print(f"성공 {len(results)}개:")
    for p in results:
        print(f"  - {p}")
    if failed:
        print(f"\n건너뜀 {len(failed)}개 (에러):")
        for fp, err in failed:
            print(f"  - {fp.name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
