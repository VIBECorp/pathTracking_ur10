#!/usr/bin/env bash
# 로봇 기록 경로 → 최적 경로 자동화 파이프라인 (쉘 래퍼)
#
# 사용법:
#   conda activate track   # Get Optimized Trajectory 단계에서 ray 사용
#   ./run_optimization_pipeline.sh <파일 또는 폴더>
#   ./run_optimization_pipeline.sh trajectory_from_real/trajectory_20260128_194007.json
#   ./run_optimization_pipeline.sh trajectory_from_real
#
# 추가 옵션은 run_optimization_pipeline.py 인자로 전달:
#   --output_base, --skip_existing
#   --convert_mode / --mode {continuous_only|all} (Recorded to Spline 모드, 기본 continuous_only)
#   --target_duration, --target_count (기본 2.0, 10)
#   --spline_termination_max_deviation, --vel_limit_factor, --acc_limit_factor
#   --online_trajectory_time_step
# 예:
#   ./run_optimization_pipeline.sh trajectory_from_real --output_base path_opt_260128 --skip_existing
#   ./run_optimization_pipeline.sh trajectory_from_real/traj.json --mode all --target_duration 1.0 --target_count 5

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ $# -lt 1 ]]; then
    echo "사용법: $0 <로봇 기록 경로 파일|폴더> [run_optimization_pipeline.py 옵션...]"
    echo ""
    python scripts/run_optimization_pipeline.py --help
    exit 1
fi

INPUT="$1"
shift
exec python scripts/run_optimization_pipeline.py "$INPUT" "$@"
