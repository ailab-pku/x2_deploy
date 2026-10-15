#!/usr/bin/env bash
set -e -o pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_PYTHON="${X2_DEPLOY_PYTHON:-/agibot/data/home/agi/x2_deploy_venv/bin/python}"

usage() {
    cat <<'EOF'
Usage: ./deploy.sh [dry-run|check|live] [sim2real options]

  dry-run   Read robot feedback and infer without commands (default).
  check     Check feedback and absence of other command publishers; read-only.
  live      Send commands; stop native MC on PC1 beforehand:
              aima em stop-app mc

Examples:
  ./deploy.sh dry-run --max-steps 500
  ./deploy.sh check
  ./deploy.sh live --prepare-seconds 3
  ./deploy.sh live --policy assets/character.onnx

Live defaults: assets/prepare.onnx, 3-second transition, motion to its final
frame, then track that reference with the policy at 50 Hz until Ctrl+C.
Use --max-steps for a test limit. Identity IMU quaternion; no mode switching.
Live refuses to start while another HAL command publisher is present.
Normal completion / Ctrl+C retains measured joint positions with kp/kd;
this is position hold, not balance control. Faults fall back to damping
unless another controller has taken ownership.
EOF
}

case "${1:-dry-run}" in
    -h|--help|help) usage; exit 0 ;;
    dry-run|check|live) DEPLOY_MODE="${1:-dry-run}"; if (( $# )); then shift; fi ;;
    *) usage >&2; exit 2 ;;
esac

if [[ ! -x "$DEPLOY_PYTHON" ]]; then
    echo "Python interpreter not found: $DEPLOY_PYTHON" >&2
    exit 1
fi
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble setup not found" >&2
    exit 1
fi

# Source ROS before enabling nounset: its setup scripts use optional variables.
source /opt/ros/humble/setup.bash
set -u
export PYTHONPATH="$PROJECT_DIR:/agibot/software/common/local/lib/python3.10/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/agibot/software/common/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AMENT_PREFIX_PATH="/agibot/software/common${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-/agibot/software/entry/cfg/ros_dds_configuration.xml}"
export PYTHONUNBUFFERED=1
cd -- "$PROJECT_DIR"

if [[ "$DEPLOY_MODE" == dry-run ]]; then
    exec "$DEPLOY_PYTHON" scripts/sim2real.py --dry-run "$@"
fi
if [[ "$DEPLOY_MODE" == check ]]; then
    exec "$DEPLOY_PYTHON" scripts/sim2real.py --check-only "$@"
fi

echo "Live control: 3-second transition to frame 0, then 50 Hz policy (unless overridden)."
exec "$DEPLOY_PYTHON" scripts/sim2real.py "$@"
