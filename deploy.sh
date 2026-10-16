#!/usr/bin/env bash
set -e -o pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_PYTHON="${X2_DEPLOY_PYTHON:-/agibot/data/home/agi/x2_deploy_venv/bin/python}"

usage() {
    cat <<'EOF'
Usage: ./deploy.sh [sim2real options]

Waits for confirmation before preparation. Stop native MC on PC1 beforehand:
  aima em stop-app mc

Examples:
  ./deploy.sh
  ./deploy.sh --policy assets/policy.onnx

Defaults: assets/policy.onnx, confirm with y, 3-second transition to frame 0,
play motion at 50 Hz, then return to frame 0 over 3 seconds and hold until Ctrl+C.
HAL IMU axes are used directly; no mode switching.
Ctrl+C retains measured joint positions with kp/kd;
this is position hold, not balance control. Exceptions fall back to damping.
EOF
}

case "${1:-}" in
    -h|--help|help) usage; exit 0 ;;
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

echo "Live control: confirm prepare, play motion, return to frame 0 and hold until Ctrl+C."
exec "$DEPLOY_PYTHON" scripts/sim2real.py "$@"
