#!/usr/bin/env bash
set -e -o pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_PYTHON="${X2_DEPLOY_PYTHON:-/agibot/data/home/agi/x2_deploy_venv/bin/python}"

usage() {
    cat <<'EOF'
Usage: ./deploy.sh [sim2real options]

Starts preparation and RL control directly. Stop native MC on PC1 beforehand:
  aima em stop-app mc

Examples:
  ./deploy.sh
  ./deploy.sh --policy assets/policy.onnx

Defaults: assets/policy.onnx, 3-second transition, motion to its final
frame at 50 Hz, then exit automatically.
HAL IMU axes are used directly; no mode switching.
Normal completion / Ctrl+C retains measured joint positions with kp/kd;
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

echo "Live control: 3-second transition to frame 0, then 50 Hz policy."
exec "$DEPLOY_PYTHON" scripts/sim2real.py "$@"
