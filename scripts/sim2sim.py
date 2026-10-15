from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from x2_deploy.task import make_x2_beyond_mimic_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="X2 Ultra BeyondMimic sim-to-sim runner")
    parser.add_argument("--policy", help="Motion-export ONNX policy (motion is embedded)")
    parser.add_argument("--scene", help="Override the project-local MuJoCo scene")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    parser.add_argument("--max-steps", type=int, help="Stop after this many 50 Hz policy steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = make_x2_beyond_mimic_task(args.policy, args.scene)
    print(f"Policy: {task.policy_path}")
    print(f"Scene:  {task.scene_path}")
    task.create_sim_runner().run(headless=args.headless, max_steps=args.max_steps)


if __name__ == "__main__":
    main()

