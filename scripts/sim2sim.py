from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="X2 Ultra BeyondMimic sim-to-sim runner")
    parser.add_argument("--policy", type=Path, help="Motion-export ONNX policy (motion is embedded)")
    parser.add_argument("--scene", type=Path, help="Override the project-local MuJoCo scene")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    parser.add_argument("--max-steps", type=int, help="Stop after this many 50 Hz policy steps")
    args = parser.parse_args()

    from source.beyond_mimic import BeyondMimicPolicy
    from source.controller import MujocoController
    from source.runtime import run_sim2sim
    from source.x2_ultra import X2_ULTRA

    policy_path = (args.policy or X2_ULTRA.default_policy_path).expanduser().resolve()
    scene_path = (args.scene or X2_ULTRA.scene_path).expanduser().resolve()
    print(f"Policy: {policy_path}")
    print(f"Scene:  {scene_path}")
    policy = BeyondMimicPolicy(policy_path, X2_ULTRA)
    controller = MujocoController(X2_ULTRA, scene_path)
    steps = run_sim2sim(controller, policy, headless=args.headless, max_steps=args.max_steps)
    print(f"Completed {steps} policy steps ({steps * X2_ULTRA.policy_dt:.2f} s simulated)")


if __name__ == "__main__":
    main()
