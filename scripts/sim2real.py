from __future__ import annotations

import argparse
from pathlib import Path
import sys
from threading import Thread

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Original X2 Ultra BeyondMimic deploy via AimDK v1.1 / ROS 2")
    parser.add_argument("--policy", type=Path, default=PROJECT_ROOT / "assets" / "policy.onnx")
    args = parser.parse_args()

    import onnxruntime as ort
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    from source.beyond_mimic import BeyondMimicPolicy
    from source.controllers.robot_controller import AimDKController
    from source.runtime import run_sim2real
    from assets.x2_ultra import X2_ULTRA

    # Keep inference single-threaded so ROS feedback remains responsive.
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy = BeyondMimicPolicy(args.policy.expanduser().resolve(), X2_ULTRA, session_options=options)
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    controller = AimDKController(X2_ULTRA)
    executor = SingleThreadedExecutor()
    executor.add_node(controller)
    spinner = Thread(target=executor.spin, name="aimdk_feedback", daemon=True)
    spinner.start()
    print("Live: policy/targets 50 Hz, robot-side PD; original X2 Ultra")
    try:
        print(f"Motion: {policy.motion_frames} frames ({policy.motion_frames * X2_ULTRA.policy_dt:.2f} s); "
              "exit after the final frame")
        steps = run_sim2real(controller, policy)
        print(f"Completed {steps} policy steps")
    except KeyboardInterrupt:
        print("Stopping policy; retaining measured joint pose if feedback is valid")
    finally:
        controller.stop()
        executor.shutdown()
        spinner.join()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
