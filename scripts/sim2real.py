from __future__ import annotations

import argparse
import math
from pathlib import Path
import signal
import sys
from threading import Thread

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Original X2 Ultra BeyondMimic deploy via AimDK v1.1 / ROS 2")
    parser.add_argument("--policy", type=Path, default=PROJECT_ROOT / "assets" / "character.onnx")
    parser.add_argument("--dry-run", action="store_true", help="Read feedback and infer without publishing commands")
    parser.add_argument("--check-only", action="store_true", help="Check feedback and command ownership without publishing")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Limit 50 Hz steps for testing; default tracks the final motion reference until Ctrl+C")
    parser.add_argument("--prepare-seconds", type=float, default=3.0, help="Transition duration to reference frame zero")
    parser.add_argument("--state-timeout", type=float, default=0.1, help="Joint/IMU/target watchdog in seconds")
    parser.add_argument("--imu-quat", type=float, nargs=4, default=(1, 0, 0, 0), metavar=("W", "X", "Y", "Z"),
                        help="Pelvis IMU mounting quaternion: rotate sensor vectors into pelvis axes")
    args, ros_args = parser.parse_known_args()
    if (args.max_steps is not None and args.max_steps <= 0) or not all(
        math.isfinite(value) and value > 0 for value in (args.prepare_seconds, args.state_timeout)
    ):
        parser.error("Step count, preparation duration and state timeout must be positive")

    import onnxruntime as ort
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    from source.beyond_mimic import BeyondMimicPolicy
    from source.real_controller import AimDKController
    from source.runtime import run_sim2real
    from source.x2_ultra import X2_ULTRA

    # Leave CPU time for ROS feedback and the watchdog while inference runs.
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy = BeyondMimicPolicy(args.policy.expanduser().resolve(), X2_ULTRA, session_options=options)
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    controller = AimDKController(X2_ULTRA, dry_run=args.dry_run or args.check_only,
                                 imu_quat=tuple(args.imu_quat), state_timeout=args.state_timeout)
    executor = SingleThreadedExecutor()
    executor.add_node(controller)
    spinner = Thread(target=executor.spin, name="aimdk_feedback", daemon=True)

    def on_signal(signum, frame):
        # Keep ROS alive until the measured-pose hold command has been sent.
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    spinner.start()
    print(f"{'Read-only' if controller.dry_run else 'Live'}: policy/targets 50 Hz, robot-side PD; original X2 Ultra")
    try:
        if args.check_only:
            controller.wait_for_state()
            controller.check_control_ownership()
            print(f"Preflight passed: no HAL command publishers; mode={controller.mode or 'unknown'} / {controller.mode_status}")
            return
        print(f"Motion: {policy.motion_frames} frames ({policy.motion_frames * X2_ULTRA.policy_dt:.2f} s); "
              f"{'track final reference until Ctrl+C' if args.max_steps is None else f'test limit {args.max_steps} steps'}")
        steps = run_sim2real(controller, policy, max_steps=args.max_steps, prepare_seconds=args.prepare_seconds)
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
