from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .beyond_mimic import BeyondMimicPolicy

PREPARE_SECONDS = 3.0

if TYPE_CHECKING:
    from .controllers.mujoco_controller import MujocoController
    from .controllers.robot_controller import AimDKController


def run_sim2sim(
    controller: MujocoController, policy: BeyondMimicPolicy,
    *, headless: bool = False,
) -> int:
    """Restore reference frame zero, then run policy and PD control."""
    reference = policy.reset()
    root = policy.root_index
    controller.reset(
        reference.body_pos_w[root], reference.body_quat_w[root], policy.reference_joint_pos_xml,
        joint_vel=policy.reference_joint_vel_xml,
        root_lin_vel_w=reference.body_lin_vel_w[root],
        root_ang_vel_w=reference.body_ang_vel_w[root],
    )
    viewer = None
    paused = False
    step = 0

    def on_key(keycode: int) -> None:
        nonlocal paused
        if keycode == ord(" "):
            paused = not paused

    try:
        if not headless:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(controller.model, controller.data, key_callback=on_key)
            viewer.cam.elevation = -20
            print("[MuJoCo] Space: pause/resume; close the viewer to exit")
        while step < policy.motion_frames:
            if viewer is not None and not viewer.is_running():
                break
            start = time.perf_counter()
            if not paused:
                controller.step(policy.infer(controller.read_state()))
                step += 1
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, controller.robot.policy_dt - (time.perf_counter() - start)))
    finally:
        if viewer is not None:
            viewer.close()
    return step


def _wait_for_next_tick(deadline: float, dt: float) -> float:
    """Keep 50 Hz; on overrun, continue immediately and rebase the clock."""
    deadline += dt
    while time.perf_counter() < deadline:
        time.sleep(0.0002)
    return deadline


def run_sim2real(
    controller: AimDKController, policy: BeyondMimicPolicy,
) -> None:
    """Confirm, prepare, play motion, then return to frame zero until Ctrl+C."""
    try:
        answer = input("Enter prepare (3-second transition to motion frame 0)? [y/N]: ")
    except EOFError:
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        print("Cancelled before prepare; no commands sent.", flush=True)
        return

    policy.reset()
    frame_zero = policy.reference_joint_pos_xml.copy()
    controller.start()
    deadline = time.perf_counter()
    targets = frame_zero
    while policy.frame < policy.motion_frames:
        targets = policy.infer(controller.read_state())
        controller.set_target(targets)
        deadline += controller.robot.policy_dt
        while time.perf_counter() < deadline:
            time.sleep(0.0002)