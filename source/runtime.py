from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from .beyond_mimic import BeyondMimicPolicy

if TYPE_CHECKING:
    from .controller import MujocoController
    from .real_controller import AimDKController


def run_sim2sim(
    controller: MujocoController, policy: BeyondMimicPolicy,
    *, headless: bool = False, max_steps: int | None = None,
) -> int:
    """Restore reference frame zero, then run policy and PD control."""
    reference = policy.reset()
    root = policy.root_index
    controller.reset(
        reference.body_pos_w[root], reference.body_quat_w[root], reference.joint_pos,
        joint_vel=reference.joint_vel,
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
        while max_steps is None or step < max_steps:
            if viewer is not None and not viewer.is_running():
                break
            start = time.monotonic()
            if not paused:
                controller.step(policy.infer(controller.read_state()))
                step += 1
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, controller.robot.policy_dt - (time.monotonic() - start)))
    finally:
        if viewer is not None:
            viewer.close()
    return step


def run_sim2real(
    controller: AimDKController, policy: BeyondMimicPolicy,
    *, max_steps: int | None = None, prepare_seconds: float = 3.0,
) -> int:
    """Run motion once, then keep tracking its final reference until interrupted."""
    reference = policy.reset()
    controller.wait_for_state()
    controller.align_heading(reference.body_quat_w[policy.root_index])
    hold_on_exit = False
    try:
        if not controller.dry_run:
            controller.start()
            initial = controller.read_state().joint_pos
            ticks = max(1, round(prepare_seconds / controller.robot.policy_dt))
            deadline = time.monotonic()
            for tick in range(1, ticks + 1):
                controller.read_state()
                blend = tick / ticks
                blend = blend * blend * (3.0 - 2.0 * blend)
                controller.set_target(initial + blend * (reference.joint_pos - initial))
                deadline += controller.robot.policy_dt
                time.sleep(max(0.0, deadline - time.monotonic()))
            error = controller.read_state().joint_pos - reference.joint_pos
            if np.max(np.abs(error[controller.active])) > 0.2:
                raise RuntimeError("Robot did not reach reference frame zero (joint error > 0.2 rad)")
        deadline = time.monotonic()
        step = 0
        while max_steps is None or step < max_steps:
            targets = policy.infer(controller.read_state())
            if not controller.dry_run:
                controller.set_target(targets)
            step += 1
            if step == policy.motion_frames and (max_steps is None or step < max_steps):
                print(f"Motion complete ({policy.motion_frames} frames); tracking final reference at 50 Hz. Ctrl+C to stop.", flush=True)
            deadline += controller.robot.policy_dt
            time.sleep(max(0.0, deadline - time.monotonic()))
        controller.read_state()
        hold_on_exit = True
        return step
    except KeyboardInterrupt:
        hold_on_exit = True
        raise
    finally:
        controller.stop(hold_position=hold_on_exit)
