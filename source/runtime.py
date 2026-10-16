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
) -> int:
    """Run every motion frame once, then exit with measured-pose hold."""
    reference = policy.reset()
    controller.align_heading(reference.body_quat_w[policy.root_index])
    hold_on_exit = False
    try:
        controller.start()
        initial = controller.read_state().joint_pos
        ticks = max(1, round(PREPARE_SECONDS / controller.robot.policy_dt))
        deadline = time.monotonic()
        for tick in range(1, ticks + 1):
            controller.read_state()
            blend = tick / ticks
            blend = blend * blend * (3.0 - 2.0 * blend)
            controller.set_target(initial + blend * (policy.reference_joint_pos_xml - initial))
            deadline += controller.robot.policy_dt
            time.sleep(max(0.0, deadline - time.monotonic()))
        deadline = time.monotonic()
        step = 0
        while step < policy.motion_frames:
            targets = policy.infer(controller.read_state())
            controller.set_target(targets)
            step += 1
            deadline += controller.robot.policy_dt
            time.sleep(max(0.0, deadline - time.monotonic()))
        hold_on_exit = True
        return step
    except KeyboardInterrupt:
        hold_on_exit = True
        raise
    finally:
        controller.stop(hold_position=hold_on_exit)
