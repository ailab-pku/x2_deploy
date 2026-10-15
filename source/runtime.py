from __future__ import annotations

import time

from .beyond_mimic import BeyondMimicPolicy
from .controller import MujocoController


class Sim2SimRunner:
    def __init__(self, controller: MujocoController, policy: BeyondMimicPolicy):
        self.controller = controller
        self.policy = policy

    def run(self, *, headless: bool = False, max_steps: int | None = None) -> None:
        reference = self.policy.reset()
        root_pos, root_quat, joint_pos = self.policy.initial_pose()
        self.controller.reset(root_pos, root_quat, joint_pos)
        self.controller.settle(reference.joint_pos)
        if not headless:
            self.controller.launch_viewer()
            print("[MuJoCo] Space: pause/resume; close the viewer to exit")

        step = 0
        try:
            while self.controller.is_running and (max_steps is None or step < max_steps):
                self.controller.wait_while_paused()
                if not self.controller.is_running:
                    break
                start = time.monotonic()
                targets = self.policy.infer(self.controller.read_state())
                self.controller.step(targets)
                self.controller.sync()
                step += 1
                if not headless:
                    remaining = self.controller.robot.policy_dt - (time.monotonic() - start)
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            self.controller.close()

