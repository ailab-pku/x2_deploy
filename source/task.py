from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .beyond_mimic import BeyondMimicPolicy
from .controller import MujocoController
from .runtime import Sim2SimRunner
from .x2_ultra import X2_ULTRA, X2UltraConfig


@dataclass(frozen=True)
class X2BeyondMimicTask:
    robot: X2UltraConfig
    policy_path: Path
    scene_path: Path

    def create_sim_runner(self) -> Sim2SimRunner:
        controller = MujocoController(self.robot, self.scene_path)
        policy = BeyondMimicPolicy(self.policy_path, self.robot)
        return Sim2SimRunner(controller, policy)


def make_x2_beyond_mimic_task(
    policy_path: str | Path | None = None,
    scene_path: str | Path | None = None,
) -> X2BeyondMimicTask:
    return X2BeyondMimicTask(
        robot=X2_ULTRA,
        policy_path=Path(policy_path or X2_ULTRA.default_policy_path).expanduser().resolve(),
        scene_path=Path(scene_path or X2_ULTRA.scene_path).expanduser().resolve(),
    )

