# X2 Ultra BeyondMimic deployment

This project is a self-contained sim-to-sim implementation for the
`whole_body_tracking_X2_ultra` BeyondMimic policy. It deliberately does not use
the locomotion observation/history from the AimDK `sim2sim_x2.py` example.

## Observation

The 164-dimensional actor observation exactly follows the training config:

| term | dimension | order |
| --- | ---: | --- |
| reference joint position + velocity | 62 | tracking/action joint order |
| reference torso orientation relative to current torso | 6 | first two rotation-matrix columns |
| pelvis angular velocity | 3 | pelvis/body frame |
| joint position relative to default | 31 | Isaac Lab articulation order |
| joint velocity | 31 | Isaac Lab articulation order |
| previous action | 31 | tracking/action joint order |

The exported motion-policy ONNX has `obs` and `time_step` inputs and contains
the reference motion. The runner caches the next reference while evaluating the
current action, so it performs one ONNX call per 50 Hz policy tick.

## Layout

- `scripts/sim2sim.py`: command-line program entry
- `source/x2_deploy/beyond_mimic.py`: BeyondMimic observation and ONNX inference
- `source/x2_deploy/x2_ultra.py`: X2 joint order, gains, limits, and asset paths
- `source/x2_deploy/task.py`: policy/robot/backend assembly
- `source/x2_deploy/controller.py`: backend-neutral interface and MuJoCo backend
- `source/x2_deploy/assets/`: ONNX policy, project-local MJCF, and meshes

A future real-robot controller can implement `RobotController`; the observation
and policy code does not need to be copied or rewritten.

## Run

```bash
cd /home/bozhou/cbz/deploy
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python scripts/sim2sim.py
```

For a short headless validation:

```bash
.venv/bin/python scripts/sim2sim.py --headless --max-steps 100
```

Use another policy exported by this project's motion exporter with:

```bash
.venv/bin/python scripts/sim2sim.py --policy /absolute/path/to/policy.onnx
```

The policy is validated at startup for input size, observation term order,
joint order, body metadata, and action scale. A plain locomotion ONNX is rejected
instead of silently receiving the wrong observation.
