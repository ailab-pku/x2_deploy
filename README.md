# X2 Ultra BeyondMimic Deploy

基于 MuJoCo 和 ONNX Runtime 的最小 sim2sim 部署。

## 运行

依赖：Python 3.10+、`numpy`、`mujoco`、`onnxruntime`。

```bash
python -m venv .venv
.venv/bin/python -m pip install numpy mujoco onnxruntime
.venv/bin/python scripts/sim2sim.py
```

窗口中按空格暂停/继续，关闭窗口退出。无窗口运行 500 个 policy 步（10 秒仿真时间）：

```bash
.venv/bin/python scripts/sim2sim.py --headless --max-steps 500
```

可用 `--policy PATH` 指定 motion-export ONNX，`--scene PATH` 指定 MuJoCo 场景。
默认加载 `assets/models/prepare.onnx` 和 `assets/x2_ultra/scene.xml`。
代码直接使用当前配置，不额外校验模型元数据；替换 ONNX 时需保持关节顺序、obs 布局和 action 映射一致。

## 代码

- `scripts/sim2sim.py`：启动入口。
- `source/runtime.py`：参考状态初始化、policy 控制循环和交互窗口。
- `source/beyond_mimic.py`：ONNX 推理、内嵌 reference motion 和 obs 构造。
- `source/controller.py`：MuJoCo 状态读取与力矩 PD 控制。
- `source/x2_ultra.py`：关节顺序、增益、action 映射及仿真参数。
- `source/math_utils.py`：四元数运算。

## 当前 checkpoint 的约定

Reference motion 已包含在 ONNX 中，由 `time_step` 索引；运行时不读取
`assets/motions/prepare.npz`。初始位置和速度均取自 reference 第 0 帧。
ONNX 内包含 observation normalization，部署端只构造原始 obs。

164 维 obs 的顺序：

| 索引 | 内容 | 关节顺序 |
| --- | --- | --- |
| 0:62 | Reference joint position / velocity | Action 顺序 |
| 62:68 | Reference torso 相对当前 torso 的旋转矩阵前两列，按行展开 | — |
| 68:71 | Pelvis 局部角速度 | — |
| 71:102 | 当前关节位置减 nominal default | Isaac Lab articulation 顺序 |
| 102:133 | 当前关节速度 | Isaac Lab articulation 顺序 |
| 133:164 | 上一步原始 action，初始为零 | Action 顺序 |

`prepare.onnx` 训练时，`randomize_joint_default_pos` 将 articulation 顺序的 default
向量按位置赋给了 action 顺序的 offset。当前 `action_offset` 保留这一映射以匹配
checkpoint；它与用于 obs 的、按关节名称对齐的 default 不同。训练端修复该事件并
重新训练后，需要同步更新部署端的 action offset。

Policy 为 50 Hz，底层 PD/物理步为 200 Hz，每个 action 保持 4 个物理步，未启用执行器
延迟或启动等待。关节 armature 和各部件质量使用模型描述文件的原始值。
