# X2 Ultra BeyondMimic Deploy

基于 ONNX Runtime 的 X2 Ultra 部署，支持 MuJoCo sim2sim 和 AimDK sim2real。

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
默认加载 `assets/prepare.onnx` 和 `assets/x2_ultra/scene.xml`。
代码直接使用当前配置，不额外校验模型元数据；替换 ONNX 时需保持关节顺序、obs 布局和 action 映射一致。

## 代码

- `scripts/sim2sim.py`：启动入口。
- `scripts/sim2real.py`：原版 X2 Ultra 的 AimDK / ROS 2 真机入口。
- `source/runtime.py`：参考状态初始化、policy 控制循环和交互窗口。
- `source/beyond_mimic.py`：ONNX 推理、内嵌 reference motion 和 obs 构造。
- `source/controller.py`：MuJoCo 状态读取与力矩 PD 控制。
- `source/real_controller.py`：真机 HAL 状态、IMU 转换、目标位置/PD 参数下发与退出阻尼。
- `source/state.py`：仿真与真机共用的 policy 状态。
- `source/x2_ultra.py`：关节顺序、增益、action 映射及仿真参数。
- `source/math_utils.py`：四元数运算。

## 当前 checkpoint 的约定

Reference motion 已包含在 ONNX 中，由 `time_step` 索引。
初始位置和速度均取自 reference 第 0 帧。
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

Sim2sim 中 policy 为 50 Hz，底层 PD/物理步为 200 Hz，每个 action 保持 4 个物理步，未启用执行器
延迟或启动等待。关节 armature 和各部件质量使用模型描述文件的原始值。

## Sim2real（原版 X2 Ultra）

参考 AimDK v1.1.0 的 `py_examples/motocontrol.py`、`echo_imu_data.py` 和
`extra/x2_rl_deploy`，使用 ROS 2 Humble 的 HAL 接口。真机路径只依赖
`numpy`、`onnxruntime`、`rclpy`、`sensor_msgs` 和 SDK 构建出的 `aimdk_msgs`，不依赖 MuJoCo。

在机器人开发计算单元或开发容器中，用与 ROS 2 匹配的系统 Python 创建环境。
当前 sim2sim 的 Python 3.13 环境不适用于 Humble 的 Python 3.10 扩展。

```bash
source /opt/ros/humble/setup.bash
source /path/to/aimdk/install/local_setup.bash
python3 -m venv --system-site-packages .venv-real
.venv-real/bin/python -m pip install numpy onnxruntime
```

先读取真机反馈并运行 500 帧推理；`--dry-run` 不创建命令 publisher、不切换模式、不驱动关节：

```bash
.venv-real/bin/python scripts/sim2real.py --dry-run --max-steps 500
```

实际控制前，按官方流程在有支撑的状态下切入 `Develop_MC`，等待迁移完成：

```bash
ros2 run py_examples migrate_system_state Develop_MC
.venv-real/bin/python scripts/sim2real.py --imu-quat W X Y Z --max-steps 500
```

将 `W X Y Z` 替换为这台机器的骨盆 IMU 安装旋转四元数（标量在前）。其含义为
`v_pelvis = R(q_BS) @ v_sensor`，姿态转换为 `q_WB = q_WS * inverse(q_BS)`。
标定数据位于 PC2 的
`/agibot/data/param/protected/factory_calib/factory_calibration_params.yml` 的 `waist_imu` 段；
应核实矩阵方向后转换。仅在 HAL 已输出骨盆坐标数据或安装坐标一致时使用默认单位四元数。
不能根据 `header.frame_id=base_link` 判断数据是否已旋转，官方两个 IMU 都使用这个 frame_id。

真机状态与控制约定：

- 按官方 `leg/waist/arm/head` 四组接收状态，按名称映射到现有 action 顺序。原版头部仅 yaw 可用，
  `head_pitch` 的测量位置/速度置零，命令槽位保留且力矩为零。当前 ONNX reference 包含 head pitch，
  因此该自由度在原版机器上无法跟踪。
- 使用 `/aima/hal/imu/torso/state`（骨盆 IMU）的局部角速度；结合腰部 yaw/pitch/roll
  正运动学得到 torso 姿态。启动时只对齐 IMU 世界系与 reference 的 yaw，保留实际 roll/pitch。
- 默认用 3 秒从实测关节位置平滑过渡到 reference 第 0 帧，到位后才开始 50 Hz policy。
  `--prepare-seconds` 可调整过渡时间。
- 推理和正常命令下发均为 50 Hz：`position=target`、`stiffness=kp`、`damping=kd`、
  `velocity=0`、`effort=0`。底层控制器持续执行 PD；Python 不计算力矩，也不重发 200 Hz 命令。
  电机端的实际控制频率和力矩限幅由固件决定，SDK 此消息没有独立的力矩上限字段。
- 关节、IMU 或 policy target 超过 `--state-timeout`（默认 0.1 秒）未更新时停止 policy 控制；
  独立 watchdog 只监测状态与目标更新，不参与正常命令发布。系统模式必须持续处于
  `Develop_MC / IN_READY`，模式反馈超时为 3 秒。
- 正常结束、Ctrl+C、SIGTERM 或反馈异常时，在仍持有 `Develop_MC` 控制权的情况下连续发送
  0.2 秒阻尼命令（`effort=stiffness=0`、`damping=kd`）。HAL 会保持最后的阻尼命令。
  完成后按官方流程切回 `Ready`，脚本不自动恢复原生运控。

SDK 明确规定 HAL 无命令超时保护；SIGKILL、进程崩溃或通信断开时无法保证退出命令送达，
进程内 watchdog 不能替代机器人端的独立保护。

目前实现已做离线接口与仿真回归验证；真机的 ROS 通信、IMU 标定和实际运动仍需在机上验证。
