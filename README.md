# X2 Ultra BeyondMimic Deploy

基于 ONNX Runtime 的 X2 Ultra 部署，支持 MuJoCo sim2sim 和 AimDK sim2real。

## 运行

`onnx` 用于读取内嵌 motion 的准确帧数，确定 motion 播放结束的时刻。

依赖：Python 3.10+、`numpy`、`mujoco`、`onnx`、`onnxruntime`。

```bash
python -m venv .venv
.venv/bin/python -m pip install numpy mujoco onnx onnxruntime
.venv/bin/python scripts/sim2sim.py
```

motion 播放完后自动退出；窗口中按空格暂停/继续，关闭窗口可提前退出。无窗口播放完整 motion：

```bash
.venv/bin/python scripts/sim2sim.py --headless
```

可用 `--policy PATH` 指定 motion-export ONNX，MuJoCo 场景固定使用 `assets/x2_ultra/scene.xml`。
默认加载 `assets/policy.onnx` 和 `assets/x2_ultra/scene.xml`；请将新训练导出的 ONNX 放到该路径，或使用 `--policy` 指定路径。
部署直接使用 `assets/x2_ultra.py` 中的配置；替换 ONNX 时需保持关节顺序、action scale 和 obs 布局一致。

## 代码

- `scripts/sim2sim.py`：启动入口。
- `scripts/sim2real.py`：原版 X2 Ultra 的 AimDK / ROS 2 真机入口。
- `source/runtime.py`：参考状态初始化、policy 控制循环和交互窗口。
- `source/beyond_mimic.py`：ONNX 推理、内嵌 reference motion 和 obs 构造。
- `source/controllers/mujoco_controller.py`：MuJoCo 状态读取与力矩 PD 控制。
- `source/controllers/robot_controller.py`：真机 HAL 状态、IMU 转换、目标位置/PD 参数下发与退出阻尼。
- `source/__init__.py`：共用的 `RobotState` 和四元数运算。
- `assets/x2_ultra.py`：关节顺序、增益、action 映射及仿真参数。

## 当前 checkpoint 的约定

Reference motion 已包含在 ONNX 中，由 `time_step` 索引。
初始位置和速度均取自 reference 第 0 帧。
ONNX 内包含 observation normalization，部署端只构造原始 obs。

164 维 obs 的顺序：

| 索引 | 内容 | 关节顺序 |
| --- | --- | --- |
| 0:62 | Reference joint position / velocity | Isaac Lab articulation 顺序 |
| 62:68 | Reference torso 相对当前 torso 的旋转矩阵前两列，按行展开 | — |
| 68:71 | Pelvis 局部角速度 | — |
| 71:102 | 当前关节位置减 nominal default | Isaac Lab articulation 顺序 |
| 102:133 | 当前关节速度 | Isaac Lab articulation 顺序 |
| 133:164 | 上一步原始 action，初始为零 | Isaac Lab articulation 顺序 |

当前训练仓库统一使用两套顺序：Isaac Lab articulation 顺序和 XML/MuJoCo 顺序。
ONNX 中的 reference、观测关节项、previous action 和 policy 输出均为 Isaac Lab 顺序；
deploy 在进入 MuJoCo 或 HAL 前按关节名称转换为 XML 顺序。

Sim2sim 中 policy 为 50 Hz，底层 PD/物理步为 200 Hz，每个 action 保持 4 个物理步，未启用执行器
延迟或启动等待。关节 armature 和各部件质量使用模型描述文件的原始值。

## Sim2real（原版 X2 Ultra）

参考 AimDK v1.1.0 的 `py_examples/motocontrol.py`、`echo_imu_data.py` 和
`extra/x2_rl_deploy`，使用 ROS 2 Humble 的 HAL 接口。真机路径只依赖
`numpy`、`onnx`、`onnxruntime`、`rclpy`、`sensor_msgs` 和 SDK 构建出的 `aimdk_msgs`，不依赖 MuJoCo。

在机器人开发计算单元或开发容器中，用与 ROS 2 匹配的系统 Python 创建环境。
当前 sim2sim 的 Python 3.13 环境不适用于 Humble 的 Python 3.10 扩展。

```bash
source /opt/ros/humble/setup.bash
source /path/to/aimdk/install/local_setup.bash
python3 -m venv --system-site-packages .venv-real
.venv-real/bin/python -m pip install numpy onnx onnxruntime
```

当前旧款 X2 沿用 `~/x2_deploy` 的接管方式：先在 PC1 手动停止原生运控，
再启动本脚本。停止 MC 不一定改变系统模式名称，因此不再要求显示 `Develop_MC`。
脚本直接创建四组 command publisher，不检查其他发布者。ROS 2 允许同一 topic 有多个
publisher，不会因其他控制器正在发布而自动报错；原生运控仍需手动停止。

```bash
# 在 PC1 执行；部署脚本不会自动执行：
aima em stop-app mc

# 在部署项目目录执行：
./deploy.sh
```

`deploy.sh` 自动加载 ROS、AIMDK、DDS 配置和已有 Python 环境，启动后先阻塞等待确认。
终端输入 `y` 或 `yes` 后才进入 prepare；输入其他内容、输入结束（EOF）或 Ctrl+C 可取消，
确认前不会创建 HAL command publisher 或发送电机命令。
也可在已加载 ROS/SDK 的环境中直接运行：

```bash
.venv-real/bin/python scripts/sim2real.py
```

IMU 直接使用 HAL 的骨盆姿态和角速度，不额外施加安装旋转，沿用当前机器的单位旋转约定。

真机状态与控制约定：

- 按官方 `leg/waist/arm/head` 四组接收状态，按名称映射到 XML 顺序；policy 计算使用 Isaac Lab 顺序。原版头部仅 yaw 可用，
  `head_pitch` 的测量位置/速度置零，命令槽位保留且力矩为零。当前 ONNX reference 包含 head pitch，
  因此该自由度在原版机器上无法跟踪。
- 使用 `/aima/hal/imu/torso/state`（骨盆 IMU）的局部角速度；结合腰部 yaw/pitch/roll
  正运动学得到 torso 姿态。启动时只对齐 IMU 世界系与 reference 的 yaw，保留实际 roll/pitch。
- 确认 prepare 后等待四组关节和 IMU 首批反馈（最多 10 秒），再对齐 heading 和接管。
  用 3 秒从实测关节位置平滑过渡到 reference 第 0 帧，随后自动开始 50 Hz policy。
- 从 ONNX 内嵌 motion 表读取总帧数，以 50 Hz 将第 0 帧到最后一帧各执行一次。
  播放结束后，从最后一次 policy 下发的 target 用 3 秒平滑过渡回启动时保存的第 0 帧位置，
  然后停止 policy 推理，以 50 Hz 持续下发第 0 帧位置和原有 kp/kd，直到 Ctrl+C。
  motion 时长为帧数 × 0.02 秒，不含 prepare、回程和保持时间。
- 推理和正常命令下发均为 50 Hz：`position=target`、`stiffness=kp`、`damping=kd`、
  `velocity=0`、`effort=0`。底层控制器持续执行 PD；Python 不计算力矩，也不重发 200 Hz 命令。
  电机端的实际控制频率和力矩限幅由固件决定，SDK 此消息没有独立的力矩上限字段。
- Ctrl+C 时，取一次退出时缓存的实测关节位置，保留 kp/kd，
  连续发送 0.2 秒保持命令后退出。HAL 持续保持最后命令，不再运行 policy，
  不会自动卸掉刚度，也没有动态平衡能力；运动中或单脚支撑时不能据此保证站立。
- 退出时若未设置保持姿态，则发送 0.2 秒阻尼（`effort=stiffness=0`、`damping=kd`）。
  尚未创建 command publisher 时退出不会发送命令。
  脚本不自动恢复原生 MC，需要现场按官方流程交还控制权。

SDK 明确规定 HAL 无命令超时保护；SIGKILL、进程崩溃或通信断开时无法保证退出命令送达。

此前版本已完成 sim2sim 和 sim2real 验证；本次交互、回程与保持流程仅做离线验证。
