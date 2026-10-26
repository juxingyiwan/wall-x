# ReBot / B601：安装、训练与部署

[English](rebot.md) · [总导航](../README.zh-CN.md)

本页支持有人值守的一台 B601 单臂、两路相机及兼容的 Wall-X Flow checkpoint。命令在 Linux 仓库根目录执行。GPU 在实验机时使用本地推理，也可采用模型服务器加网络客户端。准备 B601 演示数据、训练 checkpoint 和机器配置。

<a id="environment"></a>

## 1. 安装所需环境

训练、服务端或本地推理使用新的模型环境。准备兼容的 NVIDIA 驱动和主机 C/C++ 工具。构建脚本要求当前 Conda 环境内部具有 `nvcc`：

```bash
conda env create -f environments/python310.yml -n wallx-rebot-model
conda activate wallx-rebot-model
conda install -c nvidia cuda-toolkit=12.8
conda install -c conda-forge ffmpeg=8
test -x "$CONDA_PREFIX/bin/nvcc"
"$CONDA_PREFIX/bin/nvcc" --version
python -m pip install -r environments/requirements/rebot-model.txt
python -m pip check
export CUDA_HOME="$CONDA_PREFIX"
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
export FLASH_ATTN_CUDA_ARCHS=$(python -c 'import torch; print(torch.cuda.get_device_capability()[0] * 10)')
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install 'dmuon @ git+https://github.com/X-Square-Robot/dmuon.git@157f522b8570764c473b1a18939f4ebb36da5878'
WALLX_BUILD_ENV=wallx-rebot-model TORCH_CUDA_ARCH_LIST='8.0;8.6' bash deployment/environment/build_wallx.sh
python deployment/environment/validate.py --require-cuda
python -m pip check
```

上面的架构示例面向 A800 / RTX 3090，需按实际 GPU 选择。应通过 CUDA 扩展/服务检查，且 `pip check` 无冲突。依赖包含 Torch 2.10、Transformers 4.57.6、LeRobot 0.4.4、Hub 0.35.3、Rerun 0.26.2 和硬件包，不要叠加原生根 requirements。保留分配的 GPU 可见性，不同 Torch/CUDA ABI 分别构建源码。

仅通过网络推理时，机器人主机可使用以下独立客户端环境：

```bash
conda env create -f environments/python310.yml -n wallx-rebot-client
conda activate wallx-rebot-client
conda install -c conda-forge ffmpeg=8
python -m pip install -c environments/constraints/rebot-linux-py310-cu128.txt './deployment/client[vision,hardware]'
python -m pip check
python -m wallx_client doctor
```

`doctor` 不打开设备、不加载模型。hardware extra 包含 LeRobot 及其依赖。仅在 CPU 开发机查看 CLI 时，可在 Python 3.10/3.12 环境安装 `deployment/client[dev,vision]`；操作真机时安装 hardware extra。

<a id="assets"></a>

## 2. 准备模型资源

在模型环境下载以下基础权重和 processor，用于初始化训练。评估时使用已训练的任务 checkpoint 及其配置和归一化参数。

```bash
hf download X-Square-Robot/wall-oss-0.5 \
  --revision b7dc23a82dbf70a65859c9fdd596703234597028 \
  --local-dir /path/to/wall-oss-0.5
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

应得到 Wall-OSS 的 `config.json`、`model.safetensors` 和 Qwen 的 processor/tokenizer 文件。`model.config_path` 指向 Wall-OSS JSON，`model.processor_path` 指向 Qwen 目录。通过 `checkpoint.resume_from` 加载完整 Wall-OSS 权重时，保留 `model.pretrained_path: null`。

已有兼容的 B601 训练 checkpoint 时，保留它自己的配置、processor 和 normalizer，直接进入[部署配置](#configuration)。

<a id="data"></a>

## 3. 准备演示数据

使用兼容的 LeRobot 数据集，包含元数据、图像/视频及 parquet 列 `episode_index`、`frame_index`、`task_index`、`timestamp`、`observation.state`、`action`。state/action 每行含 7 个有限值，每个 episode 内帧连续、时间递增、任务固定。将前方/侧面相机键映射到训练视图。在模型环境运行：

```bash
mkdir -p deployment/local
test -e deployment/local/train.yml || cp workspace/example/b601.yml deployment/local/train.yml
```

新训练模板 `workspace/example/b601.yml` 使用 8 张 GPU × 每卡 batch 8 × 累积 1，有效 batch 64；平方根任务采样、32 步关节增量动作、AdamW `5e-5`、warmup 500、更新上限 16,000。验证比例为 0.1，统计范围为 `statistics_scope: train_episodes`。此前真机验证的增量 checkpoint 使用比例 0.01 和 `statistics_scope: full_dataset`，恢复时使用 checkpoint 保存的统计量与划分。[Padding 消融](../README.zh-CN.md#action-padding)使用绝对动作 checkpoint。七个 `command_to_physical_scale` 值定义 B601 适配器的动作转换。

修改全部占位值：数据集 `root`、逻辑 `repo_id`、模型配置、processor、初始权重文件和输出目录。训练 YAML 不自动展开 shell 变量。精确恢复需保持 `data.num_workers: 0`。模板只用训练 episode 计算统计量，准备输出目录必须是新的：

```bash
python -m scripts.prepare_b601 --config deployment/local/train.yml --output-dir /path/to/new-b601-comparison
python -m scripts.validate_b601_data --directory /path/to/new-b601-comparison
```

应生成 `absolute.yml`、`delta.yml`、独立统计量、`episode_split.json`、`preparation.json`。验证器使用真实数据/processor 检查输入与还原标签，不加载模型权重和硬件。准备工具设置 32 步 horizon 和 16,000 次更新上限，训练前核对并一致调整两组配置。绝对/增量对照使用相同划分、采样和预算。

验证比例范围为 $[0, 1)$，不超过 `1e-6` 时跳过验证；否则每任务至少留出一条 episode，只有一条 episode 的任务不支持正验证比例。默认统计范围为训练 episode；`statistics_scope: full_dataset` 还会纳入留出的 episode。

显式选择全数据统计时，可用下面命令冻结统计量、另行导出划分，训练前将 `data.norm_stats_path` 和 `b601.split_manifest` 指向生成文件：

```bash
python -m scripts.prepare_b601 --config deployment/local/train.yml --statistics-output /path/to/frozen-statistics.json
python -m scripts.prepare_b601 --config deployment/local/train.yml --split-output /path/to/new-episode-split.json
```

<a id="train"></a>

## 4. 训练与恢复

```bash
NPROC=8
torchrun --nproc_per_node="$NPROC" -m scripts.train_b601 \
  --config /path/to/new-b601-comparison/delta.yml --log_to_file
```

将 `NPROC` 设为分配的 GPU 数，配套调整 batch/累积步数。使用 `scripts.train_b601` 执行 B601 数据/checkpoint 流程。日志和完整 checkpoint 写入 `checkpoint.save_path`。`max_optimizer_steps` 限制已完成的更新数，epoch 可能更早结束。

```bash
NPROC=8
torchrun --nproc_per_node="$NPROC" -m scripts.train_b601 \
  --config /path/to/new-b601-comparison/delta.yml \
  --resume /path/to/complete-checkpoint --max-updates 18000 --log_to_file
```

精确恢复需要完成标记 `b601_training_state.json`、各 rank 状态及其绑定文件。保持源码、数据、划分/统计、GPU 数、batch、随机状态、预处理、优化器、调度器不变，数据 worker 为 0。仅更新/epoch 上限可在不改变绑定轨迹的情况下延长，这不会延长余弦调度周期。使用旧权重或新实验时，在新配置将 `checkpoint.resume_from` 设为权重文件，不传 `--resume`。

<a id="configuration"></a>

## 5. 配置机器人与 checkpoint

支持的控制规格为 30 Hz、两路 640 × 480 相机、32 步 × 26 维模型动作块（6 个臂关节、1 个夹爪和 19 个 padding 通道）。按硬件提供方说明配置 B601 驱动、CAN 接口和相机权限；确认标定，并与其他控制该机械臂的软件共用互斥锁文件。检查设备：

```bash
ip -details link show can0
ls -l /dev/v4l/by-id/
```

将完整推理资源放到 `<checkpoint_root>/<checkpoint_id>`：指定权重、保存的训练配置、normalizer、模型/processor 资源，以及版本化 B601 manifest 和其绑定的划分/统计文件。优化器状态用于恢复训练，推理无需加载。计算权重哈希并创建私有配置：

```bash
sha256sum /path/to/model.safetensors
mkdir -p deployment/local
test -e deployment/local/profile.yml || cp deployment/configs/b601.example.yml deployment/local/profile.yml
test -e deployment/local/client.yml || cp deployment/configs/client.yml deployment/local/client.yml
test -e deployment/machines/standalone.local.yml || cp deployment/machines/standalone.example.yml deployment/machines/standalone.local.yml
```

| 文件 | 使用前设置 |
| --- | --- |
| `deployment/local/profile.yml` | `model.checkpoint_id`、`model.epoch`、`model.step`、真实权重 SHA256、`norm_key`、padding；`client_config: client.yml` 选择相邻私有客户端文件 |
| `deployment/local/client.yml` | `task` 填完整训练指令，设置输出目录、控制/RTC/显示选项 |
| `deployment/machines/standalone.local.yml` | checkpoint 根目录、真实 CAN/相机设备、已验证的相机角色及既有共享锁路径 |

`checkpoint_id` 的格式为 `experiment/epoch_step`，后缀必须与 `epoch`、`step` 完全一致。例如 checkpoint 目录为 `my_experiment/3_800` 时，在完整配置的 `model` 节中同步设置：

```yaml
model:
  checkpoint_id: my_experiment/3_800
  epoch: 3
  step: 800
```

`epoch` 从 0 开始，`step` 为正整数；沿用实际 checkpoint 的目录名称。将 `checkpoint_sha256` 替换为上面 `sha256sum` 输出的 64 位值，`norm_key` 使用该 checkpoint 保存的键。

先将采集图像与训练视图核对，再设 `camera_mapping_verified: true`；相机角色使用不同的稳定设备路径。机器覆盖文件负责主机/设备设置，任务/控制/RTC 放在客户端 YAML；显式 CLI 可覆盖受支持值。检查实际生效配置：

```bash
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml config
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- config
```

Rerun 实时显示默认开启；本地窗口需要解锁桌面。纯 SSH 使用 `--display-data=false`，或以 `--display-ip HOST --display-port PORT` 连接已有查看器，将它们作为客户端 `preflight`/`run` 参数传入。录像独立控制，只有 `--save-rollout=false` 才关闭。以下是重要客户端默认值，实际可由 YAML/CLI 覆盖：

| 字段 | 默认值 |
| --- | --- |
| `policy.horizon` | `32` |
| `policy.action_dims` | `[6, 1, 19]` |
| `policy.padding` | `"zero"` |
| `control.hz` | `30` |
| `control.execution_horizon` | `16` |
| `control.duration_s` | `300.0` |
| `control.startup_home` | `true` |
| `shutdown.home_timeout_s` | `65.0` |
| `rtc.enabled` | `false` |
| `rtc.guidance_mode` | `"lerobot"` |
| `display_data` | `true` |
| `save_rollout` | `true` |
| `display_port` | `9876` |

<a id="replay"></a>

## 6. 采集、检查与回放

采集读取 CAN 反馈和相机，不使能电机、不加载策略。首次确认角色时先核对物理设备映射与相机画面；capture 要求映射已声明验证。以下在机器人主机运行：

```bash
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml client --output /path/to/new-capture capture
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local check
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- replay --observation /path/to/new-capture/observation.npz
```

应生成 `observation.npz`、相机 PNG 和采集报告，核对状态单位与相机顺序。`local check` 在不访问机器人的情况下验证/加载模型，replay 对保存观测推理。

<a id="network"></a>

## 7. 可选网络推理

本地推理跳过本节。GPU 主机将 `deployment/machines/cluster.example.yml` 复制到新的 `deployment/machines/local.yml`；机器人主机使用 `robot.example.yml`。填写 checkpoint 路径、端点和设备，共用同一模型配置。保留分配的 `CUDA_VISIBLE_DEVICES`，不要设置冲突的 `cuda_id`。在 GPU 主机运行：

```bash
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml server --dry-run
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml server
```

配置机器人客户端端点以连通服务端。远程主机使用默认回环地址时需要现场提供隧道。使用本仓库服务端，它实现了 `joint_state_v1`、模型身份/padding 校验及请求 ID。在机器人主机使用对应机器配置采集，再回放：

```bash
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client doctor
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client --output /path/to/new-network-capture capture
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client replay --observation /path/to/new-network-capture/observation.npz
```

<a id="operate"></a>

## 8. 预检、同步运行与 RTC

清空工作区，确保物理急停可操作，由现场人员在交互终端运行。预检读取设备并调用推理，不使能电机；`run` 会使能运动，并可能在启动和结束时归位。以下只选一种后端：

```bash
# Local backend
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- preflight --rtc false
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- run --duration 30 --rtc false
# Network backend: robot host
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client preflight --rtc false
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client run --duration 30 --rtc false
```

同步验证成功后，使用 RTC 参数重新预检，再运行。RTC 使用原生单请求 Flow 后端：

```bash
# Local RTC
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- preflight --rtc true --guidance-mode lerobot
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- run --duration 30 --rtc true --guidance-mode lerobot
# Network RTC: robot host
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client preflight --rtc true --guidance-mode lerobot
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client run --duration 30 --rtc true --guidance-mode lerobot
```

终端按键：`p` 暂停/继续，`s` 标记成功并结束，`f` 标记失败并结束，`q` 或 Ctrl+C 中止。示例明确设置 30 秒回合预算，包含启动后的暂停时间，不包含结束归位；通常默认值为 300 秒。软件停止可能执行受保护的归位，需要立即停止时使用物理急停。

归位检查名义前向扫掠，必要时协调回收，再完成原生臂归位并验证夹爪零位。65 秒超时分别应用于各主要派发阶段。运行前检查完整归位路径中的障碍物。电机/CAN/温度故障可能阻止归位，清理仍会尝试失能；进程崩溃、断电或控制器阻塞时，使用硬件停止，并在重启前确认机器人状态。

相机专项诊断不访问 CAN、策略或电机。使用未占用的输出目录，先关闭其他相机程序；它生成时序/配置报告，不保存图像或录像。

```bash
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml client --output /path/to/new-camera-check camera-test --duration 60
```

<a id="results"></a>

## 9. 查看结果与排错

配置的 `output_root` 保存配置、`report.json`、事件及已开启的媒体记录。`stop_outcome` 记录停止原因/人工标记，`outcome` 还包含清理故障。`shutdown.homing` 记录归位完成、失败或跳过；`shutdown.disable: commands_sent` 记录已发送失能命令，电机状态在现场确认。对于 `aborted`、`timeout`、`budget_exhausted`，根据现场观察或录像标注任务结果。

| 现象 | 处理 |
| --- | --- |
| 占位配置或模型身份被拒绝 | 检查 checkpoint 根目录/ID、实际加载权重哈希、normalizer 键、padding 和绑定资源 |
| 相机映射/时效失败 | 检查前/侧视图、稳定设备路径、时间戳、权限和线缆带宽；`wallx-client --help` 可查看相机诊断命令 |
| SSH 中查看器不可用 | 传 `--display-data=false` 或配置已有查看器，录像独立控制 |
| 服务可连接但回放失败 | 检查协议元数据和保存观测格式 |
| RTC 被拒绝 | 核对显式 executor/RTC 配置、30 Hz 时基及服务端 guidance 能力 |
| 归位/反馈/温度故障 | 查看停止诊断，检查物理系统并恢复健康反馈，不绕过保护 |
