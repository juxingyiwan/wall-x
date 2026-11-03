# ReBot / B601：安装、训练与部署

[English](rebot.md) · [总导航](../README.zh-CN.md)

适用于 B601 单臂与两路相机。所有命令在 Linux 仓库根目录执行，将 `/path/to/...` 替换为实际路径。

<a id="environment"></a>

## 1. 安装环境

准备 Conda、NVIDIA GPU 驱动和 C/C++ 编译工具。在运行模型的 GPU 主机上安装，训练和本地推理共用此环境：

```bash
conda env create -f environments/python310.yml -n wallx-rebot-model
conda activate wallx-rebot-model
conda install --override-channels -c nvidia -c conda-forge cuda-toolkit=12.8
export CC=/usr/bin/gcc CXX=/usr/bin/g++
conda install --override-channels -c conda-forge ffmpeg=8
python -m pip install -r environments/requirements/rebot-model.txt
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=$(python -c 'import torch; print(".".join(map(str, torch.cuda.get_device_capability())))')
export FLASH_ATTN_CUDA_ARCHS=$(python -c 'import torch; print(torch.cuda.get_device_capability()[0] * 10)')
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install 'dmuon @ git+https://github.com/X-Square-Robot/dmuon.git@157f522b8570764c473b1a18939f4ebb36da5878'
WALLX_BUILD_ENV=wallx-rebot-model bash deployment/environment/build_wallx.sh
```

两个架构变量自动选择当前 GPU 的编译目标。后续打开终端先执行 `conda activate wallx-rebot-model`。

<details>
<summary>GPU 在另一台主机上：安装机器人客户端环境</summary>

GPU 主机安装上面的模型环境；机器人主机仅需安装客户端环境。两端均使用本仓库：

```bash
conda env create -f environments/python310.yml -n wallx-rebot-client
conda activate wallx-rebot-client
conda install --override-channels -c conda-forge ffmpeg=8
python -m pip install -c environments/constraints/rebot-linux-py310-cu128.txt './deployment/client[vision,hardware]'
```

</details>

<a id="assets"></a>

## 2. 准备模型资源

在模型环境中下载 Wall-OSS-0.5 基础权重和 Qwen processor：

```bash
hf download X-Square-Robot/wall-oss-0.5 \
  --revision b7dc23a82dbf70a65859c9fdd596703234597028 \
  --local-dir /path/to/wall-oss-0.5
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

部署需要训练好的 B601 任务 checkpoint。将完整目录放在 `<checkpoint_root>/<checkpoint_id>`，保留权重、`config.yml`、归一化参数、B601 manifest 及绑定的统计/划分文件，并确保模型和 processor 路径在推理主机上可读取。

已有任务 checkpoint 时直接进入[部署配置](#configuration)；需要训练新任务时完成下一节。

<a id="train"></a>

## 3. 训练任务模型（可选）

<a id="data"></a>

<details>
<summary>准备演示数据并训练 B601 模型</summary>

准备含元数据及图像/视频的 LeRobot 演示数据集。parquet 包含 `episode_index`、`frame_index`、`task_index`、`timestamp`、`observation.state`、`action`；state/action 每帧各 7 个值，前/侧视图与训练配置对应。

```bash
mkdir -p deployment/local
test -e deployment/local/train.yml || cp workspace/example/b601.yml deployment/local/train.yml
```

修改 `deployment/local/train.yml` 中的数据集 `root`、`repo_id`、相机映射、`model.config_path`、`model.processor_path`、`checkpoint.resume_from` 和 `checkpoint.save_path`。模型配置和初始权重取自 Wall-OSS，processor 使用 Qwen；保留 `model.pretrained_path: null`。训练 YAML 中填写实际路径，不能使用 shell 变量。

模板使用 32 步关节增量动作、训练 episode 统计量和 0.1 验证比例。填写数据和模型路径后，生成训练配置、统计量与划分文件；准备目录需为新目录：

```bash
python -m scripts.prepare_b601 --config deployment/local/train.yml --output-dir /path/to/new-b601-prepared
```

训练使用生成的 `delta.yml`，保留其引用的统计量和 `episode_split.json`。

### 训练与继续训练

```bash
NPROC=8
torchrun --nproc_per_node="$NPROC" -m scripts.train_b601 \
  --config /path/to/new-b601-prepared/delta.yml --log_to_file
```

将 `NPROC` 设为已分配的 GPU 数，并相应调整每卡 batch 和梯度累积。模板为 8 张 GPU、每卡 batch 8、累积 1，更新上限 16,000；在生成的配置中确认预算。checkpoint 保存到生成配置的 `checkpoint.save_path`。集群训练在已分配的计算节点执行。

继续被中断的训练时使用完整 checkpoint：

```bash
torchrun --nproc_per_node="$NPROC" -m scripts.train_b601 \
  --config /path/to/new-b601-prepared/delta.yml \
  --resume /path/to/complete-checkpoint --log_to_file
```

完整恢复需保留 `b601_training_state.json` 及绑定文件，沿用原数据、统计量、划分、GPU 数、batch 和优化器设置，并保持 `data.num_workers: 0`。训练完成后回到[部署配置](#configuration)，使用新 checkpoint 的身份与哈希。

</details>

<a id="configuration"></a>

## 4. 配置模型与机器人

首次部署时创建以下三个配置文件；已配置的机器沿用现有文件：

```bash
mkdir -p deployment/local
test -e deployment/local/profile.yml || cp deployment/configs/b601.example.yml deployment/local/profile.yml
test -e deployment/local/client.yml || cp deployment/configs/client.yml deployment/local/client.yml
test -e deployment/machines/standalone.local.yml || cp deployment/machines/standalone.example.yml deployment/machines/standalone.local.yml
```

修改新建的本机配置：

| 文件 | 必须填写的内容 |
| --- | --- |
| `deployment/local/profile.yml` | checkpoint 身份、权重 SHA256、归一化键；保留 `client_config: client.yml` 和 `padding: zero` |
| `deployment/local/client.yml` | `task` 填完整训练指令，`output_root` 填录像与结果保存目录的绝对路径 |
| `deployment/machines/standalone.local.yml` | `checkpoint_root`、CAN 接口、两路相机的稳定设备路径、相机角色和该机械臂既有的共享锁路径 |

例如 checkpoint 目录为 `my_experiment/3_800`，在完整配置的 `model` 节修改以下字段。`epoch` 从 0 开始，`step` 与目录名称一致：

```yaml
model:
  checkpoint_id: my_experiment/3_800
  epoch: 3
  step: 800
```

将下面计算的值填入 `model.checkpoint_sha256`，将 checkpoint 保存的归一化键填入 `model.norm_key`：

```bash
sha256sum /path/to/checkpoints/my_experiment/3_800/model.safetensors
```

机器人需完成标定、CAN 配置和相机访问权限设置。相机使用 640 × 480、30 Hz；将 `front_view`、`side_view` 与训练视图一一对应，确认后设置 `camera_mapping_verified: true`。模型动作规格保持 32 步、`[6, 1, 19]`，控制频率保持 30 Hz。

<a id="operate"></a>

## 5. 运行任务

<a id="task"></a>

### 5.1 选择任务

本仓库演示使用的 B601 三任务模型通过完整英文指令选择任务：

| 任务 | 传给模型的完整指令 |
| --- | --- |
| 方块入碗 | `Grasp the cube to the bowl` |
| 笔入盒 | `Grasp the pen to the box` |
| 方块入抽屉 | `Grasp the cube into the closed top drawer` |

下面按任务分别列出完整启动命令，选择目标任务对应的一条执行；`--task` 必须放在 `run` 前面。这三个任务共用同一个三任务 checkpoint，任务选择在同步与 RTC 模式下均生效。使用自行训练的模型时，填写该模型训练时使用的任务指令。

如果希望固定默认任务，在 `deployment/local/client.yml` 顶层设置：

```yaml
task: "Grasp the cube to the bowl"
```

省略命令行 `--task` 时使用这个默认值；传入 `--task` 时仅覆盖本次运行，不修改配置文件。网络推理时，任务在机器人端选择。

### 5.2 本地推理

在机器人主机激活模型环境，连接并开启机器人和相机，在有人值守的交互终端执行。清空动作及归位路径，保持物理急停可操作。Rerun 默认开启，本地桌面需解锁。

选择本轮任务，执行对应的一条命令：

**方块入碗 — `Grasp the cube to the bowl`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/standalone.local.yml \
  local -- --task "Grasp the cube to the bowl" run --duration 30
```

**笔入盒 — `Grasp the pen to the box`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/standalone.local.yml \
  local -- --task "Grasp the pen to the box" run --duration 30
```

**方块入抽屉 — `Grasp the cube into the closed top drawer`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/standalone.local.yml \
  local -- --task "Grasp the cube into the closed top drawer" run --duration 30
```

每条本地启动命令都会加载模型，自动检查相机、CAN 反馈与模型推理，通过后才使能电机、完成启动归位并执行任务。正常结束会归位并发送失能命令。等待归位完成后再开始下一轮。

在 `deployment/local/client.yml` 设置以下运行选项；RTC 设为开启后仍使用同一条任务启动命令：

| 字段 | 默认值 | 用途 |
| --- | --- | --- |
| `control.duration_s` | `300.0` | 每轮时长；上述命令用 `--duration 30` 临时改为 30 秒，去掉该参数即使用配置值 |
| `rtc.enabled` | `false` | 同步推理；改为 `true` 开启 RTC |
| `display_data` | `true` | 显示 Rerun；无桌面的 SSH 会话设为 `false` |
| `save_rollout` | `true` | 保存录像；与实时显示分别控制 |

终端按键：`p` 暂停/继续，`s` 标记成功并结束，`f` 标记失败并结束，`q` 或 Ctrl+C 中止。软件停止会进入收尾流程；需要立即停止运动时使用物理急停。

<a id="network"></a>

### 5.3 网络推理（GPU 在另一台主机上）

两端使用本仓库及相同的 `deployment/local/profile.yml` 模型身份，各自保留相邻的 `client.yml`。机器人端的客户端文件决定任务、RTC 和录像。创建机器配置并填写：

| 主机 | 模板 → 本机配置 | 设置 |
| --- | --- | --- |
| GPU 主机 | `deployment/machines/cluster.example.yml` → `deployment/machines/server.local.yml` | `checkpoint_root`；`server.host` 为机器人可访问的内网地址，`server.port` 为监听端口 |
| 机器人主机 | `deployment/machines/robot.example.yml` → `deployment/machines/robot.local.yml` | CAN、相机、共享锁，以及指向 GPU 主机的 `client.policy.uri`，格式为 `ws://<GPU_HOST>:<PORT>` |

在 GPU 主机激活模型环境并启动服务；集群上在已分配的计算节点执行，保留其 GPU 可见性：

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/server.local.yml server
```

服务就绪后，在机器人主机激活客户端环境并运行。操作按键、RTC 配置和结果位置与[本地运行](#operate)相同：

选择本轮任务，执行对应的一条命令：

**方块入碗 — `Grasp the cube to the bowl`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/robot.local.yml \
  client --task "Grasp the cube to the bowl" run --duration 30
```

**笔入盒 — `Grasp the pen to the box`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/robot.local.yml \
  client --task "Grasp the pen to the box" run --duration 30
```

**方块入抽屉 — `Grasp the cube into the closed top drawer`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/robot.local.yml \
  client --task "Grasp the cube into the closed top drawer" run --duration 30
```

<a id="results"></a>

## 6. 查看结果

每轮在 `output_root` 下创建独立目录，保存配置、`report.json`、事件和录像。`stop_outcome` 记录结束原因或人工成功/失败标记，`shutdown.homing` 记录归位结果。到时结束只代表时长用完，任务是否成功按现场观察或录像判断。
