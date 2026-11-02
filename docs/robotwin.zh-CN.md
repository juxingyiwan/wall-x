# RoboTwin 2.0：安装、训练与评估

[English](robotwin.md) · [总导航](../README.zh-CN.md)

所有命令在 Linux 仓库根目录执行，将 `/path/to/...` 替换为实际路径。

<a id="environment"></a>

## 1. 安装环境

准备 Conda、NVIDIA GPU 驱动、C/C++ 编译工具及系统工具 unzip。模型与模拟器分别使用独立环境；模拟器需要 NVIDIA/Vulkan 图形支持。

<a id="model"></a>

### 1.1 模型环境

在将要运行模型的 GPU 上安装，集群使用已分配的计算节点：

```bash
conda env create -f environments/python310.yml -n wallx-robotwin-model
conda activate wallx-robotwin-model
conda install --override-channels -c conda-forge ffmpeg=8
conda install --override-channels -c nvidia -c conda-forge cuda-toolkit=12.8
export CC=/usr/bin/gcc CXX=/usr/bin/g++
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environments/requirements/lerobot-model.txt
```

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_ATTN_CUDA_ARCHS=$(python -c 'import torch; print(torch.cuda.get_device_capability()[0] * 10)')
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install 'dmuon @ git+https://github.com/X-Square-Robot/dmuon.git@157f522b8570764c473b1a18939f4ebb36da5878'
MAX_JOBS=8 python -m pip install --no-build-isolation -e .
```

安装 RoboTwin、cuRobo 与数据转换所需的源码依赖：

```bash
bash scripts/setup_robotwin.sh
python -m pip install h5py
python -m pip install --no-deps -e third_party/XPolicyLab
```

安装脚本固定 RoboTwin `2eeec322d95799f537cbfe5f291a8220d965ccb8`、cuRobo `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` 和 XPolicyLab `fa431ecd893ee706883e64fe5fe1464ec8cd928d`。

<a id="simulator"></a>

### 1.2 模拟器环境

```bash
conda env create -f environments/python310.yml -n wallx-robotwin-sim
conda activate wallx-robotwin-sim
conda install --override-channels -c conda-forge libgl libegl
conda install --override-channels -c nvidia -c conda-forge cuda-toolkit=12.1
export CC=/usr/bin/gcc CXX=/usr/bin/g++
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/requirements/robotwin-sim.txt
```

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install --no-build-isolation --no-deps -e third_party/RoboTwin/envs/curobo
python scripts/patch_robotwin_sim.py
```

模型环境用于数据转换、训练与评估入口；模拟器环境由评估程序通过独立进程调用。

<a id="assets"></a>

## 2. 准备模型与模拟器资源

切回模型环境，下载 Wall-OSS-0.5 基础权重和 Qwen processor：

```bash
conda activate wallx-robotwin-model
hf download X-Square-Robot/wall-oss-0.5 \
  --revision b7dc23a82dbf70a65859c9fdd596703234597028 \
  --local-dir /path/to/wall-oss-0.5
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

按[官方资源说明](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)下载 `objects`、`embodiments` 和 `background_texture`，放入 `third_party/RoboTwin/assets/`。随机化评估需要完整 unseen 纹理 `0.png` 至 `999.png`。在可写的资源目录中更新机器人路径：

```bash
conda activate wallx-robotwin-sim
(cd third_party/RoboTwin && python script/update_embodiment_config_path.py)
```

已有训练好的任务 checkpoint 时，保留它的配置和归一化参数，直接进入[评估](#evaluate)。

<a id="data"></a>

## 3. 准备训练数据

### 3.1 双任务入门数据

从 [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) 下载固定版本的两项任务：

```bash
conda activate wallx-robotwin-model
hf download TianxingChen/RoboTwin2.0 \
  dataset/adjust_bottle/demo_clean.zip dataset/click_bell/demo_clean.zip \
  --repo-type dataset --revision 3dc3b798668feb99ac61cc9086d84cbcc3d79186 \
  --local-dir /path/to/robotwin-downloads
mkdir -p /path/to/raw/adjust_bottle /path/to/raw/click_bell
unzip /path/to/robotwin-downloads/dataset/adjust_bottle/demo_clean.zip -d /path/to/raw/adjust_bottle
unzip /path/to/robotwin-downloads/dataset/click_bell/demo_clean.zip -d /path/to/raw/click_bell
```

保留压缩包的全部文件及目录层级：

```text
raw/
  adjust_bottle/adjust_bottle/aloha_agilex/data/episode_0000000.hdf5
  adjust_bottle/adjust_bottle/aloha_agilex/seed.txt
  click_bell/click_bell/aloha_agilex/data/episode_0000000.hdf5
  click_bell/click_bell/aloha_agilex/seed.txt
```

转换到新目录并复制训练模板：

```bash
python scripts/convert_robotwin.py \
  --raw-root /path/to/raw --output /path/to/robotwin_clean \
  --tasks adjust_bottle click_bell \
  --episodes-per-task 5 --validation-per-task 1 --seed 10222
mkdir -p workspace/local/robotwin2
cp workspace/example/robotwin.yml workspace/local/robotwin2/train.yml
```

生成 10 个 episode（8 个训练、2 个验证）及数据、视频和 `meta/episode_split.json`。接下来进入[配置与训练](#train)；五任务或全量数据使用下面的替代路线。

<a id="full-data"></a>

<details>
<summary>可选：五任务 / 全量数据准备</summary>

### 3.2 五任务 / 全量数据

全量来源为 [robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot)：50 个任务、27,500 条训练 episode、6,077,247 帧。源数据约 443 GB，转换时另留一份视频副本的空间。

**下载并生成转换清单。** 清单由转换器读取，记录任务、episode 和帧数：

```bash
conda activate wallx-robotwin-model
hf download robbyant/robotwin-clean-and-aug-lerobot \
  --repo-type dataset --revision 595d4acc7e026aa57afba218f4170c6d79e39103 \
  --include 'lerobot_robotwin_eef_*/*' --local-dir /path/to/source-lerobot
python -m wall_x.robotwin.audit_source \
  --source-root /path/to/source-lerobot --output /path/to/frame-count-audit.json
```

仅准备五任务时，在上面的清单命令中添加 `--tasks open_microwave click_bell stack_blocks_three place_shoe put_object_cabinet`；后续采集循环自动读取选定任务。

**采集独立验证数据。** 在已安装的模拟器环境和已分配的 GPU 上，为每个任务采集一条专家成功 episode；种子区间从 500000 开始，与评估区间分开：

```bash
conda activate wallx-robotwin-sim
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mapfile -t TASKS < <(python -c 'import json; print("\n".join(sorted({r["task"] for r in json.load(open("/path/to/frame-count-audit.json"))["rows"]})))')
task_index=0
for task in "${TASKS[@]}"; do
  python -m wall_x.robotwin.collect_development \
    --robotwin-root "$PWD/third_party/RoboTwin" --task "$task" \
    --seed "$((500000 + task_index * 1000))" --max-attempts 100 \
    --output "/path/to/independent-dev/$task" || break
  task_index=$((task_index + 1))
done
```

每个任务使用新输出目录。完成后生成 `complete.json`、`vectors.npz` 和三路视频；这些文件共同作为转换输入。

**转换数据。** 全量数据执行：

```bash
conda activate wallx-robotwin-model
python -m wall_x.robotwin.convert_lerobot \
  --source-root /path/to/source-lerobot --audit /path/to/frame-count-audit.json \
  --dev-root /path/to/independent-dev --video-storage copy \
  --output /path/to/prepared-full-dataset
```

五任务数据改用下面的命令，将清单中的源目录传给转换器：

```bash
conda activate wallx-robotwin-model
mapfile -t FOLDERS < <(python -c 'import json; print("\n".join(r["folder"] for r in json.load(open("/path/to/frame-count-audit.json"))["rows"]))')
python -m wall_x.robotwin.convert_lerobot \
  --source-root /path/to/source-lerobot --audit /path/to/frame-count-audit.json \
  --dev-root /path/to/independent-dev --video-storage copy \
  --folders "${FOLDERS[@]}" --output /path/to/prepared-five-task-dataset
```

全量数据生成 27,500 条训练和 50 条验证 episode；五任务数据生成 2,750 条训练和 5 条验证 episode。输出包含 `meta/episode_split.json`、`meta/robotwin_provenance.json` 和 `meta/video_sources.json`。复制全量模板，再进入[配置与训练](#train)：

```bash
mkdir -p workspace/local/robotwin2
cp workspace/example/robotwin_full_v1.yml workspace/local/robotwin2/train.yml
```

将数据路径指向刚生成的目录。全量模板使用 8 张 GPU、有效 batch 256 和 25,000 次更新；五任务训练按[报告实验设置](#reported-settings)修改任务列表和训练预算。

</details>

<a id="train"></a>

## 4. 配置并训练

在 `workspace/local/robotwin2/train.yml` 中填写以下路径。模型结构和初始权重取自 Wall-OSS，processor 使用 Qwen，保留 `model.pretrained_path: null`：

| 字段 | 内容 |
| --- | --- |
| `model.config_path` | Wall-OSS 目录中的 config.json |
| `model.processor_path` | Qwen processor 目录 |
| `checkpoint.resume_from` | Wall-OSS 目录中的 model.safetensors |
| `checkpoint.save_path` | 训练输出目录 |

双任务数据配置如下；全量或五任务数据替换为对应的目录和数据集键：

```yaml
data:
  lerobot_config:
    repo_id: robotwin_clean
    root: /path/to/robotwin_clean
  norm_stats_path: /path/to/robotwin_clean_norm_stats.json
  episode_split_path: /path/to/robotwin_clean/meta/episode_split.json
```

`repo_id` 是归一化参数使用的键，不含点号或斜杠；评估沿用同一键。保留模板的 `adapter: robotwin`、DOF 布局、三路 224 × 224 图像、归一化方式和 32 步预测。计算训练集统计量后启动训练：

```bash
conda activate wallx-robotwin-model
python scripts/compute_robotwin_norm_stats.py --train_config workspace/local/robotwin2/train.yml
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_robotwin.py --config workspace/local/robotwin2/train.yml --log_to_file
```

统计量写入 `data.norm_stats_path`，日志和 checkpoint 写入 `checkpoint.save_path`。双任务模板使用 4 张 GPU、有效 batch 16 和 200 次更新；全量模板将命令中的进程数改为 8。按 GPU 分配同时调整进程数、batch 和累积步数。

<a id="resume"></a>

### 继续训练

将 `checkpoint.resume_from` 改为完整 checkpoint 目录，再执行同一命令。保留权重、优化器、调度器、随机数状态、配置、归一化参数和 `robotwin_resume/`，沿用原数据划分、GPU 数、batch 和累积设置。`max_training_steps` 是累计更新上限。

<a id="evaluate"></a>

## 5. 评估模型

在模型环境中执行，填写模拟器环境的 Python 绝对路径。下面评估双任务 checkpoint，每任务 50 回合：

```bash
conda activate wallx-robotwin-model
python scripts/infer_robotwin.py \
  --checkpoint-path /path/to/complete/robotwin-checkpoint \
  --norm-key robotwin_clean --robotwin-root "$PWD/third_party/RoboTwin" \
  --sim-python /path/to/wallx-robotwin-sim/bin/python \
  --output /path/to/evaluation/clean \
  --tasks adjust_bottle click_bell --task-config demo_clean \
  --episodes 50 --num-workers 1 --execute-steps 16 \
  --instruction-type seen --instruction-count 10 --seed 10222
```

随机化评估将 `--task-config demo_clean` 改为 `--task-config demo_randomized`，并使用另一个输出目录。评估其他模型时，任务列表和 `--norm-key` 与其训练数据保持一致。

继续中断的评估时，在原命令后添加 `--resume`，保持 checkpoint、评估设置和 worker 数不变。

<a id="results"></a>

## 6. 查看结果

输出目录保存 `results.json` 和 MP4 录像。成功与策略失败均计入成功率分母；专家无效场景单列，运行错误会中止评估。

<a id="reported-settings"></a>

## 7. 复现 README 中的五任务结果

[已报告结果](../README.zh-CN.md#robotwin-results)来自独立的五任务训练。`workspace/example/robotwin.yml` 是 200 次更新的入门模板；`workspace/example/robotwin_full_v1.yml` 及其兼容副本面向全部已准备任务、25,000 次更新。五任务训练使用下述配置。

开始新的五任务训练时，复制全量模板，仅选择表中的五个任务，每任务 50 条 clean + 500 条 randomized 训练 episode，另加独立开发 episode。可用全量转换器的 `--folders` 选择源目录，或在已有准备数据上设置 `data.lerobot_config.episodes` 及对应的 `data.episode_split_path`。将 `data.lerobot_config.expected_tasks` 设为这五个任务 ID，episode ID 从当前转换数据中选择。保持每卡 batch 32 × 8 张 GPU、累积 1、AdamW `5e-5`、warmup 500、seed 10222，将 `hyperparams.scheduler.num_training_steps: 30000`。显式关闭 action/state clipping，保持三路 224 × 224 图像和 horizon 32，为新选择重新计算仅训练集统计量。

已报告训练从先前 checkpoint 续训，在 30,000 次更新后使用保存的共用 normalizer 评估，最终配置关闭 action/state clipping。评估已有 checkpoint 时，使用它保存的配置与统计量。

评估使用 `demo_clean` 和 `demo_randomized`、unseen 指令（`--instruction-type unseen --instruction-count 100`），从场景种子 100000 开始选择每任务、每种设置各 100 个专家有效场景，设置 `--execute-steps 16`。保存的面板拆成 10 个分片、每片 10 回合，模型种子为 10222 + 分片编号（0–9）。回放时复用原面板、分片种子和权重。新评估设置 `--episodes 100`。运行 README 表中的全部五个任务 ID，每种设置写入独立输出目录，失败保留在分母中。
