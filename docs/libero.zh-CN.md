# LIBERO：安装、训练与评估

[English](libero.md) · [总导航](../README.zh-CN.md)

所有命令在 Linux 仓库根目录执行，将 `/path/to/...` 替换为实际路径。

<a id="environment"></a>

## 1. 安装环境

准备 Conda、NVIDIA GPU 驱动和 C/C++ 编译工具。在将要运行模型的 GPU 上安装；集群使用已分配的计算节点。

```bash
conda env create -f environments/python310.yml -n wallx-libero
conda activate wallx-libero
conda install --override-channels -c conda-forge ffmpeg=8 libgl libegl mesalib=25.0.2
conda install --override-channels -c nvidia -c conda-forge cuda-toolkit=12.8
export CC=/usr/bin/gcc CXX=/usr/bin/g++
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environments/requirements/lerobot-model.txt -r requirements-libero.txt
```

在同一环境中编译模型依赖：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_ATTN_CUDA_ARCHS=$(python -c 'import torch; print(torch.cuda.get_device_capability()[0] * 10)')
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install 'dmuon @ git+https://github.com/X-Square-Robot/dmuon.git@157f522b8570764c473b1a18939f4ebb36da5878'
MAX_JOBS=8 python -m pip install --no-build-isolation -e .
```

后续打开终端先执行 `conda activate wallx-libero`。以下步骤依次准备资源、训练模型并评估；已有任务 checkpoint 时，完成第 2 节后直接进入[评估](#evaluate)。

<a id="assets"></a>

## 2. 准备模型与模拟器

下载 Wall-OSS-0.5 基础权重和 Qwen processor：

```bash
hf download X-Square-Robot/wall-oss-0.5 \
  --revision b7dc23a82dbf70a65859c9fdd596703234597028 \
  --local-dir /path/to/wall-oss-0.5
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

获取评估所需的 LIBERO 源码和资源：

```bash
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
export WALLX_LIBERO_ROOT="$PWD/third_party/LIBERO"
python scripts/libero_resources.py
```

资源目录应包含 `libero/libero/assets`、`bddl_files` 和 `init_files`。评估使用训练产出的任务 checkpoint，保留其中的训练配置和归一化参数。

<a id="data"></a>

## 3. 准备数据与训练配置

```bash
hf download lerobot/libero --repo-type dataset \
  --revision a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --local-dir /path/to/libero_all
mkdir -p workspace/local/libero
cp workspace/example/libero.yml workspace/local/libero/train.yml
```

修改复制模板中的路径，其他参数沿用模板。下面是需要填写的部分：

```yaml
model:
  config_path: /path/to/wall-oss-0.5/config.json
  processor_path: /path/to/Qwen2.5-VL-3B-Instruct
  pretrained_path: null
data:
  lerobot_config:
    repo_id: libero_all
    root: /path/to/libero_all
    split_mode: episode
  train_test_split: 0.95
  norm_stats_path: /path/to/libero_norm_stats.json
checkpoint:
  save_path: /path/to/libero_checkpoints
  resume_from: /path/to/wall-oss-0.5/model.safetensors
```

`repo_id` 是归一化参数使用的数据集键，不能含点号；推理时沿用同一键。模板使用 7 维动作、两路图像 `observation.images.image` 和 `observation.images.image2`，按 episode 划分 95% 训练集和 5% 验证集。

```bash
python scripts/compute_norm_stats.py --train_config workspace/local/libero/train.yml
```

统计量写入 `data.norm_stats_path`，仅使用训练 episode。更换数据或划分后重新计算。

<a id="train"></a>

## 4. 训练模型

模板使用 8 张 GPU、每卡 batch 16、累积 1，有效 batch 为 128。按分配的 GPU 调整进程数、batch、累积步数和 FSDP 配置，然后启动：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m wall_x.trainer.fsdp_trainer.train_fsdp \
  --config workspace/local/libero/train.yml
```

日志和 checkpoint 写入 `checkpoint.save_path`。继续训练时，将 `checkpoint.resume_from` 改为完整 checkpoint 目录，保留配置、归一化参数和数据，再执行同一命令。权重文件用于初始化新训练。

继续旧训练时，保持 `data.lerobot_config.split_mode: legacy_frame` 和原统计量，以及原有 worker 数和分布式拓扑；新训练使用 `episode` 划分并计算新统计量。

### 复现 README 中的对比

两种初始化均使用有效 batch 128、AdamW `5e-5`、warmup 1,000、30,000 步余弦调度、seed 10222、horizon 10 和两路 256 × 256 图像，在 25,000 次更新时评估。`num_training_steps` 同时限制训练更新数。报告实验使用 `legacy_frame` 和原有统计量。

Qwen 初始化保留 `model.config_path` 作为结构配置，将 `model.pretrained_path` 指向 Qwen 目录，设置 `checkpoint.resume_from: null`，使用 4 张 GPU × 每卡 batch 16 × 累积 2。

<a id="evaluate"></a>

## 5. 评估模型

激活模型环境，指定一个具体任务 checkpoint，运行四个标准套件，每任务评估 50 回合：

```bash
conda activate wallx-libero
WALLX_LIBERO_ROOT="$PWD/third_party/LIBERO" MUJOCO_GL=egl \
  ALL_SUITES=1 NUM_TRIALS_PER_TASK=50 LOG_DIR=/path/to/evaluation \
  bash scripts/run_libero.sh /path/to/training_checkpoint
```

使用 CPU 渲染时将 `MUJOCO_GL=egl` 改为 `MUJOCO_GL=osmesa NUMBA_DISABLE_JIT=1`，模型推理仍使用 GPU。评估启动器沿用已分配的 GPU。

<a id="results"></a>

## 6. 查看结果

每个套件的结果保存在 `/path/to/evaluation/<suite>/`，包括 `report.json`、终端日志和回合记录。成功率为 `successes / attempted`；`overall.attempted` 包含执行错误，错误同时计入 `overall.failed`。
