# LIBERO：安装、训练与评估

[English](libero.md) · [总导航](../README.zh-CN.md)

本页适用于 Wall-OSS-0.5 和 LeRobot 格式的 LIBERO 演示。所有命令在 Linux 的仓库根目录执行。安装后可直接[评估已有 checkpoint](#evaluate)，也可按顺序完成训练。数据、权重和结果放在源码跟踪目录之外。

<a id="environment"></a>

## 1. 安装独立环境

需要 Conda、Python 3.10、支持的 NVIDIA GPU/驱动、主机 C/C++ 编译工具及匹配 Torch 2.10 的 CUDA 工具链。驱动本身不提供 `nvcc`。下面的依赖组合使用 LeRobot 0.4.4、Transformers 4.57.6、Hub 0.35.3；不要在这个环境叠加安装根目录的原生 `requirements.txt`。

```bash
conda env create -f environments/python310.yml -n wallx-libero
conda activate wallx-libero
conda install -c conda-forge ffmpeg=8 libgl libegl mesalib
python -m pip install -r environments/requirements/lerobot-model.txt -r requirements-libero.txt
python -m pip check
```

仅训练时可省略 `-r requirements-libero.txt`，仿真评估前再安装。如果尚未配置匹配的工具链，先按 Torch 的 CUDA 构建版本安装并设置 `CUDA_HOME`，再继续。

准备兼容工具链后，将 `CUDA_HOME` 设为其根目录（工具链安装在当前环境中时使用 `$CONDA_PREFIX`）。在集群保留调度器提供的 `CUDA_VISIBLE_DEVICES`，在已有算力分配内执行 GPU 构建：

```bash
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available()'
export FLASH_ATTN_CUDA_ARCHS=$(python -c 'import torch; print(torch.cuda.get_device_capability()[0] * 10)')
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install 'dmuon @ git+https://github.com/X-Square-Robot/dmuon.git@157f522b8570764c473b1a18939f4ebb36da5878'
MAX_JOBS=8 python -m pip install --no-build-isolation -e .
python -m pip check
python -c 'import wall_x, flash_attn; from wall_x.model.core.ops import _cuda_ext_bin; print(wall_x.__file__, flash_attn.__version__)'
```

应无依赖冲突，打印当前工程路径和 FlashAttention 2.8.3，且 CUDA 扩展能导入。编译内存不足时降低 `MAX_JOBS`。不同 Torch/CUDA ABI 使用独立源码构建或匹配的 wheel。

<a id="assets"></a>

## 2. 准备模型与模拟器资源

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

进行 rollout 评估时，获取以下版本的 LIBERO 源码和资源：

```bash
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
export WALLX_LIBERO_ROOT="$PWD/third_party/LIBERO"
python scripts/libero_resources.py
```

应能找到 `libero/libero/assets`、`bddl_files`、`init_files`。单独分发的资源按该 LIBERO 版本的说明获取并记录来源。支持 GPU 渲染时使用 `MUJOCO_GL=egl`，CPU 渲染使用 `osmesa`；策略推理仍需要 CUDA。除非显式设置 `CUDA_ID`，启动器会保留分配的 GPU 可见性。UUID/MIG 分配需选择兼容的 EGL 设备或使用 OSMesa。

<a id="data"></a>

## 3. 准备演示数据与训练配置

```bash
hf download lerobot/libero --repo-type dataset \
  --revision a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --local-dir /path/to/libero_all
mkdir -p workspace/local/libero
cp workspace/example/libero.yml workspace/local/libero/train.yml
```

修改复制模板中的全部占位路径。保留 7 维真实动作及虚拟通道 padding，按数据集核对相机映射。下面仅列路径和配置片段，其余内容继续使用模板：

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

`repo_id` 是保存的 normalizer 键，不是文件路径，不能含点号；`root` 用于选择目录，可以含点号。推理时沿用同一键。设选中的 episode 数为 $N$，训练比例为 $r$（`train_test_split`），则前 $\lfloor N r \rfloor$ 个用于训练，其余用于验证。

```bash
python scripts/compute_norm_stats.py --train_config workspace/local/libero/train.yml
```

应在配置指定位置生成含 `episode_split` 的统计 JSON，只有训练 episode 参与计算。episode 选择、顺序或划分比例改变后，重新计算统计量。

<a id="train"></a>

## 4. 训练与继续运行

模板使用 8 张 GPU × 每卡 16 样本 × 1 次累积，有效 batch 为 128；AdamW 学习率 `5e-5`、warmup 1,000 步、余弦调度 30,000 步、seed 10222、horizon 10、两路 256 × 256 图像。当前训练器也用 `num_training_steps` 限制更新数；[已报告对比](../README.zh-CN.md#libero-results)使用 25,000 次更新的 checkpoint。数据相机键为 `observation.images.image` 和 `observation.images.image2`，其他导出版本先核对 `meta/info.json`。

对照 Qwen 初始化时，保留 `model.config_path` 作为结构配置，将 `model.pretrained_path` 指向 Qwen 目录，设置 `checkpoint.resume_from: null`。采用 4 张 GPU × 每卡 16 × 累积 2 保持有效 batch 128。新训练使用 `split_mode: episode`；已报告实验使用 `legacy_frame` 和原有统计量。

```bash
torchrun --standalone --nproc_per_node=8 \
  -m wall_x.trainer.fsdp_trainer.train_fsdp \
  --config workspace/local/libero/train.yml
```

按分配的 GPU 调整进程数、batch、累积步数和 FSDP 配置。日志和 checkpoint 使用 `checkpoint.save_path`。`checkpoint.resume_from` 为权重文件时初始化新训练；继续运行时改为完整 checkpoint 目录，保留配置、normalizer 和资源后执行同一命令。checkpoint 根目录可能自动选择记录步数最大的子目录，复现评估时请指定具体 checkpoint。

旧训练继续运行时，保留原统计量，并显式设置 `data.lerobot_config.split_mode: legacy_frame`。该模式保留历史的 episode 前缀加 95/5 帧划分。改用 `episode` 需计算新统计量，再从权重文件初始化新训练。恢复会还原记录的进度并检查划分；复现时保持原有 worker 数量和分布式拓扑。

<a id="evaluate"></a>

## 5. 评估已训练的 checkpoint

准备已训练的 LIBERO checkpoint，以及对应的训练 YAML 和 normalizer。在模型环境先运行一个回合：

```bash
export WALLX_LIBERO_ROOT=/path/to/LIBERO
export MUJOCO_GL=egl
source scripts/libero_rendering.sh
python scripts/infer_libero.py --checkpoint-path /path/to/training_checkpoint \
  --task-suite-name libero_spatial --task-indices 0 --num-trials-per-task 1 \
  --num-workers 1 --log-dir /path/to/evaluation/single-episode
```

之后可选择 smoke（task 0、5 回合、1 个 worker）或全部四个标准套件：

```bash
WALLX_LIBERO_ROOT=/path/to/LIBERO SMOKE=1 \
  bash scripts/run_libero.sh /path/to/training_checkpoint
WALLX_LIBERO_ROOT=/path/to/LIBERO ALL_SUITES=1 NUM_TRIALS_PER_TASK=50 \
  LOG_DIR=/path/to/evaluation \
  bash scripts/run_libero.sh /path/to/training_checkpoint
```

显式 CLI 参数覆盖评估 YAML，未指定的参数保留 YAML 值。可在不加载模型和模拟器的情况下检查配置：

```bash
python scripts/infer_libero.py --config /path/to/eval.yml \
  --task-suite-name libero_goal --num-trials-per-task 50 --print-config
```

<a id="results"></a>

## 6. 查看输出与排错

每个套件生成 `report.json` 和回合日志/录像。成功率为 `successes / attempted`；`overall.attempted` 包含执行错误，执行错误同时计入 `overall.failed`。比较结果时记录代码/权重/统计量哈希、模拟器/数据版本、任务、回合数、种子、动作预算和渲染器。

| 现象 | 处理 |
| --- | --- |
| CUDA/编译器/扩展错误 | 对照 Torch CUDA、`nvcc`、`CUDA_HOME`，在选定环境重编译 |
| 缺少 `libGL`/EGL/OSMesa | 检查图形库和 `MUJOCO_GL`，在导入模拟器前完成渲染设置 |
| LIBERO 源码缺失或选错 | 设置 `WALLX_LIBERO_ROOT`，清除旧 `LIBERO_PATH`，重跑资源检查 |
| normalizer 缺失或划分冲突 | 使用保存的数据集键和匹配配置/统计量；仅在新训练时重算 |
| 权重选错 | 指定具体文件；目录选择优先 `pytorch_model_fsdp.bin`，其次 `model.safetensors` |
| 报告包含执行错误 | 先检查回合错误，再计算或比较 benchmark 分数 |
