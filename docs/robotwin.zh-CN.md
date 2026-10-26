# RoboTwin 2.0：安装、训练与评估

[English](robotwin.md) · [总导航](../README.zh-CN.md)

本页使用 Wall-OSS-0.5 Flow 模型、三路相机及双臂演示。命令在 Linux 仓库根目录运行，模型和模拟器使用独立环境。入门示例从两个任务的数据转换开始，完成短训练与策略评估。

<a id="model"></a>

## 1. 安装模型环境

准备 NVIDIA 驱动、C/C++ 编译工具和匹配 Torch 2.10 的工具链。LeRobot 依赖组合使用 Transformers 4.57.6、Hub 0.35.3；不要叠加安装根目录的原生 `requirements.txt`。

```bash
conda env create -f environments/python310.yml -n wallx-robotwin-model
conda activate wallx-robotwin-model
conda install -c conda-forge ffmpeg=8
conda install -c nvidia cuda-toolkit=12.8
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environments/requirements/lerobot-model.txt
python -m pip check
```

上述命令在当前环境安装 CUDA 12.8 工具链及 cu128 PyTorch。每次重新激活环境后，设置工具链路径；在集群保留调度器提供的 `CUDA_VISIBLE_DEVICES`，在已分配的 GPU 上执行构建：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda == "12.8"; assert torch.cuda.is_available()'
export FLASH_ATTN_CUDA_ARCHS=$(python -c 'import torch; print(torch.cuda.get_device_capability()[0] * 10)')
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install 'dmuon @ git+https://github.com/X-Square-Robot/dmuon.git@157f522b8570764c473b1a18939f4ebb36da5878'
MAX_JOBS=8 python -m pip install --no-build-isolation -e .
python -m pip check
python -c 'import wall_x, flash_attn; from wall_x.model.core.ops import _cuda_ext_bin; print(wall_x.__file__, flash_attn.__version__)'
```

应无依赖冲突，打印当前工程路径和 FlashAttention 2.8.3，且 CUDA 扩展能导入。编译内存不足时降低 `MAX_JOBS`。不同 Torch/CUDA ABI 使用独立源码构建或匹配的 wheel。

<a id="assets"></a>

## 2. 下载模型与源码依赖

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

```bash
bash scripts/setup_robotwin.sh
python -m pip install h5py
python -m pip install --no-deps -e third_party/XPolicyLab
python scripts/convert_robotwin.py --help
python scripts/train_robotwin.py --help
```

安装脚本仅获取源码：RoboTwin `2eeec322d95799f537cbfe5f291a8220d965ccb8`、cuRobo `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`、XPolicyLab `fa431ecd893ee706883e64fe5fe1464ec8cd928d`。已有目录版本不匹配时会拒绝继续。XPolicyLab 提供 HDF5 图像解码器；两条 help 命令应正常退出，不启动模拟器。

<a id="simulator"></a>

## 3. 安装模拟器与资源

```bash
conda env create -f environments/python310.yml -n wallx-robotwin-sim
conda activate wallx-robotwin-sim
conda install -c conda-forge libgl
conda install -c nvidia cuda-toolkit=12.1
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/requirements/robotwin-sim.txt
```

模拟器使用 CUDA 12.1 和 cu121 PyTorch。每次重新激活模拟器环境后设置工具链路径，再编译 cuRobo 并应用兼容补丁：

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda == "12.1"'
python -m pip install --no-build-isolation --no-deps -e third_party/RoboTwin/envs/curobo
python scripts/patch_robotwin_sim.py
python -m pip check
```

按[官方资源说明](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)下载 `objects`、`embodiments`、`background_texture`，放入 `third_party/RoboTwin/assets/`。embodiment 目录需要可写。随机化评估需要完整 unseen 纹理 `0.png` 至 `999.png`。

```bash
(cd third_party/RoboTwin && python script/update_embodiment_config_path.py)
python -c 'import sapien, mplib, curobo; print("Simulator imports OK")'
vulkaninfo --summary
```

应成功导入模拟器，Vulkan 应列出可用 GPU。SAPIEN 需要兼容的 NVIDIA/Vulkan 图形库；缺少标准 vendor 目录时，按实际驱动配置 `VK_ICD_FILENAMES`，必要时再设置 `__EGL_VENDOR_LIBRARY_FILENAMES`。MuJoCo/OSMesa 选项不能配置 SAPIEN。不要在模拟器环境安装 Wall-X 模型包。

<a id="data"></a>

## 4. 准备公开的双任务数据

使用 [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) 的固定版本 `3dc3b798668feb99ac61cc9086d84cbcc3d79186`。切回模型环境下载两个压缩包：

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

如缺少 `unzip`，先安装该系统工具。保留压缩包的全部元数据，转换前核对解压路径：

```text
raw/
  adjust_bottle/adjust_bottle/aloha_agilex/data/episode_0000000.hdf5
  adjust_bottle/adjust_bottle/aloha_agilex/seed.txt
  click_bell/click_bell/aloha_agilex/data/episode_0000000.hdf5
  click_bell/click_bell/aloha_agilex/seed.txt
```

```bash
python scripts/convert_robotwin.py \
  --raw-root /path/to/raw --output /path/to/robotwin_clean \
  --tasks adjust_bottle click_bell \
  --episodes-per-task 5 --validation-per-task 1 --seed 10222
mkdir -p workspace/local/robotwin2
cp workspace/example/robotwin.yml workspace/local/robotwin2/train.yml
```

输出目录必须尚不存在。应生成 10 个 episode（8 个训练、2 个验证），以及 `data/`、`videos/`、`meta/info.json`、`meta/episode_split.json`、`meta/robotwin_provenance.json`。采样频率读取 HDF5 元数据（本示例为 15 Hz），录像编码 fps 不等于数据采样频率。转换器保留已对齐的动作目标和完整 episode。

<a id="train"></a>

## 5. 配置统计量并训练

修改复制模板中的全部路径，包括 `model.config_path`、`processor_path`、`pretrained_path`、初始化用的 `checkpoint.resume_from` 和输出 `checkpoint.save_path`。保留 `adapter: robotwin`、DOF 布局、224 × 224 等比例补边图像、归一化策略、32 步预测和 16 步执行。设置：

```yaml
data:
  lerobot_config:
    repo_id: robotwin_clean
    root: /path/to/robotwin_clean
  norm_stats_path: /path/to/robotwin_clean_norm_stats.json
  episode_split_path: /path/to/robotwin_clean/meta/episode_split.json
```

该片段用于补充模板。`repo_id` 是独立于路径的稳定 normalizer 键，不含点号和斜杠。仅使用训练 episode 计算统计量，再检查初始化：

```bash
python scripts/compute_robotwin_norm_stats.py --train_config workspace/local/robotwin2/train.yml
python -m wall_x.robotwin.preflight --config workspace/local/robotwin2/train.yml
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_robotwin.py --config workspace/local/robotwin2/train.yml --log_to_file
```

统计命令写入 `data.norm_stats_path`；预检应显示 `FSDPTrainer._resume_from_single_file` 和正确初始权重。小模板使用 4 张 GPU、有效 batch 16、最多 200 次更新；按分配同时调整进程数和 batch。日志和 checkpoint 写入 `checkpoint.save_path`。使用 RoboTwin 专用训练入口，并设置 `adapter: robotwin`。

<a id="resume"></a>

## 6. 恢复训练

将 `checkpoint.resume_from` 设为完整 checkpoint 目录后执行同一训练命令。保留数据、划分、统计量、GPU 数量、batch/累积、优化器和调度器。`checkpoint.resume_model` 不会被加载。单独的 `.safetensors` 文件会开始新的训练轨迹。

完整保留 `model.safetensors`、`optimizer.pt`、`scheduler.pt`、`rng_state.pt`、配置/normalizer 及新格式 `robotwin_resume/` 元数据。`max_training_steps` 是累计更新上限，调度器和 epoch 限制可能更早结束训练。新 checkpoint 后缀表示已完成更新数；旧 checkpoint 保留原名称和元数据，由兼容加载器解释。

<a id="evaluate"></a>

## 7. 评估与断点续评

在模型环境运行，并传入模拟器环境实际的 Python 路径：

```bash
python scripts/infer_robotwin.py \
  --checkpoint-path /path/to/complete/robotwin-checkpoint \
  --norm-key robotwin_clean --robotwin-root /path/to/RoboTwin \
  --sim-python /path/to/wallx-robotwin-sim/bin/python \
  --output /path/to/evaluation/clean-smoke \
  --tasks adjust_bottle click_bell --task-config demo_clean \
  --episodes 1 --num-workers 1 --execute-steps 16 \
  --instruction-type seen --instruction-count 10 --seed 10222 --dry-run
```

Dry run 检查保存配置与 normalizer 兼容性，不加载 GPU 权重、不启动模拟器。去掉 `--dry-run` 后每个任务运行一回合，应生成 `results.json` 和 MP4 录像。报告记录每回合的任务结果。父进程加载一个模型，每个 worker 运行一个模拟器进程。

完整评估需增大 `--episodes`，记录任务、数据/checkpoint 哈希、模型与场景种子、执行长度及指令设置。`demo_randomized` 使用另一个输出目录。成功和策略失败都计入分母，专家无效场景单列，基础设施错误会中止运行。默认每任务 50 回合、场景种子 100000、模型种子 10222。始终显式指定所需任务和训练 normalizer 键。

固定场景/指令面板时，仅选择一个任务并添加 `--scene-manifest /path/to/panel.json`。首轮生成清单，其他 checkpoint 在独立输出目录复用。续评使用原参数加 `--resume`；保持 checkpoint、协议、worker 数及划分不变，未完成 batch 会重跑。回合数可增加，但不能小于已完成的分母。

<a id="reported-settings"></a>

### 五任务结果对应的设置

[已报告结果](../README.zh-CN.md#robotwin-results)来自独立的五任务训练。`workspace/example/robotwin.yml` 是 200 次更新的冒烟模板；`workspace/example/robotwin_full_v1.yml` 及其兼容副本面向全部已准备任务、25,000 次更新。五任务训练使用下述配置。

开始新的五任务训练时，复制全量模板，仅选择表中的五个任务，每任务 50 条 clean + 500 条 randomized 训练 episode，另加独立开发 episode。可用全量转换器的 `--folders` 选择源目录，或在已有准备数据上设置 `data.lerobot_config.episodes` 及对应的 `data.episode_split_path`。将 `data.lerobot_config.expected_tasks` 设为这五个任务 ID，episode ID 从当前转换数据中选择。保持每卡 batch 32 × 8 张 GPU、累积 1、AdamW `5e-5`、warmup 500、seed 10222，将 `hyperparams.scheduler.num_training_steps: 30000`。显式关闭 action/state clipping，保持三路 224 × 224 图像和 horizon 32，为新选择重新计算仅训练集统计量。

已报告训练从先前 checkpoint 续训，在 30,000 次更新后使用保存的共用 normalizer 评估，最终配置关闭 action/state clipping。评估已有 checkpoint 时，使用它保存的配置与统计量。

评估使用 `demo_clean` 和 `demo_randomized`、unseen 指令（`--instruction-type unseen --instruction-count 100`），从场景种子 100000 开始选择每任务、每种设置各 100 个专家有效场景，设置 `--execute-steps 16`。保存的面板拆成 10 个分片、每片 10 回合，模型种子为 10222 + 分片编号（0–9）。回放时复用原面板、分片种子和权重。新评估设置 `--episodes 100`。运行 README 表中的全部五个任务 ID，每种设置写入独立输出目录，失败保留在分母中。

<a id="full-data"></a>

## 8. 全量数据路线

全量模板 `workspace/example/robotwin_full_v1.yml` 使用 8 张 GPU、有效 batch 256 和 25,000 次更新。数据为 [robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot) 的固定版本 `595d4acc7e026aa57afba218f4170c6d79e39103`：50 个任务、27,500 条训练 episode、6,077,247 帧，文件时间基准为 50 Hz。首次验证流程可使用上面的双任务路线。

**下载并生成清单。** 以下命令下载 LeRobot v2.1 源目录，并从 `meta/info.json` 与 `meta/episodes.jsonl` 生成转换器的 audit JSON。清单检查 clean/randomized 任务配对、连续 episode 编号、每任务 50/500 条及总帧数，并保存元数据 SHA256。源数据约 443 GB；使用 `copy` 转换时另留一份视频副本的空间。

```bash
conda activate wallx-robotwin-model
hf download robbyant/robotwin-clean-and-aug-lerobot \
  --repo-type dataset --revision 595d4acc7e026aa57afba218f4170c6d79e39103 \
  --include 'lerobot_robotwin_eef_*/*' --local-dir /path/to/source-lerobot
python -m wall_x.robotwin.audit_source \
  --source-root /path/to/source-lerobot --output /path/to/frame-count-audit.json
```

**采集独立开发集。** 完成第 3 节模拟器与资源安装后，在已分配的 GPU 上运行以下命令。每个任务从独立种子区间采集一条专家成功的 `demo_clean` episode；种子从 500000 起，与本指南评测的 100000 起始区间分开。循环遇到采集错误即停止，可根据输出继续未完成任务。

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

每个任务输出 `complete.json`（指令、成功种子、拒绝种子与帧数）、`vectors.npz` 和 `face_view.mp4`、`left_wrist_view.mp4`、`right_wrist_view.mp4`。视频为 640 × 480、50 Hz 文件时间基准。采集沿用专家的 `save_freq=15`：以当前观测配下一采样姿态作为动作目标，末帧不单独作为样本；转换器不再次平移动作。50 Hz 是与源数据一致的文件时间基准，不代表物理仿真中的采样频率。每个任务的输出目录必须是新目录，`complete.json` 写入后才表示该条采集完成。

**转换与训练。** 回到模型环境执行：

```bash
conda activate wallx-robotwin-model
python -m wall_x.robotwin.convert_lerobot \
  --source-root /path/to/source-lerobot --audit /path/to/frame-count-audit.json \
  --dev-root /path/to/independent-dev --video-storage copy \
  --output /path/to/prepared-full-dataset
```

应得到 27,500 条训练 episode 和 50 条独立 validation episode，以及 `meta/episode_split.json`、`meta/robotwin_provenance.json`、`meta/video_sources.json`。使用新输出目录，将全量模板中的数据路径和 split 路径指向该目录，然后按第 5 节重新计算训练集统计量并训练。保留视频来源清单；`copy` 保留输入，`move` 会移动同一文件系统中的源视频，重建已移动的数据可传 `--video-index`。

仅准备五任务时，为 audit 命令添加 `--tasks open_microwave click_bell stack_blocks_three place_shoe put_object_cabinet`，采集循环会读取这五个任务。转换命令还需通过 `--folders` 传入该 audit 的全部 `rows[].folder`（五任务各两个目录），以启用子集模式；应得到 2,750 条训练及 5 条 validation episode。训练与评测参数使用[五任务设置](#reported-settings)。

```bash
mapfile -t FOLDERS < <(python -c 'import json; print("\n".join(r["folder"] for r in json.load(open("/path/to/frame-count-audit.json"))["rows"]))')
python -m wall_x.robotwin.convert_lerobot \
  --source-root /path/to/source-lerobot --audit /path/to/frame-count-audit.json \
  --dev-root /path/to/independent-dev --video-storage copy \
  --folders "${FOLDERS[@]}" --output /path/to/prepared-five-task-dataset
```

<a id="troubleshooting"></a>

## 9. 排错

| 现象 | 处理 |
| --- | --- |
| HDF5 解码器或文件缺失 | 确认 XPolicyLab 安装在模型环境，核对实际解压层级 |
| 视频损坏或帧数不符 | 训练前检查源视频解码和帧数；修复或更换损坏的源视频并保留哈希 |
| 模拟器导入/渲染错误 | 核对 `--sim-python`、固定版本、OpenGL/Vulkan 库和完整资源 |
| normalizer/布局不匹配 | 使用 checkpoint 的键、配置、统计量，保留其 clipping 和 padding 策略 |
| 拒绝恢复 | 恢复原 checkpoint、划分及训练/评估协议 |
| 显存不足 | 按实际分配降低 batch/worker 等配置；模拟器 worker 不分摊模型 |
