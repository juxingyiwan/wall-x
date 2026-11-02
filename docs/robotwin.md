# RoboTwin 2.0: installation, training and evaluation

[中文](robotwin.zh-CN.md) · [Home](../README.md)

Run all commands from the repository root on Linux and replace `/path/to/...` with your paths.

<a id="environment"></a>

## 1. Install the environments

Install Conda, an NVIDIA GPU driver, C/C++ build tools and the unzip system utility. Use separate model and simulator environments; the simulator requires NVIDIA/Vulkan graphics support.

<a id="model"></a>

### 1.1 Model environment

Build on the GPU that will run the model; use an allocated compute node on a cluster:

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

Install the RoboTwin, cuRobo and data-conversion source dependencies:

```bash
bash scripts/setup_robotwin.sh
python -m pip install h5py
python -m pip install --no-deps -e third_party/XPolicyLab
```

The setup script pins RoboTwin to `2eeec322d95799f537cbfe5f291a8220d965ccb8`, cuRobo to `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` and XPolicyLab to `fa431ecd893ee706883e64fe5fe1464ec8cd928d`.

<a id="simulator"></a>

### 1.2 Simulator environment

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

Use the model environment for data conversion, training and the evaluation entry point. Evaluation starts simulator processes in the simulator environment.

<a id="assets"></a>

## 2. Prepare model and simulator resources

Switch to the model environment and download the Wall-OSS-0.5 foundation weights and Qwen processor:

```bash
conda activate wallx-robotwin-model
hf download X-Square-Robot/wall-oss-0.5 \
  --revision b7dc23a82dbf70a65859c9fdd596703234597028 \
  --local-dir /path/to/wall-oss-0.5
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

Download `objects`, `embodiments` and `background_texture` following the [official asset instructions](https://robotwin-platform.github.io/doc/usage/robotwin-install.html), and place them in `third_party/RoboTwin/assets/`. Randomized evaluation requires the complete unseen texture set from `0.png` to `999.png`. Update robot paths in the writable asset directory:

```bash
conda activate wallx-robotwin-sim
(cd third_party/RoboTwin && python script/update_embodiment_config_path.py)
```

With a trained task checkpoint, retain its configuration and normalizers and continue to [evaluation](#evaluate).

<a id="data"></a>

## 3. Prepare training data

### 3.1 Two-task starter dataset

Download the pinned two-task dataset from [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0):

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

Keep all archive files and the directory structure:

```text
raw/
  adjust_bottle/adjust_bottle/aloha_agilex/data/episode_0000000.hdf5
  adjust_bottle/adjust_bottle/aloha_agilex/seed.txt
  click_bell/click_bell/aloha_agilex/data/episode_0000000.hdf5
  click_bell/click_bell/aloha_agilex/seed.txt
```

Convert into a new directory and copy the training template:

```bash
python scripts/convert_robotwin.py \
  --raw-root /path/to/raw --output /path/to/robotwin_clean \
  --tasks adjust_bottle click_bell \
  --episodes-per-task 5 --validation-per-task 1 --seed 10222
mkdir -p workspace/local/robotwin2
cp workspace/example/robotwin.yml workspace/local/robotwin2/train.yml
```

This creates 10 episodes (8 training, 2 validation), data, videos and `meta/episode_split.json`. Continue to [configuration and training](#train), or use the alternative below for five-task or full-data training.

<a id="full-data"></a>

<details>
<summary>Optional: five-task / full dataset preparation</summary>

### 3.2 Five-task / full dataset

The full source is [robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot): 50 tasks, 27,500 training episodes and 6,077,247 frames. Allow about 443 GB for the source and space for a second video copy during conversion.

**Download and generate the conversion manifest.** The converter reads this manifest for tasks, episodes and frame counts:

```bash
conda activate wallx-robotwin-model
hf download robbyant/robotwin-clean-and-aug-lerobot \
  --repo-type dataset --revision 595d4acc7e026aa57afba218f4170c6d79e39103 \
  --include 'lerobot_robotwin_eef_*/*' --local-dir /path/to/source-lerobot
python -m wall_x.robotwin.audit_source \
  --source-root /path/to/source-lerobot --output /path/to/frame-count-audit.json
```

For five-task preparation, add `--tasks open_microwave click_bell stack_blocks_three place_shoe put_object_cabinet` to the manifest command above; the collection loop below reads the selected tasks.

**Collect independent validation data.** Use the installed simulator environment and an allocated GPU to collect one successful expert episode per task. Seeds start at 500000, separately from evaluation:

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

Use a new output directory for each task. Completed collection produces `complete.json`, `vectors.npz` and three videos, all used by the converter.

**Convert the dataset.** For the full dataset, run:

```bash
conda activate wallx-robotwin-model
python -m wall_x.robotwin.convert_lerobot \
  --source-root /path/to/source-lerobot --audit /path/to/frame-count-audit.json \
  --dev-root /path/to/independent-dev --video-storage copy \
  --output /path/to/prepared-full-dataset
```

For five-task data, use this command instead to pass the source folders from the manifest:

```bash
conda activate wallx-robotwin-model
mapfile -t FOLDERS < <(python -c 'import json; print("\n".join(r["folder"] for r in json.load(open("/path/to/frame-count-audit.json"))["rows"]))')
python -m wall_x.robotwin.convert_lerobot \
  --source-root /path/to/source-lerobot --audit /path/to/frame-count-audit.json \
  --dev-root /path/to/independent-dev --video-storage copy \
  --folders "${FOLDERS[@]}" --output /path/to/prepared-five-task-dataset
```

The full dataset contains 27,500 training and 50 validation episodes; the five-task dataset contains 2,750 training and 5 validation episodes. Output includes `meta/episode_split.json`, `meta/robotwin_provenance.json` and `meta/video_sources.json`. Copy the full-data template and continue to [configuration and training](#train):

```bash
mkdir -p workspace/local/robotwin2
cp workspace/example/robotwin_full_v1.yml workspace/local/robotwin2/train.yml
```

Point the data paths to the newly generated directory. The full-data template uses 8 GPUs, effective batch 256 and 25,000 updates. For five-task training, use the task list and training budget in the [reported experiment settings](#reported-settings).

</details>

<a id="train"></a>

## 4. Configure and train

Fill in these paths in `workspace/local/robotwin2/train.yml`. Use Wall-OSS for the model architecture and initial weights, Qwen for the processor, and retain `model.pretrained_path: null`:

| Field | Value |
| --- | --- |
| `model.config_path` | config.json in the Wall-OSS directory |
| `model.processor_path` | Qwen processor directory |
| `checkpoint.resume_from` | model.safetensors in the Wall-OSS directory |
| `checkpoint.save_path` | Training output directory |

Use these data settings for the two-task dataset. For full or five-task data, substitute its directory and dataset key:

```yaml
data:
  lerobot_config:
    repo_id: robotwin_clean
    root: /path/to/robotwin_clean
  norm_stats_path: /path/to/robotwin_clean_norm_stats.json
  episode_split_path: /path/to/robotwin_clean/meta/episode_split.json
```

`repo_id` is the normalizer key and must contain neither dots nor slashes; use the same key for evaluation. Retain the template’s `adapter: robotwin`, DOF layout, three 224 × 224 views, normalization and 32-step prediction horizon. Compute training-set statistics, then launch training:

```bash
conda activate wallx-robotwin-model
python scripts/compute_robotwin_norm_stats.py --train_config workspace/local/robotwin2/train.yml
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_robotwin.py --config workspace/local/robotwin2/train.yml --log_to_file
```

Statistics are written to `data.norm_stats_path`; logs and checkpoints go to `checkpoint.save_path`. The two-task template uses 4 GPUs, effective batch 16 and 200 updates. For the full-data template, change the command’s process count to 8. Adjust process count, batch size and accumulation together for your GPU allocation.

<a id="resume"></a>

### Resume training

Set `checkpoint.resume_from` to a complete checkpoint directory and rerun the command. Keep the weights, optimizer, scheduler, RNG state, configuration, normalizers and `robotwin_resume/`, with the original data split, GPU count, batch size and accumulation. `max_training_steps` is the cumulative update limit.

<a id="evaluate"></a>

## 5. Evaluate the model

Run in the model environment and supply the absolute path to the simulator environment’s Python. This evaluates the two-task checkpoint for 50 episodes per task:

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

For randomized evaluation, replace `--task-config demo_clean` with `--task-config demo_randomized` and use a separate output directory. For another model, match the task list and `--norm-key` to its training data.

To continue an interrupted evaluation, add `--resume` to the original command and retain the checkpoint, evaluation settings and worker count.

<a id="results"></a>

## 6. View results

The output directory contains `results.json` and MP4 recordings. Successes and policy failures both count toward the success-rate denominator. Expert-invalid scenes are listed separately; execution errors stop evaluation.

<a id="reported-settings"></a>

## 7. Reproduce the five-task README results

The [reported results](../README.md#robotwin-results) use a separate five-task run. `workspace/example/robotwin.yml` is a 200-update starter preset; `workspace/example/robotwin_full_v1.yml` and its compatibility copy target 25,000 updates on all prepared tasks. Use the settings below for the five-task run.

For a new five-task run, copy the full preset and select only the five listed tasks: 50 clean + 500 randomized training episodes per task and independent development episodes. Select source folders with the full-data converter's `--folders`, or set `data.lerobot_config.episodes` and a matching `data.episode_split_path` on an existing prepared dataset. Set `data.lerobot_config.expected_tasks` to those five task IDs and select episode IDs from the current converted dataset. Keep batch 32 × eight GPUs, accumulation 1, AdamW `5e-5`, warmup 500 and seed 10222; set `hyperparams.scheduler.num_training_steps: 30000`. Keep explicit action/state clipping disabled, three 224 × 224 views and horizon 32. Recompute training-only statistics for this new selection.

The reported run resumed from an earlier checkpoint and was evaluated at 30,000 updates using its saved shared normalizer. Its final configuration disables action/state clipping. Evaluate an existing checkpoint with its saved configuration and statistics.

Evaluation used `demo_clean` and `demo_randomized`, unseen instructions (`--instruction-type unseen --instruction-count 100`), 100 expert-valid scenes/task/condition selected from scene seed 100000, and `--execute-steps 16`. The stored panel was split into ten shards of ten episodes, with model seed 10222 + shard index (0–9). Replay the saved panels with their shard seeds and weights. For a new evaluation, set `--episodes 100`. Run all five task IDs listed in the README, write each condition to a separate output directory, and retain failures in the denominator.
