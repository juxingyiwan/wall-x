# RoboTwin 2.0: installation, training and evaluation

[中文](robotwin.zh-CN.md) · [Home](../README.md)

This guide uses the Wall-OSS-0.5 Flow model, three cameras and dual-arm demonstrations. Run commands from the repository root on Linux. Use separate model and simulator environments. The quick start converts two tasks, runs a short training job and evaluates the resulting policy.

<a id="model"></a>

## 1. Install the model environment

Provide an NVIDIA driver, C/C++ build tools and a toolkit matching Torch 2.10. The LeRobot profile selects Transformers 4.57.6 and Hub 0.35.3; do not install the native root `requirements.txt` into it.

```bash
conda env create -f environments/python310.yml -n wallx-robotwin-model
conda activate wallx-robotwin-model
conda install -c conda-forge ffmpeg=8
python -m pip install -r environments/requirements/lerobot-model.txt
python -m pip check
```

With a compatible toolkit available, set `CUDA_HOME` to its root (use `$CONDA_PREFIX` for a toolkit installed in this environment). Keep scheduler-provided `CUDA_VISIBLE_DEVICES` on a cluster. Run the GPU build inside an allocation:

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

Expected: no broken requirements; this checkout and FlashAttention 2.8.3 are printed, and the CUDA extension imports. Lower `MAX_JOBS` if compilation exhausts memory. Use separate source builds or wheels for different Torch/CUDA ABIs.

<a id="assets"></a>

## 2. Download model and source dependencies

Download the following foundation weights and processor in the model environment to initialize training. Evaluation uses a trained task checkpoint with its saved configuration and normalizers.

```bash
hf download X-Square-Robot/wall-oss-0.5 \
  --revision b7dc23a82dbf70a65859c9fdd596703234597028 \
  --local-dir /path/to/wall-oss-0.5
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

Expected: Wall-OSS `config.json` and `model.safetensors`, plus the Qwen processor/tokenizer files. Set `model.config_path` to the Wall-OSS JSON and `model.processor_path` to the Qwen directory. When `checkpoint.resume_from` supplies complete Wall-OSS weights, keep `model.pretrained_path: null`.

```bash
bash scripts/setup_robotwin.sh
python -m pip install h5py
python -m pip install --no-deps -e third_party/XPolicyLab
python scripts/convert_robotwin.py --help
python scripts/train_robotwin.py --help
```

The setup script fetches source only: RoboTwin `2eeec322d95799f537cbfe5f291a8220d965ccb8`, cuRobo `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`, XPolicyLab `fa431ecd893ee706883e64fe5fe1464ec8cd928d`. It rejects an existing checkout at another revision. XPolicyLab provides the HDF5 image decoder; the two help commands should exit successfully without starting a simulator.

<a id="simulator"></a>

## 3. Install the simulator and assets

```bash
conda env create -f environments/python310.yml -n wallx-robotwin-sim
conda activate wallx-robotwin-sim
conda install -c conda-forge libgl
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/requirements/robotwin-sim.txt
```

Set `CUDA_HOME` to a locally provisioned toolkit compatible with this cu121 Torch build, then compile cuRobo and apply the version-checked compatibility patch:

```bash
nvcc --version
python -m pip install --no-build-isolation --no-deps -e third_party/RoboTwin/envs/curobo
python scripts/patch_robotwin_sim.py
python -m pip check
```

Download `objects`, `embodiments` and `background_texture` using the [official asset instructions](https://robotwin-platform.github.io/doc/usage/robotwin-install.html), placing them under `third_party/RoboTwin/assets/`. The embodiment directory must be writable. Randomized evaluation requires all unseen textures `0.png` through `999.png`.

```bash
(cd third_party/RoboTwin && python script/update_embodiment_config_path.py)
python -c 'import sapien, mplib, curobo; print("Simulator imports OK")'
vulkaninfo --summary
```

Expected: simulator imports succeed and Vulkan lists a usable GPU. SAPIEN requires compatible NVIDIA/Vulkan graphics libraries. Where vendor directories are unavailable, configure `VK_ICD_FILENAMES` and, if needed, `__EGL_VENDOR_LIBRARY_FILENAMES` for the actual driver. MuJoCo/OSMesa options do not configure SAPIEN. Do not install the Wall-X model package in this simulator environment.

<a id="data"></a>

## 4. Prepare the public two-task dataset

Use [TianxingChen/RoboTwin2.0](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`. In the model environment, download both archives:

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

Install the system `unzip` utility if needed. Retain all archive metadata. Verify these resulting paths before conversion:

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

The output directory must be new. Expected: ten episodes (eight training, two validation), `data/`, `videos/`, `meta/info.json`, `meta/episode_split.json` and `meta/robotwin_provenance.json`. Source timing is read from the HDF5 metadata (15 Hz in this example); video encoding fps is not the dataset sampling frequency. The converter preserves already aligned action targets and whole episodes.

<a id="train"></a>

## 5. Configure statistics and train

Edit every path in the copied template. Set `model.config_path`, `processor_path`, `pretrained_path`, initial `checkpoint.resume_from`, and `checkpoint.save_path`. Keep `adapter: robotwin`, the DOF layout, 224 × 224 images with aspect-preserving padding, normalization policy, 32 predicted actions and 16 executed actions. Set:

```yaml
data:
  lerobot_config:
    repo_id: robotwin_clean
    root: /path/to/robotwin_clean
  norm_stats_path: /path/to/robotwin_clean_norm_stats.json
  episode_split_path: /path/to/robotwin_clean/meta/episode_split.json
```

This excerpt supplements the template. `repo_id` is a stable normalizer key, separate from the path; avoid dots and slashes. Compute statistics only from training episodes, then check initialization:

```bash
python scripts/compute_robotwin_norm_stats.py --train_config workspace/local/robotwin2/train.yml
python -m wall_x.robotwin.preflight --config workspace/local/robotwin2/train.yml
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/train_robotwin.py --config workspace/local/robotwin2/train.yml --log_to_file
```

The statistics command writes `data.norm_stats_path`; preflight must show `FSDPTrainer._resume_from_single_file` and the intended initial weights. The small template uses four GPUs, effective batch 16 and a 200-update ceiling. Adapt process count and batch settings together to your allocation. Logs/checkpoints go to `checkpoint.save_path`. Use the dedicated RoboTwin training entry with `adapter: robotwin`.

<a id="resume"></a>

## 6. Resume training

Set `checkpoint.resume_from` to a complete checkpoint directory and rerun the same training command. Preserve dataset, split, statistics, GPU count, batch/accumulation, optimizer and scheduler. `checkpoint.resume_model` is not loaded. A single `.safetensors` file starts a new training trajectory.

Keep `model.safetensors`, `optimizer.pt`, `scheduler.pt`, `rng_state.pt`, saved config/normalizers and new-format `robotwin_resume/` metadata together. `max_training_steps` is a cumulative update cap; scheduler and epoch limits can stop earlier. New checkpoint suffixes count completed updates; retain older checkpoint names/metadata for their compatibility loader.

<a id="evaluate"></a>

## 7. Evaluate and continue an evaluation

Run in the model environment; pass the simulator environment’s actual Python path:

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

The dry run checks saved configuration and normalizer compatibility without loading GPU weights or starting the simulator. Remove `--dry-run` for one episode per task. Expected: `results.json` and rollout MP4 files. Read task success in the report. The parent loads one model and each worker runs a simulator process.

For a full evaluation, increase `--episodes` and record task IDs, data/checkpoint hashes, model seed, scene seed, execution length and instruction settings. Use a different output directory for `demo_randomized`. Successes and policy failures count in the denominator; expert-invalid scenes are separate; infrastructure errors stop the run. Defaults are 50 episodes/task, scene seed 100000 and model seed 10222. Always pass your own tasks and training normalizer key.

Add `--scene-manifest /path/to/panel.json` with exactly one task for a fixed scene/instruction panel. The first run records it; other checkpoints reuse it in separate output directories. Resume with the original arguments and `--resume`. Keep checkpoint, protocol, workers and partitioning unchanged; an unfinished batch reruns. Episode counts may grow but cannot fall below the completed denominator.

<a id="reported-settings"></a>

### Settings behind the five-task results

The [reported results](../README.md#robotwin-results) use a separate five-task run. `workspace/example/robotwin.yml` is a 200-update smoke preset; `workspace/example/robotwin_full_v1.yml` and its compatibility copy target 25,000 updates on all prepared tasks. Use the settings below for the five-task run.

For a new five-task run, copy the full preset and select only the five listed tasks: 50 clean + 500 randomized training episodes per task and independent development episodes. Select source folders with the full-data converter's `--folders`, or set `data.lerobot_config.episodes` and a matching `data.episode_split_path` on an existing prepared dataset. Set `data.lerobot_config.expected_tasks` to those five task IDs and select episode IDs from the current converted dataset. Keep batch 32 × eight GPUs, accumulation 1, AdamW `5e-5`, warmup 500 and seed 10222; set `hyperparams.scheduler.num_training_steps: 30000`. Keep explicit action/state clipping disabled, three 224 × 224 views and horizon 32. Recompute training-only statistics for this new selection.

The reported run resumed from an earlier checkpoint and was evaluated at 30,000 updates using its saved shared normalizer. Its final configuration disables action/state clipping. Evaluate an existing checkpoint with its saved configuration and statistics.

Evaluation used `demo_clean` and `demo_randomized`, unseen instructions (`--instruction-type unseen --instruction-count 100`), 100 expert-valid scenes/task/condition selected from scene seed 100000, and `--execute-steps 16`. The stored panel was split into ten shards of ten episodes, with model seed 10222 + shard index (0–9). Replay the saved panels with their shard seeds and weights. For a new evaluation, set `--episodes 100`. Run all five task IDs listed in the README, write each condition to a separate output directory, and retain failures in the denominator.

<a id="full-data"></a>

## 8. Optional prepared full-data route

The full template `workspace/example/robotwin_full_v1.yml` uses eight GPUs, effective batch 256 and 25,000 updates. It targets [robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot) revision `595d4acc7e026aa57afba218f4170c6d79e39103` (27,500 source episodes, 6,077,247 training frames, 50 Hz). Prepare an audit JSON and independently collected development data with the fields listed below. For a quick start, use the two-task route above.

Required inputs: one LeRobot v2.1 directory per source folder; audit fields `repo`, `revision`, `rows[{folder, task}]`; per-task development `complete.json` with `instruction`, `frames`, `seed`, `task_config`; `vectors.npz` with finite aligned `[frames,16]` state/action arrays (`xyz,WXYZ,gripper` per arm); three aligned 50 Hz videos named `face_view.mp4`, `left_wrist_view.mp4`, `right_wrist_view.mp4`.

```bash
python -m wall_x.robotwin.convert_lerobot \
  --source-root /path/to/source-lerobot --audit /path/to/frame-count-audit.json \
  --dev-root /path/to/independent-dev --video-storage copy \
  --output /path/to/prepared-full-dataset
```

Use a new output directory, edit the full template and recompute its training statistics. Keep `meta/video_sources.json` with the data. Use `copy` to preserve inputs; `move` consumes source files on the same filesystem. Rebuilding relocated data additionally accepts `--video-index`.

<a id="troubleshooting"></a>

## 9. Resolve failures

| Symptom | Action |
| --- | --- |
| HDF5 decoder or file not found | Confirm XPolicyLab is installed in the model environment and check the exact extracted directory layout |
| Broken video/frame count | Check source decoding and frame counts before training; repair or replace corrupted source videos and retain their hashes |
| Simulator import/render error | Check `--sim-python`, pinned versions, OpenGL/Vulkan libraries and complete assets |
| Normalizer/layout mismatch | Use the checkpoint’s key/config/statistics; keep its clipping and padding policy |
| Resume rejected | Restore the original checkpoint, split and evaluation/training protocol |
| Insufficient GPU memory | Reduce allocated batch/workers as appropriate; simulator workers do not distribute the model |
