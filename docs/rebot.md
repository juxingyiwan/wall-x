# ReBot / B601: installation, training and deployment

[中文](rebot.zh-CN.md) · [Home](../README.md)

This guide supports one attended B601 arm, two cameras and compatible Wall-X Flow checkpoints. Commands run from the repository root on Linux. Use local inference when the GPU is on the robot host, or a model server plus a network client. Prepare demonstrations, a trained checkpoint and machine configuration for your B601.

<a id="environment"></a>

## 1. Install the selected environments

For training, serving or local inference, create a new model environment. Provide a compatible NVIDIA driver and host C/C++ tools. This build helper requires `nvcc` inside the active Conda environment:

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

The architecture example above targets A800 / RTX 3090; select the actual GPU architecture(s). Expected: CUDA-extension/serving checks pass and `pip check` reports no broken requirements. The profile includes Torch 2.10, Transformers 4.57.6, LeRobot 0.4.4, Hub 0.35.3, Rerun 0.26.2 and hardware packages. Do not combine it with the native root requirements. Keep allocated GPU visibility and use separate source builds for different Torch/CUDA ABIs.

For network inference only, the robot host can use this separate client environment instead of the model environment:

```bash
conda env create -f environments/python310.yml -n wallx-rebot-client
conda activate wallx-rebot-client
conda install -c conda-forge ffmpeg=8
python -m pip install -c environments/constraints/rebot-linux-py310-cu128.txt './deployment/client[vision,hardware]'
python -m pip check
python -m wallx_client doctor
```

`doctor` opens no devices and loads no model. The hardware extra includes LeRobot and its dependencies. To inspect the CLI on a CPU-only development machine, install `deployment/client[dev,vision]` in a Python 3.10/3.12 environment; use the hardware extra for physical operation.

<a id="assets"></a>

## 2. Prepare model assets

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

If you already have a compatible trained B601 checkpoint, keep its own configuration, processor and normalizers and continue at [deployment configuration](#configuration).

<a id="data"></a>

## 3. Prepare demonstrations

Use a compatible LeRobot dataset with metadata, images/videos and parquet columns `episode_index`, `frame_index`, `task_index`, `timestamp`, `observation.state`, `action`. State/action rows contain seven finite values, frames are contiguous within each episode, timestamps increase and the task stays constant. Map the front/side camera keys to the training views. Run in the model environment:

```bash
mkdir -p deployment/local
test -e deployment/local/train.yml || cp workspace/example/b601.yml deployment/local/train.yml
```

The `workspace/example/b601.yml` new-run preset uses eight GPUs × batch 8 × accumulation 1 = effective batch 64, square-root task sampling, 32-step joint-delta actions, AdamW `5e-5`, 500 warmup steps and a 16,000-update limit. Its validation ratio is 0.1 with `statistics_scope: train_episodes`. The previously tested delta checkpoint used ratio 0.01 and `statistics_scope: full_dataset`; resume with the checkpoint’s saved statistics and split. The [padding ablation](../README.md#action-padding) used absolute-action checkpoints. The seven `command_to_physical_scale` values define the B601 adapter’s action conversion.

Edit every placeholder: dataset `root`, logical `repo_id`, model config, processor, initial weight file and output directories. Training YAML does not expand shell variables. Keep `data.num_workers: 0` for exact resume. The template selects statistics from training episodes only. Choose a new preparation directory:

```bash
python -m scripts.prepare_b601 --config deployment/local/train.yml --output-dir /path/to/new-b601-comparison
python -m scripts.validate_b601_data --directory /path/to/new-b601-comparison
```

Expected: `absolute.yml`, `delta.yml`, separate statistics, `episode_split.json` and `preparation.json`. The validator compares real data/processor inputs and reconstructed labels without model weights or hardware. The helper selects a 32-step horizon and 16,000-update cap; review and adjust both controls consistently before training. Compare absolute/delta with the same split, sampling and budget.

The validation ratio is in $[0, 1)$; values at most `1e-6` skip validation. Otherwise each task reserves at least one episode; a one-episode task cannot support a positive validation ratio. The default scope uses training episodes; `statistics_scope: full_dataset` also includes held-out episodes.

For an explicitly selected full-dataset scope, freeze statistics and optionally export a split with the following commands; set `data.norm_stats_path` and `b601.split_manifest` to the resulting files before training:

```bash
python -m scripts.prepare_b601 --config deployment/local/train.yml --statistics-output /path/to/frozen-statistics.json
python -m scripts.prepare_b601 --config deployment/local/train.yml --split-output /path/to/new-episode-split.json
```

<a id="train"></a>

## 4. Train and resume

```bash
NPROC=8
torchrun --nproc_per_node="$NPROC" -m scripts.train_b601 \
  --config /path/to/new-b601-comparison/delta.yml --log_to_file
```

Set `NPROC` to your allocated GPU count; adjust batch/accumulation accordingly. Use `scripts.train_b601` for B601 data/checkpoint orchestration. Logs and complete checkpoints go to `checkpoint.save_path`. `max_optimizer_steps` caps completed updates; epochs can stop earlier.

```bash
NPROC=8
torchrun --nproc_per_node="$NPROC" -m scripts.train_b601 \
  --config /path/to/new-b601-comparison/delta.yml \
  --resume /path/to/complete-checkpoint --max-updates 18000 --log_to_file
```

Exact resume needs the completion marker `b601_training_state.json`, per-rank state and its bound files. Keep source, data, split/statistics, GPU count, batch, RNG, preprocessing, optimizer and scheduler settings unchanged, with zero data workers. Only update/epoch caps may extend without changing the bound trajectory; this does not extend the cosine schedule. For old weights or a new experiment, set `checkpoint.resume_from` to a weights file in a new config and omit `--resume`.

<a id="configuration"></a>

## 5. Configure the robot and checkpoint

The supported control contract is 30 Hz, two 640 × 480 cameras and action chunks of 32 steps × 26 dimensions (six arm joints, one gripper and 19 padding channels). Install/configure the B601 driver, CAN interface and camera permissions according to the hardware provider. Confirm calibration and shared exclusive lock files with other software controlling this arm. Inspect device availability:

```bash
ip -details link show can0
ls -l /dev/v4l/by-id/
```

Keep the complete inference bundle under `<checkpoint_root>/<checkpoint_id>`: selected weights, saved training config, normalizers, model/processor assets, and versioned B601 manifest plus its bound split/statistics. Optimizer state is needed for resume, not inference. Compute the weight digest and create private configuration files:

```bash
sha256sum /path/to/model.safetensors
mkdir -p deployment/local
test -e deployment/local/profile.yml || cp deployment/configs/b601.example.yml deployment/local/profile.yml
test -e deployment/local/client.yml || cp deployment/configs/client.yml deployment/local/client.yml
test -e deployment/machines/standalone.local.yml || cp deployment/machines/standalone.example.yml deployment/machines/standalone.local.yml
```

| File | Set before use |
| --- | --- |
| `deployment/local/profile.yml` | `model.checkpoint_id`, `model.epoch`, `model.step`, actual weight SHA256, `norm_key`, padding; `client_config: client.yml` selects the adjacent private client file |
| `deployment/local/client.yml` | Full training instruction in `task`, output directory, control/RTC/display settings |
| `deployment/machines/standalone.local.yml` | Checkpoint root, actual CAN/camera devices, verified camera roles and the existing shared lock paths |

`checkpoint_id` has the form `experiment/epoch_step`; its suffix must match `epoch` and `step` exactly. For a checkpoint directory named `my_experiment/3_800`, set these fields together in the complete profile’s `model` section:

```yaml
model:
  checkpoint_id: my_experiment/3_800
  epoch: 3
  step: 800
```

`epoch` is zero-based and `step` is a positive integer; retain the actual checkpoint directory name. Replace `checkpoint_sha256` with the 64-character value from `sha256sum` above, and use the checkpoint’s saved `norm_key`.

Inspect captured images against the training views before setting `camera_mapping_verified: true`. Different camera roles need distinct stable device paths. Machine overrides select host/device settings; task/control/RTC belong in the client YAML. Explicit CLI options override supported values. Inspect the effective configuration:

```bash
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml config
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- config
```

Live Rerun display defaults to enabled. Unlock the desktop for a local window. Over plain SSH use `--display-data=false`, or `--display-ip HOST --display-port PORT` for an existing viewer; pass these as client `preflight`/`run` options. Recording is independent and stays enabled unless `--save-rollout=false` is supplied. Important client defaults (your YAML/CLI may override them):

| Field | Default |
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

## 6. Capture, check and replay

Capture reads CAN feedback and cameras without enabling motors or loading a policy. For initial role verification, use the physical device mapping and camera inspection; capture requires the mapping to be declared verified. Run on the robot host:

```bash
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml client --output /path/to/new-capture capture
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local check
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- replay --observation /path/to/new-capture/observation.npz
```

Expected: `observation.npz`, camera PNGs and a capture report; check state units and camera order. `local check` validates/loads the model without robot access; replay runs inference on the saved observation.

<a id="network"></a>

## 7. Optional network inference

Skip this section for local inference. On the GPU host, copy `deployment/machines/cluster.example.yml` to a new `deployment/machines/local.yml`; on the robot host use `robot.example.yml` instead. Fill in checkpoint paths/endpoints/devices and use the same model profile. Keep allocated `CUDA_VISIBLE_DEVICES` and omit a conflicting `cuda_id`. Run on the GPU host:

```bash
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml server --dry-run
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml server
```

Configure the robot client’s endpoint to reach the server. The loopback default requires a site-provided tunnel for a remote host. Use this repository’s server, which implements `joint_state_v1`, model identity/padding checks and request IDs. On the robot host, capture using its machine file, then replay:

```bash
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client doctor
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client --output /path/to/new-network-capture capture
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client replay --observation /path/to/new-network-capture/observation.npz
```

<a id="operate"></a>

## 8. Preflight, synchronous operation and RTC

Clear the workspace, keep the physical emergency stop accessible and use an interactive terminal with an on-site operator. Preflight reads devices and invokes inference without enabling motors. `run` enables motion and may home both at startup and shutdown. Choose only one backend:

```bash
# Local backend
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- preflight --rtc false
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- run --duration 30 --rtc false
# Network backend: robot host
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client preflight --rtc false
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client run --duration 30 --rtc false
```

After synchronous validation, repeat preflight with RTC before operating. RTC uses the native single-request Flow backend:

```bash
# Local RTC
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- preflight --rtc true --guidance-mode lerobot
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml local -- run --duration 30 --rtc true --guidance-mode lerobot
# Network RTC: robot host
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client preflight --rtc true --guidance-mode lerobot
bash wallx --config deployment/local/profile.yml --machine deployment/machines/local.yml client run --duration 30 --rtc true --guidance-mode lerobot
```

In the terminal: `p` pauses/resumes; `s` marks success and stops; `f` marks failure and stops; `q` or Ctrl+C aborts. These examples set a 30-second episode budget, including pauses after startup and excluding shutdown homing. The normal default is 300 seconds. Software stopping may perform guarded homing; use the physical stop for an immediate emergency.

Homing checks a nominal forward sweep, may execute a coordinated return, completes the native arm finish and verifies gripper zero. The 65-second timeout applies to each major dispatched stage. Check the full return path for obstacles. Motor/CAN/temperature faults can prevent homing; cleanup still attempts disable. For a crash, power loss or blocked controller, use the hardware stop and verify the robot’s state before restarting.

For a camera-only diagnostic (no CAN, policy or motor access), use an unused output directory. It writes timing/configuration reports without images or video; close other camera users first.

```bash
bash wallx --config deployment/local/profile.yml --machine deployment/machines/standalone.local.yml client --output /path/to/new-camera-check camera-test --duration 60
```

<a id="results"></a>

## 9. Inspect results and troubleshoot

The configured `output_root` contains configuration, `report.json`, events and enabled media. `stop_outcome` records the stop reason/operator label; `outcome` also includes cleanup failures. `shutdown.homing` records completion, failure or skipping; `shutdown.disable: commands_sent` records that disable commands were sent. Verify motor state on the robot. For `aborted`, `timeout` or `budget_exhausted`, label the task result using on-site observation or video.

| Symptom | Action |
| --- | --- |
| Placeholder or model identity rejected | Check checkpoint root/ID, exact loaded weight hash, normalizer key, padding and bound assets |
| Camera mapping/staleness failure | Inspect actual front/side images, stable device paths, timestamps, permissions and cable bandwidth; use `wallx-client --help` for camera diagnostics |
| Viewer unavailable over SSH | Pass `--display-data=false` or configure an existing viewer; recording is independent |
| Server connects but replay fails | Check protocol metadata and saved observation format |
| RTC rejected | Match explicit executor/RTC settings, 30 Hz timebase and supported server guidance capability |
| Homing/feedback/temperature fault | Read shutdown diagnostics, check the physical system and restore healthy feedback; do not bypass protections |
