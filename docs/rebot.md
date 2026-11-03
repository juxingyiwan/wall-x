# ReBot / B601: installation, training and deployment

[中文](rebot.zh-CN.md) · [Home](../README.md)

For one B601 arm and two cameras. Run all commands from the repository root on Linux and replace `/path/to/...` with your paths.

<a id="environment"></a>

## 1. Install the environment

Install Conda, an NVIDIA GPU driver and C/C++ build tools. Install this environment on the GPU host for both training and local inference:

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

The two architecture variables select the current GPU’s build target. Run `conda activate wallx-rebot-model` in each new terminal.

<details>
<summary>GPU on another host: install the robot client environment</summary>

Install the model environment above on the GPU host. The robot host only needs the client environment. Use this repository on both hosts:

```bash
conda env create -f environments/python310.yml -n wallx-rebot-client
conda activate wallx-rebot-client
conda install --override-channels -c conda-forge ffmpeg=8
python -m pip install -c environments/constraints/rebot-linux-py310-cu128.txt './deployment/client[vision,hardware]'
```

</details>

<a id="assets"></a>

## 2. Prepare model resources

Download the Wall-OSS-0.5 foundation weights and Qwen processor in the model environment:

```bash
hf download X-Square-Robot/wall-oss-0.5 \
  --revision b7dc23a82dbf70a65859c9fdd596703234597028 \
  --local-dir /path/to/wall-oss-0.5
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

Deployment requires a trained B601 task checkpoint. Place the complete directory at `<checkpoint_root>/<checkpoint_id>`, retaining weights, `config.yml`, normalizers, the B601 manifest and bound statistics/split files. Model and processor paths must be readable on the inference host.

With a task checkpoint, continue to [deployment configuration](#configuration). For a new task, complete the training section below.

<a id="train"></a>

## 3. Train a task model (optional)

<a id="data"></a>

<details>
<summary>Prepare demonstrations and train a B601 model</summary>

Prepare a LeRobot demonstration dataset with metadata and images/videos. Its parquet columns must include `episode_index`, `frame_index`, `task_index`, `timestamp`, `observation.state` and `action`. State/action each have seven values per frame, with front/side views matching the training configuration.

```bash
mkdir -p deployment/local
test -e deployment/local/train.yml || cp workspace/example/b601.yml deployment/local/train.yml
```

Edit the dataset `root`, `repo_id`, camera mapping, `model.config_path`, `model.processor_path`, `checkpoint.resume_from` and `checkpoint.save_path` in `deployment/local/train.yml`. Use the Wall-OSS config and initial weights, and the Qwen processor; keep `model.pretrained_path: null`. Training YAML requires actual paths, not shell variables.

The template uses 32-step joint-delta actions, training-episode statistics and a validation ratio of 0.1. Fill in the data and model paths, then generate the training configurations, statistics and split in a new preparation directory:

```bash
python -m scripts.prepare_b601 --config deployment/local/train.yml --output-dir /path/to/new-b601-prepared
```

Train with the generated `delta.yml`, retaining its referenced statistics and `episode_split.json`.

### Train and resume

```bash
NPROC=8
torchrun --nproc_per_node="$NPROC" -m scripts.train_b601 \
  --config /path/to/new-b601-prepared/delta.yml --log_to_file
```

Set `NPROC` to the allocated GPU count and adjust per-GPU batch size and gradient accumulation accordingly. The preset uses eight GPUs, batch 8 per GPU, accumulation 1 and a 16,000-update cap; confirm the budget in the generated configuration. Checkpoints go to its `checkpoint.save_path`. On a cluster, run on an allocated compute node.

To resume an interrupted run, use a complete checkpoint:

```bash
torchrun --nproc_per_node="$NPROC" -m scripts.train_b601 \
  --config /path/to/new-b601-prepared/delta.yml \
  --resume /path/to/complete-checkpoint --log_to_file
```

For full resume, keep `b601_training_state.json` and its bound files, preserve the original data, statistics, split, GPU count, batch and optimizer settings, and keep `data.num_workers: 0`. After training, return to [deployment configuration](#configuration) with the new checkpoint identity and digest.

</details>

<a id="configuration"></a>

## 4. Configure the model and robot

Create these three configuration files for a new deployment; keep the existing files on a configured machine:

```bash
mkdir -p deployment/local
test -e deployment/local/profile.yml || cp deployment/configs/b601.example.yml deployment/local/profile.yml
test -e deployment/local/client.yml || cp deployment/configs/client.yml deployment/local/client.yml
test -e deployment/machines/standalone.local.yml || cp deployment/machines/standalone.example.yml deployment/machines/standalone.local.yml
```

Edit the new machine-local configuration:

| File | Required settings |
| --- | --- |
| `deployment/local/profile.yml` | Checkpoint identity, weight SHA256 and normalization key; keep `client_config: client.yml` and `padding: zero` |
| `deployment/local/client.yml` | Set `task` to the full training instruction and `output_root` to an absolute directory for recordings and results |
| `deployment/machines/standalone.local.yml` | `checkpoint_root`, CAN interface, stable paths for both cameras, camera roles and the arm's existing shared lock paths |

For a checkpoint directory named `my_experiment/3_800`, edit these fields in the complete profile’s `model` section. `epoch` is zero-based and `step` must match the directory name:

```yaml
model:
  checkpoint_id: my_experiment/3_800
  epoch: 3
  step: 800
```

Set `model.checkpoint_sha256` to the digest below and `model.norm_key` to the normalization key saved with the checkpoint:

```bash
sha256sum /path/to/checkpoints/my_experiment/3_800/model.safetensors
```

Complete robot calibration, CAN setup and camera permissions. Use cameras at 640 × 480 and 30 Hz. Match `front_view` and `side_view` to the training views, then set `camera_mapping_verified: true`. Keep the model action specification at 32 steps with `[6, 1, 19]` dimensions and the control rate at 30 Hz.

<a id="operate"></a>

## 5. Run a task

<a id="task"></a>

### 5.1 Select a task

The B601 three-task model used in this repository's demonstrations selects tasks through full English instructions:

| Task | Full instruction sent to the model |
| --- | --- |
| Cube into bowl | `Grasp the cube to the bowl` |
| Pen into box | `Grasp the pen to the box` |
| Cube into drawer | `Grasp the cube into the closed top drawer` |

Complete launch commands for each task are listed below. Run the one for your chosen task; `--task` must appear before `run`. All three tasks share the same three-task checkpoint, and task selection works in both synchronous and RTC modes. For your own trained model, use its training task instruction.

To set a persistent default, edit the top-level field in `deployment/local/client.yml`:

```yaml
task: "Grasp the cube to the bowl"
```

Omitting `--task` uses this default. Passing `--task` overrides it for the current run without changing the configuration file. For network inference, select the task on the robot host.

### 5.2 Local inference

Activate the model environment on the robot host, connect and power on the robot and cameras, and use an interactive terminal with an on-site operator. Clear the task and homing paths and keep the physical emergency stop accessible. Rerun is enabled by default; unlock the local desktop.

Choose the task for this episode and run its command:

**Cube into bowl — `Grasp the cube to the bowl`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/standalone.local.yml \
  local -- --task "Grasp the cube to the bowl" run --duration 30
```

**Pen into box — `Grasp the pen to the box`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/standalone.local.yml \
  local -- --task "Grasp the pen to the box" run --duration 30
```

**Cube into drawer — `Grasp the cube into the closed top drawer`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/standalone.local.yml \
  local -- --task "Grasp the cube into the closed top drawer" run --duration 30
```

Each local launch command loads the model and automatically checks the cameras, CAN feedback and inference before enabling the motors, homing and executing the task. Normal shutdown homes the arm and sends disable commands. Wait for homing to finish before starting another episode.

Set these runtime options in `deployment/local/client.yml`; enabling RTC uses the same task launch command:

| Field | Default | Purpose |
| --- | --- | --- |
| `control.duration_s` | `300.0` | Episode duration; the commands above temporarily select 30 seconds with `--duration 30`. Omit that option to use the configured duration |
| `rtc.enabled` | `false` | Synchronous inference; set to `true` to enable RTC |
| `display_data` | `true` | Show Rerun; set to `false` for an SSH session without a desktop |
| `save_rollout` | `true` | Save recordings; controlled independently of the live display |

Terminal keys: `p` pauses/resumes, `s` marks success and stops, `f` marks failure and stops, and `q` or Ctrl+C aborts. Software stopping enters shutdown handling; use the physical emergency stop when motion must stop immediately.

<a id="network"></a>

### 5.3 Network inference (GPU on another host)

Use this repository and the same model identity in `deployment/local/profile.yml` on both hosts, each with an adjacent `client.yml`. The robot-side client file selects the task, RTC and recording options. Create the machine configurations and fill in:

| Host | Template → machine configuration | Settings |
| --- | --- | --- |
| GPU host | `deployment/machines/cluster.example.yml` → `deployment/machines/server.local.yml` | `checkpoint_root`; set `server.host` to a private address reachable by the robot and `server.port` to the listening port |
| Robot host | `deployment/machines/robot.example.yml` → `deployment/machines/robot.local.yml` | CAN, cameras, shared locks and `client.policy.uri` pointing to the GPU host as `ws://<GPU_HOST>:<PORT>` |

Activate the model environment and start the server on the GPU host. On a cluster, use an allocated compute node and preserve its GPU visibility:

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/server.local.yml server
```

Once the server is ready, activate the client environment on the robot host and run. Operator keys, RTC settings and result locations are the same as for [local operation](#operate):

Choose the task for this episode and run its command:

**Cube into bowl — `Grasp the cube to the bowl`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/robot.local.yml \
  client --task "Grasp the cube to the bowl" run --duration 30
```

**Pen into box — `Grasp the pen to the box`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/robot.local.yml \
  client --task "Grasp the pen to the box" run --duration 30
```

**Cube into drawer — `Grasp the cube into the closed top drawer`**

```bash
bash wallx --config deployment/local/profile.yml \
  --machine deployment/machines/robot.local.yml \
  client --task "Grasp the cube into the closed top drawer" run --duration 30
```

<a id="results"></a>

## 6. View results

Each episode creates a separate directory under `output_root` containing configuration, `report.json`, events and recordings. `stop_outcome` records the stop reason or operator success/failure label; `shutdown.homing` records the homing result. Reaching the time limit means the episode budget expired; judge task success from observation or video.
