# Wall-X: LIBERO, RoboTwin 2.0 and ReBot

[中文](README.zh-CN.md) · [Home](README.md)

Built on [X-Square Robot / Wall-X](https://github.com/X-Square-Robot/wall-x), this repository provides training, evaluation and deployment workflows for LIBERO, RoboTwin 2.0 and ReBot / B601.

<a id="workflows"></a>

## Choose one workflow

| Workflow | Complete guide |
| --- | --- |
| LIBERO: install, prepare data, train, resume and evaluate | [English](docs/libero.md) · [中文](docs/libero.zh-CN.md) |
| RoboTwin 2.0: model/simulator setup, convert, train and evaluate | [English](docs/robotwin.md) · [中文](docs/robotwin.zh-CN.md) |
| ReBot / B601: install, prepare/train, local/network inference, sync and RTC | [English](docs/rebot.md) · [中文](docs/rebot.zh-CN.md) |

Each workflow uses its own Conda environment; RoboTwin has a separate simulator environment. The guides cover setup, resources, commands and troubleshooting.

Clone this repository, run the guide commands from its root and replace placeholder paths. Model training and inference require Linux, an NVIDIA GPU and CUDA. Each guide explains how to obtain foundation weights, prepare data and use checkpoints.

<a id="experiments"></a>

## Experimental results

Training configurations and evaluation procedures are documented in the corresponding guides.

<a id="libero-results"></a>

### LIBERO

Both initializations use the Wall-X action architecture and full-parameter training: effective batch 128, seed 10222, a 30,000-step cosine schedule and evaluation at 25,000 updates. Each suite contains 10 tasks with 50 trials/task (2,000 trials per initialization). The Qwen initialization loads the pretrained vision-language backbone with a newly initialized action expert. See the [training and evaluation settings](docs/libero.md#train).

| Suite | Wall-OSS-0.5 initialization | Qwen2.5-VL-3B initialization |
| --- | --- | --- |
| Spatial | 98.8% (494/500) | 98.2% (491/500) |
| Object | 99.6% (498/500) | 100.0% (500/500) |
| Goal | 98.4% (492/500) | 98.8% (494/500) |
| Long | 96.0% (480/500) | 94.6% (473/500) |
| **Overall** | **98.2% (1964/2000)** | **97.9% (1958/2000)** |

Each initialization uses one training run.

![LIBERO training flow loss for Wall-OSS-0.5 and Qwen2.5-VL-3B initializations](assets/libero_training_loss.png)

Training flow loss through 25,000 steps, with a logarithmic vertical axis. Faint lines show logged interval averages; solid lines show a trailing 25-point mean (about 250 steps).

<a id="robotwin-results"></a>

### RoboTwin 2.0

Five-task training uses 50 clean + 500 randomized demonstrations per task (2,750 total), effective batch 256 and a checkpoint at 30,000 completed updates. Evaluation uses unseen instructions, 32 predicted / 16 executed actions and **100 expert-valid trials per task and condition**: 500 Clean + 500 Randomized trials. See the [evaluation settings](docs/robotwin.md#reported-settings) for scenes and seeds.

| Task | Clean successes/trials | Randomized successes/trials |
| --- | --- | --- |
| `open_microwave` | 67/100 | 85/100 |
| `click_bell` | 71/100 | 92/100 |
| `stack_blocks_three` | 79/100 | 78/100 |
| `place_shoe` | 97/100 | 96/100 |
| `put_object_cabinet` | 76/100 | 82/100 |
| **Five-task mean** | **78.0% (390/500)** | **86.6% (433/500)** |

Success rates use valid evaluation episodes across the five tasks. Policy failures count in the denominator; expert-invalid scenes are excluded before evaluation.

<a id="rebot-results"></a>

### ReBot: Wall-OSS-0.5 with and without RTC

Wall-OSS-0.5 on the real robot: **RTC (real-time chunking) enabled on the left, disabled on the right**. Click the preview to watch the original side-by-side comparison.

[![Wall-OSS-0.5 comparison: RTC enabled on the left, disabled on the right](assets/rebot_walloss05_rtc_comparison.jpg)](assets/rebot_walloss05_rtc_comparison.mp4)

[Play comparison](assets/rebot_walloss05_rtc_comparison.mp4) · [Operating steps](docs/rebot.md#operate) · [Action-padding notes and ablation](#action-padding)

<a id="upstream"></a>

## Wall-X documentation

Official documentation from revision `ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f`, with the official homepages saved as `README.upstream.*`. Use the workflow guides above for this repository.

| Original material | Location |
| --- | --- |
| Official overview, models, community and citation | [English](README.upstream.md) · [中文](README.upstream.zh-CN.md) |
| Original usage and script documentation | [Workspace](workspace/README.md) · [Scripts](scripts/README.md) |
| Original hardware examples | [Flowers](workspace/TRAIN_ARRANGE_3_FLOWERS.md) · [RTX 5090 setup](workspace/rtx5090/README.md) · [Deployment](workspace/rtx5090/DEPLOY.md) · [Original notes](workspace/rtx5090/local/NOTES.md) |

<a id="contribute"></a>

## Contribute and develop

See [Contributing](CONTRIBUTING.md) ([中文](CONTRIBUTING.zh-CN.md)) for tests and PR preparation, and [Technical development notes](#development) ([中文](README.zh-CN.md#development)) for implementation details. The project uses [Apache-2.0](LICENSE); third-party attribution is in the [source notices](NOTICE.txt) and [client notices](deployment/client/NOTICE.txt). Models and datasets use their respective licenses.

<a id="development"></a>

## Technical development notes

This section covers component ownership, implementation analysis and homing bounds. See the [three workflows](#workflows) for operating procedures and [Contributing](CONTRIBUTING.md) for checks.

<a id="ownership"></a>

### Component ownership

| Component | Responsibility |
| --- | --- |
| `wall_x/model`, `data`, `config`, `trainer`, `_vendor/harrix` | Upstream core; shared padding, loader and recovery patches remain here |
| `wall_x/robotwin`, `scripts/*robotwin*` | RoboTwin conversion/training/evaluation and isolated simulator workers |
| `wall_x/data/b601*`, `wall_x/trainer/adapters/b601*` | B601 preparation, representation and recovery |
| `deployment/`, `deployment/client/` | Model backends/launcher and independently packaged hardware controller |
| `environments/`, `workspace/example/` | Dependency profiles and public configuration templates |

<a id="implementation-analysis"></a>

### Wall-X improvements and analysis

<a id="action-padding"></a>

#### Action padding: training/inference alignment

This experiment was motivated by action jitter observed during ReBot deployment and targets a mismatch already present in [upstream Wall-X ce2a1ae](https://github.com/X-Square-Robot/wall-x/blob/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f/wall_x/model/qact/qwen2_5/modeling_qwen2_5_vl_act.py#L2080-L2092). B601 uses seven physical action channels (six arm joints and a gripper), padded to the model's 26 dimensions. Training normalizes the physical channels before appending zeros; the upstream inference path instead maps raw zero to -1 when [checkpoint statistics already cover all 26 dimensions](https://github.com/X-Square-Robot/wall-x/blob/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f/wall_x/model/core/action/normalizer.py#L98-L109), with `min=0, delta=1` in the virtual channels. Although excluded from the loss, virtual channels still enter the shared action encoder and can affect physical outputs. The ablation measures the effect of restoring the training-time zero endpoint on predicted-action smoothness.

<a id="padding-video"></a>

##### Real-robot comparison

Wall-OSS-0.5 was trained on 100 demonstrations for 12 epochs to open a drawer and place a block inside. RTC is enabled in both runs; only action padding alignment changes: aligned on the left, unaligned on the right.

[![Action padding on the robot: aligned on the left, unaligned on the right, with RTC enabled in both runs](assets/rebot_walloss05_padding_comparison.jpg)](assets/rebot_walloss05_padding_comparison.mp4)

[Play the comparison video](assets/rebot_walloss05_padding_comparison.mp4)

<a id="padding-offline"></a>

##### Offline prediction ablation

Six checkpoints were replayed on recorded robot observations with matched weights, observations and noise within each pair. Only the virtual action endpoint changed from -1 to 0; state/masks were fixed, with RTC, smoothing and clipping disabled. Across 900 paired predictions, the zero endpoint reduced the metric in all 900 pairs.

For each prediction, smoothness is measured by the mean absolute second difference over the first $K = 16$ steps and $D = 6$ arm joints:

$$
J = \frac{1}{D(K-2)}
    \sum_{j=1}^{D}\sum_{k=0}^{K-3}
    \left|q_{k+2,j} - 2q_{k+1,j} + q_{k,j}\right|.
$$

Here $q_{k,j}$ is the predicted angle of joint $j$ at step $k$, in degrees. The mean covers 14 second differences per joint (84 terms in total). The unit is degrees/sample-step²; lower is smoother. Values below pool the corresponding checkpoints by pair count.

| Task | Pairs | Endpoint -1 | Endpoint 0 | Reduction |
| --- | --- | --- | --- | --- |
| Bowl | 360 | 2.958 | 0.262 | 91.1% |
| Drawer | 270 | 2.334 | 0.206 | 91.2% |
| Pen | 270 | 2.557 | 0.263 | 89.7% |

These checkpoints used absolute joint actions; the current B601 template uses delta actions. See [ReBot configuration](docs/rebot.md#data).

<a id="homing"></a>

### Homing bounds

The following constants control the nominal homing sweep and gripper-zero checks. Check the full arm path for obstacles before operation. Keep the values aligned with source and test feedback-loss and timeout paths with fake devices:

| Source constant | Value | Meaning |
| --- | --- | --- |
| `front_home.FORWARD_LIMIT_M` | `0.5` | Forward plane (m) |
| `front_home.CLEARANCE_M` | `0.03` | Nominal margin (m) |
| `zero_servo.ZERO_PRECISION_DEG` | `0.5` | Maximum gripper zero error (degrees) |
| `zero_servo.MAX_REFINEMENT_S` | `10.0` | Refinement deadline (s) |

<a id="source"></a>

### Authoritative implementation references

Use [padding](wall_x/inference_padding.py), [checkpoint selection](wall_x/checkpoint_weights.py), the [client configuration](deployment/client/src/wallx_client/config.py), [wire protocol](deployment/client/src/wallx_client/protocol.py) and [sample manifest](deployment/client/docs/examples/server_manifest.json) when changing behavior. Inspect actual parsers through `--help`. Retain third-party attribution in [NOTICE](NOTICE.txt) and the [client notice](deployment/client/NOTICE.txt).
