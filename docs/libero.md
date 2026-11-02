# LIBERO: installation, training and evaluation

[中文](libero.zh-CN.md) · [Home](../README.md)

Run all commands from the repository root on Linux and replace `/path/to/...` with your paths.

<a id="environment"></a>

## 1. Install the environment

Install Conda, an NVIDIA GPU driver and C/C++ build tools. Build on the GPU that will run the model; use an allocated compute node on a cluster.

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

Build the model dependencies in the same environment:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASH_ATTN_CUDA_ARCHS=$(python -c 'import torch; print(torch.cuda.get_device_capability()[0] * 10)')
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install 'dmuon @ git+https://github.com/X-Square-Robot/dmuon.git@157f522b8570764c473b1a18939f4ebb36da5878'
MAX_JOBS=8 python -m pip install --no-build-isolation -e .
```

Activate `conda activate wallx-libero` in each new terminal. The remaining steps prepare resources, train and evaluate. With a task checkpoint, complete section 2 and continue to [evaluation](#evaluate).

<a id="assets"></a>

## 2. Prepare the model and simulator

Download the Wall-OSS-0.5 foundation weights and Qwen processor:

```bash
hf download X-Square-Robot/wall-oss-0.5 \
  --revision b7dc23a82dbf70a65859c9fdd596703234597028 \
  --local-dir /path/to/wall-oss-0.5
hf download Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --local-dir /path/to/Qwen2.5-VL-3B-Instruct
```

Obtain the LIBERO source and resources for evaluation:

```bash
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
export WALLX_LIBERO_ROOT="$PWD/third_party/LIBERO"
python scripts/libero_resources.py
```

The resource tree must contain `libero/libero/assets`, `bddl_files` and `init_files`. Evaluation uses a trained task checkpoint with its training configuration and normalizers.

<a id="data"></a>

## 3. Prepare data and training configuration

```bash
hf download lerobot/libero --repo-type dataset \
  --revision a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --local-dir /path/to/libero_all
mkdir -p workspace/local/libero
cp workspace/example/libero.yml workspace/local/libero/train.yml
```

Update the paths in the copied template and retain its other settings. Fill in these fields:

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

`repo_id` is the dataset key used by the normalizer and must not contain dots; use the same key for inference. The template uses 7-dimensional actions, two views named `observation.images.image` and `observation.images.image2`, and a 95%/5% training/validation split by episode.

```bash
python scripts/compute_norm_stats.py --train_config workspace/local/libero/train.yml
```

Statistics are written to `data.norm_stats_path` using only training episodes. Recompute them after changing the data or split.

<a id="train"></a>

## 4. Train the model

The template uses 8 GPUs, batch 16 per GPU and accumulation 1, giving an effective batch of 128. Adjust process count, batch size, accumulation and FSDP settings to your GPU allocation, then launch:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m wall_x.trainer.fsdp_trainer.train_fsdp \
  --config workspace/local/libero/train.yml
```

Logs and checkpoints are written to `checkpoint.save_path`. To resume, set `checkpoint.resume_from` to a complete checkpoint directory, retain its configuration, normalizers and data, and rerun the command. A weights file initializes a new training run.

When continuing an older run, retain `data.lerobot_config.split_mode: legacy_frame`, its original statistics, worker count and distributed topology. New training uses an `episode` split with newly computed statistics.

### Reproduce the README comparison

Both initializations use effective batch 128, AdamW `5e-5`, 1,000 warmup steps, a 30,000-step cosine schedule, seed 10222, horizon 10 and two 256 × 256 views, evaluated at 25,000 updates. `num_training_steps` also caps training updates. The reported runs use `legacy_frame` and their original statistics.

For Qwen initialization, retain `model.config_path` for the architecture, point `model.pretrained_path` to the Qwen directory, set `checkpoint.resume_from: null`, and use 4 GPUs × batch 16 per GPU × accumulation 2.

<a id="evaluate"></a>

## 5. Evaluate the model

Activate the model environment and select a specific task checkpoint. Run the four standard suites with 50 trials per task:

```bash
conda activate wallx-libero
WALLX_LIBERO_ROOT="$PWD/third_party/LIBERO" MUJOCO_GL=egl \
  ALL_SUITES=1 NUM_TRIALS_PER_TASK=50 LOG_DIR=/path/to/evaluation \
  bash scripts/run_libero.sh /path/to/training_checkpoint
```

For CPU rendering, replace `MUJOCO_GL=egl` with `MUJOCO_GL=osmesa NUMBA_DISABLE_JIT=1`; model inference still uses a GPU. The evaluation launcher retains the allocated GPUs.

<a id="results"></a>

## 6. View results

Results for each suite are saved under `/path/to/evaluation/<suite>/`, including `report.json`, console logs and episode records. Success rate is `successes / attempted`; `overall.attempted` includes execution errors, which also count toward `overall.failed`.
