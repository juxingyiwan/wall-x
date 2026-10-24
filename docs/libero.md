# LIBERO: installation, training and evaluation

[中文](libero.zh-CN.md) · [Home](../README.md)

Use this guide for Wall-OSS-0.5 with LeRobot-format LIBERO demonstrations. All commands run from the repository root on Linux. Choose [existing-checkpoint evaluation](#evaluate) after setup, or follow the full training sequence. Keep datasets, checkpoints and results outside the tracked source tree.

<a id="environment"></a>

## 1. Install an independent environment

Requirements: Conda, Python 3.10, a supported NVIDIA GPU/driver, host C/C++ build tools and a CUDA toolkit matching Torch 2.10. The driver alone does not provide `nvcc`. The following profile uses LeRobot 0.4.4, Transformers 4.57.6 and Hub 0.35.3; do not also install the native root `requirements.txt` into this environment.

```bash
conda env create -f environments/python310.yml -n wallx-libero
conda activate wallx-libero
conda install -c conda-forge ffmpeg=8 libgl libegl mesalib
python -m pip install -r environments/requirements/lerobot-model.txt -r requirements-libero.txt
python -m pip check
```

For training only, omit `-r requirements-libero.txt`; install it before simulator evaluation. If a matching toolkit is not already provisioned, install one for the selected Torch CUDA build and set `CUDA_HOME` before continuing.

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

## 2. Prepare model and simulator resources

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

For rollout evaluation, obtain the following LIBERO source revision and its assets:

```bash
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
export WALLX_LIBERO_ROOT="$PWD/third_party/LIBERO"
python scripts/libero_resources.py
```

Expected: `libero/libero/assets`, `bddl_files` and `init_files` are found. Obtain separately distributed assets using that LIBERO revision’s instructions. Record their provenance. Set `MUJOCO_GL=egl` for supported GPU rendering or `osmesa` for CPU rendering; policy inference still needs CUDA. The launcher preserves allocated GPU visibility unless you explicitly set `CUDA_ID`. For UUID/MIG allocations, choose a compatible EGL device or OSMesa.

<a id="data"></a>

## 3. Prepare demonstrations and a training configuration

```bash
hf download lerobot/libero --repo-type dataset \
  --revision a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 \
  --local-dir /path/to/libero_all
mkdir -p workspace/local/libero
cp workspace/example/libero.yml workspace/local/libero/train.yml
```

Edit every placeholder in the copied template. Preserve its 7 physical action dimensions and virtual padding. Check camera mappings against the dataset. The following is only a path/configuration excerpt; retain the rest of the template:

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

`repo_id` is the saved normalizer key, not a filesystem path; it must not contain a dot. `root` selects the directory and may contain dots. Keep the same key for inference. For $N$ selected episodes and training fraction $r$ (`train_test_split`), the first $\lfloor N r \rfloor$ episodes train the model; the remainder validate it.

```bash
python scripts/compute_norm_stats.py --train_config workspace/local/libero/train.yml
```

Expected: the configured statistics JSON contains `episode_split`. Only training episodes contribute statistics. Recompute these statistics after changing episode selection, order or split ratio.

<a id="train"></a>

## 4. Train and continue a run

The preset uses eight GPUs × 16 samples/GPU × one accumulation step = effective batch 128, AdamW at `5e-5`, 1,000 warmup steps, a 30,000-step cosine schedule, seed 10222, horizon 10 and two 256 × 256 views. `num_training_steps` also caps the current trainer; choose the checkpoint at 25,000 updates for the [reported comparison](../README.md#libero-results). The dataset keys are `observation.images.image` and `observation.images.image2`; check `meta/info.json` before using another export.

For the Qwen initialization comparison, retain `model.config_path` for architecture, set `model.pretrained_path` to the Qwen directory and `checkpoint.resume_from: null`. Use four GPUs × batch 16 × accumulation 2 to retain effective batch 128. New runs use `split_mode: episode`; the reported runs used `legacy_frame` and their original statistics.

```bash
torchrun --standalone --nproc_per_node=8 \
  -m wall_x.trainer.fsdp_trainer.train_fsdp \
  --config workspace/local/libero/train.yml
```

Select process count, batch size, accumulation and FSDP settings for your GPU allocation. Logs/checkpoints use `checkpoint.save_path`. A weights file in `checkpoint.resume_from` initializes a new run. To continue, set it to a complete checkpoint directory and rerun the same command, preserving configuration, normalizers and assets. A checkpoint root may select the latest recorded step; use a concrete checkpoint for reproducible evaluation.

For older runs, retain the original statistics and explicitly set `data.lerobot_config.split_mode: legacy_frame`. That mode retains the historical episode-prefix plus 95/5 frame split. Switching to `episode` requires new statistics and a new run initialized from a weight file. Resume restores recorded progress and checks the split; retain the original worker count and distributed topology for reproducibility.

<a id="evaluate"></a>

## 5. Evaluate a trained checkpoint

Use a trained LIBERO checkpoint with its saved training YAML and normalizers. In the model environment, first run one episode:

```bash
export WALLX_LIBERO_ROOT=/path/to/LIBERO
export MUJOCO_GL=egl
source scripts/libero_rendering.sh
python scripts/infer_libero.py --checkpoint-path /path/to/training_checkpoint \
  --task-suite-name libero_spatial --task-indices 0 --num-trials-per-task 1 \
  --num-workers 1 --log-dir /path/to/evaluation/single-episode
```

Then choose a smoke run (task 0, five trials, one worker) or all four standard suites:

```bash
WALLX_LIBERO_ROOT=/path/to/LIBERO SMOKE=1 \
  bash scripts/run_libero.sh /path/to/training_checkpoint
WALLX_LIBERO_ROOT=/path/to/LIBERO ALL_SUITES=1 NUM_TRIALS_PER_TASK=50 \
  LOG_DIR=/path/to/evaluation \
  bash scripts/run_libero.sh /path/to/training_checkpoint
```

Explicit CLI values override evaluation YAML; omitted options retain YAML values. Inspect a YAML launch without loading a model or simulator:

```bash
python scripts/infer_libero.py --config /path/to/eval.yml \
  --task-suite-name libero_goal --num-trials-per-task 50 --print-config
```

<a id="results"></a>

## 6. Read outputs and resolve failures

Each suite writes `report.json` and episode logs/recordings. The success rate is `successes / attempted`; `overall.attempted` includes execution errors, which are also counted in `overall.failed`. For comparisons, record code/weight/statistics hashes, simulator/data versions, tasks, trials, seeds, action budgets and renderer.

| Symptom | Action |
| --- | --- |
| CUDA/compiler/extension error | Compare Torch CUDA, `nvcc` and `CUDA_HOME`; rebuild in the selected environment |
| Missing `libGL`/EGL/OSMesa | Check installed graphics libraries and `MUJOCO_GL`; keep rendering setup before simulator imports |
| Missing or wrong LIBERO checkout | Set `WALLX_LIBERO_ROOT`; unset stale `LIBERO_PATH`; rerun the resource check |
| Missing normalizer or split mismatch | Use the saved logical dataset key and matching config/statistics; recompute only for a new training run |
| Wrong checkpoint weights | Pass the concrete file; directory selection prefers `pytorch_model_fsdp.bin` before `model.safetensors` |
| Execution failures in a report | Inspect episode errors before calculating or comparing benchmark scores |
