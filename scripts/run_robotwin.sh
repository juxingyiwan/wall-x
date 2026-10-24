#!/usr/bin/env bash
set -euo pipefail

# Like run_libero.sh: one launcher owns the model and its simulator workers.
# CHECKPOINT_PATH, ROBOTWIN_ROOT and OUTPUT_DIR are required. SIM_PYTHON may
# select a different environment for the simulator, without any network service.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo 'CHECKPOINT_PATH=/ckpt ROBOTWIN_ROOT=/RoboTwin OUTPUT_DIR=/results bash scripts/run_robotwin.sh'
    echo 'Optional: PYTHON_BIN SIM_PYTHON CUDA_ID TASKS TASK_CONFIG NUM_TRIALS_PER_TASK NUM_WORKERS SEED EXECUTE_STEPS (default 16) RESUME'
    exit 0
fi
export PYTHONPATH="${SOURCE_ROOT}:${PYTHONPATH:-}"
# Explicit CUDA_ID wins; otherwise preserve Slurm/user visibility, even if empty.
export CUDA_VISIBLE_DEVICES="${CUDA_ID-${CUDA_VISIBLE_DEVICES-0}}"
args=(
    --checkpoint-path "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"
    --robotwin-root "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT}"
    --output "${OUTPUT_DIR:?Set OUTPUT_DIR}"
    --norm-key "${NORM_KEY:-robotwin2_full27500}"
    --task-config "${TASK_CONFIG:-demo_clean}"
    --episodes "${NUM_TRIALS_PER_TASK:-50}"
    --num-workers "${NUM_WORKERS:-1}"
    --seed "${SEED:-10222}"
    --instruction-type "${INSTRUCTION_TYPE:-seen}"
    --instruction-count "${INSTRUCTION_COUNT:-10}"
    --execute-steps "${EXECUTE_STEPS:-16}"
)
[[ -z "${TRAIN_CONFIG_PATH:-}" ]] || args+=(--train-config-path "$TRAIN_CONFIG_PATH")
[[ -z "${SIM_PYTHON:-}" ]] || args+=(--sim-python "$SIM_PYTHON")
[[ -z "${SCENE_MANIFEST:-}" ]] || args+=(--scene-manifest "$SCENE_MANIFEST")
[[ "${RESUME:-0}" != "1" ]] || args+=(--resume)
if [[ -n "${TASKS:-}" ]]; then
    read -r -a task_names <<< "$TASKS"
    args+=(--tasks "${task_names[@]}")
fi
exec "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/infer_robotwin.py" "${args[@]}" "$@"
