#!/usr/bin/env bash
set -euo pipefail

# CLI reference: scripts/README.md. End-to-end workflow: docs/libero.md.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
    cat <<'EOF'
Usage: run_libero.sh [CHECKPOINT_PATH]
Provide CHECKPOINT_PATH or CONFIG=/path/to/eval.yaml.
Optional: WALLX_LIBERO_ROOT (legacy alias LIBERO_PATH), CUDA_ID, MUJOCO_GL,
ALL_SUITES=1, TASK_SUITES, TASK_SUITE_NAME, NUM_TRIALS_PER_TASK, NUM_WORKERS,
NORM_KEY, LOG_DIR, ROLLOUT_BASE, SMOKE=1, PRINT_CONFIG=1.
Explicit environment options override YAML; omitted options retain YAML values.
EOF
}
case "${1:-}" in -h|--help) usage; exit 0 ;; esac
if [[ $# -gt 1 ]]; then usage >&2; exit 2; fi
if [[ $# -eq 1 ]]; then CHECKPOINT_PATH="$1"; fi
if [[ -z "${CHECKPOINT_PATH:-}" && -z "${CONFIG:-}" ]]; then
    usage >&2
    exit 2
fi

if [[ ${CUDA_ID+x} ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_ID}"
elif [[ ! ${CUDA_VISIBLE_DEVICES+x} ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi
if [[ -d "${SOURCE_ROOT}/wall_x" ]]; then
    export PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi
source "${SCRIPT_DIR}/libero_rendering.sh"

if [[ "${SKIP_LIBERO_DEP_CHECK:-0}" != 1 && "${PRINT_CONFIG:-0}" != 1 ]]; then
    "${PYTHON_BIN}" - <<'PY'
import importlib.util

missing = []
for module in ("libero.libero", "robosuite", "mujoco", "OpenGL", "bddl", "gym", "h5py"):
    try:
        present = importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        present = False
    if not present:
        missing.append(module)
if missing:
    raise SystemExit("Missing LIBERO dependencies: " + ", ".join(missing) + ". See docs/libero.md#assets.")
PY
fi

base_args=()
if [[ -n "${CONFIG:-}" ]]; then base_args+=(--config "${CONFIG}"); fi
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then base_args+=(--checkpoint-path "${CHECKPOINT_PATH}"); fi
for pair in \
    TRAIN_CONFIG_PATH:train-config-path NORM_KEY:norm-key ARCHITECTURE:architecture \
    ACTION_MODE:action-mode ACTION_HORIZON:action-horizon SEED:seed \
    INITIAL_STATES_PATH:initial-states-path NUM_TRIALS_PER_TASK:num-trials-per-task \
    TASK_INDICES:task-indices MAX_INFER_TIMES:max-infer-times NUM_WORKERS:num-workers \
    MAX_BATCH_SIZE:max-batch-size WS_PORT:ws-port DRIVER_MODE:driver-mode; do
    var="${pair%%:*}"
    if [[ ${!var+x} ]]; then base_args+=("--${pair#*:}" "${!var}"); fi
done
if [[ "${SMOKE:-0}" == 1 ]]; then base_args+=(--smoke); fi
if [[ ${DET+x} ]]; then
    if [[ "${DET}" == 1 ]]; then base_args+=(--deterministic-model); else base_args+=(--no-deterministic-model); fi
fi
if [[ ${SKIP_INTERMEDIATE_RENDER+x} ]]; then
    if [[ "${SKIP_INTERMEDIATE_RENDER}" == 1 ]]; then base_args+=(--skip-intermediate-render); else base_args+=(--no-skip-intermediate-render); fi
fi

# Read YAML defaults through the same merger used by the actual inference entry.
merged="$("${PYTHON_BIN}" "${SCRIPT_DIR}/infer_libero.py" "${base_args[@]}" --print-config)"
defaults="$(printf '%s' "${merged}" | "${PYTHON_BIN}" -c 'import sys,yaml; c=yaml.safe_load(sys.stdin); print(c["env"]["libero"]["task_suite_name"]); print(c["runtime"]["log_dir"])')"
mapfile -t default_values <<< "${defaults}"
suites=("${TASK_SUITE_NAME:-${default_values[0]}}")
explicit_suite=0
if [[ "${ALL_SUITES:-0}" == 1 ]]; then
    suites=(libero_spatial libero_object libero_goal libero_10)
    explicit_suite=1
fi
if [[ -n "${TASK_SUITES:-}" ]]; then
    read -r -a suites <<< "${TASK_SUITES//,/ }"
    explicit_suite=1
fi
if [[ ${TASK_SUITE_NAME+x} ]]; then explicit_suite=1; fi
rollout_override="${WALLX_ROLLOUT_DIR:-}"
for suite in "${suites[@]}"; do
    args=("${base_args[@]}")
    if [[ "${explicit_suite}" == 1 ]]; then args+=(--task-suite-name "${suite}"); fi
    log_dir="${LOG_DIR:-${ROLLOUT_BASE:+${ROLLOUT_BASE}/json}}"
    log_dir="${log_dir:-${default_values[1]}}"
    rollout_dir="${rollout_override:-${ROLLOUT_BASE:+${ROLLOUT_BASE}/videos}}"
    if [[ ${#suites[@]} -gt 1 ]]; then
        log_dir="${log_dir}/${suite}"
        if [[ -n "${rollout_dir}" ]]; then rollout_dir="${rollout_dir}/${suite}"; fi
    fi
    args+=(--log-dir "${log_dir}")
    if [[ -n "${rollout_dir}" ]]; then
        export WALLX_ROLLOUT_DIR="${rollout_dir}"
        mkdir -p "${rollout_dir}"
    fi
    if [[ "${PRINT_CONFIG:-0}" == 1 ]]; then
        printf '%s\n' '---'
        "${PYTHON_BIN}" "${SCRIPT_DIR}/infer_libero.py" "${args[@]}" --print-config
    else
        mkdir -p "${log_dir}"
        echo "=== LIBERO eval: suite=${suite} log_dir=${log_dir} ==="
        PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/infer_libero.py" "${args[@]}" 2>&1 | tee -a "${log_dir}/console.log"
    fi
done
