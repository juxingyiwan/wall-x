#!/usr/bin/env bash
# Source before importing MuJoCo or PyOpenGL. Override with MUJOCO_GL=osmesa.
case "${MUJOCO_GL:-egl}" in
    egl|osmesa) export MUJOCO_GL="${MUJOCO_GL:-egl}" ;;
    *) echo "Unsupported MUJOCO_GL=${MUJOCO_GL}; choose egl or osmesa." >&2
       return 2 2>/dev/null || exit 2 ;;
esac
export PYOPENGL_PLATFORM="${MUJOCO_GL}"

_wallx_scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_wallx_root="$(cd "${_wallx_scripts}/.." && pwd)"
# Resolve source and installed layouts through the same sibling helper.
_wallx_selected="$("${PYTHON_BIN:-python}" "${_wallx_scripts}/libero_resources.py" --repo-root "${_wallx_root}" --print-root)" || { return 2 2>/dev/null || exit 2; }
mapfile -t _wallx_paths <<< "${_wallx_selected}"
export LIBERO_CONFIG_PATH="${_wallx_paths[0]}"
export LIBERO_PATH="${_wallx_paths[1]}"
export PYTHONPATH="${LIBERO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ "${MUJOCO_GL}" == egl ]]; then
    # Optional environment-local NVIDIA runtime; otherwise use system discovery.
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        _wallx_vendor="${CONDA_PREFIX}/etc/glvnd/egl_vendor.d/10_wallx_nvidia.json"
        if [[ -f "${_wallx_vendor}" ]]; then
            export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-${_wallx_vendor}}"
        fi
        if [[ -d "${CONDA_PREFIX}/lib/wallx-nvidia-535.274.02" ]]; then
            export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/wallx-nvidia-535.274.02:${LD_LIBRARY_PATH}"
        fi
    fi
    # robosuite checks the device against the physical CUDA_VISIBLE_DEVICES IDs.
    _wallx_cuda="${CUDA_VISIBLE_DEVICES:-${CUDA_ID:-0}}"
    _wallx_first="${_wallx_cuda%%,*}"
    if [[ "${_wallx_first}" =~ ^[0-9]+$ ]]; then
        export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${_wallx_first}}"
    fi
else
    unset MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_FILENAMES
fi
printf 'LIBERO rendering: %s\n' "${MUJOCO_GL}"
unset _wallx_root _wallx_scripts _wallx_selected _wallx_paths _wallx_vendor _wallx_cuda _wallx_first
