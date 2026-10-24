#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
prefix="$(python -c 'import pathlib,sys; print(pathlib.Path(sys.prefix).resolve())')"
expected_env="${WALLX_BUILD_ENV:-wallx-unified}"
if [[ "$(basename "$prefix")" != "$expected_env" ]]; then
    echo "Activate $expected_env first; set WALLX_BUILD_ENV to select a dedicated build environment." >&2
    exit 2
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME="$prefix" PATH="$prefix/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6}" MAX_JOBS="${MAX_JOBS:-4}"
unset PYTHONPATH
test -x "$CUDA_HOME/bin/nvcc"
mkdir -p build/deployment-wheels
python setup.py build_ext --inplace
python -m pip wheel --no-deps --no-build-isolation \
    --wheel-dir build/deployment-wheels .
# Dependencies are resolved separately; this wheel only replaces Wall-X itself.
python -m pip install --no-deps --force-reinstall \
    build/deployment-wheels/wall_x-1.1.0-*.whl
python -m pip check
