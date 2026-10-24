#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Shared profile supplies the task. Extra arguments are passed to the run command.
exec bash "$root/wallx" --machine "${WALLX_MACHINE:-$root/deployment/machines/local.yml}" client run "$@"
