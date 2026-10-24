#!/usr/bin/env python3
"""Run LIBERO evaluation through harrix.

The script accepts either a full harrix EvalConfig YAML or a checkpoint path
plus common command-line overrides. It intentionally bypasses the legacy
Wall-X inference stack.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

LIBERO_DEFAULT_MAX_INFER_TIMES = {
    "libero_spatial": 22,
    "libero_object": 28,
    "libero_goal": 30,
    "libero_10": 52,
    "libero_90": 40,
}


def _ensure_local_harrix_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    harrix_python = repo_root / "third_party" / "harrix" / "python"
    if harrix_python.is_dir():
        sys.path.insert(0, str(harrix_python))


def _parse_task_indices(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(x) for x in value.split(",") if x.strip()]


def _resolve_max_infer_times(
    task_suite_name: str | None, max_infer_times: int | None
) -> int:
    if max_infer_times is not None:
        return max_infer_times
    suite = task_suite_name or "libero_spatial"
    return LIBERO_DEFAULT_MAX_INFER_TIMES.get(suite, 22)


def _load_or_build_raw_config(args: argparse.Namespace) -> dict:
    if args.config is not None:
        with open(args.config, encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        if not isinstance(raw, dict):
            raise ValueError("Evaluation YAML must contain a mapping")
    else:
        raw = {}
    model = raw.setdefault("model", {})
    env = raw.setdefault("env", {})
    libero = env.setdefault("libero", {})
    runtime = raw.setdefault("runtime", {})
    debug = raw.setdefault("debug", {})
    for section, defaults in (
        (
            model,
            {
                "checkpoint_path": None,
                "train_config_path": None,
                "norm_key": "libero_all",
                "cam_names": ["face_view", "right_wrist_view"],
                "architecture": "qwen2_5",
                "action_mode": "flow",
                "action_horizon": None,
            },
        ),
        (env, {"seed": 42}),
        (
            libero,
            {
                "task_suite_name": "libero_spatial",
                "initial_states_path": "DEFAULT",
                "num_trials_per_task": 50,
                "skip_intermediate_render": True,
            },
        ),
        (
            runtime,
            {
                "num_workers": 1,
                "max_batch_size": 1,
                "ws_port": 8765,
                "log_dir": "/tmp/harrix_libero_eval",
                "driver_mode": "in_process",
            },
        ),
        (debug, {"deterministic_model": False}),
    ):
        for key, default in defaults.items():
            value = getattr(args, key)
            if value is not None:
                section[key] = value
            else:
                section.setdefault(key, default)
    env["type"] = "libero"
    if not model.get("checkpoint_path"):
        raise ValueError("--checkpoint-path is required when YAML does not provide it")
    # A new suite gets its own default rollout budget unless explicitly given.
    if (
        args.max_infer_times is not None
        or args.task_suite_name is not None
        or libero.get("max_infer_times") is None
    ):
        libero["max_infer_times"] = _resolve_max_infer_times(
            libero["task_suite_name"], args.max_infer_times
        )
    if args.task_indices is not None:
        libero["task_indices"] = _parse_task_indices(args.task_indices)
    if args.smoke:
        libero.update(task_indices=[0], num_trials_per_task=5)
        runtime.update(num_workers=1, max_batch_size=1)
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=None, help="Optional harrix EvalConfig YAML."
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--train-config-path", default=None)
    parser.add_argument("--norm-key", default=None)
    parser.add_argument("--architecture", default=None)
    parser.add_argument("--action-mode", default=None)
    parser.add_argument("--cam-names", nargs="+", default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--task-suite-name", default=None)
    parser.add_argument("--initial-states-path", default=None)
    parser.add_argument("--num-trials-per-task", type=int, default=None)
    parser.add_argument(
        "--task-indices", default=None, help="Comma-separated task ids."
    )
    parser.add_argument(
        "--max-infer-times",
        type=int,
        default=None,
        help=(
            "Number of model action chunks per episode. Defaults are suite-specific "
            "and match the internal LIBERO evaluator."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--ws-port", type=int, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--driver-mode", choices=["in_process"], default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--deterministic-model", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--skip-intermediate-render",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the merged configuration without loading models or simulators.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = _load_or_build_raw_config(args)
    if args.print_config:
        print(yaml.safe_dump(raw, sort_keys=False), end="")
        return 0
    from libero_resources import configure_libero

    configure_libero(Path(__file__).resolve().parents[1])
    _ensure_local_harrix_on_path()

    from wall_x._vendor.harrix.eval_config import (
        autofill_from_checkpoint,
        load_eval_config,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(raw, f, sort_keys=False)
        tmp_config = f.name

    try:
        cfg = autofill_from_checkpoint(load_eval_config(tmp_config))
    finally:
        Path(tmp_config).unlink()
    if cfg.runtime.driver_mode != "in_process":
        raise ValueError("Only driver_mode='in_process' is supported")
    from wall_x._vendor.harrix.drivers.inproc import run

    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
