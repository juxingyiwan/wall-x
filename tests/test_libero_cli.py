import importlib.util
import sys
from pathlib import Path

import yaml

spec = importlib.util.spec_from_file_location(
    "infer_libero", Path(__file__).resolve().parents[1] / "scripts/infer_libero.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def merged(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["infer_libero.py", *args])
    return cli._load_or_build_raw_config(cli.parse_args())


def test_yaml_defaults_and_explicit_overrides(monkeypatch, tmp_path):
    config = tmp_path / "eval.yaml"
    raw = {
        "model": {
            "checkpoint_path": "/weights",
            "norm_key": "old",
            "cam_names": ["custom"],
        },
        "env": {
            "seed": 1,
            "libero": {
                "task_suite_name": "libero_10",
                "max_infer_times": 123,
                "num_trials_per_task": 7,
            },
        },
        "runtime": {"num_workers": 2, "log_dir": "/old"},
        "debug": {"deterministic_model": True},
    }
    config.write_text(yaml.safe_dump(raw))
    result = merged(monkeypatch, "--config", str(config))
    assert result["env"]["libero"]["max_infer_times"] == 123
    assert result["model"]["cam_names"] == ["custom"]
    assert result["runtime"]["num_workers"] == 2
    result = merged(
        monkeypatch,
        "--config",
        str(config),
        "--task-suite-name",
        "libero_goal",
        "--num-trials-per-task",
        "2",
        "--num-workers",
        "3",
        "--seed",
        "99",
        "--norm-key",
        "new",
        "--log-dir",
        "/new",
        "--cam-names",
        "front",
        "wrist",
        "--action-horizon",
        "4",
        "--no-deterministic-model",
        "--no-skip-intermediate-render",
    )
    assert result["env"]["libero"] == {
        "task_suite_name": "libero_goal",
        "max_infer_times": 30,
        "num_trials_per_task": 2,
        "initial_states_path": "DEFAULT",
        "skip_intermediate_render": False,
    }
    assert result["env"]["seed"] == 99
    assert result["model"]["norm_key"] == "new"
    assert result["model"]["cam_names"] == ["front", "wrist"]
    assert result["model"]["action_horizon"] == 4
    assert result["runtime"]["num_workers"] == 3
    assert result["runtime"]["log_dir"] == "/new"
    assert result["debug"]["deterministic_model"] is False


def test_cli_only_and_smoke(monkeypatch):
    result = merged(
        monkeypatch,
        "--checkpoint-path",
        "/weights",
        "--smoke",
        "--num-workers",
        "20",
        "--num-trials-per-task",
        "100",
    )
    assert result["env"]["libero"]["task_indices"] == [0]
    assert result["env"]["libero"]["num_trials_per_task"] == 5
    assert result["runtime"]["num_workers"] == 1
