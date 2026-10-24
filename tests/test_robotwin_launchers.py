"""Launchers preserve scheduler visibility and the 32-predict/16-execute protocol."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "visible, override, expected",
    [
        (None, None, "0"),
        ("3", None, "3"),
        ("2,5", None, "2,5"),
        ("", None, ""),
        ("3", "7", "7"),
    ],
)
def test_launcher_preserves_gpu_selection(tmp_path, visible, override, expected):
    probe = tmp_path / "python-probe"
    probe.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "print(json.dumps({'gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), 'argv': sys.argv[1:]}))\n"
    )
    probe.chmod(0o700)
    env = dict(
        os.environ,
        PYTHON_BIN=str(probe),
        CHECKPOINT_PATH="fixture",
        ROBOTWIN_ROOT="fixture",
        OUTPUT_DIR=str(tmp_path / "results"),
    )
    for key in ("CUDA_VISIBLE_DEVICES", "CUDA_ID", "EXECUTE_STEPS", "ACTION_HORIZON"):
        env.pop(key, None)
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    if override is not None:
        env["CUDA_ID"] = override
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", str(root / "scripts" / "run_robotwin.sh")],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    actual = json.loads(result.stdout.splitlines()[-1])
    assert actual["gpu"] == expected
    assert actual["argv"][actual["argv"].index("--execute-steps") + 1] == "16"
