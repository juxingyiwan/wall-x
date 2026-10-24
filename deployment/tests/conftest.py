"""Test the checkout even when an older editable installation is present."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "deployment/client/src")]


@pytest.fixture(autouse=True)
def isolated_launch_environment(monkeypatch):
    # Launcher tests inspect configuration or stub the model; they never execute
    # CUDA. A nonexistent device keeps the fixtures independent of host Slurm
    # state while also preventing accidental access to a physical GPU.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-wallx-test-only")
