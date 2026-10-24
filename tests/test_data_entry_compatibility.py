"""Keep established data commands and Python imports usable after relocation."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "arguments,expected",
    [
        (["-m", "scripts.prepare_b601", "--help"], "--statistics-output"),
        (["scripts/prepare_b601.py", "--help"], "--split-output"),
        (
            ["-m", "tools.sqrt_sampling", "small=100", "large=900"],
            "small\t100\t25.0000%",
        ),
    ],
)
def test_existing_cli(arguments, expected):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout


def test_previous_python_imports_preserve_outputs():
    from scripts.prepare_b601 import make_stats as previous_stats
    from tools.sqrt_sampling import DistributedSquareRootSampler as PreviousSampler
    from wall_x.data.b601_prepare import make_stats
    from wall_x.data.sampling import DistributedSquareRootSampler

    # Existing callers retain the implementations, including sampler class identity.
    assert previous_stats is make_stats
    assert PreviousSampler is DistributedSquareRootSampler
