"""Check that a non-editable Python package contains the training entry graph."""

import ast
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Distribution, find_packages


def test_built_python_package_can_import_trainer(tmp_path):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "setup.py").read_text(encoding="utf-8"))
    setup = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "setup"
    )
    discovery = next(k.value for k in setup.keywords if k.arg == "packages")
    assert isinstance(discovery, ast.Call) and discovery.func.id == "find_packages"
    options = {k.arg: ast.literal_eval(k.value) for k in discovery.keywords}
    packages = find_packages(where=str(root), **options)
    distribution = Distribution(
        {
            "name": "wallx-package-test",
            "packages": packages,
            "package_dir": {"": str(root)},
        }
    )
    distribution.script_name = str(root / "setup.py")
    build = distribution.get_command_obj("build_py")
    build.build_lib = str(tmp_path / "package")
    build.ensure_finalized()
    build.run()
    code = (
        "import pathlib,os,sys,wall_x; "
        "assert pathlib.Path(wall_x.__file__).is_relative_to(pathlib.Path(os.environ['PYTHONPATH'])); "
        "from wall_x.trainer.fsdp_trainer import FSDPTrainer; "
        "from wall_x.trainer.scheduler.scheduler import get_scheduler; "
        "from wall_x.data.sampling import DistributedSquareRootSampler; "
        "from wall_x.data.b601_prepare import make_stats; "
        "from wall_x.data.backends.lerobot.bridge import build_dataset_config, wrap_dataset; "
        "assert len(list(DistributedSquareRootSampler([0, 0, 1], seed=17))) == 3; "
        "assert not any(n == 'tools' or n.startswith('tools.') or n.startswith('scripts.') for n in sys.modules); "
        "print(FSDPTrainer.__name__, get_scheduler.__name__, make_stats.__module__)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": build.build_lib, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
