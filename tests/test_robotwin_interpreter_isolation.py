"""A wheel must share only Wall-X source with the simulator interpreter."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from wall_x.robotwin import env


@pytest.mark.skipif(sys.platform == "win32", reason="Worker passes POSIX pipe FDs")
@pytest.mark.parametrize("separate_python", [False, True])
def test_wheel_worker_does_not_add_model_site_packages(
    tmp_path, monkeypatch, separate_python
):
    python = (
        os.environ.get("ROBOTWIN_TEST_SIM_PYTHON")
        if separate_python
        else sys.executable
    )
    if python is None:
        pytest.skip("Set ROBOTWIN_TEST_SIM_PYTHON to check a separate simulator")
    model_site = tmp_path / "model-site-packages"
    package = model_site / "wall_x"
    shutil.copytree(
        Path(env.__file__).resolve().parents[1],
        package,
        ignore=shutil.ignore_patterns("__pycache__", "*.so"),
    )
    # If the launcher exports the model's site-packages, this masks the sim's
    # own NumPy (and would also mask ABI-sensitive packages such as Torch).
    (model_site / "numpy.py").write_text(
        "raise RuntimeError('Model environment leaked into simulator')\n"
    )
    factories = tmp_path / "factories"
    factories.mkdir()
    (factories / "probe.py").write_text(
        "import numpy, sys, wall_x\n"
        "class Probe:\n"
        "    def __init__(self, config): pass\n"
        "    def reset_episode(self, value):\n"
        "        return dict(numpy=numpy.__file__, wall_x=wall_x.__file__, paths=sys.path)\n"
        "    def shutdown(self): pass\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(factories))
    monkeypatch.setattr(env, "__file__", str(package / "robotwin/env.py"))
    worker = env.SubprocEnvHandle(
        {}, python=python, timeout=30, env_factory="probe:Probe"
    )
    try:
        result = worker.call("reset_episode")
        assert Path(result["wall_x"]).parent == package
        assert not Path(result["numpy"]).is_relative_to(model_site)
        assert str(model_site) not in result["paths"]
    finally:
        worker.shutdown()
