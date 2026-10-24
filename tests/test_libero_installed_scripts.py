import ast
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from setuptools import Distribution

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def installed(tmp_path, monkeypatch):
    tree = ast.parse((ROOT / "setup.py").read_text())
    scripts = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PUBLIC_SCRIPTS" for t in n.targets)
    )
    monkeypatch.chdir(ROOT)
    distribution = Distribution(
        {"name": "wallx-scripts-test", "scripts": scripts, "packages": []}
    )
    distribution.script_name = "setup.py"
    distribution.get_command_obj("egg_info").egg_base = str(tmp_path)
    build = distribution.get_command_obj("build_scripts")
    build.build_dir = str(tmp_path / "build")
    build.ensure_finalized()
    build.run()
    install = distribution.get_command_obj("install_scripts")
    install.build_dir = build.build_dir
    install.install_dir = str(tmp_path / "installed/bin")
    install.skip_build = True
    install.ensure_finalized()
    install.run()
    return Path(install.install_dir)


@pytest.mark.parametrize(
    "visible,override,expected",
    [
        ("3", None, "3"),
        ("2,3", None, "2,3"),
        ("GPU-example", None, "GPU-example"),
        ("", None, ""),
        (None, None, "0"),
        ("3", "7", "7"),
    ],
)
def test_installed_entry_preserves_scheduler_and_yaml(
    installed, tmp_path, visible, override, expected
):
    libero = tmp_path / "LIBERO"
    for folder in ("assets", "bddl_files", "init_files"):
        (libero / "libero/libero" / folder).mkdir(parents=True)
    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"checkpoint_path": "/example/weights", "norm_key": "custom"},
                "env": {
                    "libero": {
                        "task_suite_name": "libero_goal",
                        "num_trials_per_task": 9,
                    }
                },
                "runtime": {"num_workers": 3, "log_dir": str(tmp_path / "logs")},
            }
        )
    )
    capture = tmp_path / "cuda.txt"
    shim = tmp_path / "python-shim"
    shim.write_text(
        '#!/bin/bash\nprintf "%s" "$CUDA_VISIBLE_DEVICES" > "$CAPTURE"\nexec '
        + shlex.quote(sys.executable)
        + ' "$@"\n'
    )
    shim.chmod(0o755)
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "CONFIG": str(config),
        "WALLX_LIBERO_ROOT": str(libero),
        "MUJOCO_GL": "osmesa",
        "PYTHON_BIN": str(shim),
        "CAPTURE": str(capture),
        "PRINT_CONFIG": "1",
    }
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    if override is not None:
        env["CUDA_ID"] = override
    proc = subprocess.run(
        ["bash", str(installed / "run_libero.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert capture.read_text() == expected
    result = yaml.safe_load(proc.stdout.split("LIBERO rendering: osmesa\n", 1)[1])
    assert result["env"]["libero"]["task_suite_name"] == "libero_goal"
    assert result["env"]["libero"]["num_trials_per_task"] == 9
    assert result["runtime"]["num_workers"] == 3
    assert result["model"]["norm_key"] == "custom"
    env.update(ALL_SUITES="1", NUM_TRIALS_PER_TASK="2")
    proc = subprocess.run(
        ["bash", str(installed / "run_libero.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    results = list(
        yaml.safe_load_all(proc.stdout.split("LIBERO rendering: osmesa\n", 1)[1])
    )
    assert [r["env"]["libero"]["task_suite_name"] for r in results] == [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    ]
    assert len({r["runtime"]["log_dir"] for r in results}) == 4
    assert all(r["env"]["libero"]["num_trials_per_task"] == 2 for r in results)
