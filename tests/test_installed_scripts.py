"""Exercise the installed launcher without CUDA, simulator imports or weights."""

import ast
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from setuptools import Distribution

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("bash") is None, reason="Bash launcher requires bash")
@pytest.mark.parametrize("backend", ["egl", "osmesa"])
def test_installed_libero_launcher_loads_rendering_helper(
    tmp_path, monkeypatch, backend
):
    # Read the real script manifest without importing setup.py's CUDA extension.
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    scripts = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PUBLIC_SCRIPTS"
            for target in node.targets
        )
    )
    monkeypatch.chdir(ROOT)
    distribution = Distribution(
        {"name": "wallx-launcher-test", "scripts": scripts, "packages": []}
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

    # Intercept only Python inference; the installed shell files run unmodified.
    capture = tmp_path / "capture.py"
    capture.write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['CAPTURE_PATH']).write_text(json.dumps({\n"
        "    'args': sys.argv[1:],\n"
        "    'gl': os.environ.get('MUJOCO_GL'),\n"
        "    'platform': os.environ.get('PYOPENGL_PLATFORM'),\n"
        "    'device': os.environ.get('MUJOCO_EGL_DEVICE_ID'),\n"
        "}))\n",
        encoding="utf-8",
    )
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "python"
    shim.write_text(
        "#!/bin/sh\n"
        f'case "$*" in *--print-config*) exec {shlex.quote(sys.executable)} "$@" ;; esac\n'
        f'case "$1" in */infer_libero.py) exec {shlex.quote(sys.executable)} {shlex.quote(str(capture))} "$@" ;; *) exec {shlex.quote(sys.executable)} "$@" ;; esac\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    libero = tmp_path / "LIBERO"
    for folder in ("assets", "bddl_files", "init_files"):
        (libero / "libero/libero" / folder).mkdir(parents=True)
    report_path = tmp_path / "invocation.json"
    checkpoint = str(tmp_path / "checkpoint")
    result = subprocess.run(
        ["bash", str(Path(install.install_dir) / "run_libero.sh"), checkpoint],
        cwd=tmp_path,
        env={
            "PATH": f"{shim_dir}{os.pathsep}{os.defpath}",
            "HOME": str(tmp_path),
            "WALLX_LIBERO_ROOT": str(libero),
            "MUJOCO_GL": backend,
            "CUDA_ID": "3",
            "SKIP_LIBERO_DEP_CHECK": "1",
            "LOG_DIR": str(tmp_path / "logs"),
            "CAPTURE_PATH": str(report_path),
        },
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["gl"] == report["platform"] == backend
    assert report["device"] == ("3" if backend == "egl" else None)
    args = report["args"]
    assert Path(args[0]) == Path(install.install_dir) / "infer_libero.py"
    assert args[args.index("--checkpoint-path") + 1] == checkpoint
