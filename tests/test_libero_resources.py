import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "libero_resources",
    Path(__file__).resolve().parents[1] / "scripts/libero_resources.py",
)
resources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resources)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    repo = tmp_path / "wall-x"
    benchmark = repo / "third_party/LIBERO/libero/libero"
    for name in ("assets", "bddl_files", "init_files"):
        (benchmark / name).mkdir(parents=True)
    monkeypatch.setenv("WALLX_CACHE_ROOT", str(tmp_path / "cache"))
    for key in (
        "WALLX_LIBERO_ROOT",
        "WALLX_LIBERO_CONFIG_PATH",
        "WALLX_LIBERO_DATA_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LIBERO_PATH", "")
    monkeypatch.setattr(sys, "path", list(sys.path))
    return repo


def test_ignores_cosmos_ambient_config_without_modifying_it(
    layout, tmp_path, monkeypatch
):
    cosmos = tmp_path / "cosmos-config"
    cosmos.mkdir()
    original = "assets: /cosmos/assets\n"
    (cosmos / "config.yaml").write_text(original)
    monkeypatch.setenv("LIBERO_CONFIG_PATH", str(cosmos))
    monkeypatch.setenv("LIBERO_PATH", "/cosmos/LIBERO-plus")
    monkeypatch.setenv("WALLX_LIBERO_ROOT", str(layout / "third_party/LIBERO"))
    path = resources.configure_libero(layout)
    config = yaml.safe_load((path / "config.yaml").read_text())
    assert config["assets"] == str(layout / "third_party/LIBERO/libero/libero/assets")
    assert (cosmos / "config.yaml").read_text() == original
    import os

    assert os.environ["LIBERO_CONFIG_PATH"] == str(path)
    assert os.environ["LIBERO_PATH"] == str(layout / "third_party/LIBERO")


def test_fails_for_missing_local_assets_instead_of_global_fallback(layout):
    (layout / "third_party/LIBERO/libero/libero/assets").rmdir()
    with pytest.raises(FileNotFoundError, match="Missing LIBERO assets"):
        resources.configure_libero(layout)


def test_rejects_already_loaded_foreign_libero(layout, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "libero.libero",
        SimpleNamespace(__file__="/cosmos/LIBERO/libero/libero/__init__.py"),
    )
    with pytest.raises(RuntimeError, match="already imported from another project"):
        resources.configure_libero(layout)


def test_legacy_path_is_honored_and_stale_path_fails(layout, monkeypatch, tmp_path):
    root = layout / "third_party/LIBERO"
    monkeypatch.setenv("LIBERO_PATH", str(root))
    config = resources.configure_libero(tmp_path / "installed")
    assert yaml.safe_load((config / "config.yaml").read_text())[
        "benchmark_root"
    ] == str(root / "libero/libero")
    monkeypatch.setenv("LIBERO_PATH", "/missing/stale/libero")
    with pytest.raises(FileNotFoundError, match="stale"):
        resources.configure_libero(layout)


def test_config_identity_includes_resource_and_data_roots(
    layout, monkeypatch, tmp_path
):
    monkeypatch.setenv("WALLX_LIBERO_ROOT", str(layout / "third_party/LIBERO"))
    first = resources.configure_libero(layout)
    before = (first / "config.yaml").read_text()
    monkeypatch.setenv("WALLX_LIBERO_DATA_ROOT", str(tmp_path / "other-data"))
    second = resources.configure_libero(layout)
    assert first != second
    assert (first / "config.yaml").read_text() == before
    monkeypatch.setenv("WALLX_LIBERO_CONFIG_PATH", str(first))
    with pytest.raises(ValueError, match="different resources"):
        resources.configure_libero(layout)


def test_two_processes_cannot_replace_each_others_config(tmp_path):
    import os
    import subprocess

    procs = []
    for name in ("a", "b"):
        root = tmp_path / name
        for folder in ("assets", "bddl_files", "init_files"):
            (root / "libero/libero" / folder).mkdir(parents=True)
        procs.append(
            subprocess.Popen(
                [sys.executable, str(Path(resources.__file__))],
                env={
                    **os.environ,
                    "WALLX_LIBERO_ROOT": str(root),
                    "WALLX_CACHE_ROOT": str(tmp_path / "cache"),
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    results = [proc.communicate(timeout=20) for proc in procs]
    assert all(proc.returncode == 0 for proc in procs), results
    configs = [Path(out.strip()) for out, err in results]
    assert configs[0] != configs[1]
    for name, config in zip(("a", "b"), configs):
        assert yaml.safe_load((config / "config.yaml").read_text())["assets"] == str(
            tmp_path / name / "libero/libero/assets"
        )
