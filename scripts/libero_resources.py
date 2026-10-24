#!/usr/bin/env python3
"""Select validated LIBERO resources without reading global ~/.libero."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml


def libero_root(repo_root: Path) -> Path:
    """The explicit Wall-X variable wins over the legacy LIBERO_PATH alias."""
    value = os.environ.get("WALLX_LIBERO_ROOT") or os.environ.get("LIBERO_PATH")
    return Path(value or repo_root / "third_party/LIBERO").expanduser().resolve()


def configure_libero(repo_root: Path | None = None) -> Path:
    repo_root = (repo_root or Path(__file__).resolve().parents[1]).resolve()
    root = libero_root(repo_root)
    benchmark = root / "libero/libero"
    cache = (
        Path(os.environ.get("WALLX_CACHE_ROOT", Path.home() / ".cache/wall-x"))
        .expanduser()
        .resolve()
    )
    data = (
        Path(os.environ.get("WALLX_LIBERO_DATA_ROOT", cache / "datasets/libero"))
        .expanduser()
        .resolve()
    )
    resources = {
        "benchmark_root": str(benchmark),
        "assets": str(benchmark / "assets"),
        "bddl_files": str(benchmark / "bddl_files"),
        "init_states": str(benchmark / "init_files"),
        "datasets": str(data),
    }
    for key in ("benchmark_root", "assets", "bddl_files", "init_states"):
        if not Path(resources[key]).is_dir():
            raise FileNotFoundError(
                f"Missing LIBERO {key}: {resources[key]}. Set WALLX_LIBERO_ROOT to a complete LIBERO checkout; clear an inherited LIBERO_PATH if it is stale."
            )
    for name in ("libero", "libero.libero"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError(
                    f"{name} was already imported from another project: {module.__file__}"
                )
    identity = hashlib.sha256(
        json.dumps(resources, sort_keys=True).encode()
    ).hexdigest()
    config_dir = (
        Path(
            os.environ.get(
                "WALLX_LIBERO_CONFIG_PATH", cache / "libero/config" / identity
            )
        )
        .expanduser()
        .resolve()
    )
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    contents = yaml.safe_dump(resources, sort_keys=False)
    # Publish once: concurrent runs cannot replace another process's config.
    with tempfile.NamedTemporaryFile(
        "w", dir=config_dir, delete=False, encoding="utf-8"
    ) as stream:
        stream.write(contents)
        temp_path = Path(stream.name)
    try:
        try:
            os.link(temp_path, config_file)
        except FileExistsError:
            if yaml.safe_load(config_file.read_text(encoding="utf-8")) != resources:
                raise ValueError(
                    f"LIBERO config already selects different resources: {config_file}. Choose a separate WALLX_LIBERO_CONFIG_PATH."
                ) from None
    finally:
        temp_path.unlink()
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["LIBERO_PATH"] = str(root)
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(root), *[p for p in paths if p and p != str(root)]]
    )
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    return config_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--print-root", action="store_true")
    args = parser.parse_args()
    print(configure_libero(args.repo_root))
    if args.print_root:
        print(os.environ["LIBERO_PATH"])
