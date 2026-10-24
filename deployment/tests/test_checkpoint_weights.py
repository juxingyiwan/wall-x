"""Checkpoint preflight must hash the file selected by the native loaders."""

import hashlib
import subprocess
import sys

import pytest

from deployment.cli import verify_checkpoint
from wall_x.checkpoint_weights import resolve_checkpoint_weights


@pytest.mark.parametrize(
    "names,selected",
    [
        (["model.safetensors"], "model.safetensors"),
        (["pytorch_model_fsdp.bin"], "pytorch_model_fsdp.bin"),
        (["model.safetensors", "pytorch_model_fsdp.bin"], "pytorch_model_fsdp.bin"),
    ],
)
def test_verify_selected_checkpoint(tmp_path, names, selected):
    (tmp_path / "config.yml").write_text("{}")
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    assert resolve_checkpoint_weights(tmp_path) == tmp_path / selected
    deployment = {
        "checkpoint_path": str(tmp_path),
        "model": {"checkpoint_sha256": hashlib.sha256(selected.encode()).hexdigest()},
    }
    verify_checkpoint(deployment)
    deployment["model"]["checkpoint_sha256"] = hashlib.sha256(b"other weights").hexdigest()
    with pytest.raises(ValueError, match="SHA256"):
        verify_checkpoint(deployment)


def test_coexisting_safetensors_pin_cannot_authorize_fsdp(tmp_path):
    (tmp_path / "config.yml").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"approved")
    (tmp_path / "pytorch_model_fsdp.bin").write_bytes(b"different model")
    with pytest.raises(ValueError, match="pytorch_model_fsdp.bin"):
        verify_checkpoint(
            {
                "checkpoint_path": str(tmp_path),
                "model": {"checkpoint_sha256": hashlib.sha256(b"approved").hexdigest()},
            }
        )


def test_missing_weights_fail_clearly(tmp_path):
    (tmp_path / "config.yml").write_text("{}")
    with pytest.raises(FileNotFoundError, match="No pytorch_model_fsdp.bin or model.safetensors"):
        verify_checkpoint(
            {"checkpoint_path": str(tmp_path), "model": {"checkpoint_sha256": "a" * 64}}
        )


def test_preflight_does_not_import_model_or_hardware_dependencies(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    (tmp_path / "config.yml").write_text("{}")
    (tmp_path / "pytorch_model_fsdp.bin").write_bytes(b"fixture")
    code = f"""
import importlib.util, sys
spec = importlib.util.spec_from_file_location('entry', {str(root / "deployment/cli.py")!r})
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
entry.verify_checkpoint({{'checkpoint_path': {str(tmp_path)!r},
    'model': {{'checkpoint_sha256': {hashlib.sha256(b"fixture").hexdigest()!r}}}}})
assert not ({{'torch', 'safetensors', 'transformers', 'motorbridge', 'can'}} & sys.modules.keys())
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
