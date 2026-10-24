"""Resolve one weight file before loading weights or checkpoint metadata."""

import hashlib
from pathlib import Path


def resolve_checkpoint_weights(path: str | Path) -> Path:
    """Explicit files win; directories retain FSDP-before-safetensors precedence."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.suffix not in {".bin", ".pt", ".pth", ".safetensors"}:
            raise ValueError(f"Unsupported checkpoint weight format: {path}")
        return path
    for name in ("pytorch_model_fsdp.bin", "model.safetensors"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No pytorch_model_fsdp.bin or model.safetensors under {path}"
    )


def checkpoint_sha256(weights: str | Path) -> str:
    """Hash one selected weight file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(weights).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
