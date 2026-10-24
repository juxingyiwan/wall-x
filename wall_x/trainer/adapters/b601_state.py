"""B601-only training state; native trainer/checkpoint formats stay unchanged."""

import copy
import hashlib
import itertools
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wall_x.data.b601 import (
    LAYOUT,
    checkpoint_config,
    digest,
    preprocessing_contract,
    processor_checksums,
)
from wall_x.data.b601_split import resolve_split


def canonicalize_config(cfg):
    """Repair YAML key sorting only for the known, explicitly declared layout."""
    opts = (cfg.data.lerobot_config or {}).get("b601")
    if opts is None:
        raise ValueError("B601Trainer requires explicit data.lerobot_config.b601")
    for key in ("dof_config", "agent_pos_config", "ar_dof_config"):
        layout = getattr(cfg.task, key)
        if layout is None and key == "ar_dof_config":
            layout = LAYOUT
        if dict(layout) != LAYOUT:
            raise ValueError(f"B601 requires the known 6+1+19 {key}")
        setattr(cfg.task, key, dict(LAYOUT))
        cfg._raw_yaml.setdefault("task", {})[key] = dict(LAYOUT)


def capture_rng():
    result = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state()
    return result


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


class CursorBatchSampler:
    """Skip indices, not decoded samples; only the trainer advances the cursor."""

    def __init__(self, base, accumulation):
        self.base = base
        self.stop = len(base) - len(base) % accumulation
        self.start = 0

    def __iter__(self):
        return itertools.islice(iter(self.base), self.start, self.stop)

    def __len__(self):
        return self.stop - self.start


def resumable_loader(loader, accumulation, seed):
    # Worker prefetch has its own state; do not silently pretend it is restored.
    if loader.num_workers != 0:
        raise ValueError("Exact B601 resume currently requires data.num_workers: 0")
    batches = CursorBatchSampler(loader.batch_sampler, accumulation)
    if not batches.stop:
        raise ValueError("B601 epoch has no complete optimizer update")
    result = DataLoader(
        loader.dataset,
        batch_sampler=batches,
        collate_fn=loader.collate_fn,
        num_workers=0,
        pin_memory=loader.pin_memory,
        # Iterator construction must not consume the model's Torch RNG.
        generator=torch.Generator().manual_seed(seed),
    )
    return result, batches


def input_pipeline_contract(cfg):
    """Bind effective native bridge/collator settings, including their defaults."""
    from wall_x.data.backends.lerobot.bridge import build_dataset_config
    from wall_x.data.backends.lerobot.config import LerobotConfig

    flat, _ = build_dataset_config(cfg)
    data = flat["data"]
    vision = LerobotConfig.from_yaml_dict(flat)
    noise = flat["noise_scheduler"]
    return {
        "version": 1,
        "max_length": data.get("max_length", 768),
        "padding_side": flat["padding_side"],
        "state_bins": flat["state_bins"],
        "use_fast_tokenizer": flat["use_fast_tokenizer"],
        "use_state_string_representation": flat["use_state_string_representation"],
        "action_horizon": cfg.task.action_horizon,
        "action_horizon_flow": cfg.task.action_horizon_flow,
        "min_pixels": vision.min_pixels,
        "max_pixels": vision.max_pixels,
        "image_factor": vision.image_factor,
        "noise_scheduler": {
            key: float(noise.get(key, default))
            for key, default in (
                ("beta_alpha", 1.5),
                ("beta_beta", 1.0),
                ("time_shift", 1.0),
                ("s", 0.999),
            )
        },
    }


def runtime_contract(cfg, world_size, batches):
    config = checkpoint_config(cfg)
    artifact = json.loads(Path(cfg.data.norm_stats_path).read_text())
    return {
        "world_size": world_size,
        "batch_size_per_gpu": cfg.hyperparams.batch_size_per_gpu,
        "gradient_accumulation_steps": cfg.hyperparams.gradient_accumulation_steps,
        "batches_per_epoch": batches,
        "seed": cfg.hyperparams.seed,
        "task_sampling": cfg.data.task_sampling,
        "split_sha256": digest(resolve_split(config)),
        "statistics_contract": artifact["b601_contract"],
        "preprocessing": preprocessing_contract(config),
        "input_pipeline": input_pipeline_contract(cfg),
        "optimizer": config["hyperparams"]["optimizer"],
        "scheduler": config["hyperparams"]["scheduler"],
        "distributed": config["distributed"],
        "flow_loss_weight": cfg.model.flow_loss_weight,
        "ar_loss_weight": cfg.model.ar_loss_weight,
        "model_type": cfg.model_type,
        "model_settings": {
            k: v
            for k, v in config["model"].items()
            if k not in ("config_path", "processor_path", "pretrained_path")
        },
        "model_config_sha256": (
            file_sha256(model_config_file(cfg.model.config_path))
            if cfg.model.config_path
            else None
        ),
        "processor_files": (
            processor_checksums(cfg.model.processor_path)
            if cfg.model.processor_path
            else {}
        ),
    }


def model_config_file(source):
    """Resolve the same JSON selected by the native config loader, without weights."""
    source = str(source)
    if source.endswith(".json") or Path(source).is_file():
        return Path(source)
    from transformers.configuration_utils import get_configuration_file
    from transformers.utils.hub import cached_file, extract_commit_hash

    path = Path(cached_file(source, "config.json"))
    config = json.loads(path.read_text())
    if "configuration_files" in config:
        name = get_configuration_file(config["configuration_files"])
        path = Path(
            cached_file(source, name, _commit_hash=extract_commit_hash(str(path), None))
        )
    return path


def load_scaler_state(checkpoint, scaler, contract):
    """Validate exact AMP restoration before changing any live training state."""
    precision = contract["distributed"]
    required = precision["use_amp"] and not precision["bf16"]
    enabled = scaler is not None and scaler.is_enabled()
    if required != enabled:
        raise ValueError("B601 GradScaler precision contract mismatch")
    if not required:
        return None
    path = Path(checkpoint) / "grad_scaler.pt"
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        # Use PyTorch's own schema check without changing the active scaler.
        probe = copy.deepcopy(scaler)
        probe.load_state_dict(state)
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        ValueError,
        RuntimeError,
        KeyError,
        TypeError,
    ) as exc:
        raise ValueError(f"Invalid B601 GradScaler state in {path}: {exc}") from exc
    return state


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def code_checksums():
    root = Path(__file__).resolve().parents[3]
    paths = (
        "wall_x/trainer/adapters/b601_state.py",
        "wall_x/trainer/adapters/b601_trainer.py",
        "wall_x/data/b601.py",
        "wall_x/data/b601_split.py",
        "wall_x/data/backends/lerobot/b601_adapter.py",
        "wall_x/trainer/fsdp_trainer/fsdp_trainer.py",
        "wall_x/trainer/fsdp_trainer/checkpoint_io.py",
    )
    return {name: file_sha256(root / name) for name in paths}


def validate_dataset_sources(cfg):
    artifact = json.loads(Path(cfg.data.norm_stats_path).read_text())
    root = Path(cfg.data.lerobot_config["root"]).resolve()
    for source in artifact["b601_contract"]["source_files"]:
        path = (root / source["path"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != source["sha256"]:
            raise ValueError("B601 dataset no longer matches the frozen statistics")


def load_runtime(checkpoint, rank, world_size):
    root = Path(checkpoint)
    marker = root / "b601_training_state.json"
    if not marker.exists():
        raise ValueError(
            "Checkpoint lacks exact B601 training state; use model.safetensors "
            "explicitly for weights-only initialization, not exact resume"
        )
    meta = json.loads(marker.read_text())
    for name, checksum in meta["metadata_files"].items():
        if file_sha256(root / name) != checksum:
            raise ValueError("B601 checkpoint metadata checksum mismatch")
    if meta["version"] != 1 or meta["world_size"] != world_size:
        raise ValueError(
            "B601 exact resume requires the saved state version/world size"
        )
    name = f"b601_training_rank{rank}.pt"
    if file_sha256(root / name) != meta["rank_files"][name]:
        raise ValueError("B601 training state checksum mismatch")
    state = torch.load(root / name, map_location="cpu", weights_only=False)
    if "input_pipeline" not in state["contract"]:
        raise ValueError(
            "Checkpoint predates complete B601 input/sampling validation; use "
            "model.safetensors for weights-only initialization, not exact resume"
        )
    for key in ("completed_updates", "next_epoch", "next_batch"):
        if state[key] != meta[key]:
            raise ValueError("B601 rank progress mismatch")
    if state["rank"] != rank or state["micro_step"] != 0:
        raise ValueError("B601 checkpoint must be at an optimizer boundary")
    if (
        torch.load(root / "global_step.pth", weights_only=True)["global_step"]
        != state["completed_updates"]
    ):
        raise ValueError("B601 checkpoint update count mismatch")
    for name in ("optimizer.pt", "scheduler.pt", "model.safetensors"):
        if not (root / name).exists():
            raise ValueError(f"Incomplete B601 checkpoint: missing {name}")
    return copy.deepcopy(state)
