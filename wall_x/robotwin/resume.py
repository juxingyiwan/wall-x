"""RoboTwin batch cursors and reproducible data randomness, isolated from Wall-X."""

import hashlib
import itertools
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def capture_rng():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def data_seed(*parts):
    encoded = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63)


@contextmanager
def seeded_cpu(seed):
    """Do not initialize CUDA in workers or advance the model's CPU RNG."""
    previous = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    try:
        torch.random.default_generator.manual_seed(seed)
        np.random.seed(seed % (2**32))
        random.seed(seed)
        yield
    finally:
        torch.set_rng_state(previous[0])
        np.random.set_state(previous[1])
        random.setstate(previous[2])


class KeyedDataset:
    def __init__(self, dataset, seed, rank, stream):
        self.dataset, self.seed, self.rank, self.stream = dataset, seed, rank, stream

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, batch, position, index = key
        seed = data_seed(self.seed, self.rank, self.stream, epoch, batch, position)
        with seeded_cpu(seed):
            return key, self.dataset[index]


class KeyedCollator:
    def __init__(self, collator, seed, rank, stream):
        self.collator, self.seed, self.rank, self.stream = collator, seed, rank, stream

    def __call__(self, samples):
        keys, values = zip(*samples)
        epoch, batch = keys[0][:2]
        if any(key[:2] != (epoch, batch) for key in keys):
            raise ValueError("RoboTwin batch contains inconsistent replay keys")
        seed = data_seed(self.seed, self.rank, self.stream, epoch, batch, "collate")
        previous = getattr(self.collator, "np_rng", None)
        try:
            # The native collator seeds default_rng() from system entropy.
            self.collator.np_rng = np.random.default_rng(seed)
            with seeded_cpu(seed):
                return self.collator(list(values))
        finally:
            self.collator.np_rng = previous


class CursorBatchSampler:
    def __init__(self, sampler):
        self.original = sampler
        self.start = 0
        self.end = len(sampler)

    def __len__(self):
        return self.end - self.start

    def __iter__(self):
        epoch = getattr(self.original.sampler, "epoch", 0)
        # Skip indices, not decoded videos. Prefetch never determines the
        # checkpoint cursor; only batches delivered to the trainer do.
        batches = itertools.islice(iter(self.original), self.start, self.end)
        for batch, indices in enumerate(batches, self.start):
            yield [(epoch, batch, pos, int(index)) for pos, index in enumerate(indices)]


class ReplayableLoader:
    """Keep native loop lengths while delivering only unconsumed batches."""

    def __init__(self, original, seed, rank, stream):
        self.original = original
        self.batch_sampler = CursorBatchSampler(original.batch_sampler)
        self.next_batch = 0
        self.loader = DataLoader(
            KeyedDataset(original.dataset, seed, rank, stream),
            batch_sampler=self.batch_sampler,
            collate_fn=KeyedCollator(original.collate_fn, seed, rank, stream),
            num_workers=original.num_workers,
            pin_memory=original.pin_memory,
            persistent_workers=original.persistent_workers,
            prefetch_factor=original.prefetch_factor,
            # Worker creation must not advance the restored model RNG.
            generator=torch.Generator().manual_seed(data_seed(seed, rank, stream)),
        )

    def configure(self, start=0, accumulation=1):
        total = len(self.original.batch_sampler)
        self.batch_sampler.end = total - total % accumulation
        if not 0 <= start <= self.batch_sampler.end or start % accumulation:
            raise ValueError("Invalid RoboTwin resume batch cursor")
        self.batch_sampler.start = self.next_batch = start

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __len__(self):
        return self.batch_sampler.end

    def __iter__(self):
        self.next_batch = self.batch_sampler.start
        for batch in self.loader:
            self.next_batch += 1
            yield batch


def resume_protocol(cfg, loader, world_size):
    """Reject changes that silently change batches, loss, or the LR trajectory."""
    raw = cfg._raw_yaml
    data = dict(raw["data"])
    data.pop("num_workers", None)  # Keyed randomness is independent of workers.
    files = {}
    for key in ("episode_split_path", "norm_stats_path"):
        path = data.get(key)
        if path:
            files[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        "world_size": world_size,
        "seed": cfg.hyperparams.seed,
        "batch_size_per_gpu": cfg.hyperparams.batch_size_per_gpu,
        "gradient_accumulation_steps": cfg.hyperparams.gradient_accumulation_steps,
        "batches_per_epoch": len(loader),
        "dataset_frames": len(loader.original.dataset),
        "data": data,
        "data_file_hashes": files,
        "task": raw.get("task"),
        "model": raw.get("model"),
        "optimizer": raw["hyperparams"].get("optimizer"),
        "scheduler": raw["hyperparams"].get("scheduler"),
        "val_interval": cfg.logging.val_interval,
        "validate_first": cfg.checkpoint.validate_first,
    }


def read_state(path, rank):
    directory = Path(path) / "robotwin_resume"
    if not directory.exists():
        return None
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise ValueError("Incomplete RoboTwin checkpoint: resume metadata missing")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("version") != 1:
        raise ValueError("Unsupported RoboTwin resume metadata version")
    for index in range(metadata["protocol"]["world_size"]):
        if not (directory / f"rank_{index:05d}.pt").is_file():
            raise ValueError("Incomplete RoboTwin checkpoint: rank RNG state missing")
    rng = torch.load(
        directory / f"rank_{rank:05d}.pt", map_location="cpu", weights_only=False
    )
    return metadata, rng


def legacy_cursor(path, loader, accumulation, world_size):
    """Recover a legacy epoch cursor without pretending RNG is exact."""
    path = Path(path)

    def field(name, key):
        return int(torch.load(path / name, map_location="cpu", weights_only=False)[key])

    if field("world_size.pth", "world_size") != world_size:
        raise ValueError("Legacy RoboTwin resume requires the original GPU count")
    step = field("global_step.pth", "global_step")
    epoch = field("current_epoch.pth", "current_epoch")
    if path.name == f"{epoch}_{step}":
        completed = step + 1
        cursor = completed * accumulation - epoch * len(loader)
    elif path.name == str(epoch):
        completed = step
        cursor = len(loader)
        if completed * accumulation != (epoch + 1) * len(loader):
            raise ValueError("Cannot infer legacy RoboTwin epoch-end cursor")
    else:
        raise ValueError("Cannot infer legacy RoboTwin checkpoint cursor from its name")
    if not 0 < cursor <= len(loader) or cursor % accumulation:
        raise ValueError(
            "Legacy checkpoint progress disagrees with the current data layout"
        )
    return epoch, cursor, completed


def validate_legacy_config(path, cfg):
    class SavedConfigLoader(yaml.SafeLoader):
        pass

    SavedConfigLoader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: loader.construct_sequence(node),
    )
    saved = yaml.load((Path(path) / "config.yml").read_text(), Loader=SavedConfigLoader)
    previous = saved.get("_raw_yaml", saved)
    current = cfg._raw_yaml
    for key in ("task", "model", "data"):
        before, after = dict(previous[key]), dict(current[key])
        if key == "data":
            before.pop("num_workers", None)
            after.pop("num_workers", None)
        if before != after:
            raise ValueError(f"Legacy RoboTwin resume configuration changed: {key}")
    for key in (
        "seed",
        "batch_size_per_gpu",
        "gradient_accumulation_steps",
        "optimizer",
        "scheduler",
    ):
        if previous["hyperparams"].get(key) != current["hyperparams"].get(key):
            raise ValueError(f"Legacy RoboTwin resume configuration changed: {key}")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
