"""Deterministic B601 episode selection without a separate experiment file."""

import copy
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import numpy as np

from wall_x.utils.validation import is_builtin_int

VALIDATION_RATIO_EPS = 1e-6


def _validation_count(size, ratio):
    if type(ratio) not in (int, float) or not np.isfinite(ratio) or not 0 <= ratio < 1:
        raise ValueError("validation_ratio must be finite and satisfy 0 <= ratio < 1")
    if ratio <= VALIDATION_RATIO_EPS:
        return 0
    # Decimal avoids flooring e.g. 100 * 0.29 to 28 due to binary roundoff.
    return max(1, int(Decimal(size) * Decimal(str(ratio))))


def stratified_split(
    episodes, seed=10222, validation_episodes_per_task=None, validation_ratio=None
):
    if validation_ratio is not None:
        if validation_episodes_per_task is not None:
            raise ValueError(
                "Set only validation_ratio or validation_episodes_per_task"
            )
        _validation_count(1, validation_ratio)
    if validation_episodes_per_task is not None and (
        not is_builtin_int(validation_episodes_per_task)
        or validation_episodes_per_task < 1
    ):
        raise ValueError("validation_episodes_per_task must be a positive integer")
    rng = np.random.default_rng(seed)
    groups = {}
    for episode, values in sorted(episodes.items()):
        groups.setdefault(int(values["task_index"][0]), []).append(episode)
    train, val, counts = [], [], []
    for ids in groups.values():
        ids = rng.permutation(ids).tolist()
        if validation_ratio is not None:
            count = _validation_count(len(ids), validation_ratio)
        else:
            count = (
                validation_episodes_per_task
                if validation_episodes_per_task is not None
                else max(1, round(len(ids) * 0.1))
            )
        if count >= len(ids):
            raise ValueError("Each task must retain at least one training episode")
        val.extend(ids[:count])
        train.extend(ids[count:])
        counts.append(count)
    split = {
        "version": 1,
        "seed": seed,
        "train_episodes": sorted(train),
        "validation_episodes": sorted(val),
    }
    if validation_episodes_per_task is not None:
        split["validation_episodes_per_task"] = validation_episodes_per_task
    elif validation_ratio is not None and len(set(counts)) == 1:
        # Keep the exact existing assignment/checksum when 1% still means one
        # episode per task. The requested ratio itself is saved in config.yml.
        split["validation_episodes_per_task"] = counts[0]
    return split


@lru_cache(maxsize=8)
def _from_numeric_sources(sources, seed, count, ratio):
    import pyarrow.parquet as pq

    tasks = {}
    for filename, _size, _modified in sources:
        cols = pq.read_table(filename, columns=["episode_index", "task_index"])
        episode_ids = np.asarray(cols["episode_index"], dtype=np.int64)
        task_ids = np.asarray(cols["task_index"], dtype=np.int64)
        for episode in np.unique(episode_ids):
            values = np.unique(task_ids[episode_ids == episode])
            if len(values) != 1 or episode < 0:
                raise ValueError("B601 split requires one task per valid episode")
            episode, task = int(episode), int(values[0])
            if episode in tasks and tasks[episode] != task:
                raise ValueError("B601 episode has inconsistent tasks across files")
            tasks[episode] = task
    if not tasks:
        raise ValueError("No B601 episode/task rows found")
    return stratified_split(
        {ep: {"task_index": [task]} for ep, task in tasks.items()}, seed, count, ratio
    )


def resolve_split(config):
    """Read an explicit saved split, or derive it from seed and validation ratio.

    Only episode/task columns are read; no normalization is fitted and no file
    is created. Saved checkpoints embed the resolved lists for offline reload.
    """
    from wall_x.data.b601 import read_split

    data = config.get("data", {})
    lerobot = data.get("lerobot_config") or config.get("lerobot_config") or {}
    opts = lerobot["b601"]
    explicit = opts.get("split_manifest")
    if explicit is not None:
        return read_split(explicit)
    count = opts.get("validation_episodes_per_task")
    ratio = opts.get("validation_ratio")
    if ratio is None and (not is_builtin_int(count) or count < 1):
        raise ValueError(
            "Automatic B601 split requires validation_ratio or validation_episodes_per_task >= 1"
        )
    if ratio is not None:
        if count is not None:
            raise ValueError(
                "Set only validation_ratio or validation_episodes_per_task"
            )
        _validation_count(1, ratio)
    seed = config.get("hyperparams", {}).get(
        "seed", config.get("seed", data.get("seed", 10222))
    )
    if not is_builtin_int(seed) or seed < 0:
        raise ValueError("Automatic B601 split requires a nonnegative integer seed")
    root = Path(lerobot.get("root") or lerobot["repo_id"]).resolve()
    sources = []
    for path in sorted((root / "data").rglob("*.parquet")):
        stat = path.stat()
        sources.append((str(path), stat.st_size, stat.st_mtime_ns))
    if not sources:
        raise ValueError(f"No B601 numeric dataset files under {root / 'data'}")
    return read_split(
        copy.deepcopy(_from_numeric_sources(tuple(sources), seed, count, ratio))
    )
