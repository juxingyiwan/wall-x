"""Explicit episode splits shared by training and normalization."""

import json
from pathlib import Path

import numpy as np


def load_episode_split(path, available=None):
    """Read disjoint, nonempty episode IDs and optionally require exact coverage."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result = []
    for name in ("train", "validation"):
        ids = data.get(name)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Episode split requires a nonempty {name} list")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in ids):
            raise ValueError(f"{name} episode IDs must be nonnegative integers")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate episode IDs in {name}")
        result.append(ids)
    train, validation = result
    if set(train) & set(validation):
        raise ValueError("Training and validation episodes overlap")
    if available is not None:
        expected = set(int(v) for v in available)
        actual = set(train) | set(validation)
        if actual != expected:
            raise ValueError(
                f"Episode split coverage mismatch: missing={sorted(expected-actual)}, "
                f"unknown={sorted(actual-expected)}"
            )
    return train, validation


def split_frame_indices(episode_ids, path):
    """Return frame indices without decoding images or splitting trajectories."""
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    train, validation = load_episode_split(path, np.unique(episode_ids))
    return (
        np.flatnonzero(np.isin(episode_ids, train)).tolist(),
        np.flatnonzero(np.isin(episode_ids, validation)).tolist(),
    )


def episode_slices(episode_ids, frame_indices=None):
    """Validate chronological contiguous episodes and return their slices."""
    ids = np.asarray(episode_ids, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("Expected a nonempty vector of episode IDs")
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    starts = ids[boundaries[:-1]]
    if len(np.unique(starts)) != len(starts):
        raise ValueError("Episode rows must be contiguous")
    frames = None if frame_indices is None else np.asarray(frame_indices)
    if frames is not None and frames.shape != ids.shape:
        raise ValueError("frame_index and episode_index lengths differ")
    slices = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if frames is not None and not np.array_equal(
            frames[start:end], np.arange(end - start)
        ):
            raise ValueError(
                "Each episode must contain consecutive frames starting at 0"
            )
        slices.append(slice(int(start), int(end)))
    return slices
