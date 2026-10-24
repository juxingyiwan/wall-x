"""Square-root task sampling for LeRobot training.

For task i with n_i training frames, choose it with probability
sqrt(n_i) / sum_j sqrt(n_j), then choose one of its frames uniformly.
The equivalent per-frame weight is 1 / sqrt(n_i). Sampling uses replacement
and preserves the original number of samples per epoch.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Hashable

import torch
from torch.utils.data import Sampler, Subset


def square_root_probabilities(counts: dict[Hashable, int]) -> dict[Hashable, float]:
    """Return normalized task probabilities from positive frame counts."""
    if not counts or any(int(n) != n or n <= 0 for n in counts.values()):
        raise ValueError("counts must contain positive integer frame counts")
    roots = {task: math.sqrt(n) for task, n in counts.items()}
    total = sum(roots.values())
    return {task: value / total for task, value in roots.items()}


def lerobot_train_task_ids(train_dataset: Subset) -> list[int]:
    """Read task IDs for the active training split without decoding videos."""
    base = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
    hf_dataset = getattr(base, "hf_dataset", None)
    if hf_dataset is None or "task_index" not in hf_dataset.column_names:
        raise ValueError("LeRobot dataset must have a loaded task_index column")
    task_column = hf_dataset["task_index"]
    if len(task_column) != len(base):
        raise ValueError("LeRobot task_index column does not match dataset length")
    indices = (
        train_dataset.indices if isinstance(train_dataset, Subset) else range(len(base))
    )
    return [int(task_column[index]) for index in indices]


class DistributedSquareRootSampler(Sampler[int]):
    """Deterministic, replacement-based square-root sampler for DDP.

    All ranks generate the same global draw and take disjoint positions in it.
    An index can repeat because replacement is required to oversample small
    tasks; each rank still receives the same number of samples.
    """

    def __init__(
        self,
        task_ids: Sequence[Hashable],
        *,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if not task_ids:
            raise ValueError("task_ids must contain at least one sample")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.counts = Counter(task_ids)
        self.probabilities = square_root_probabilities(dict(self.counts))
        self.weights = torch.tensor(
            [1.0 / math.sqrt(self.counts[task]) for task in task_ids],
            dtype=torch.double,
        )
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(task_ids) // num_replicas
        if self.num_samples == 0:
            raise ValueError("Training split has fewer samples than DDP ranks")
        self.total_size = self.num_samples * num_replicas

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=generator
        ).tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
