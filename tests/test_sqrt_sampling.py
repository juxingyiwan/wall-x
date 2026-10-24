"""Behavioral checks for task balancing and DDP epoch sampling."""

import pytest

from wall_x.data.sampling import (
    DistributedSquareRootSampler,
    lerobot_train_task_ids,
    square_root_probabilities,
)


def test_square_root_probabilities_and_invalid_counts():
    probabilities = square_root_probabilities({"small": 100, "large": 900})
    assert probabilities == {"small": 0.25, "large": 0.75}
    with pytest.raises(ValueError):
        square_root_probabilities({"empty": 0})


def test_distributed_sampling_is_repeatable_and_balances_tasks():
    task_ids = [0] * 100 + [1] * 900
    samplers = [
        DistributedSquareRootSampler(task_ids, num_replicas=8, rank=rank, seed=17)
        for rank in range(8)
    ]
    assert all(len(sampler) == 125 for sampler in samplers)
    first_draw = [list(sampler) for sampler in samplers]
    assert first_draw == [list(sampler) for sampler in samplers]
    for sampler in samplers:
        sampler.set_epoch(1)
    assert first_draw != [list(sampler) for sampler in samplers]

    # Aggregate several independent epochs across all ranks to test the
    # actual draw distribution rather than just the closed-form formula.
    selected = []
    for epoch in range(20):
        for sampler in samplers:
            sampler.set_epoch(epoch)
            selected.extend(task_ids[index] for index in sampler)
    small_share = selected.count(0) / len(selected)
    assert small_share == pytest.approx(0.25, abs=0.02)


def test_lerobot_training_subset_reads_task_index_without_getitem():
    from torch.utils.data import Subset

    class ColumnDataset:
        column_names = ["task_index"]

        def __getitem__(self, key):
            assert key == "task_index"
            return [3, 4, 3, 5]

    class BaseDataset:
        hf_dataset = ColumnDataset()

        def __len__(self):
            return 4

        def __getitem__(self, index):
            raise AssertionError("Video/frame decoding must not be called")

    subset = Subset(BaseDataset(), [3, 0, 2])
    assert lerobot_train_task_ids(subset) == [5, 3, 3]
