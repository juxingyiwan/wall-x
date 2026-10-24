"""A validation pass must not switch an in-flight training iterator's split."""

import pytest
from torch.utils.data import default_collate

from wall_x.data.backends.lerobot import loader
from wall_x.robotwin import collator
from wall_x.robotwin.dataset import RobotwinPreprocessedDataset


class ScalarDataset(RobotwinPreprocessedDataset):
    def __getitem__(self, index):
        return self._dataset[index]


@pytest.mark.parametrize("num_workers", [0, 1])
def test_train_and_validation_iterators_keep_their_own_split(monkeypatch, num_workers):
    dataset = object.__new__(ScalarDataset)
    dataset.train_dataset = list(range(12))
    dataset.val_dataset = [100, 101, 102]
    dataset._train()
    dataset.config = {"batch_size_per_gpu": 1, "num_workers": num_workers}
    dataset.dataload_config = {}
    dataset.norm_stats = {}
    dataset.lerobot_config = {}
    dataset.seed, dataset.rank, dataset.world_size = 42, 0, 1
    monkeypatch.setattr(loader, "DataCollator", lambda *args: default_collate)
    monkeypatch.setattr(collator, "RobotwinDataCollator", lambda *args: default_collate)

    train, _ = dataset.get_train_dataloader()
    train_iterator = iter(train)
    train_values = [next(train_iterator).item()]

    validation, _ = dataset.get_val_dataloader()
    validation_iterator = iter(validation)
    validation_values = [next(validation_iterator).item()]
    train_values.extend(batch.item() for batch in train_iterator)

    # Creating the next epoch's training loader must not switch validation either.
    next_epoch, _ = dataset.get_train_dataloader()
    validation_values.extend(batch.item() for batch in validation_iterator)

    assert sorted(train_values) == list(range(12))
    assert validation_values == [100, 101, 102]
    assert sorted(batch.item() for batch in next_epoch) == list(range(12))
