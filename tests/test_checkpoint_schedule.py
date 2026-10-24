"""Checkpoint boundaries must include the last requested optimizer update."""

from types import SimpleNamespace

from wall_x.robotwin.config import RobotwinLoggingConfig as LoggingConfig
from wall_x.robotwin.trainer import RobotwinFSDPTrainer as FSDPTrainer


def checkpoint_updates(logging, total_updates):
    trainer = SimpleNamespace(cfg=SimpleNamespace(logging=logging))
    saved = []
    for completed in range(1, total_updates + 1):
        trainer.global_step = completed
        if FSDPTrainer._should_save_checkpoint(trainer):
            saved.append(completed)
    return saved


def test_fifty_thousand_updates_save_ten_exact_checkpoints():
    logging = LoggingConfig(save_interval=5000, save_on_completed_steps=True)
    assert checkpoint_updates(logging, 50000) == list(range(5000, 50001, 5000))


def test_legacy_checkpoint_numbering_is_preserved_by_default():
    logging = LoggingConfig(save_interval=5000)
    assert checkpoint_updates(logging, 10001) == [5001, 10001]


def test_completed_updates_include_first_update_and_respect_threshold():
    logging = LoggingConfig(save_interval=1, save_on_completed_steps=True)
    assert checkpoint_updates(logging, 3) == [1, 2, 3]
    logging.ignore_until_interval = 2
    assert checkpoint_updates(logging, 3) == [2, 3]


def test_zero_interval_disables_step_checkpoints():
    logging = LoggingConfig(save_interval=0, save_on_completed_steps=True)
    assert checkpoint_updates(logging, 3) == []
