"""Independent update limits preserve the original learning-rate schedule."""

from types import SimpleNamespace

import pytest
import torch

from wall_x.config.hyperparams_config import (
    ConstantSchedulerConfig,
    CosineSchedulerConfig,
)
from wall_x.robotwin.config import RobotwinHyperParams as TrainHyperParams
from wall_x.robotwin.config import RobotwinLoggingConfig as LoggingConfig
from wall_x.robotwin.trainer import RobotwinFSDPTrainer as FSDPTrainer
from wall_x.trainer.scheduler.scheduler import get_scheduler


def scheduler_trace(hp, steps):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=5e-5)
    sched = get_scheduler(
        optimizer=opt,
        lr_scheduler_type="cosine",
        num_warmup_steps=500,
        num_training_steps=hp.scheduler.num_training_steps,
        peak_lr=5e-5,
        end_lr=1e-6,
    )
    trace = []
    for _ in range(steps):
        trace.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return trace


def test_20k_stop_keeps_every_lr_of_original_30k_schedule():
    old = TrainHyperParams(
        scheduler=CosineSchedulerConfig(
            num_training_steps=30000, num_warmup_steps=500, min_lr=1e-6
        )
    )
    new = TrainHyperParams(
        scheduler=CosineSchedulerConfig(
            num_training_steps=30000, num_warmup_steps=500, min_lr=1e-6
        ),
        max_training_steps=20000,
    )
    assert old.training_step_limit == 30000 and new.training_step_limit == 20000
    assert scheduler_trace(old, 20000) == scheduler_trace(new, 20000)
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(
            logging=LoggingConfig(save_interval=2500, save_on_completed_steps=True)
        )
    )
    saved = []
    for completed in range(1, new.training_step_limit + 1):
        trainer.global_step = completed
        if FSDPTrainer._should_save_checkpoint(trainer):
            saved.append(completed)
    assert saved == list(range(2500, 20001, 2500))


def test_uncapped_and_legacy_scheduler_limits():
    assert (
        TrainHyperParams(scheduler=ConstantSchedulerConfig()).training_step_limit == 0
    )
    assert (
        TrainHyperParams(
            scheduler=ConstantSchedulerConfig(), max_training_steps=20000
        ).training_step_limit
        == 20000
    )
    assert (
        TrainHyperParams(
            scheduler=CosineSchedulerConfig(num_training_steps=10000),
            max_training_steps=20000,
        ).training_step_limit
        == 10000
    )


@pytest.mark.parametrize("limit", [0, -1, True, "20000", 2.5])
def test_invalid_update_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="max_training_steps"):
        TrainHyperParams(max_training_steps=limit)
