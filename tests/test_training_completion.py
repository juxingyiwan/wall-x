import json
from types import SimpleNamespace

import pytest
import torch

from wall_x.trainer.adapters.base_adapter import ModelAdapter
from wall_x.trainer.fsdp_trainer import base_trainer, checkpoint_io


@pytest.mark.parametrize(
    "updates,limit,interval,prior,expected",
    [
        (1, 1, 1, None, [1]),
        (5, 3, 20, None, [3]),
        (3, 3, 1, 3, []),
        (3, 3, 20, 2, [3]),
        (2, 0, 20, None, [2]),
    ],
)
def test_fit_saves_completed_terminal_update(
    monkeypatch, updates, limit, interval, prior, expected
):
    monkeypatch.setattr(base_trainer, "barrier", lambda: None)
    saves = []
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(
            checkpoint=SimpleNamespace(validate_first=False),
            debug=SimpleNamespace(profile=False),
            logging=SimpleNamespace(epoch_save_interval=interval),
        ),
        nvtx=False,
        start_epoch=0,
        num_epoch=1,
        global_step=0,
        num_training_steps=limit,
        val_loop=lambda: None,
        _last_saved_step=prior,
    )

    def train_loop(epoch, profiler):
        trainer.global_step += min(updates, limit) if limit else updates

    trainer.train_loop = train_loop
    trainer.save_checkpoint = lambda epoch, step: saves.append(step)
    base_trainer.DistributedTrainer.fit(trainer)
    assert saves == expected


@pytest.mark.parametrize("directory", [False, True])
def test_metadata_copies_actual_model_json(tmp_path, directory):
    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"tiny"}')
    out = tmp_path / "checkpoint"
    out.mkdir()
    checkpoint_io._save_training_checkpoint_metadata(
        ckpt_path=str(out),
        config={
            "model": {
                "config_path": str(source if directory else source / "config.json")
            }
        },
        epoch=0,
        global_step=1,
        seed=42,
        world_size=1,
        normalizer_action=torch.nn.Identity(),
        normalizer_propri=torch.nn.Identity(),
        log_fn=lambda *a, **kw: None,
    )
    assert json.loads((out / "config.json").read_text()) == {"model_type": "tiny"}


def test_missing_model_config_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model config"):
        checkpoint_io._save_training_checkpoint_metadata(
            ckpt_path=str(tmp_path),
            config={"model": {"config_path": str(tmp_path / "missing")}},
            epoch=0,
            global_step=1,
            seed=42,
            world_size=1,
            normalizer_action=None,
            normalizer_propri=None,
            log_fn=lambda *a, **kw: None,
        )


@pytest.mark.parametrize("complete,epoch,offset", [(False, 2, 8), (True, 3, 0)])
def test_resume_counter_and_epoch_semantics(tmp_path, complete, epoch, offset):
    (tmp_path / "train_progress.json").write_text(
        json.dumps(
            {
                "version": 1,
                "completed_updates": 7,
                "epoch": 2,
                "epoch_complete": complete,
                "batches_consumed": 8,
            }
        )
    )
    state = ModelAdapter.load_step_and_epoch(None, str(tmp_path), False)
    assert state == {
        "global_step": 7,
        "start_epoch": epoch,
        "batches_consumed": offset,
        "completed_updates": True,
    }


def test_legacy_counter_is_still_identified(tmp_path):
    torch.save({"global_step": 6}, tmp_path / "global_step.pth")
    torch.save({"current_epoch": 2}, tmp_path / "current_epoch.pth")
    state = ModelAdapter.load_step_and_epoch(None, str(tmp_path), False)
    assert state == {"global_step": 6, "start_epoch": 3}


def test_final_epoch_still_validates(monkeypatch):
    monkeypatch.setattr(base_trainer, "barrier", lambda: None)
    validations = []
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(
            checkpoint=SimpleNamespace(validate_first=False),
            debug=SimpleNamespace(profile=False),
            logging=SimpleNamespace(epoch_save_interval=10),
        ),
        nvtx=False,
        start_epoch=0,
        num_epoch=1,
        global_step=0,
        num_training_steps=0,
        val_loop=lambda: validations.append(True),
        save_checkpoint=lambda *a: None,
    )
    trainer.train_loop = lambda *a, **kw: setattr(trainer, "global_step", 1)
    base_trainer.DistributedTrainer.fit(trainer)
    assert validations == [True]


@pytest.mark.parametrize("scenario", ["libero", "robotwin"])
@pytest.mark.parametrize(
    "batches,limit,interval,expected_saves,complete",
    [(5, 1, 10, 1, False), (4, 2, 2, 1, True), (6, 2, 1, 2, False)],
)
def test_real_cpu_optimizer_loop_saves_final_weights(
    tmp_path, batches, limit, interval, expected_saves, complete, scenario
):
    import contextlib
    from pathlib import Path

    from safetensors.torch import load_file

    from wall_x.config.loader import load_config
    from wall_x.trainer.fsdp_trainer.fsdp_trainer import FSDPTrainer

    class Timer:
        def __call__(self, *args, **kwargs):
            return self

        def start(self, *args, **kwargs):
            pass

        def stop(self, *args, **kwargs):
            pass

    cfg = load_config(
        Path(__file__).resolve().parents[1] / "workspace/example/libero.yml"
    )
    model_json = tmp_path / "config.json"
    model_json.write_text("{}")
    cfg.model.config_path = str(model_json)
    cfg.model.processor_path = None
    cfg.data.norm_stats_path = None
    cfg.checkpoint.save_path = str(tmp_path / "checkpoints")
    cfg.checkpoint.validate_first = False
    cfg.hyperparams.optimizer.enable_grad_clip = False
    cfg.logging.save_interval = interval
    cfg.logging.ignore_until_interval = 0
    cfg.logging.epoch_save_interval = 10
    if scenario == "robotwin":
        from wall_x.robotwin.trainer import RobotwinFSDPTrainer

        trainer = object.__new__(RobotwinFSDPTrainer)
        cfg.logging.save_on_completed_steps = True
        cfg.data.lerobot_config["adapter"] = "robotwin"
        trainer._pending_rng = None
        trainer._pending_validation = trainer._pending_epoch_validation = False
        trainer._resume_batch = 0
        trainer._resume_protocol = {"world_size": 1}
    else:
        trainer = object.__new__(FSDPTrainer)
    trainer.cfg = cfg
    trainer.model = torch.nn.Linear(1, 1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer, lambda step: 1.0
    )
    trainer.adapter = SimpleNamespace(
        forward=lambda model, batch, **kw: model(batch["x"]).square().mean(),
        extract_loss=lambda output: output,
        collect_grad_norms=lambda *a, **kw: None,
    )
    trainer.strategy = SimpleNamespace(
        params_sharded=False, no_sync=lambda model: contextlib.nullcontext()
    )
    trainer.train_dataloader = [
        {"x": torch.tensor([[float(i + 1)]])} for i in range(batches)
    ]
    if scenario == "robotwin":
        from wall_x.robotwin.resume import ReplayableLoader

        original = torch.utils.data.DataLoader(trainer.train_dataloader, batch_size=1)
        trainer.train_dataloader = ReplayableLoader(original, 42, 0, "train")
    trainer.dataset = SimpleNamespace(
        get_train_dataloader=lambda: trainer.train_dataloader
    )
    trainer.device = "cpu"
    trainer.use_amp = False
    trainer.nvtx = False
    trainer.start_epoch = trainer.global_step = trainer.micro_step = (
        trainer.initial_step
    ) = 0
    trainer.rank = 0
    trainer.world_size = 1
    trainer.seed = 42
    trainer.grad_accum_steps = 2
    trainer.grad_scaler = None
    trainer.num_epoch = 1
    trainer.num_training_steps = limit
    trainer.show_time_details = False
    trainer.timers = Timer()
    trainer.normalizer_action = trainer.normalizer_propri = torch.nn.Identity()
    trainer._frozen_prefixes = None
    trainer.log = trainer._log_training_metrics = trainer.training_log = (
        lambda *a, **kw: None
    )
    trainer._should_validate = lambda: False
    trainer.val_loop = lambda: None
    trainer.fit()
    checkpoints = list((tmp_path / "checkpoints").iterdir())
    assert len(checkpoints) == expected_saves
    final = tmp_path / "checkpoints" / f"0_{limit}"
    assert trainer.global_step == limit
    progress = json.loads((final / "train_progress.json").read_text())
    assert progress["completed_updates"] == limit
    if scenario == "robotwin":
        from wall_x.robotwin.resume import read_state

        metadata, _ = read_state(final, 0)
        assert metadata["completed_updates"] == limit
        assert metadata["next_batch"] == limit * 2
    assert progress["batches_consumed"] == limit * 2
    assert progress["epoch_complete"] is complete
    for key, value in load_file(str(final / "model.safetensors")).items():
        torch.testing.assert_close(value, trainer.model.state_dict()[key])
    assert torch.load(final / "scheduler.pt")["last_epoch"] == limit
