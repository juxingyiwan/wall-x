"""Exercise exact-resume boundaries with real CPU optimizer/scaler state."""

import copy
import importlib.util
import json
import logging
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from b601_training_fixture import Harness, fixture_config
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP

from wall_x.config import load_config
from wall_x.data.backends.lerobot.build import load_lerobot_normalizers
from wall_x.trainer.adapters.b601_state import restore_rng, runtime_contract
from wall_x.trainer.adapters.vla_model_adapter import VLAdapter

# The Qwen package __init__ eagerly imports CUDA model kernels. Load its actual
# standalone configuration module so these CPU tests still use the native class.
_config_path = (
    Path(__file__).resolve().parents[1]
    / "wall_x/model/qact/qwen2_5/configuration_qwen2_5_vl.py"
)
_config_spec = importlib.util.spec_from_file_location(
    "b601_test_qwen_config", _config_path
)
_config_module = importlib.util.module_from_spec(_config_spec)
_config_spec.loader.exec_module(_config_module)
Qwen2_5_VLConfig = _config_module.Qwen2_5_VLConfig


class ConfigOnlyAdapter(VLAdapter):
    """Use the real native config loader without importing CUDA model kernels."""

    MODEL_TYPE = "qwen2_5"

    @classmethod
    def model_class(cls):
        return None

    @classmethod
    def config_class(cls):
        return Qwen2_5_VLConfig


def scaled_trainer(cfg, resume=None):
    trainer = Harness(copy.deepcopy(cfg))
    trainer.grad_scaler = torch.amp.GradScaler(
        "cpu", init_scale=1024, growth_interval=3
    )
    if resume:
        trainer.cfg.checkpoint.resume_from = str(resume)
        trainer.resume_from_checkpoint()
        trainer.load_dataset()
        restore_rng(trainer._resume_state["rng"])
    return trainer


def scaled_checkpoint(tmp_path):
    cfg = fixture_config(tmp_path, updates=2)
    cfg.distributed.use_amp = True
    cfg.distributed.bf16 = False
    trainer = scaled_trainer(cfg)
    trainer.fit()
    return cfg, trainer, tmp_path / "ckpts/0_2"


def test_scaler_resume_matches_uninterrupted_updates(tmp_path):
    cfg = fixture_config(tmp_path)
    cfg.distributed.use_amp = True
    cfg.distributed.bf16 = False
    continuous = scaled_trainer(cfg)
    continuous.cfg.checkpoint.save_path = str(tmp_path / "continuous")
    continuous.fit()
    before = scaled_trainer(cfg)
    before.num_training_steps = 2
    before.fit()
    saved = before.grad_scaler.state_dict()
    assert saved["_growth_tracker"] == 2
    resumed = scaled_trainer(cfg, tmp_path / "ckpts/0_2")
    assert resumed.grad_scaler.state_dict() == saved
    resumed.cfg.checkpoint.save_path = str(tmp_path / "resumed")
    resumed.fit()
    assert continuous.trace == before.trace + resumed.trace
    assert continuous.grad_scaler.state_dict() == resumed.grad_scaler.state_dict()
    for a, b in zip(continuous.parameters_snapshot(), resumed.parameters_snapshot()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("damage", ["missing", "corrupt", "empty", "changed"])
def test_incomplete_scaler_refused_before_model_restore(tmp_path, damage):
    cfg, before, checkpoint = scaled_checkpoint(tmp_path)
    path = checkpoint / "grad_scaler.pt"
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"broken checkpoint")
    elif damage == "empty":
        torch.save({}, path)
    else:
        state = before.grad_scaler.state_dict()
        state["scale"] *= 2
        torch.save(state, path)
    resumed = scaled_trainer(cfg)
    weights = resumed.parameters_snapshot()
    initial_scaler = resumed.grad_scaler.state_dict()
    resumed.cfg.checkpoint.resume_from = str(checkpoint)
    with pytest.raises(ValueError, match="grad_scaler|GradScaler|metadata checksum"):
        resumed.resume_from_checkpoint()
    assert resumed.grad_scaler.state_dict() == initial_scaler
    assert not resumed.optimizer.state
    for a, b in zip(weights, resumed.parameters_snapshot()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("damage", [None, "missing", "empty", "corrupt"])
def test_legacy_scaler_without_checksum_still_requires_valid_state(tmp_path, damage):
    cfg, before, checkpoint = scaled_checkpoint(tmp_path)
    marker = checkpoint / "b601_training_state.json"
    meta = json.loads(marker.read_text())
    meta["metadata_files"].pop("grad_scaler.pt", None)
    marker.write_text(json.dumps(meta))
    if damage == "missing":
        (checkpoint / "grad_scaler.pt").unlink()
    elif damage == "empty":
        torch.save({}, checkpoint / "grad_scaler.pt")
    elif damage == "corrupt":
        (checkpoint / "grad_scaler.pt").write_bytes(b"invalid pickle")
    if damage:
        with pytest.raises(ValueError, match="GradScaler|grad_scaler"):
            scaled_trainer(cfg, checkpoint)
    else:
        resumed = scaled_trainer(cfg, checkpoint)
        assert resumed.grad_scaler.state_dict() == before.grad_scaler.state_dict()


def test_bf16_resume_does_not_require_scaler_file(tmp_path):
    cfg = fixture_config(tmp_path, updates=2)
    before = Harness(copy.deepcopy(cfg))
    before.fit()
    checkpoint = tmp_path / "ckpts/0_2"
    assert not (checkpoint / "grad_scaler.pt").exists()
    resumed = Harness(cfg, resume=checkpoint)
    assert resumed.global_step == 2


def test_directory_config_load_save_resume_and_fingerprint(tmp_path):
    cfg = fixture_config(tmp_path / "fixture", updates=2)
    directory = tmp_path / "model"
    Qwen2_5_VLConfig().save_pretrained(directory)
    config_file = directory / "config.json"
    cfg.model.config_path = str(config_file)
    expected = runtime_contract(cfg, 1, 6)
    cfg.model.config_path = str(directory)
    adapter = ConfigOnlyAdapter(cfg=cfg, logger=logging.getLogger(__name__))
    assert adapter.build_model_config().model_type == "qwen2_5_vl"
    assert runtime_contract(cfg, 1, 6) == expected
    before = Harness(copy.deepcopy(cfg))
    before.fit()
    checkpoint = tmp_path / "fixture/ckpts/0_2"
    assert (checkpoint / "config.json").read_bytes() == config_file.read_bytes()
    reloaded = load_config(str(checkpoint / "config.yml"))
    assert runtime_contract(reloaded, 1, 6) == before._training_contract
    assert Harness(copy.deepcopy(cfg), resume=checkpoint).global_step == 2
    config_file.write_text(config_file.read_text() + "\n")
    with pytest.raises(ValueError, match="contract mismatch"):
        Harness(copy.deepcopy(cfg), resume=checkpoint)


def test_directory_versioned_config_uses_native_selection(tmp_path):
    cfg = fixture_config(tmp_path / "fixture", updates=2)
    directory = tmp_path / "model"
    Qwen2_5_VLConfig().save_pretrained(directory)
    config_file = directory / "config.json"
    content = json.loads(config_file.read_text())
    content["configuration_files"] = ["config.4.0.0.json"]
    config_file.write_text(json.dumps(content))
    selected = directory / "config.4.0.0.json"
    selected.write_text(json.dumps({**content, "audit_marker": "selected"}))
    cfg.model.config_path = str(selected)
    expected = runtime_contract(cfg, 1, 6)
    cfg.model.config_path = str(directory)
    adapter = ConfigOnlyAdapter(cfg=cfg, logger=logging.getLogger(__name__))
    assert adapter.build_model_config().audit_marker == "selected"
    assert runtime_contract(cfg, 1, 6) == expected
    trainer = Harness(cfg)
    trainer.fit()
    assert (
        tmp_path / "fixture/ckpts/0_2/config.json"
    ).read_bytes() == selected.read_bytes()


@pytest.fixture
def single_rank_group(tmp_path):
    dist.init_process_group(
        "gloo",
        init_method=(tmp_path / "rendezvous").as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def normalizer_trainer(cfg, *, ddp, resume=None):
    trainer = Harness(copy.deepcopy(cfg))
    loaded = load_lerobot_normalizers(trainer.cfg)
    trainer.normalizer_action, trainer.normalizer_propri = loaded[:2]
    trainer.model.action_preprocessor = torch.nn.Module()
    trainer.model.action_preprocessor.normalizer_action = trainer.normalizer_action
    trainer.model.action_preprocessor.normalizer_propri = trainer.normalizer_propri
    if ddp:
        trainer.model = DDP(trainer.model)
    if resume:
        trainer.cfg.checkpoint.resume_from = str(resume)
        trainer.resume_from_checkpoint()
        trainer.load_dataset()
        restore_rng(trainer._resume_state["rng"])
    return trainer


@pytest.mark.parametrize("ddp", [False, True])
def test_normalizer_resume_matches_continuous_training(
    tmp_path, single_rank_group, ddp
):
    cfg = fixture_config(tmp_path / "fixture", updates=4)
    continuous = normalizer_trainer(cfg, ddp=ddp)
    continuous.cfg.checkpoint.save_path = str(tmp_path / "continuous")
    continuous.fit()
    before = normalizer_trainer(cfg, ddp=ddp)
    # Successful log messages may also contain failure words in user paths.
    before.cfg.checkpoint.save_path = str(tmp_path / "failed-experiment")
    before.num_training_steps = 2
    before.fit()
    checkpoint = tmp_path / "failed-experiment/0_2"
    assert (
        len(
            [
                key
                for key in load_file(checkpoint / "model.safetensors")
                if "normalizer" in key
            ]
        )
        == 4
    )
    resumed = normalizer_trainer(cfg, ddp=ddp, resume=checkpoint)
    assert resumed.global_step == 2
    torch.testing.assert_close(
        resumed.optimizer.state_dict(), before.optimizer.state_dict(), atol=0, rtol=0
    )
    assert resumed.lr_scheduler.state_dict() == before.lr_scheduler.state_dict()
    resumed.cfg.checkpoint.save_path = str(tmp_path / "resumed")
    resumed.fit()
    assert continuous.trace == before.trace + resumed.trace
    for left, right in zip(
        continuous.parameters_snapshot(), resumed.parameters_snapshot()
    ):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    torch.testing.assert_close(
        resumed.optimizer.state_dict(),
        continuous.optimizer.state_dict(),
        atol=0,
        rtol=0,
    )
    assert resumed.lr_scheduler.state_dict() == continuous.lr_scheduler.state_dict()


@pytest.mark.parametrize(
    "damage",
    [
        "missing_weight",
        "unexpected_weight",
        "optimizer_groups",
        "scheduler_type",
        "weight_shape",
    ],
)
def test_normalizer_does_not_hide_restore_failures(tmp_path, single_rank_group, damage):
    cfg = fixture_config(tmp_path / "fixture", updates=2)
    before = normalizer_trainer(cfg, ddp=True)
    before.fit()
    checkpoint = tmp_path / "fixture/ckpts/0_2"
    if damage in ("missing_weight", "unexpected_weight", "weight_shape"):
        path = checkpoint / "model.safetensors"
        state = load_file(path)
        if damage == "missing_weight":
            del state["layer.bias"]
        elif damage == "unexpected_weight":
            state["unexpected.weight"] = torch.zeros(1)
        else:
            state["layer.bias"] = torch.zeros(3)
        save_file(state, path)
    elif damage == "optimizer_groups":
        path = checkpoint / "optimizer.pt"
        state = torch.load(path, weights_only=False)
        state["param_groups"][0]["params"] = []
        torch.save(state, path)
    else:
        torch.save(None, checkpoint / "scheduler.pt")
    resumed = normalizer_trainer(cfg, ddp=True)
    resumed.cfg.checkpoint.resume_from = str(checkpoint)
    inner = resumed.model.module
    hooks = tuple(inner._load_state_dict_post_hooks)
    with pytest.raises(
        (ValueError, RuntimeError), match="partial .*restore|size mismatch"
    ):
        resumed.resume_from_checkpoint()
    assert tuple(inner._load_state_dict_post_hooks) == hooks
