import copy
import dataclasses
import json

import pytest
import torch
import yaml
from b601_training_fixture import Harness, fixture_config

from wall_x.config import load_config
from wall_x.data.b601 import checkpoint_config, load_b601_norm_stats
from wall_x.data.backends.lerobot.b601_adapter import B601DataCollator
from wall_x.data.backends.lerobot.build import (
    _build_flat_config,
    load_lerobot_normalizers,
)
from wall_x.data.backends.lerobot.loader import DataCollator
from wall_x.trainer.adapters.b601_state import (
    file_sha256,
    input_pipeline_contract,
    load_runtime,
    runtime_contract,
)


def configured_fixture(tmp_path):
    cfg = fixture_config(tmp_path)
    cfg._raw_data.update(
        max_length=1024,
        state_bins=512,
        min_pixels=784,
        max_pixels=200704,
        image_factor=28,
    )
    cfg.data.noise_scheduler = {
        "beta_alpha": 1.25,
        "beta_beta": 1.1,
        "time_shift": 2.0,
        "s": 0.95,
    }
    return cfg


def test_checkpoint_preserves_backend_fields_and_cli_overrides(tmp_path):
    cfg = configured_fixture(tmp_path)
    cfg.data.padding_side = "right"
    cfg.hyperparams.max_optimizer_steps = 2
    trainer = Harness(cfg)
    trainer.fit()
    path = tmp_path / "ckpts/0_2/config.yml"
    saved = yaml.safe_load(path.read_text())
    assert "_raw_data" not in saved and "_raw_yaml" not in saved
    assert saved["data"]["max_length"] == 1024
    assert saved["data"]["padding_side"] == "right"
    assert saved["hyperparams"]["max_optimizer_steps"] == 2
    reloaded = load_config(str(path))
    assert input_pipeline_contract(reloaded) == input_pipeline_contract(cfg)
    assert runtime_contract(reloaded, 1, 6) == runtime_contract(cfg, 1, 6)
    assert _build_flat_config(reloaded)[0]["data"]["max_length"] == 1024
    assert load_lerobot_normalizers(reloaded)
    second = checkpoint_config(reloaded)
    assert "_raw_yaml" not in second and "_raw_data" not in second
    assert second["data"] == saved["data"]


def test_typed_values_win_over_stale_raw_yaml(tmp_path):
    cfg = fixture_config(tmp_path)
    cfg._raw_yaml = copy.deepcopy(cfg._raw_yaml)
    cfg._raw_data["max_length"] = 1024
    cfg._raw_yaml["data"]["max_length"] = 768
    cfg._raw_yaml["data"]["padding_side"] = "left"
    cfg.data.padding_side = "right"
    cfg.data.noise_scheduler = {"time_shift": 3.0}
    merged = checkpoint_config(dataclasses.asdict(cfg))
    assert merged["data"]["max_length"] == cfg._raw_data["max_length"]
    assert merged["data"]["padding_side"] == "right"
    assert merged["data"]["noise_scheduler"] == {"time_shift": 3.0}


def test_cleared_backend_fields_and_empty_typed_dicts_stay_cleared(tmp_path):
    cfg = fixture_config(tmp_path)
    cfg._raw_yaml = copy.deepcopy(cfg._raw_yaml)
    cfg._raw_yaml["data"]["max_length"] = 1024
    cfg._raw_data.pop("max_length", None)
    cfg._raw_data["noise_scheduler"] = {"beta_alpha": 0.5, "time_shift": 3.0}
    cfg.data.noise_scheduler = {}
    merged = checkpoint_config(cfg)
    assert "max_length" not in merged["data"]
    assert merged["data"]["noise_scheduler"] == {}
    path = tmp_path / "cleared.yml"
    path.write_text(yaml.safe_dump(merged, sort_keys=False))
    assert input_pipeline_contract(load_config(str(path))) == input_pipeline_contract(
        cfg
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_length", 768),
        ("padding_side", "right"),
        ("state_bins", 128),
        ("beta_alpha", 0.5),
        ("beta_beta", 2.0),
        ("time_shift", 3.0),
        ("s", 0.9),
        ("min_pixels", 3136),
        ("max_pixels", 100352),
        ("image_factor", 14),
    ],
)
def test_changed_inputs_or_sampling_reject_exact_resume(tmp_path, field, value):
    cfg = configured_fixture(tmp_path)
    before = Harness(copy.deepcopy(cfg))
    before.num_training_steps = 2
    before.fit()
    changed = copy.deepcopy(cfg)
    if field in ("beta_alpha", "beta_beta", "time_shift", "s"):
        changed.data.noise_scheduler[field] = value
    elif field == "padding_side":
        changed.data.padding_side = value
    else:
        changed._raw_data[field] = value
    assert runtime_contract(cfg, 1, 6) != runtime_contract(changed, 1, 6)
    with pytest.raises(ValueError, match="contract mismatch|manifest mismatch"):
        Harness(changed, resume=tmp_path / "ckpts/0_2")


def test_explicit_defaults_bind_actual_collator_and_rng(tmp_path, monkeypatch):
    cfg = fixture_config(tmp_path)
    implicit = input_pipeline_contract(cfg)
    cfg.data.noise_scheduler = dict(implicit["noise_scheduler"])
    assert input_pipeline_contract(cfg) == implicit
    flat, lerobot = _build_flat_config(cfg)
    numeric = load_b601_norm_stats(cfg.data.norm_stats_path, cfg.data.key_mappings)
    monkeypatch.setattr(DataCollator, "load_processor", lambda self: None)
    collator = B601DataCollator(flat, flat["data"], numeric, lerobot)
    assert collator.state_dict()["sampling"] == implicit["noise_scheduler"]
    collator.sample_time(5, "cpu", torch.float32)
    state = collator.state_dict()
    expected = collator.sample_time(5, "cpu", torch.float32)
    collator.load_state_dict(state)
    torch.testing.assert_close(
        collator.sample_time(5, "cpu", torch.float32), expected, atol=0, rtol=0
    )
    collator.time_shift = 3.0
    with pytest.raises(ValueError, match="time-sampling settings mismatch"):
        collator.load_state_dict(state)


@pytest.mark.parametrize(
    "save_interval,epoch_interval", [(4000, 20), (6, 20), (4000, 1)]
)
def test_epoch_limit_saves_and_finished_resume_does_not_resave(
    tmp_path, save_interval, epoch_interval
):
    cfg = fixture_config(tmp_path)
    cfg.hyperparams.num_epoch = 1
    cfg.logging.save_interval = save_interval
    cfg.logging.epoch_save_interval = epoch_interval
    trainer = Harness(copy.deepcopy(cfg))
    trainer.fit()
    path = tmp_path / "ckpts/0_6"
    assert trainer.global_step == 6 and trainer.next_epoch == 1
    assert list((tmp_path / "ckpts").iterdir()) == [path]
    state = load_runtime(path, 0, 1)
    assert state["completed_updates"] == 6 and state["next_batch"] == 0
    assert state["next_epoch"] == 1
    resumed = Harness(cfg, resume=path)
    resumed.cfg.checkpoint.save_path = str(tmp_path / "no-new-updates")
    resumed.fit()
    assert resumed.global_step == 6 and not resumed.trace
    assert not (tmp_path / "no-new-updates").exists()


def test_exception_does_not_save_failed_or_partial_update(tmp_path):
    cfg = fixture_config(tmp_path)
    cfg.logging.save_interval = 1
    trainer = Harness(cfg)
    forward = trainer.adapter.forward

    def fail_second(*args, **kwargs):
        if trainer.global_step == 1:
            raise RuntimeError("injected second-update failure")
        return forward(*args, **kwargs)

    trainer.adapter.forward = fail_second
    with pytest.raises(RuntimeError, match="second-update failure"):
        trainer.fit()
    paths = list((tmp_path / "ckpts").iterdir())
    assert paths == [tmp_path / "ckpts/0_1"]
    assert load_runtime(paths[0], 0, 1)["completed_updates"] == 1


def test_old_checkpoint_without_input_binding_is_not_exact_resume(tmp_path):
    trainer = Harness(fixture_config(tmp_path))
    trainer.num_training_steps = 2
    trainer.fit()
    path = tmp_path / "ckpts/0_2"
    rank_file = path / "b601_training_rank0.pt"
    state = torch.load(rank_file, weights_only=False)
    state["contract"].pop("input_pipeline")
    torch.save(state, rank_file)
    marker = path / "b601_training_state.json"
    meta = json.loads(marker.read_text())
    meta["rank_files"][rank_file.name] = file_sha256(rank_file)
    marker.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="predates complete B601 input"):
        load_runtime(path, 0, 1)
