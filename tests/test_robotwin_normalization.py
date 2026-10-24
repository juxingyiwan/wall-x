"""End-to-end contracts for RoboTwin clipping, padding and saved policies."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from transformers import BatchEncoding

from wall_x.model.core.action.normalizer import Normalizer
from wall_x.robotwin import collator as module
from wall_x.robotwin.normalization import (
    adapt_normalizers,
    resolve_normalization,
    scale_quantiles,
)


def raw_config(action=False, state=False, legacy=False):
    lerobot = {"adapter": "robotwin", "repo_id": "fixture"}
    if not legacy:
        lerobot["normalization"] = {
            "clip_action_targets": action,
            "clip_state_inputs": state,
        }
    return {
        "data": {"lerobot_config": lerobot},
        "dof_total_dim": 4,
        "agent_pos_total_dim": 4,
    }


def sample():
    return {
        "action": torch.tensor([[-6.0, 5.0], [6.0, -1.0]]),
        "agent_pos": torch.tensor([-6.0, 5.0]),
        "image_inputs": [],
        "text": "move",
    }


def normalizer():
    stats = SimpleNamespace(
        min=torch.tensor([-2.0, 1.0]), delta=torch.tensor([4.0, 2.0])
    )
    return Normalizer.from_lerobot_norm_stats(stats, "fixture")


def make_collator(monkeypatch, config):
    def load_processor(self):
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 7)
        )
        self.train_action_tokenizer = None

    monkeypatch.setattr(module.DataCollator, "load_processor", load_processor)
    monkeypatch.setattr(module, "replace_action_token", lambda text, *args: text)
    monkeypatch.setattr(
        module,
        "preprocesser_call",
        lambda **kwargs: BatchEncoding({"input_ids": torch.tensor([[7]])}),
    )
    stats = SimpleNamespace(
        min=torch.tensor([-2.0, 1.0]), delta=torch.tensor([4.0, 2.0])
    )
    return module.RobotwinDataCollator(
        config, {}, {"action": stats, "state": stats}, config["data"]["lerobot_config"]
    )


@pytest.mark.parametrize(
    "clip_action,clip_state",
    [(False, False), (True, True), (False, True), (True, False)],
)
def test_train_inference_policy_agrees_and_padding_stays_zero(
    monkeypatch, clip_action, clip_state
):
    config = raw_config(clip_action, clip_state)
    batch = make_collator(monkeypatch, config)([sample()])
    action_norm, state_norm = adapt_normalizers(config, normalizer(), normalizer())
    for source, key, norm, clip in [
        (sample()["action"][None], "action_chunk", action_norm, clip_action),
        (sample()["agent_pos"][None, None], "proprioception", state_norm, clip_state),
    ]:
        infer = norm.normalize_data(source, ["fixture"])
        torch.testing.assert_close(infer, batch[key][..., :2], atol=0, rtol=0)
        assert batch[key][..., 2:].count_nonzero() == 0
        assert (
            batch["dof_mask" if key == "action_chunk" else "agent_pos_mask"][
                ..., 2:
            ].count_nonzero()
            == 0
        )
        if not clip:
            torch.testing.assert_close(
                norm.unnormalize_data(infer, ["fixture"]), source, atol=0, rtol=0
            )
            assert infer.abs().max() > 1
        else:
            assert infer.abs().max() == 1


def test_native_and_old_robotwin_are_unchanged(monkeypatch):
    config = raw_config(legacy=True)
    norm_a, norm_s = normalizer(), normalizer()
    a, s = adapt_normalizers(config, norm_a, norm_s)
    assert a is norm_a and s is norm_s
    result = make_collator(monkeypatch, config)([sample()])
    assert result["action_chunk"].abs().max() == 1
    assert result["proprioception"].abs().max() == 1
    # Native normalizers remain bounded even after creating a RoboTwin one.
    adapt_normalizers(raw_config(), norm_a, norm_s)
    assert (
        norm_s.normalize_data(sample()["agent_pos"][None, None], ["fixture"])
        .abs()
        .max()
        == 1
    )


def test_earlier_action_only_config_keeps_state_clipping(monkeypatch):
    config = raw_config(legacy=True)
    config["data"]["clip_action_targets"] = False
    result = make_collator(monkeypatch, config)([sample()])
    assert result["action_chunk"].abs().max() > 1
    assert result["proprioception"].abs().max() == 1
    action, state = normalizer(), normalizer()
    actual_action, actual_state = adapt_normalizers(config, action, state)
    assert actual_action is action and actual_state is state


@pytest.mark.parametrize("key", ["clip_action_targets", "clip_state_inputs"])
@pytest.mark.parametrize("value", ["false", None, 0])
def test_malformed_policy_fails_loudly(key, value):
    config = raw_config()
    config["data"]["lerobot_config"]["normalization"][key] = value
    with pytest.raises(ValueError, match="boolean"):
        resolve_normalization(config)


def test_ambiguous_or_unsupported_modes_are_rejected():
    config = raw_config()
    config["data"]["clip_action_targets"] = True
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_normalization(config)
    config = raw_config()
    config["use_fast_tokenizer"] = True
    with pytest.raises(ValueError, match="use_fast_tokenizer"):
        resolve_normalization(config)
    config = raw_config()
    config["task"] = {"use_state_string_representation": True}
    with pytest.raises(ValueError, match="continuous state"):
        resolve_normalization(config)


def test_degenerate_range_is_finite_and_parameter_keys_are_compatible():
    x = scale_quantiles(
        torch.tensor([-2.0, 2.0]), torch.zeros(2), torch.tensor([0.0, 2.0]), clip=False
    )
    torch.testing.assert_close(x, torch.tensor([-5.0, 1.0]))
    original = normalizer()
    _, new = adapt_normalizers(raw_config(), original, original)
    assert new.state_dict().keys() == original.state_dict().keys()
    for key in original.state_dict():
        torch.testing.assert_close(
            new.state_dict()[key], original.state_dict()[key], atol=0, rtol=0
        )


def test_direct_inference_input_keeps_state_tails_and_padding(monkeypatch):
    from wall_x.robotwin import adapter as infer

    obj = infer.RobotwinInferAdapter.__new__(infer.RobotwinInferAdapter)
    obj._train_config = raw_config()
    obj._train_config["agent_pos_config"] = {"joint": 2, "action_padding": 2}
    obj._train_config["data"]["max_length"] = 128
    obj._normalizer_action, obj._normalizer_propri = adapt_normalizers(
        obj._train_config, normalizer(), normalizer()
    )
    obj._norm_key, obj._cam_names, obj._device = "fixture", [], "cpu"
    obj._processor = SimpleNamespace(
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 7)
    )
    monkeypatch.setattr(
        infer,
        "preprocesser_call",
        lambda **kwargs: BatchEncoding({"input_ids": torch.tensor([[7]])}),
    )
    obs = {
        "proprioception": np.array([[[-6.0, 5.0, 0.0, 0.0]]], dtype=np.float32),
        "agent_pos_mask": np.array([[[1.0, 1.0, 0.0, 0.0]]], dtype=np.float32),
        "dof_mask": np.array([[[1.0, 1.0, 0.0, 0.0]]], dtype=np.float32),
    }
    result = obj._construct_model_input([obs], [""], [""])
    torch.testing.assert_close(
        result["proprioception"],
        torch.tensor([[[-3.0, 3.0, 0.0, 0.0]]]),
        atol=0,
        rtol=0,
    )
    assert result["agent_pos_mask"][..., 2:].count_nonzero() == 0


def test_new_checkpoint_config_roundtrip_preserves_policy_and_stop_cap(tmp_path):
    from wall_x.config.loader import save_config
    from wall_x.robotwin.config import load_config

    config = raw_config()
    config.update(
        task={"dof_config": {"joint": 2, "action_padding": 2}},
        model_type="qwen2_5",
        model={"config_path": "unused", "processor_path": "unused"},
        hyperparams={
            "max_training_steps": 20000,
            "scheduler": {"num_training_steps": 200000, "num_warmup_steps": 1000},
        },
        logging={"save_on_completed_steps": True},
    )
    config["data"]["episode_split_path"] = "/unused/split.json"
    path = tmp_path / "input.yml"
    path.write_text(yaml.safe_dump(config))
    cfg = load_config(str(path))
    saved = save_config(cfg, str(tmp_path / "checkpoint"))
    restored = load_config(saved)
    assert resolve_normalization(yaml.safe_load(open(saved))) == resolve_normalization(
        config
    )
    assert restored.hyperparams.training_step_limit == 20000
    assert restored.hyperparams.scheduler.num_training_steps == 200000
    assert restored.hyperparams.scheduler.num_warmup_steps == 1000
    assert restored.logging.save_on_completed_steps is True


def test_legacy_action_only_saved_config_keeps_legacy_inference(tmp_path):
    from wall_x.config.loader import save_config
    from wall_x.robotwin.config import load_config

    raw = raw_config(legacy=True)
    raw.update(
        model_type="qwen2_5",
        task={"dof_config": {"joint": 2}},
        model={"config_path": "unused", "processor_path": "unused"},
    )
    raw["data"]["clip_action_targets"] = False
    path = tmp_path / "input.yml"
    path.write_text(yaml.safe_dump(raw))
    saved = save_config(load_config(str(path)), str(tmp_path / "checkpoint"))
    restored = yaml.safe_load(open(saved))
    assert restored["data"]["clip_action_targets"] is False
    assert "normalization" not in restored["data"]["lerobot_config"]
    a, s = normalizer(), normalizer()
    actual_a, actual_s = adapt_normalizers(restored, a, s)
    assert actual_a is a and actual_s is s
