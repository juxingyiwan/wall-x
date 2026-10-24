"""Regression tests for normalized virtual padding in flow generation."""

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "flow_padding_normalizer", ROOT / "wall_x/model/core/action/normalizer.py"
)
normalizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(normalizer_module)
Normalizer = normalizer_module.Normalizer
build_padding = normalizer_module.build_flow_padding_action


def make_normalizer(width, name="robot", shift=0):
    minimum = torch.arange(width, dtype=torch.float32) - 3 + shift
    delta = torch.arange(width, dtype=torch.float32) + 2
    stats = SimpleNamespace(min=minimum, delta=delta)
    return Normalizer.from_lerobot_norm_stats(stats, name)


@pytest.mark.parametrize("real_width", [7, 14, 20])
@pytest.mark.parametrize("padded_stats", [False, True])
def test_virtual_endpoint_matches_training_with_short_or_legacy_stats(
    real_width, padded_stats
):
    normalizer = make_normalizer(real_width)
    if padded_stats:
        normalizer_module.pad_normalizer_to_dim(normalizer, 26, "action")
    before = {k: v.clone() for k, v in normalizer.state_dict().items()}
    mask = torch.zeros(1, 32, 26)
    mask[..., :real_width] = 1
    result = build_padding(
        normalizer,
        ["robot"],
        mask,
        {"joints": real_width, "action_padding": 26 - real_width},
    )
    assert result.shape == (1, 1, 26)
    assert result[..., real_width:].count_nonzero() == 0
    expected = (
        (-before["min.robot"][:real_width] / before["delta.robot"][:real_width]) * 2 - 1
    ).clamp(-1, 1)
    torch.testing.assert_close(result[0, 0, :real_width], expected)
    for key, value in normalizer.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_virtual_noise_path_matches_training_at_every_timestep():
    normalizer = make_normalizer(7)
    normalizer_module.pad_normalizer_to_dim(normalizer, 26, "action")
    mask = torch.zeros(1, 32, 26)
    mask[..., :7] = 1
    padding = build_padding(
        normalizer, "robot", mask, {"joints": 7, "action_padding": 19}
    )
    noise = torch.randn(1, 32, 26, generator=torch.Generator().manual_seed(7))
    velocity = padding - noise
    for t in (0.0, 0.1, 0.5, 0.9, 0.999):
        inferred = noise + t * velocity
        training = (1 - t) * noise  # The normalized virtual training target is zero.
        torch.testing.assert_close(inferred[..., 7:], training[..., 7:])


def test_missing_real_joint_retains_raw_zero_endpoint():
    normalizer = make_normalizer(26)
    mask = torch.zeros(1, 10, 26)
    mask[..., :7] = 1
    mask[..., 2] = 0
    result = build_padding(
        normalizer, ["robot"], mask, {"joints": 7, "action_padding": 19}
    )
    legacy = normalizer.normalize_data(torch.zeros(1, 1, 26), ["robot"])
    torch.testing.assert_close(result[..., :7], legacy[..., :7], rtol=0, atol=0)
    assert result[..., 2].item() != 0
    assert result[..., 7:].count_nonzero() == 0


def test_no_virtual_padding_preserves_existing_normalization():
    normalizer = make_normalizer(26)
    mask = torch.ones(1, 10, 26)
    mask[..., 6] = 0
    expected = normalizer.normalize_data(torch.zeros(1, 1, 26), ["robot"])
    result = build_padding(normalizer, ["robot"], mask, {"joints": 26})
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_batches_use_their_own_stats_and_broadcast_over_horizon():
    normalizer = make_normalizer(26)
    other = make_normalizer(26, "other", shift=3)
    normalizer.min["other"] = other.min["other"]
    normalizer.delta["other"] = other.delta["other"]
    mask = torch.zeros(2, 32, 26)
    mask[..., :7] = 1
    result = build_padding(
        normalizer, ["robot", "other"], mask, {"joints": 7, "action_padding": 19}
    )
    expected = normalizer.normalize_data(torch.zeros(2, 1, 26), ["robot", "other"])
    assert result.shape == (2, 1, 26)
    assert (result - torch.zeros_like(mask)).shape == mask.shape
    torch.testing.assert_close(result[..., :7], expected[..., :7])
    assert result[..., 7:].count_nonzero() == 0


def test_rejects_config_width_mismatch():
    with pytest.raises(ValueError, match="width"):
        build_padding(
            make_normalizer(26), ["robot"], torch.zeros(1, 32, 26), {"joints": 7}
        )


def test_rejects_active_virtual_dimensions():
    with pytest.raises(ValueError, match="inactive"):
        build_padding(
            make_normalizer(26),
            ["robot"],
            torch.ones(1, 32, 26),
            {"joints": 7, "action_padding": 19},
        )


@pytest.mark.parametrize("as_numpy", [False, True])
def test_explicit_padding_preserves_real_channels_and_input(as_numpy):
    values = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    if as_numpy:
        values = values.numpy()
    before = torch.as_tensor(values).clone()
    result = normalizer_module.zero_virtual_padding(
        values, {"left": 2, "action_padding": 2, "right": 2}
    )
    result = torch.as_tensor(result)
    torch.testing.assert_close(torch.as_tensor(values), before)
    torch.testing.assert_close(result[..., :2], before[..., :2])
    torch.testing.assert_close(result[..., 4:], before[..., 4:])
    assert result[..., 2:4].count_nonzero() == 0


@pytest.mark.parametrize(
    "robot_class",
    ["Robot", "EX001RobotPreprocessor", "DesktopRobotPreprocessor", "DesktopRobotVGA"],
)
@pytest.mark.parametrize("real_width", [7, 20])
def test_native_robot_masks_follow_layout_without_hardware(robot_class, real_width):
    from wall_x._vendor.harrix.serving._wallx_infer import robot

    cls = getattr(robot, robot_class)
    instance = cls.__new__(cls)
    instance.config = SimpleNamespace(
        train_config={
            "task": {
                "dof_config": {"joints": real_width, "action_padding": 26 - real_width}
            }
        },
        action_horizon=32,
    )
    mask = instance._get_dof_mask()
    assert mask.shape == (1, 32, 26)
    assert np.all(mask[..., :real_width] == 1)
    assert np.all(mask[..., real_width:] == 0)


@pytest.mark.parametrize("mode", ["zero", "original"])
@pytest.mark.parametrize(
    "adapter_kind,real_width", [("serving", 7), ("serving", 20), ("libero", 8)]
)
def test_native_input_builder_matches_normalize_then_pad(
    monkeypatch, adapter_kind, real_width, mode
):
    layout = {"joints": real_width, "action_padding": 26 - real_width}
    train_config = {
        "task": {"agent_pos_config": layout, "dof_config": layout},
        "data": {},
    }
    raw = np.arange(26, dtype=np.float32).reshape(1, 1, 26) / 10
    raw[..., real_width:] = 0
    state_mask = np.ones_like(raw)
    state_mask[..., 2] = 0  # Unavailable real state must not be treated as padding.
    action_mask = np.zeros((1, 32, 26), dtype=np.float32)
    action_mask[..., :real_width] = 1
    normalizer = make_normalizer(real_width, "fixture")
    normalizer_module.pad_normalizer_to_dim(normalizer, 26, "state")
    expected = normalizer.normalize_data(torch.from_numpy(raw), ["fixture"])
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 7)
    )

    if adapter_kind == "serving":
        from wall_x._vendor.harrix.serving._wallx_infer import model_wrapper as module

        wrapper = module.WallxModelWrapper.__new__(module.WallxModelWrapper)
        wrapper.config = SimpleNamespace(
            train_config=train_config, model_device="cpu", padding=mode
        )
        wrapper.norm_key = "fixture"
        wrapper.normalizer_propri = normalizer
        wrapper.logger = logging.getLogger("padding_test")
        wrapper.model = SimpleNamespace(processor=processor)
        wrapper._resize_images = lambda _: []
        monkeypatch.setattr(module, "ENABLE_FAST_PREPROCESS", False)
        monkeypatch.setattr(
            module, "preprocesser_call", lambda **kw: {"input_ids": torch.tensor([[7]])}
        )
        state = SimpleNamespace(
            agent_pos=raw, agent_pos_mask=state_mask, dof_mask=action_mask
        )
        inputs = wrapper.construct_model_input(
            [{"robot_state_action_data": state}], ["prefix"], ["postfix"]
        )
    else:
        from wall_x._vendor.harrix.adapters import qwen_vlact as module

        adapter = module.QwenVLActInferAdapter.__new__(module.QwenVLActInferAdapter)
        adapter._padding = mode
        adapter._train_config = train_config
        adapter._norm_key = "fixture"
        adapter._normalizer_propri = normalizer
        adapter._processor = processor
        adapter._cam_names = []
        adapter._device = "cpu"
        monkeypatch.setattr(
            module, "preprocesser_call", lambda **kw: {"input_ids": torch.tensor([[7]])}
        )
        inputs = adapter._construct_model_input(
            [
                {
                    "proprioception": raw,
                    "agent_pos_mask": state_mask,
                    "dof_mask": action_mask,
                }
            ],
            ["prefix"],
            ["postfix"],
        )
    torch.testing.assert_close(
        inputs["proprioception"][..., :real_width], expected[..., :real_width]
    )
    if mode == "zero":
        assert inputs["proprioception"][..., real_width:].count_nonzero() == 0
    else:
        torch.testing.assert_close(inputs["proprioception"], expected, rtol=0, atol=0)
    assert inputs["agent_pos_mask"][..., real_width:].count_nonzero() == 0
    assert inputs["agent_pos_mask"][..., 2].count_nonzero() == 0
