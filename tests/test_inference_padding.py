"""Padding policy preserves upstream normalization and native/RTC time semantics."""

import pytest
import torch
from rtc_model_fixture import SyntheticModel
from test_flow_action_padding import build_padding, make_normalizer, normalizer_module

from wall_x.inference_padding import apply_inference_padding, validate_padding
from wall_x.model.core.action.normalizer import normalize_data_with_virtual_tail
from wall_x.model.core.action.rtc_contract import RTCCondition


@pytest.mark.parametrize("mode", ["zero", "original"])
@pytest.mark.parametrize("short_stats", [False, True])
@pytest.mark.parametrize("real", [7, 14, 20])
def test_endpoint_against_upstream_and_correction(mode, short_stats, real):
    normalizer = make_normalizer(real)
    if not short_stats:
        normalizer_module.pad_normalizer_to_dim(normalizer, 26, "action")
    mask = torch.zeros(1, 32, 26)
    mask[..., :real] = 1
    layout = {"joints": real, "action_padding": 26 - real}
    expected = normalize_data_with_virtual_tail(
        normalizer, torch.zeros(1, 1, 26), ["robot"], mask
    )
    result = build_padding(normalizer, ["robot"], mask, layout, mode=mode)
    if mode == "zero":
        expected[..., real:] = 0
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    if mode == "original" and not short_stats:
        assert torch.all(result[..., real:] == -1)
    noise = torch.randn(1, 32, 26)
    for t in [0, 0.25, 0.75, 0.999]:
        torch.testing.assert_close(
            noise + t * (result - noise), (1 - t) * noise + t * expected
        )


@pytest.mark.parametrize(
    "value", [None, True, 0, {}, {"mode": "zero"}, "aligned_zero", "bad"]
)
def test_invalid_policy_fails_before_inference(value):
    with pytest.raises(ValueError, match="padding"):
        validate_padding(value)


@pytest.mark.parametrize("mode", ["zero", "original"])
def test_only_explicit_virtual_channels_are_changed(mode):
    x = torch.tensor([[[4.0, -0.3, 8.0, -0.7]]])
    before = x.clone()
    result = apply_inference_padding(
        x, {"arm": 1, "action_padding": 2, "missing_real": 1}, mode
    )
    assert torch.equal(x, before)
    assert result[..., 0] == 4 and result[..., 3] == -0.7
    assert torch.equal(
        result[..., 1:3], torch.zeros(1, 1, 2) if mode == "zero" else x[..., 1:3]
    )


@pytest.mark.parametrize("guided", [False, True])
def test_real_model_method_chain_uses_policy_without_changing_noise_or_real_mask(
    guided,
):
    results = []
    noise = torch.randn(1, 8, 5, generator=torch.Generator().manual_seed(123))
    for mode in ("zero", "original"):
        model = SyntheticModel(horizon=8, dim=5, real=3)
        model.inference_padding = mode
        captured = []
        prepare = model._prepare_flow_action_inputs

        def trace(*a, **kw):
            ctx = prepare(*a, **kw)
            captured.append(ctx)
            return ctx

        model._prepare_flow_action_inputs = trace
        condition = (
            RTCCondition(
                prefix=torch.zeros(1, 8, 5),
                weights=torch.tensor([1.0, 1.0, 0.75, 0.5, 0.0, 0.0, 0.0, 0.0]).reshape(
                    1, 8, 1
                ),
            )
            if guided
            else None
        )
        result = model.run(condition=condition, noise=noise, num_inference_timesteps=3)
        assert torch.equal(model.last_noise, noise)
        endpoint = captured[0]["padding_action"]
        assert torch.all(endpoint[..., 3:] == (0 if mode == "zero" else -1))
        assert torch.all(captured[0]["dof_mask"][..., :3] == 1)
        assert torch.all(captured[0]["dof_mask"][..., 3:] == 0)
        assert torch.isfinite(result).all()
        results.append(result)
    assert not torch.equal(*results)
