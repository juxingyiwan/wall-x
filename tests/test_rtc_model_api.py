"""Public adapter -> real preparation -> real sampler regression tests (CPU)."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from rtc_model_fixture import ROOT, SyntheticModel

from wall_x.model.adapters.qwen_flow import QWEN_RTC_CAPABILITY
from wall_x.model.adapters.rtc_preprocessing import build_condition, make_condition
from wall_x.model.core.action import flow
from wall_x.model.core.action.rtc_contract import NOISE_AT_ZERO


def condition(model, mode="lerobot"):
    return make_condition(
        dict(
            prefix=torch.full((5, model.real), 1.25),
            committed_steps=2,
            guidance_horizon=5,
            max_guidance=1.0,
            guidance_mode=mode,
        ),
        model.action_preprocessor.normalizer_action,
        model.key,
        model.inputs()["dof_mask"],
        model.horizon,
        model.dim,
        "cpu",
    )


def public_adapter(monkeypatch, model):
    from wall_x._vendor.harrix.adapters import qwen_vlact as module

    obj = module.QwenVLActInferAdapter.__new__(module.QwenVLActInferAdapter)
    obj._model = model
    obj._action_mode = "flow"
    obj._action_horizon, obj._action_dim = model.horizon, model.dim
    obj._num_inference_timesteps = 10
    obj._train_config = {}
    obj._normalizer_action = model.action_preprocessor.normalizer_action
    obj._get_flow_prompt = lambda _: ("prefix", "postfix")
    obj._construct_model_input = lambda obs, *args: model.inputs(len(obs))
    # Observation/decoding is specific to an environment, outside the RTC contract.
    monkeypatch.setattr(
        module, "encode_proprio", lambda observation, *args: observation
    )
    monkeypatch.setattr(module, "decode_chunk", lambda actions, *args: actions[0])
    return obj


@pytest.mark.parametrize("shape", [(8, 3, 3), (16, 14, 14), (32, 26, 7)])
@pytest.mark.parametrize("mode", ["lerobot", "pi_vjp"])
def test_public_adapter_rtc_noise_and_zero_native_parity(monkeypatch, shape, mode):
    model = SyntheticModel(*shape)
    adapter = public_adapter(monkeypatch, model)
    c = condition(model, mode)
    noise = np.linspace(-1, 1, model.horizon * model.dim, dtype=np.float64).reshape(
        model.horizon, model.dim
    )
    payload = dict(observation={}, instruction="task", noise=noise, rtc_condition=c)
    assert adapter.supports_rtc() and model.supports_rtc()
    cap = adapter.model_rtc_capability()
    assert cap.time_convention == NOISE_AT_ZERO
    assert cap.max_batch_size == 1 and cap.backends == ("native_pytorch",)
    torch.manual_seed(1)
    actual = adapter.predict_batch([payload])[0]
    torch.manual_seed(912)
    repeated = adapter.predict_batch([payload])[0]
    np.testing.assert_array_equal(actual, repeated)
    torch.testing.assert_close(
        model.last_noise, torch.from_numpy(noise[None]).float(), rtol=0, atol=0
    )
    native = adapter.predict_batch([{**payload, "rtc_condition": None}])[0]
    zero = adapter.predict_batch(
        [{**payload, "rtc_condition": replace(c, weights=c.weights * 0)}]
    )[0]
    np.testing.assert_array_equal(native, zero)
    assert (
        np.max(abs(actual - native)) > 0.001
    )  # Condition must actually reach sampling.
    assert model.parameter.grad is None
    np.testing.assert_array_equal(noise, payload["noise"])


@pytest.mark.parametrize("mode", [None, "lerobot", "pi_vjp"])
def test_explicit_noise_is_used_without_consuming_rng(mode):
    model = SyntheticModel()
    c = None if mode is None else condition(model, mode)
    zeros, ones = torch.zeros(1, 8, 3), torch.ones(1, 8, 3)
    a = model.run(condition=c, noise=zeros, seed=601)
    torch.testing.assert_close(model.last_noise, zeros, rtol=0, atol=0)
    expected_rng = torch.Generator().manual_seed(601).get_state()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    b = model.run(condition=c, noise=zeros, seed=602)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    changed = model.run(condition=c, noise=ones)
    assert not torch.equal(a, changed)
    assert torch.equal(zeros, torch.zeros_like(zeros))


def test_omitted_noise_preserves_native_rng_and_batch_behavior():
    model = SyntheticModel()
    generator = torch.Generator().manual_seed(601)
    expected = torch.randn(2, 8, 3, generator=generator)
    torch.manual_seed(601)
    result = model.generate_flow_action(
        action_horizon=8, action_dim=3, **model.inputs(2)
    )
    torch.testing.assert_close(model.last_noise, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), generator.get_state())
    assert result["predict_action"].shape == (2, 8, 3)


def test_noise_with_token_embedding_and_proprioception_preparation():
    model = SyntheticModel()
    forward = model.model

    class Network:
        def __call__(self, *args, **kwargs):
            return forward(*args, **kwargs)

        def embed_tokens(self, ids):
            return torch.zeros(*ids.shape, model.dim)

    model.model = Network()
    model.config.use_state_string_representation = False
    model.action_token_id_set["propri_token_id"] = 2
    calls = []

    def proprioception_proj(proprioception, dataset_names, mask, use_history):
        calls.append(proprioception.clone())
        return proprioception

    model.action_preprocessor.proprioception_proj = proprioception_proj
    inputs = model.inputs()
    inputs.pop("inputs_embeds")
    inputs["input_ids"][0, 0] = 2
    inputs.update(
        proprioception=torch.ones(1, 1, 3), agent_pos_mask=torch.ones(1, 1, 3)
    )
    noise = torch.ones(1, 8, 3)
    result = model.generate_flow_action(
        action_horizon=8, action_dim=3, noise=noise, **inputs
    )
    assert len(calls) == 1 and result["predict_action"].shape == noise.shape
    torch.testing.assert_close(model.last_noise, noise, rtol=0, atol=0)


@pytest.mark.parametrize(
    "noise,error",
    [
        (torch.zeros(8, 3), ValueError),
        (torch.zeros(1, 8, 4), ValueError),
        (torch.zeros(1, 8, 3, dtype=torch.int64), TypeError),
        (np.zeros((1, 8, 3)), TypeError),
        (torch.full((1, 8, 3), float("nan")), ValueError),
        (torch.full((1, 8, 3), 1e99, dtype=torch.float64), ValueError),
    ],
)
def test_invalid_model_noise_rejected_before_denoising(noise, error):
    model = SyntheticModel()
    with pytest.raises(error, match="noise"):
        model.run(noise=noise)
    assert model.calls == 0


@pytest.mark.parametrize(
    "invalid", ["batch", "attention", "inference_mode", "unknown_mode", "wire"]
)
def test_unsupported_model_requests_fail_before_denoising(invalid):
    model = SyntheticModel()
    c = condition(model, "pi_vjp")
    kwargs = dict(rtc_condition=c)
    if invalid == "attention":
        kwargs["output_attentions"] = True
    elif invalid == "unknown_mode":
        kwargs["rtc_condition"] = replace(c, guidance_mode="unknown")
    elif invalid == "wire":
        kwargs["rtc"] = {"enabled": True}
    with torch.inference_mode(invalid == "inference_mode"):
        with pytest.raises((ValueError, RuntimeError)):
            model.generate_flow_action(
                action_horizon=8,
                action_dim=3,
                **model.inputs(2 if invalid == "batch" else 1),
                **kwargs,
            )
    assert model.calls == 0


def test_custom_backend_requires_explicit_rtc_support():
    class OtherBackend(SyntheticModel):
        def _execute_flow_action(self, ctx):
            raise AssertionError("unsupported backend must not run")

    model = OtherBackend()
    assert not model.supports_rtc()
    with pytest.raises(ValueError, match="Custom flow backend"):
        model.run(condition=condition(model))
    with pytest.raises(ValueError, match="cuda_graph"):
        QWEN_RTC_CAPABILITY.validate(
            condition(model), batch_size=1, backend="cuda_graph"
        )


def test_public_model_predict_action_dispatch():
    model = SyntheticModel()
    c = condition(model)
    noise = torch.zeros(1, 8, 3)
    actual, _ = model.predict_action(
        "flow",
        action_horizon=8,
        action_dim=3,
        rtc_condition=c,
        noise=noise,
        **model.inputs(),
    )
    torch.testing.assert_close(
        actual, model.run(condition=c, noise=noise), rtol=0, atol=0
    )
    for mode in ("fast", "spatial_token", "dllm"):
        with pytest.raises(ValueError, match="flow chunk predictor"):
            model.predict_action(mode, rtc_condition=c)


def test_public_unsupported_mode_batch_wire_and_mixed_noise(monkeypatch):
    model = SyntheticModel()
    adapter = public_adapter(monkeypatch, model)
    p = dict(observation={}, instruction="task", rtc_condition=condition(model))
    for payloads in ([p, p], [{**p, "rtc": {"enabled": True}}]):
        with pytest.raises(ValueError):
            adapter.predict_batch(payloads)
    adapter._action_mode = "ar"
    assert not adapter.supports_rtc()
    with pytest.raises(ValueError):
        adapter.predict_batch([p])
    adapter._action_mode = "flow"
    p = {**p, "rtc_condition": None}
    with pytest.raises(ValueError, match="every payload"):
        adapter.predict_batch([p, {**p, "noise": np.zeros((8, 3))}])
    assert model.calls == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: replace(c, prefix=c.prefix[..., :2]),
        lambda c: replace(c, weights=torch.ones(1, 8, 3)),
        lambda c: replace(c, weights=c.weights + 2),
        lambda c: replace(c, prefix=c.prefix * float("nan")),
        lambda c: replace(c, max_guidance=float("inf")),
    ],
)
def test_bad_model_space_condition_rejected(mutate):
    model = SyntheticModel()
    with pytest.raises(ValueError):
        model.run(condition=mutate(condition(model)))
    assert model.calls == 0


def test_custom_normalizer_channel_mapping_and_renamed_stats():
    model = SyntheticModel()
    raw = torch.arange(10.0).reshape(5, 2)
    mask = torch.tensor([1.0, 0.0, 1.0]).reshape(1, 1, 3)
    mean, std = torch.tensor([2.0, 4.0]), torch.tensor([0.5, 2.0])
    c = build_condition(
        raw,
        normalize_prefix=lambda x: (x - mean) / std,
        dof_mask=mask,
        horizon=8,
        action_dim=3,
        device="cpu",
        committed_steps=2,
        guidance_horizon=5,
    )
    torch.testing.assert_close(c.prefix[0, :5, [0, 2]], (raw - mean) / std)
    assert not c.prefix[..., 1].any() and not c.prefix[:, 5:].any()
    assert c.prefix.max() > 1  # Never clip an already commanded prefix.
    normalizer = model.action_preprocessor.normalizer_action
    normalizer.min["renamed"] = normalizer.min[model.key]
    normalizer.delta["renamed"] = normalizer.delta[model.key]
    old = condition(model)
    model.key = "renamed"
    new = condition(model)
    torch.testing.assert_close(
        model.run(condition=old), model.run(condition=new), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "times,convention",
    [
        ([1.0, 0.5, 0.0], NOISE_AT_ZERO),
        ([0.0, 0.5, 0.4], NOISE_AT_ZERO),
        ([0.0, 0.5, 1.1], NOISE_AT_ZERO),
        ([0.0, float("nan"), 1.0], NOISE_AT_ZERO),
        ([0.0, 0.5, 1.0], "action_at_zero"),
        ([0.0, 0.5, 1.0], None),
    ],
)
def test_wrong_flow_time_fails_before_model(times, convention):
    class Adapter:
        time_convention = convention

        def denoise(self, t, x):
            raise AssertionError("time contract must be checked first")

    with pytest.raises(ValueError, match="RTC"):
        flow.sample_flow(
            Adapter(),
            torch.zeros(1, 8, 3),
            torch.tensor(times),
            torch.ones(1, 1, 3),
            condition(SyntheticModel()),
        )


def test_checkpoint_probe_hooks_execute_reference_and_restore(monkeypatch):
    from deployment.validation.rtc_guidance_probe import (
        load_reference,
        reference_guidance,
        traced_guidance,
        zero_guidance,
    )

    model = SyntheticModel(32, 26, 7)
    adapter = public_adapter(monkeypatch, model)
    c = condition(model)
    p = dict(
        observation={}, instruction="task", noise=np.zeros((32, 26)), rtc_condition=c
    )
    original = flow.guided_velocity
    upstream, digest = load_reference(ROOT)
    assert len(digest) == 64
    reference = reference_guidance(
        upstream, dict(committed_steps=2, guidance_horizon=5)
    )
    results = {}
    for name, callback in (
        ("zero", zero_guidance),
        ("reference", reference),
        ("current", original),
    ):
        with traced_guidance(callback, expect_guidance=True) as trace:
            results[name] = adapter.predict_batch([p])[0]
        assert trace.calls == 10
        assert flow.guided_velocity is original
    with traced_guidance(original, expect_guidance=False) as trace:
        native = adapter.predict_batch([{**p, "rtc_condition": None}])[0]
    assert trace.calls == 0
    np.testing.assert_array_equal(results["zero"], native)
    np.testing.assert_allclose(
        results["current"], results["reference"], rtol=0, atol=1e-6
    )
    assert np.max(abs(results["current"] - native)) > 0.001
    with pytest.raises(RuntimeError, match="dispatch mismatch"):
        with traced_guidance(original, expect_guidance=True):
            pass
    with pytest.raises(RuntimeError, match="inference failed"):
        with traced_guidance(original, expect_guidance=True):
            raise RuntimeError("inference failed")
    assert flow.guided_velocity is original
