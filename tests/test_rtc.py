"""CPU-only RTC mathematics and protocol tests; no checkpoint or robot needed."""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wall_x.model.core.action import rtc
from wall_x.model.core.action import rtc_model as adapter

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wire = load("rtc_test_wire", "wall_x/_vendor/harrix/serving/rtc.py")


@pytest.mark.parametrize("t", [0.0, 0.3, 0.999])
def test_full_endpoint_jacobian_and_no_parameter_grads(t):
    x = torch.ones(1, 4, 3)
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    condition = rtc.RTCCondition(
        torch.ones_like(x) * 3, torch.ones(1, 4, 1), 5, "pi_vjp"
    )
    mask = torch.tensor([1, 1, 0]).view(1, 1, 3)
    time = torch.tensor(t)
    with torch.no_grad():
        actual = rtc.guided_velocity(
            lambda time, z: parameter * z, x, time, condition, mask
        )
    jacobian = 1 + 2 * (1 - t)
    gain = min(5, (t * t + (1 - t) ** 2) / max(1e-8, t * (1 - t)))
    expected = 2 * x + gain * jacobian * (3 - jacobian * x) * mask
    torch.testing.assert_close(actual, expected)
    assert parameter.grad is None and not actual.requires_grad


@pytest.mark.parametrize("mode", ["lerobot", "pi_vjp"])
def test_zero_weight_equals_original_velocity(mode):
    x = torch.randn(1, 32, 26)
    c = rtc.RTCCondition(torch.randn_like(x), torch.zeros(1, 32, 1), guidance_mode=mode)
    with torch.no_grad():
        got = rtc.guided_velocity(
            lambda t, z: z.sin(), x, torch.tensor(0.3), c, torch.ones_like(x)
        )
    torch.testing.assert_close(got, x.sin())


@pytest.mark.parametrize("mode", ["lerobot", "pi_vjp"])
def test_mask_keeps_virtual_tail_unguided(mode):
    x = torch.randn(1, 32, 26)
    mask = torch.zeros(1, 1, 26)
    mask[..., :7] = 1
    c = rtc.RTCCondition(torch.randn_like(x), torch.ones(1, 32, 1), guidance_mode=mode)
    got = rtc.guided_velocity(lambda t, z: z * 2, x, torch.tensor(0.3), c, mask)
    torch.testing.assert_close(got[..., 7:], (2 * x)[..., 7:])


@pytest.mark.parametrize("mask_horizon", [1, 32])
@pytest.mark.parametrize("prefix_steps", [16, 32])
def test_raw_prefix_inverse_normalization_is_not_clipped(mask_horizon, prefix_steps):
    mask = torch.tensor([1] * 7 + [0] * 19).view(1, 1, 26).repeat(1, mask_horizon, 1)
    normalizer = SimpleNamespace(
        min={"test": torch.arange(7.0)}, delta={"test": torch.ones(7) * 2}
    )
    raw = np.ones((prefix_steps, 7)) * 20
    r = dict(prefix=raw, committed_steps=10, guidance_horizon=16, max_guidance=5)
    c = rtc.make_condition(r, normalizer, "test", mask, 32, 26, "cpu")
    torch.testing.assert_close(
        c.prefix[0, :16, :7],
        torch.tensor(raw[:16], dtype=torch.float32) - torch.arange(7.0) - 1,
    )
    assert c.prefix[..., 7:].count_nonzero() == 0
    assert torch.all(c.weights[:, :10] == 1)
    assert torch.all((c.weights[:, 10:16] > 0) & (c.weights[:, 10:16] < 1))
    assert c.weights[:, 16:].count_nonzero() == 0
    # Zero padding happens after normalization, not in physical action units.
    assert c.prefix[:, 16:].count_nonzero() == 0


@pytest.mark.parametrize("t", [0.0, 0.3, 0.999])
def test_compact_prefix_preserves_guidance_with_temporally_coupled_denoiser(t):
    generator = torch.Generator().manual_seed(71)
    raw = torch.randn(32, 7, generator=generator)
    normalizer = SimpleNamespace(
        min={"test": torch.arange(7.0)}, delta={"test": torch.arange(1.0, 8.0)}
    )
    mask = torch.ones(1, 32, 7)
    request = dict(
        prefix=raw[:20], committed_steps=14, guidance_horizon=20, max_guidance=5
    )
    compact = rtc.make_condition(request, normalizer, "test", mask, 32, 7, "cpu")
    legacy = rtc.RTCCondition(
        (2 * (raw - normalizer.min["test"]) / normalizer.delta["test"] - 1)[None],
        compact.weights,
        compact.max_guidance,
        compact.guidance_mode,
    )
    x = torch.randn(1, 32, 7, generator=generator)

    def denoise(time, z):
        return z.sin() + 0.1 * z.mean(dim=1, keepdim=True)

    before = rtc.guided_velocity(denoise, x, torch.tensor(t), legacy, mask)
    after = rtc.guided_velocity(denoise, x, torch.tensor(t), compact, mask)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def request():
    return dict(
        version=1,
        enabled=True,
        epoch=0,
        plan_id=1,
        model_dt_s=1 / 30,
        prefix=np.zeros((32, 7)).tolist(),
        committed_steps=14,
        guidance_horizon=20,
        max_guidance=5,
    )


CAP = dict(
    version=1, supported=True, model_dt_s=1 / 30, action_horizon=32, real_action_dim=7
)


@pytest.mark.parametrize("prefix_steps", [20, 32])
def test_client_controls_activation_per_request_without_mutation(prefix_steps):
    r = request()
    r["prefix"] = r["prefix"][:prefix_steps]
    assert wire.validate_request(None, CAP, "flow") is None
    assert wire.validate_request({"enabled": False}, CAP, "flow") is None
    assert wire.validate_request(r, CAP, "flow") == r
    assert wire.validate_request(None, CAP, "flow") is None
    assert wire.acknowledgement(r)["mode"] == "guided"
    for cap, mode in [(dict(CAP, supported=False), "flow"), (CAP, "ar")]:
        with pytest.raises(ValueError):
            wire.validate_request(r, cap, mode)


@pytest.mark.parametrize(
    "change",
    [
        dict(prefix=[[float("nan")] * 7] * 32),
        dict(prefix=np.zeros((19, 7)).tolist()),
        dict(prefix=np.zeros((21, 7)).tolist()),
        dict(committed_steps=32),
        dict(guidance_horizon=32),
        dict(max_guidance=float("inf")),
        dict(model_dt_s=1 / 15),
        dict(enabled=1),
    ],
)
def test_invalid_prefix_or_timebase_rejected(change):
    with pytest.raises(ValueError):
        wire.validate_request(dict(request(), **change), CAP, "flow")


@pytest.mark.parametrize("mode", ["lerobot", "pi_vjp"])
def test_cached_adapter_rebuilds_first_step_grad_and_detaches_cache(mode):
    class Preprocessor:
        action_hidden_size = 3

        @staticmethod
        def step(timestep, noisy_action, dof_mask):
            assert torch.is_grad_enabled() == (mode == "pi_vjp")
            assert noisy_action.requires_grad == (mode == "pi_vjp")
            return noisy_action * 2, None

        @staticmethod
        def action_proj_back(hidden):
            return hidden

    class Model:
        config = SimpleNamespace(use_x_pred=False)
        action_preprocessor = Preprocessor()
        action_token_id_set = {"action_token_id": 1}
        steps = 0

        @staticmethod
        def model(inputs_embeds, past_key_values, use_cache, **kwargs):
            if not use_cache:
                assert past_key_values.key_cache[0].grad_fn is None
            return SimpleNamespace(
                last_hidden_state=inputs_embeds,
                past_key_values=SimpleNamespace(
                    key_cache=[inputs_embeds.unsqueeze(1)],
                    value_cache=[inputs_embeds.unsqueeze(1)],
                ),
            )

        def _prepare_flow_postfix(self, ctx, cache):
            self.steps += 1
            return (
                None,
                ctx["inputs_embeds"][:, 1:],
                None,
                None,
                ctx["input_ids"][:, 1:],
                None,
                None,
                None,
            )

    m = Model()
    x = torch.ones(1, 4, 3)
    ctx = dict(
        batch_size=1,
        output_attentions=False,
        dof_mask=torch.ones(1, 1, 3),
        noisy_action=x,
        times=torch.tensor([0.0, 0.3, 0.6, 0.9]),
        padding_action=None,
        inputs_embeds=torch.zeros(1, 5, 3),
        flow_action_mask=torch.tensor([[False, True, True, True, True]]),
        input_ids=torch.tensor([[0, 1, 1, 1, 1]]),
    )
    ctx.update(
        {
            k: None
            for k in (
                "attention_mask",
                "position_ids",
                "moe_token_types",
                "start_indices",
                "end_indices",
                "positional_masks",
            )
        }
    )
    c = rtc.RTCCondition(torch.zeros_like(x), torch.zeros(1, 4, 1), guidance_mode=mode)
    with torch.no_grad():
        trajectory, _ = adapter.generate_rtc_action(m, ctx, c)
    expected = x
    for dt, actual in zip(torch.diff(ctx["times"]), trajectory):
        expected = expected + dt * 2 * expected
        torch.testing.assert_close(actual, expected)
        assert not actual.requires_grad
    assert m.steps == 1


@pytest.mark.parametrize(
    "blocked", [None, "libero", "batch", "graph", "engine", "smooth", "serialized"]
)
def test_launch_capability_uses_resolved_inference_config(blocked, monkeypatch):
    source = (ROOT / "wall_x/_vendor/harrix/serving/launch_serving.py").read_text()
    definition = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "create_policy"
    )
    resolved = SimpleNamespace(
        smooth_action=blocked == "smooth", action_horizon=32, action_dim=26
    )
    policy = SimpleNamespace(
        config=resolved, metadata={"wallx_client": {"state_dim": 7}}
    )
    args = SimpleNamespace(
        model_type="wallx",
        image_passing_mode="base64",
        serialize_actions=blocked == "serialized",
        env="libero" if blocked == "libero" else "real",
        max_batch_size=2 if blocked == "batch" else None,
        enable_cuda_graph=blocked == "graph",
        enable_experimental_engine=blocked == "engine",
    )
    monkeypatch.delenv("ENABLE_CUDA_GRAPH", raising=False)
    monkeypatch.delenv("ENABLE_EXPERIMENTAL_INFERENCE_ENGINE", raising=False)
    namespace = dict(
        Args=object,
        EnvMode=SimpleNamespace(X2ROBOT="real"),
        os=os,
        get_model_config=lambda args: SimpleNamespace(action_dim=None),
        get_wallx_policy=lambda *args: policy,
    )
    exec(
        compile(
            ast.Module(body=[definition], type_ignores=[]), "create_policy", "exec"
        ),
        namespace,
    )
    assert namespace["create_policy"](args).rtc_capability["supported"] == (
        blocked is None
    )
