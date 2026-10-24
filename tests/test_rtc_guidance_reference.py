"""Numerical parity with the pinned upstream class, not a rederived mock formula."""

import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_rtc import CAP, request, rtc, wire


def upstream_processor(gain):
    path = Path(__file__).parent / "fixtures/rtc/lerobot_b9cb121_modeling_rtc.py"
    source = path.read_bytes()
    # The adjacent fixture is the unchanged Apache-2.0 source at this commit.
    digest = (path.with_suffix(".sha256")).read_text().strip()
    assert hashlib.sha256(source).hexdigest() == digest
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "RTCProcessor"
    )
    namespace = dict(
        torch=torch,
        Tensor=torch.Tensor,
        math=math,
        RTCConfig=SimpleNamespace,
        RTCAttentionSchedule=SimpleNamespace(
            ZEROS="zeros", ONES="ones", LINEAR="linear", EXP="exp"
        ),
    )
    # Avoid importing an installed (possibly different) LeRobot package. Execute
    # the actual reference class, including its autograd statement order.
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["RTCProcessor"](
        SimpleNamespace(
            enabled=True,
            mode="guided",
            debug=False,
            execution_horizon=5,
            max_guidance_weight=gain,
            prefix_attention_schedule="linear",
        )
    )


@pytest.mark.parametrize("gain", [5.0, 10.0])
@pytest.mark.parametrize("time", [0.0, 0.0999, 0.3, 0.4995, 0.8991, 0.999])
def test_lerobot_numerical_parity_with_coupled_nonlinear_denoiser(time, gain):
    torch.manual_seed(17)
    x, prefix = torch.randn(2, 8, 3), torch.randn(2, 8, 3)
    parameter = torch.nn.Parameter(torch.tensor(1.7))
    ref = upstream_processor(gain)
    weights = ref.get_prefix_weights(3, 5, 8)[None, :, None]
    condition = rtc.RTCCondition(prefix, weights, gain, "lerobot")
    t = torch.tensor(time)

    def velocity(z):
        return parameter * z.sin() + 0.2 * z.mean(dim=1, keepdim=True)

    with torch.no_grad():
        expected = -ref.denoise_step(x, prefix, 3, 1 - t, lambda z: -velocity(z), 5)
    actual = rtc.guided_velocity(
        lambda t, z: velocity(z), x, t, condition, torch.ones_like(x)
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert parameter.grad is None and not actual.requires_grad


def test_lerobot_avoids_graph_and_backwards_even_with_grad_enabled(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("Direct-residual RTC must not traverse a backward graph")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    x = torch.ones(1, 8, 3, requires_grad=True)
    condition = rtc.RTCCondition(torch.zeros_like(x), torch.ones(1, 8, 1))

    def denoise(t, z):
        assert not torch.is_grad_enabled() and not z.requires_grad
        return z.square()

    actual = rtc.guided_velocity(
        denoise, x, torch.tensor(0.3), condition, torch.ones_like(x)
    )
    assert not actual.requires_grad


@pytest.mark.parametrize("strong,end", [(1, 2), (14, 20), (24, 30), (25, 28), (30, 31)])
def test_taper_has_exact_upstream_rounding(strong, end):
    req = dict(
        request(), guidance_mode="lerobot", committed_steps=strong, guidance_horizon=end
    )
    normalizer = SimpleNamespace(
        min={"test": torch.zeros(7)}, delta={"test": torch.ones(7)}
    )
    condition = rtc.make_condition(
        req, normalizer, "test", torch.ones(1, 32, 7), 32, 7, "cpu"
    )
    expected = upstream_processor(10).get_prefix_weights(strong, end, 32)
    torch.testing.assert_close(condition.weights.flatten(), expected, atol=0, rtol=0)


def test_legacy_requests_preserve_pi_vjp_and_explicit_modes_are_acknowledged():
    legacy = request()
    normalizer = SimpleNamespace(
        min={"test": torch.zeros(7)}, delta={"test": torch.ones(7)}
    )
    for mode in [None, "lerobot", "pi_vjp"]:
        req = legacy if mode is None else dict(legacy, guidance_mode=mode)
        assert wire.validate_request(req, CAP, "flow") == req
        condition = rtc.make_condition(
            req, normalizer, "test", torch.ones(1, 32, 7), 32, 7, "cpu"
        )
        assert condition.guidance_mode == (mode or "pi_vjp")
        assert wire.acknowledgement(req).get("guidance_mode") == mode
    with pytest.raises(ValueError, match="guidance_mode"):
        wire.validate_request(dict(legacy, guidance_mode="unknown"), CAP, "flow")
    with pytest.raises(ValueError, match="guidance_mode"):
        rtc.guided_velocity(
            lambda t, z: z,
            torch.ones(1, 1, 1),
            torch.tensor(0.2),
            rtc.RTCCondition(
                torch.ones(1, 1, 1), torch.ones(1, 1, 1), guidance_mode="unknown"
            ),
            torch.ones(1, 1, 1),
        )
