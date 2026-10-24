"""Compare RTC implementations on saved observations without opening hardware.

Cases JSON: a list of {id, source: query NPZ path, task, wire: recorded RTC request,
directions: seven hardware unit/sign scales, crop: recorded zero-based crop}.
The recorded crop is fixed for comparison, not a simulated live rollout.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def load_reference(root):
    """Load the checksum-pinned upstream implementation without installing LeRobot."""
    import torch

    reference_path = root / "tests/fixtures/rtc/lerobot_b9cb121_modeling_rtc.py"
    source = reference_path.read_bytes()
    assert (
        hashlib.sha256(source).hexdigest()
        == reference_path.with_suffix(".sha256").read_text().strip()
    )
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
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(reference_path), "exec"), namespace)
    upstream = namespace["RTCProcessor"](
        SimpleNamespace(
            enabled=True,
            mode="guided",
            debug=False,
            execution_horizon=30,
            max_guidance_weight=10.0,
            prefix_attention_schedule="linear",
        )
    )
    return upstream, hashlib.sha256(source).hexdigest()


def zero_guidance(denoise, x, t, condition, mask):
    import torch

    from wall_x.model.core.action.rtc import guided_velocity

    return guided_velocity(
        denoise, x, t, replace(condition, weights=torch.zeros_like(condition.weights)), mask
    )


def reference_guidance(upstream, wire):
    def reference(denoise, x, t, condition, mask):
        active = mask.reshape(-1, x.shape[-1])[0].bool()
        buffer = {}

        def negative_velocity(small):
            full = x.detach().clone()
            full[:, :, active] = small
            buffer["velocity"] = denoise(t, full)
            return -buffer["velocity"][:, :, active]

        upstream.rtc_config.max_guidance_weight = condition.max_guidance
        small = -upstream.denoise_step(
            x[:, :, active],
            condition.prefix[:, :, active],
            wire["committed_steps"],
            1 - t,
            negative_velocity,
            wire["guidance_horizon"],
        )
        value = buffer["velocity"].clone()
        value[:, :, active] = small
        return value.detach()

    return reference


@contextmanager
def traced_guidance(callback, *, expect_guidance):
    """Patch the actual sampler binding in this standalone, serial probe.

    Always restore it, including on inference errors. Call counts prevent a
    future refactor from silently turning the reference comparison into a no-op.
    Do not use this process-global test hook in a concurrent serving process.
    """
    from unittest.mock import patch

    from wall_x.model.core.action import flow

    trace = SimpleNamespace(calls=0)

    def counted(*args, **kwargs):
        trace.calls += 1
        return callback(*args, **kwargs)

    with patch.object(flow, "guided_velocity", counted):
        yield trace
    if bool(trace.calls) != expect_guidance:
        raise RuntimeError(f"Guidance probe dispatch mismatch: {trace.calls} calls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--machine", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 41])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root / "deployment/client/src")]
    import numpy as np
    import yaml
    from wallx_client.config import load_config
    from wallx_client.contracts import Observation
    from wallx_client.protocol import build_request
    from wallx_client.rtc import validate_ack, validate_capability

    from deployment.cli import load_deployment
    from deployment.local_runtime import local_deployment, native_policy

    if args.output.exists():
        raise FileExistsError(args.output)
    deployment = local_deployment(load_deployment(args.profile, args.machine))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "client.yml"
        path.write_text(yaml.safe_dump(deployment["client"]))
        cfg = load_config(path)
    cases = json.loads(args.cases.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, bodies, checks = [], [], []
    report = dict(
        source_root=str(root),
        checkpoint_sha256=cfg.policy.checkpoint_sha256,
        cases=cases,
        rows=rows,
        checks=checks,
        scope="Saved-input offline inference; fresh paired seeds; no hardware execution or task-success claim",
    )

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))

    with native_policy(deployment, args.output.with_suffix(".model.log")) as policy:
        import torch

        from wall_x.model.core.action.rtc import guided_velocity as original

        upstream, reference_sha256 = load_reference(root)
        report.update(
            gpu=torch.cuda.get_device_name(),
            reference_sha256=reference_sha256,
            configured_guidance_mode=cfg.rtc.guidance_mode,
            configured_max_guidance=cfg.rtc.max_guidance,
        )
        validate_capability(policy.metadata["rtc"], cfg.policy, cfg.rtc)
        for case in cases:
            source = Path(case["source"])
            case["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            with np.load(source, allow_pickle=False) as data:
                now = time.monotonic()
                obs = Observation(
                    data["state"].copy(),
                    {k: data[k].copy() for k in cfg.policy.camera_names},
                    now,
                    time.time(),
                    now,
                )
            _, body = build_request(obs, case["task"], cfg.policy, case["wire"])
            bodies.append(body)

        variants = ["native", "zero", "pi_vjp5", "lerobot10", "upstream10"]

        def infer(body, seed, variant):
            policy.reset()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            inp = copy.deepcopy(body)
            if variant == "native":
                inp.pop("rtc")
            else:
                inp["rtc"].update(
                    guidance_mode="pi_vjp" if variant == "pi_vjp5" else "lerobot",
                    max_guidance=5.0 if variant == "pi_vjp5" else 10.0,
                )
            active_wire = inp.get("rtc")
            torch.cuda.synchronize()
            start = time.perf_counter()
            callback = (
                reference_guidance(upstream, active_wire)
                if variant == "upstream10"
                else zero_guidance
                if variant == "zero"
                else original
            )
            with traced_guidance(callback, expect_guidance=variant != "native") as trace:
                response = policy.infer(inp)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if active_wire is not None:
                validate_ack(response["rtc"], active_wire)
            actions = np.asarray(response["predict_action"], dtype=np.float64).reshape(32, 26)
            assert np.isfinite(actions).all()
            return actions, elapsed, trace.calls

        try:
            for variant in variants:
                infer(bodies[0], args.seeds[0], variant)
            for case, body in zip(cases, bodies, strict=True):
                strong, end = case["wire"]["committed_steps"], case["wire"]["guidance_horizon"]
                directions, prefix = np.array(case["directions"]), np.array(case["wire"]["prefix"])
                for seed_index, seed in enumerate(args.seeds):
                    paired = {}
                    for variant in variants if seed_index % 2 == 0 else list(reversed(variants)):
                        actions, elapsed, guidance_calls = infer(body, seed, variant)
                        paired[variant] = actions
                        physical = actions[:, :7] * directions
                        error = abs(physical[:strong] - prefix[:strong] * directions)
                        delta = np.diff(physical[:, :6], axis=0)
                        regions = dict(
                            strong_internal=delta[: strong - 1],
                            strong_to_taper=delta[strong - 1 : strong],
                            taper_internal=delta[strong : end - 1],
                            taper_to_free=delta[end - 1 : end],
                            free_internal=delta[end:],
                        )
                        row = dict(
                            case=case["id"],
                            seed=seed,
                            variant=variant,
                            seconds=elapsed,
                            guidance_calls=guidance_calls,
                            arm_prefix_max_deg=float(error[:, :6].max()),
                            gripper_prefix_max_deg=float(error[:, 6].max()),
                            max_arm_step_by_region_deg={
                                k: float(abs(v).max()) if v.size else 0 for k, v in regions.items()
                            },
                            actions=actions.tolist(),
                        )
                        rows.append(row)
                        print(
                            json.dumps({k: v for k, v in row.items() if k != "actions"}), flush=True
                        )
                        save()
                    parity = float(
                        abs(
                            (paired["lerobot10"][:, :7] - paired["upstream10"][:, :7]) * directions
                        ).max()
                    )
                    unchanged = float(abs(paired["zero"] - paired["native"]).max())
                    checks.append(
                        dict(
                            case=case["id"],
                            seed=seed,
                            upstream_max_physical_error_deg=parity,
                            zero_vs_native_max_error=unchanged,
                            passed=parity < 0.05 and unchanged < 1e-5,
                        )
                    )
            report["timing_median_ms"] = {
                v: float(np.median([r["seconds"] for r in rows if r["variant"] == v]) * 1000)
                for v in variants
            }
            report["hardware_modules_loaded"] = sorted(
                {"motorbridge", "can", "lerobot_robot_seeed_b601"} & sys.modules.keys()
            )
            report["passed"] = (
                all(c["passed"] for c in checks) and not report["hardware_modules_loaded"]
            )
            save()
            print(
                "SUMMARY",
                json.dumps({k: report[k] for k in ["passed", "checks", "timing_median_ms"]}),
                flush=True,
            )
            assert report["passed"], "RTC numerical parity/native path regression"
        finally:
            save()


if __name__ == "__main__":
    main()
