"""Offline padding comparison on recorded observations; never opens robot hardware.

The local/server fixture fixes model noise only in this process. Production
protocols do not accept a seed or padding override per request. Each process
loads its policy from an explicit profile, just as deployment does.
With --baseline-repo, zero-mode client roundtrip includes a second inference;
use trace.json model_ms/infer_ms for latency comparisons, not client roundtrip.
"""

import argparse
import ast
import asyncio
import copy
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import MethodType

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "deployment/client/src")]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def plan(cases):
    return [(cases[0], guided, True) for guided in (False, True)] + [
        (case, guided, False) for case in cases for guided in (False, True)
    ]


def run_client(cfg, cases, factory, output):
    from wallx_client.contracts import Observation, ProtocolError

    rows, connections = [], {}
    try:
        for case, guided, warmup in plan(cases):
            if guided not in connections:
                rtc = replace(cfg.rtc, enabled=guided, guidance_mode="lerobot")
                connection = factory(cfg.policy, rtc=rtc)
                metadata = connection.open()
                assert metadata["padding"] == cfg.policy.padding
                connections[guided] = connection
            connection = connections[guided]
            source = Path(case["source"])
            assert hashlib.sha256(source.read_bytes()).hexdigest() == case["source_sha256"]
            with np.load(source, allow_pickle=False) as data:
                now = time.monotonic()
                obs = Observation(
                    data["state"].copy(),
                    {name: data[name].copy() for name in cfg.policy.camera_names},
                    now,
                    time.time(),
                    now,
                )
            wire = copy.deepcopy(case["wire"]) if guided else None
            if wire is not None:
                wire.update(guidance_mode="lerobot", max_guidance=10.0)
            start = time.perf_counter()
            prediction = connection.predict(
                obs, case["task"], 0 if wire is None else wire["epoch"], rtc_request=wire
            )
            assert prediction.actions.shape == (32, 7) and np.isfinite(prediction.actions).all()
            if guided:
                assert prediction.rtc["mode"] == "guided"
            rows.append(
                dict(
                    case=case["id"],
                    rtc=guided,
                    warmup=warmup,
                    padding=cfg.policy.padding,
                    actions=prediction.actions.tolist(),
                    roundtrip_ms=prediction.roundtrip_s * 1000,
                    total_ms=(time.perf_counter() - start) * 1000,
                )
            )
            save(output, dict(passed=False, rows=rows, metadata=metadata))
        # Exercise both mode directions against the real loaded model metadata.
        wrong = replace(cfg.policy, padding="original" if cfg.policy.padding == "zero" else "zero")
        bad = factory(wrong, rtc=replace(cfg.rtc, enabled=False))
        try:
            try:
                bad.open()
            except ProtocolError as exc:
                assert "padding mismatch" in str(exc)
            else:
                raise AssertionError("Mismatched padding was accepted")
        finally:
            bad.close()
    finally:
        cleanup = [c.close_with_status(3) for c in connections.values()]
    assert all(not c["timed_out"] for c in cleanup)
    assert not {"motorbridge", "can", "lerobot_robot_seeed_b601"} & sys.modules.keys()
    result = dict(passed=True, rows=rows, metadata=metadata, cleanup=cleanup, hardware_opened=False)
    save(output, result)
    return result


def prior_method(repository, relative, owner, name, namespace):
    """Compile one prior method in memory; never retain another source checkout."""
    source = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=repository, text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == owner)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            fn,
        ],
        type_ignores=[],
    )
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), scope)
    return scope[name]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("local", "server", "client"), required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--machine", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--port", type=int, default=34879)
    p.add_argument("--baseline-repo", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(a.cases.read_text())
    from wallx_client.config import load_config
    from wallx_client.transport import WallXConnection

    from deployment.cli import load_deployment
    from deployment.local_policy import LocalPolicyConnection
    from deployment.local_runtime import local_deployment, native_policy

    deployment = load_deployment(a.profile, a.machine)
    if a.backend == "local":
        deployment = local_deployment(deployment)
    client = copy.deepcopy(deployment["client"])
    client["control"].update(executor=None)
    if a.backend != "local":
        client["policy"]["uri"] = f"ws://127.0.0.1:{a.port}"
    effective = a.output / "client.yml"
    effective.write_text(yaml.safe_dump(client))
    cfg = load_config(effective)
    if a.backend == "client":
        run_client(cfg, cases, WallXConnection, a.output / "client-results.json")
        return

    import torch

    steps = plan(cases)
    with native_policy(deployment, a.output / "model.log") as policy:
        model, wrapper = policy.model_wrapper.model, policy.model_wrapper
        mode = deployment["padding"]
        assert model.inference_padding == wrapper.config.padding == mode
        captures, rows = {}, []
        prepare, generate, construct = (
            model._prepare_flow_action_inputs,
            model.generate_flow_action,
            wrapper.construct_model_input,
        )
        old_prepare, old_construct = None, None
        if mode == "zero" and a.baseline_repo:
            import wall_x._vendor.harrix.serving._wallx_infer.model_wrapper as wrapper_module
            import wall_x.model.qact.qwen2_5.modeling_qwen2_5_vl_act as native_module

            old_prepare = prior_method(
                a.baseline_repo,
                "wall_x/model/qact/qwen2_5/modeling_qwen2_5_vl_act.py",
                "Qwen2_5_VLMoEForAction",
                "_prepare_flow_action_inputs",
                vars(native_module),
            )
            old_construct = prior_method(
                a.baseline_repo,
                "wall_x/_vendor/harrix/serving/_wallx_infer/model_wrapper.py",
                "WallxModelWrapper",
                "construct_model_input",
                vars(wrapper_module),
            )

        def tensor_hash(value):
            return hashlib.sha256(
                value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()

        def traced_prepare(*args, **kwargs):
            ctx = prepare(*args, **kwargs)
            captures["noise_sha256"] = tensor_hash(ctx["noise"])
            endpoint = ctx["padding_action"][..., 7:]
            expected = 0 if mode == "zero" else -1
            assert torch.all(endpoint == expected)
            captures["virtual_endpoint"] = expected
            return ctx

        def traced_construct(*args, **kwargs):
            inputs = construct(*args, **kwargs)
            state = inputs["proprioception"][..., 7:]
            expected = 0 if mode == "zero" else -1
            assert torch.all(state == expected)
            captures["virtual_state"] = expected
            return inputs

        fixed_noise = None

        def fixed_generate(*args, **kwargs):
            kwargs["noise"] = fixed_noise
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = generate(*args, **kwargs)
            torch.cuda.synchronize()
            captures["model_ms"] = (time.perf_counter() - start) * 1000
            return result

        model._prepare_flow_action_inputs = traced_prepare
        model.generate_flow_action = fixed_generate
        wrapper.construct_model_input = traced_construct

        class Fixture:
            @property
            def metadata(self):
                return policy.metadata

            def infer(self, body):
                nonlocal fixed_noise
                case, guided, warmup = steps[len(rows)]
                assert body["instruction"] == case["task"]
                assert (body.get("rtc") is not None) == guided
                seed = 7 + cases.index(case)
                fixed_noise = torch.randn(
                    (1, 32, 26), generator=torch.Generator().manual_seed(seed)
                ).to(model.device)
                captures.clear()
                torch.manual_seed(seed)
                start = time.perf_counter()
                response = policy.infer(copy.deepcopy(body))
                torch.cuda.synchronize()
                current = dict(captures)
                current.update(
                    case=case["id"],
                    rtc=guided,
                    warmup=warmup,
                    padding=mode,
                    infer_ms=(time.perf_counter() - start) * 1000,
                    seed=seed,
                )
                if old_prepare is not None and not warmup:
                    try:
                        model._prepare_flow_action_inputs = MethodType(old_prepare, model)
                        wrapper.construct_model_input = MethodType(old_construct, wrapper)
                        torch.manual_seed(seed)
                        baseline = policy.infer(copy.deepcopy(body))
                        np.testing.assert_array_equal(
                            response["predict_action"], baseline["predict_action"]
                        )
                        current["prior_zero_bitwise_equal"] = True
                    finally:
                        model._prepare_flow_action_inputs = traced_prepare
                        wrapper.construct_model_input = traced_construct
                rows.append(current)
                save(a.output / "trace.json", rows)
                return response

        fixture = Fixture()
        if a.backend == "local":
            run_client(
                cfg,
                cases,
                lambda config, rtc: LocalPolicyConnection(config, fixture, rtc=rtc),
                a.output / "client-results.json",
            )
        else:
            from wall_x._vendor.harrix.serving.websocket_policy_server import WebsocketPolicyServer

            async def serve():
                server = WebsocketPolicyServer(
                    fixture, host="127.0.0.1", port=a.port, metadata=fixture.metadata
                )
                task = asyncio.create_task(server.run())
                try:
                    for _ in range(100):
                        if task.done():
                            await task
                        try:
                            _, writer = await asyncio.open_connection("127.0.0.1", a.port)
                            writer.close()
                            await writer.wait_closed()
                            break
                        except OSError:
                            await asyncio.sleep(0.1)
                    else:
                        raise TimeoutError("Probe server did not start")
                    save(a.output / "ready.json", dict(padding=mode, port=a.port))
                    deadline = time.monotonic() + 600
                    while len(rows) < len(steps):
                        if task.done():
                            await task
                        if time.monotonic() > deadline:
                            raise TimeoutError("Probe client deadline")
                        await asyncio.sleep(0.1)
                    # Keep the listener for the client's mismatched-mode check.
                    await asyncio.sleep(5)
                finally:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            asyncio.run(serve())
        assert len(rows) == len(steps)
        assert not {"motorbridge", "can", "lerobot_robot_seeed_b601"} & sys.modules.keys()
        save(
            a.output / "complete.json",
            dict(
                passed=True,
                padding=mode,
                gpu=torch.cuda.get_device_name(),
                rows=len(rows),
                formal_rows=sum(not row["warmup"] for row in rows),
                hardware_opened=False,
            ),
        )


if __name__ == "__main__":
    main()
