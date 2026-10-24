"""CPU assembly/timebase tests and frozen pre-refactor native flow parity."""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_b601_pipeline import artifact, make_stats, save_manifest

from wall_x._vendor.harrix.serving._wallx_infer.joint_state import joint_metadata
from wall_x._vendor.harrix.serving.rtc import build_capability
from wall_x.model.adapters.qwen_flow import generate_qwen_flow
from wall_x.model.core.action.rtc import RTCCondition

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fps", [20, 30, 60])
def test_real_metadata_and_create_policy_timebase(tmp_path, fps):
    cfg, _, episodes, split = artifact(tmp_path)
    cfg["data"]["lerobot_config"]["b601"]["fps"] = fps
    stats = make_stats(cfg, episodes, split, [])
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    save_manifest(cfg, tmp_path)
    weights = b"metadata-only-fixture"
    (tmp_path / "model.safetensors").write_bytes(weights)
    config = SimpleNamespace(
        train_config=cfg,
        checkpoint_path=tmp_path,
        action_horizon=32,
        action_dim=26,
        cam_names=["front_view", "side_view"],
        smooth_action=False,
    )
    metadata = joint_metadata(config)
    assert metadata["checkpoint_sha256"] == hashlib.sha256(weights).hexdigest()
    assert metadata["model_dt_s"] == 1 / fps
    policy = SimpleNamespace(config=config, metadata={"wallx_client": metadata})
    # Execute actual create_policy assembly with a model factory replacement;
    # metadata, manifest validation, codec contract and capability are real.
    source = (ROOT / "wall_x/_vendor/harrix/serving/launch_serving.py").read_text()
    definition = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "create_policy"
    )
    import os

    namespace = dict(
        Args=object,
        EnvMode=SimpleNamespace(X2ROBOT="real"),
        os=os,
        get_model_config=lambda _: config,
        get_wallx_policy=lambda *args: policy,
    )
    exec(
        compile(
            ast.Module(body=[definition], type_ignores=[]), "create_policy", "exec"
        ),
        namespace,
    )
    args = SimpleNamespace(
        model_type="wallx",
        image_passing_mode="base64",
        serialize_actions=False,
        env="real",
        max_batch_size=None,
        enable_cuda_graph=False,
        enable_experimental_engine=False,
    )
    cap = namespace["create_policy"](args).rtc_capability
    assert cap["supported"] == (fps == 30)
    assert cap["model_dt_s"] == 1 / fps
    assert cap["model_timebase_source"] == "validated_b601_manifest"


def test_versioned_contract_cannot_inherit_legacy_timebase():
    policy = SimpleNamespace(
        config=SimpleNamespace(action_horizon=32, action_dim=26),
        metadata={
            "wallx_client": {
                "state_dim": 7,
                "model_action_representation": "b601_joint_delta_v1",
            }
        },
    )
    assert not build_capability(policy, enabled=True)["supported"]


@pytest.mark.parametrize("x_pred", [False, True])
@pytest.mark.parametrize("cache_layers", [False, True])
@pytest.mark.parametrize("batch,attention", [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shared_sampler_matches_frozen_native(
    x_pred, cache_layers, batch, attention, dtype
):
    spec = importlib.util.spec_from_file_location(
        "frozen_native", ROOT / "tests/fixtures/rtc/wallx_5c8ced1_native_flow.py"
    )
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)

    class Preprocessor:
        action_hidden_size = 3

        @staticmethod
        def step(timestep, noisy_action, dof_mask):
            return noisy_action * 0.3 + timestep[:, None, None] * 0.1, None

        @staticmethod
        def action_proj_back(hidden):
            return hidden

    class Model:
        config = SimpleNamespace(use_x_pred=x_pred)
        action_preprocessor = Preprocessor()
        action_token_id_set = {"action_token_id": 1}

        @staticmethod
        def model(inputs_embeds, past_key_values, use_cache, **kwargs):
            cache = (
                SimpleNamespace(
                    layers=[
                        SimpleNamespace(
                            keys=inputs_embeds.unsqueeze(1),
                            values=inputs_embeds.unsqueeze(1),
                        )
                    ]
                )
                if cache_layers
                else SimpleNamespace(
                    key_cache=[inputs_embeds.unsqueeze(1)],
                    value_cache=[inputs_embeds.unsqueeze(1)],
                )
            )
            return SimpleNamespace(
                last_hidden_state=inputs_embeds,
                past_key_values=cache,
                attentions=[torch.ones(batch, 2, 4, 5)],
            )

        @staticmethod
        def _prepare_flow_postfix(ctx, cache):
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

    torch.manual_seed(81)
    x = torch.randn(batch, 4, 3)
    ctx = dict(
        batch_size=batch,
        action_horizon=4,
        action_dim=3,
        output_attentions=attention,
        dof_mask=torch.tensor([1, 1, 0]).view(1, 1, 3),
        noisy_action=x,
        noise=x.clone(),
        times=torch.linspace(0, 1, 11) * 0.999,
        padding_action=torch.zeros_like(x),
        inputs_embeds=torch.cat([torch.ones(batch, 1, 3), x * 0.3], dim=1).to(dtype),
        flow_action_mask=torch.tensor([[False, True, True, True, True]]).expand(
            batch, -1
        ),
        input_ids=torch.tensor([[0, 1, 1, 1, 1]]).expand(batch, -1),
        img_mask=torch.tensor([[True, False, False, False, False]]),
        adarms_cond=None,
    )
    ctx.update(
        {
            key: None
            for key in (
                "attention_mask",
                "position_ids",
                "moe_token_types",
                "start_indices",
                "end_indices",
                "positional_masks",
            )
        }
    )
    with torch.no_grad():
        expected, maps = frozen._generate_flow_action_vanilla(
            Model(), copy.deepcopy(ctx)
        )
        actual, actual_maps = generate_qwen_flow(Model(), copy.deepcopy(ctx))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if attention:
            torch.testing.assert_close(actual_maps, maps, rtol=0, atol=0)
        if batch == 1 and not attention:
            for mode in ("lerobot", "pi_vjp"):
                condition = RTCCondition(
                    torch.zeros_like(x), torch.zeros(1, 4, 1), guidance_mode=mode
                )
                guided, _ = generate_qwen_flow(Model(), copy.deepcopy(ctx), condition)
                torch.testing.assert_close(
                    torch.stack(guided), expected, rtol=0, atol=0
                )
