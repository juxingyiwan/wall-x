"""Actual Qwen method chain on CPU; only neural network/cache IO is substituted."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from wall_x.model.core.action.normalizer import (
    Normalizer,
    build_flow_padding_action,
    unnormalize_data_with_virtual_tail,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "wall_x/model/qact/qwen2_5/modeling_qwen2_5_vl_act.py"
owner = next(
    n
    for n in ast.parse(SOURCE.read_text()).body
    if isinstance(n, ast.ClassDef) and n.name == "Qwen2_5_VLMoEForAction"
)
METHODS = {
    "generate_flow_action",
    "_prepare_flow_action_inputs",
    "_execute_flow_action",
    "_generate_flow_action_vanilla",
    "_finalize_flow_action_output",
    "model_rtc_capability",
    "supports_rtc",
    "predict_action",
}
namespace = dict(
    torch=torch,
    build_flow_padding_action=build_flow_padding_action,
    unnormalize_data_with_virtual_tail=unnormalize_data_with_virtual_tail,
)
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *[
            n
            for n in owner.body
            if isinstance(n, ast.FunctionDef) and n.name in METHODS
        ],
    ],
    type_ignores=[],
)
exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)


class SyntheticModel:
    def __init__(self, horizon=8, dim=3, real=3, key="fixture"):
        self.horizon, self.dim, self.real, self.key = horizon, dim, real, key
        self.config = SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            use_return_dict=True,
            use_x_pred=False,
            dof_config={"arm": real, "action_padding": dim - real},
        )
        self.times_cache = {}
        self.action_token_id_set = {"action_token_id": 1}
        stats = SimpleNamespace(min=torch.zeros(dim), delta=torch.ones(dim) * 2)
        self.action_preprocessor = SimpleNamespace(
            action_hidden_size=dim,
            normalizer_action=Normalizer.from_lerobot_norm_stats(stats, key),
            get_inference_times=lambda steps, device, dtype: torch.linspace(
                0, 0.999, steps + 1, device=device, dtype=dtype
            ),
            step=lambda timestep, noisy_action, dof_mask: (
                noisy_action * 0.3 + timestep[:, None, None] * 0.1,
                None,
            ),
            action_proj_back=lambda hidden: hidden * 0.5,
        )
        self.last_noise = None
        self.calls = 0
        self.parameter = torch.nn.Parameter(torch.tensor(0.1))

    def _finalize_flow_action_output(self, trajectory, ctx):
        self.last_noise = ctx["noise"].clone()
        return namespace["_finalize_flow_action_output"](self, trajectory, ctx)

    def model(self, inputs_embeds, past_key_values, use_cache, **kwargs):
        self.calls += 1
        hidden = inputs_embeds + self.parameter * torch.tanh(
            inputs_embeds.mean(dim=1, keepdim=True)
        )
        cache = SimpleNamespace(
            key_cache=[inputs_embeds.unsqueeze(1)],
            value_cache=[inputs_embeds.unsqueeze(1)],
        )
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)

    @staticmethod
    def _prepare_flow_postfix(ctx, cache):
        return (
            ctx["position_ids"],
            ctx["inputs_embeds"],
            None,
            None,
            ctx["input_ids"],
            None,
            None,
            None,
        )

    def inputs(self, batch=1):
        h, d = self.horizon, self.dim
        mask = torch.zeros(batch, h, d)
        mask[..., : self.real] = 1
        return dict(
            input_ids=torch.tensor([[0] + [1] * h]).expand(batch, -1),
            inputs_embeds=torch.zeros(batch, h + 1, d),
            position_ids=torch.zeros(3, batch, h + 1, dtype=torch.long),
            prefix_length=1,
            start_indices=torch.zeros(1, dtype=torch.long),
            end_indices=torch.tensor([h + 1]),
            dataset_names=[self.key] * batch,
            dof_mask=mask,
        )

    def run(self, *, condition=None, noise=None, seed=601, **kwargs):
        torch.manual_seed(seed)
        return self.generate_flow_action(
            action_horizon=self.horizon,
            action_dim=self.dim,
            rtc_condition=condition,
            noise=noise,
            **self.inputs(),
            **kwargs,
        )["predict_action"]


for name in METHODS - {"_finalize_flow_action_output"}:
    setattr(SyntheticModel, name, namespace[name])
namespace["Qwen2_5_VLMoEForAction"] = SyntheticModel
