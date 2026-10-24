"""Qwen-only embedding/cache bridge shared by native and guided flow sampling."""

import torch

from wall_x.model.core.action.rtc_contract import NOISE_AT_ZERO, ModelRTCCapability

QWEN_RTC_CAPABILITY = ModelRTCCapability(
    supported=True,
    guidance_modes=("lerobot", "pi_vjp"),
    backends=("native_pytorch",),
    unsupported_reason=None,
)


class QwenFlowAdapter:
    time_convention = NOISE_AT_ZERO

    def __init__(self, model, ctx, *, guided=False):
        if guided and (ctx["batch_size"] != 1 or ctx["output_attentions"]):
            raise ValueError("RTC v1 supports batch=1 without attention-map export")
        self.model, self.ctx, self.guided = model, ctx, guided
        self.mask = ctx["dof_mask"]
        self.padding_velocity = None
        if not self.mask.all() and ctx["padding_action"] is not None:
            self.padding_velocity = ctx["padding_action"] - ctx["noisy_action"]
        self.cache = self.postfix = None
        self.attentions = []

    def denoise(self, t, noisy):
        model, ctx = self.model, self.ctx
        first = self.postfix is None
        if first:
            inputs = ctx["inputs_embeds"].detach().clone()
            action_mask = ctx["flow_action_mask"]
            kwargs = {
                key: ctx[key]
                for key in (
                    "attention_mask",
                    "position_ids",
                    "moe_token_types",
                    "start_indices",
                    "end_indices",
                    "positional_masks",
                )
            }
        else:
            pos, base, attn, moe, ids, start, end, _ = self.postfix
            inputs = base.detach().clone()
            action_mask = ids == model.action_token_id_set["action_token_id"]
            kwargs = dict(
                attention_mask=attn,
                position_ids=pos,
                moe_token_types=moe,
                start_indices=start,
                end_indices=end,
            )
        if first and not self.guided:
            # Native input preparation already embedded t=0. Preserve its exact
            # prefill, including export/batch behavior and no extra model work.
            cond = ctx["adarms_cond"]
        else:
            embed, cond = model.action_preprocessor.step(
                timestep=t.expand(noisy.shape[0]),
                noisy_action=noisy,
                dof_mask=self.mask,
            )
            inputs[action_mask] = embed.reshape(-1, inputs.shape[-1]).to(inputs.dtype)
        output = model.model(
            input_ids=None,
            inputs_embeds=inputs,
            past_key_values=self.cache,
            use_cache=first,
            output_attentions=bool(ctx["output_attentions"]),
            output_hidden_states=False,
            return_dict=True,
            adarms_conds=[None, cond],
            **kwargs,
        )
        if first:
            self.cache = output.past_key_values
        if ctx["output_attentions"]:
            self.attentions.append(output.attentions)
        hidden = output.last_hidden_state[action_mask].float()
        velocity = model.action_preprocessor.action_proj_back(
            hidden[:, : model.action_preprocessor.action_hidden_size]
        ).reshape_as(noisy)
        if getattr(model.config, "use_x_pred", False):
            velocity = (velocity - noisy) / (1 - t).clamp_min(0.05)
        if self.padding_velocity is not None:
            velocity = velocity * self.mask + self.padding_velocity * (1 - self.mask)
        return velocity

    def after_step(self, index):
        if index != 0:
            return
        if self.cache is None:
            raise RuntimeError("Qwen flow requires the native prefix KV cache")
        self.postfix = self.model._prepare_flow_postfix(self.ctx, self.cache)
        # Detach only after the first optional VJP has consumed its graph.
        if hasattr(self.cache, "key_cache"):
            for i in range(len(self.cache.key_cache)):
                self.cache.key_cache[i] = self.cache.key_cache[i].detach()
                self.cache.value_cache[i] = self.cache.value_cache[i].detach()
        else:
            for layer in self.cache.layers:
                layer.keys = layer.keys.detach()
                layer.values = layer.values.detach()

    def attention_maps(self):
        if not self.ctx["output_attentions"] or self.ctx["img_mask"] is None:
            return None
        maps = torch.stack(
            [torch.stack(values, dim=0) for values in self.attentions], dim=0
        )
        image_indices = self.ctx["img_mask"][0].nonzero(as_tuple=True)[0]
        return maps.mean(2)[:, :, image_indices]


def generate_qwen_flow(model, ctx, condition=None):
    from wall_x.model.core.action.flow import sample_flow

    if condition is not None:
        QWEN_RTC_CAPABILITY.validate(
            condition,
            batch_size=ctx["batch_size"],
            output_attentions=ctx["output_attentions"],
        )
    adapter = QwenFlowAdapter(model, ctx, guided=condition is not None)
    trajectory = sample_flow(
        adapter, ctx["noisy_action"], ctx["times"], ctx["dof_mask"], condition
    )
    # Keep the native return type, attention export and ctx update contract.
    if condition is None:
        ctx["noisy_action"] = trajectory[0]
        return torch.stack(trajectory), adapter.attention_maps()
    return trajectory, None
