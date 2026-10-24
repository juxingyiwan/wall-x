"""Frozen 5c8ced1 native vanilla sampler for behavioral regression."""
import torch
from torchdiffeq import odeint

def _generate_flow_action_vanilla(self, ctx):
    # Timing: prefill forward pass
    all_attention_maps = []
    output_attentions = True if ctx["output_attentions"] else False
    prefetch_output = self.model(
        input_ids=None,
        attention_mask=ctx["attention_mask"],
        position_ids=ctx["position_ids"],
        past_key_values=None,
        inputs_embeds=ctx["inputs_embeds"],
        moe_token_types=ctx["moe_token_types"],
        start_indices=ctx["start_indices"],
        end_indices=ctx["end_indices"],
        positional_masks=ctx["positional_masks"],
        use_cache=True,
        output_attentions=output_attentions,
        output_hidden_states=False,
        return_dict=True,
        adarms_conds=[None, ctx["adarms_cond"]],
    )
    hidden_states = prefetch_output.last_hidden_state
    prefix_kv_cache = prefetch_output.past_key_values
    if output_attentions:
        all_attention_maps.append(prefetch_output.attentions)
    action_hidden_states = hidden_states[ctx["flow_action_mask"]].to(torch.float32)
    action_pred = self.action_preprocessor.action_proj_back(
        action_hidden_states[:, : self.action_preprocessor.action_hidden_size]
    )
    if getattr(self.config, "use_x_pred", False):
        v_0 = action_pred - ctx["noise"].reshape(-1, ctx["noise"].shape[-1])
    else:
        v_0 = action_pred
    v_0 = v_0.reshape(ctx["batch_size"], ctx["action_horizon"], ctx["action_dim"])

    v_padding = None
    if (not ctx["dof_mask"].all()) and (ctx["padding_action"] is not None):
        v_padding = ctx["padding_action"] - ctx["noisy_action"]
        v_0 = (v_padding) * (1 - ctx["dof_mask"]) + v_0 * ctx["dof_mask"]
    dt = ctx["times"][1] - ctx["times"][0]
    ctx["noisy_action"] = ctx["noisy_action"] + dt * v_0

    (
        postfix_position_ids,
        postfix_inputs_embeds,
        _postfix_attention_mask,
        postfix_moe_token_types,
        postfix_input_ids,
        postfix_start_indices,
        postfix_end_indices,
        padding_mask,
    ) = self._prepare_flow_postfix(ctx, prefix_kv_cache)

    def step_with_kvcache(timestep, noisy_action):
        action_mask = (
            postfix_input_ids == self.action_token_id_set["action_token_id"]
        )
        assert action_mask.any(), "No action token found in input_ids"
        timestep = timestep.unsqueeze(0).repeat(noisy_action.shape[0])
        action_embed, adarms_cond = self.action_preprocessor.step(
            timestep=timestep, noisy_action=noisy_action, dof_mask=ctx["dof_mask"]
        )
        action_embed = action_embed.reshape(-1, postfix_inputs_embeds.shape[-1])

        # Create a temporary inputs_embeds copy for thread safety
        temp_inputs_embeds = postfix_inputs_embeds.clone()
        temp_inputs_embeds[action_mask] = action_embed.to(temp_inputs_embeds.dtype)
        transformer_outputs = self.model(
            input_ids=None,
            attention_mask=_postfix_attention_mask,
            position_ids=postfix_position_ids,
            past_key_values=prefix_kv_cache,
            inputs_embeds=temp_inputs_embeds,
            moe_token_types=postfix_moe_token_types,
            start_indices=postfix_start_indices,
            end_indices=postfix_end_indices,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
            adarms_conds=[None, adarms_cond],
        )
        if output_attentions:
            all_attention_maps.append(transformer_outputs.attentions)
        hidden_states = transformer_outputs.last_hidden_state
        action_hidden_states = hidden_states[action_mask].to(torch.float32)
        action_pred = self.action_preprocessor.action_proj_back(
            action_hidden_states[:, : self.action_preprocessor.action_hidden_size]
        )
        if getattr(self.config, "use_x_pred", False):
            # Align noisy_action (and timestep) to action_pred's shape
            B, action_horizon, action_dim = noisy_action.shape
            noisy_action_flat = noisy_action.reshape(
                -1, action_dim
            )  # [B * action_horizon, action_dim]
            timestep_expand = timestep.view(-1, 1).repeat_interleave(
                action_horizon, dim=0
            )  # [B * action_horizon, 1]
            v_t = (action_pred - noisy_action_flat) / torch.clamp(
                1 - timestep_expand, min=0.05
            )
        else:
            v_t = action_pred
        v_t = v_t.reshape(
            ctx["batch_size"], ctx["action_horizon"], ctx["action_dim"]
        )

        if (not ctx["dof_mask"].all()) and (ctx["padding_action"] is not None):
            v_t = (v_padding) * (1 - ctx["dof_mask"]) + v_t * ctx["dof_mask"]

        return v_t

    action_trajectory = odeint(
        step_with_kvcache, ctx["noisy_action"], ctx["times"][1:], method="euler"
    )

    attention_maps = None
    if output_attentions and ctx["img_mask"] is not None:
        attention_maps = [torch.stack(map, dim=0) for map in all_attention_maps]
        attention_maps = torch.stack(
            attention_maps, dim=0
        )  # [flow steps, layer depths, action tokens, all tokens]
        attention_maps = attention_maps.mean(2)
        image_mask_indices = ctx["img_mask"][0].nonzero(as_tuple=True)[0]
        attention_maps = attention_maps[:, :, image_mask_indices]
    return action_trajectory, attention_maps
