"""RoboTwin batching: preserve real action/state tails and native padding."""

import torch

from wall_x.data.backends.lerobot.loader import (
    DataCollator,
    preprocesser_call,
    replace_action_token,
)
from wall_x.robotwin.normalization import resolve_normalization, scale_quantiles


class RobotwinDataCollator(DataCollator):
    def __init__(self, config, dataload_config, stats, lerobot_config):
        policy_config = dict(config)
        policy_config["data"] = {
            **config.get("data", {}),
            "lerobot_config": lerobot_config,
        }
        self.normalization = resolve_normalization(policy_config)
        super().__init__(config, dataload_config, stats, lerobot_config)

    def __call__(self, batch):
        additional_inputs = {}

        # Tail-pad widths when dof_config / agent_pos_config (sum) is larger
        # than the lerobot action/state - typical when resuming a ckpt that
        # was pretrained on a bigger action space. Extra columns are filled
        # with zeros and their mask set to 0 so loss is not propagated.
        dof_total = int(self.config.get("dof_total_dim", 0) or 0)
        agent_pos_total = int(self.config.get("agent_pos_total_dim", 0) or 0)

        # Explicit init so the ``if action is not None`` guard and later
        # ``replace_action_token`` call stay well-defined even if a batch
        # unexpectedly omits the action / agent_pos keys. Without this the
        # loop-local variables would leak NameError on the first miss.
        action = None
        dof_mask = None
        agent_pos = None
        agent_pos_mask = None

        for key in batch[0].keys():
            if key == "agent_pos":
                agent_pos = torch.stack([item["agent_pos"] for item in batch])
                if agent_pos.dim() == 2:
                    agent_pos = agent_pos.unsqueeze(1)
                agent_pos_mask = (~torch.isnan(agent_pos)).float()
                agent_pos.nan_to_num_(nan=0.0)
                state_min_stat = self._align_norm_stat(
                    self.state_min_stat,
                    agent_pos,
                    pad_value=0.0,
                    name="state.min",
                )
                state_delta = self._align_norm_stat(
                    self.state_delta,
                    agent_pos,
                    pad_value=1.0,
                    name="state.delta",
                )
                agent_pos = scale_quantiles(
                    agent_pos,
                    state_min_stat,
                    state_delta,
                    clip=self.normalization.clip_state_inputs,
                )
                if agent_pos_total and agent_pos.shape[-1] < agent_pos_total:
                    pad_w = agent_pos_total - agent_pos.shape[-1]
                    agent_pos = torch.nn.functional.pad(agent_pos, (0, pad_w))
                    agent_pos_mask = torch.nn.functional.pad(agent_pos_mask, (0, pad_w))
                additional_inputs["proprioception"] = agent_pos
                additional_inputs["agent_pos_mask"] = agent_pos_mask
            elif key == "action":
                action = torch.stack([item["action"] for item in batch])
                if action.dim() == 2:
                    action = action.unsqueeze(1)
                dof_mask = (~torch.isnan(action)).float()
                action.nan_to_num_(nan=0.0)
                action_min_stat = self._align_norm_stat(
                    self.action_min_stat,
                    action,
                    pad_value=0.0,
                    name="action.min",
                )
                action_delta = self._align_norm_stat(
                    self.action_delta,
                    action,
                    pad_value=1.0,
                    name="action.delta",
                )
                action = scale_quantiles(
                    action,
                    action_min_stat,
                    action_delta,
                    clip=self.normalization.clip_action_targets,
                )
                if dof_total and action.shape[-1] < dof_total:
                    pad_w = dof_total - action.shape[-1]
                    action = torch.nn.functional.pad(action, (0, pad_w))
                    dof_mask = torch.nn.functional.pad(dof_mask, (0, pad_w))
                additional_inputs["action_chunk"] = action
                additional_inputs["dof_mask"] = dof_mask
            elif key == "image_inputs":
                additional_inputs["image_inputs"] = [
                    item["image_inputs"] for item in batch
                ]
            elif key == "text":
                additional_inputs["text"] = [item["text"] for item in batch]
            elif key == "frame_index":
                additional_inputs["frame_index"] = torch.stack(
                    [item["frame_index"] for item in batch]
                )
            else:
                raise NotImplementedError(
                    f"{key} input not implemented in preprocesser"
                )

        # sample noise time
        if action is not None:
            sample_time = self.sample_time(
                action.shape[0],
                device=action.device,
                dtype=torch.float32,
            )
            additional_inputs["sample_time"] = sample_time

        additional_inputs["text"] = replace_action_token(
            additional_inputs["text"],
            additional_inputs["action_chunk"],
            self.train_action_tokenizer if self.use_fast_tokenizer else None,
            additional_inputs["dof_mask"],
        )

        inputs = preprocesser_call(
            processor=self.processor,
            text=additional_inputs.pop("text"),
            images=additional_inputs.pop("image_inputs"),
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.dataload_config.get("max_length", 768),
            norm_state=(
                additional_inputs["proprioception"]
                if self.use_state_string_representation
                and "proprioception" in additional_inputs
                else None
            ),
            agent_pos_mask=additional_inputs.get("agent_pos_mask"),
            state_bins=self.state_bins,
        )

        action_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|action|>")

        # Gating token types
        additional_inputs["moe_token_types"] = inputs.input_ids == action_token_id

        inputs.update(additional_inputs)

        inputs["dataset_names"] = [self.lerobot_config["repo_id"]] * inputs[
            "action_chunk"
        ].shape[0]

        return inputs
