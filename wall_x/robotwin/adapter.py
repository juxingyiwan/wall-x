"""RoboTwin IO hooks over LIBERO's shared checkpoint and batched Flow path."""

import copy

import numpy as np
import torch
from PIL import Image

from wall_x._vendor.harrix.adapters.qwen_vlact import (
    _ACTION,
    _IMAGE_PAD,
    _PROPRI,
    _ROLE_END,
    _ROLE_START,
    _VISION_END,
    _VISION_START,
    _move_to_device,
    _normalize_real_prefix,
    get_prologue_with_embodied_information,
    preprocesser_call,
    resolve_agent_pos_config,
    resolve_state_bins,
    resolve_use_state_string_representation,
    zero_virtual_padding,
)
from wall_x._vendor.harrix.adapters.variants.qwen2_5 import Qwen2_5InferAdapter
from wall_x._vendor.harrix.utils.train_config import (
    resolve_camera_label,
    resolve_max_length,
)
from wall_x.robotwin.images import resize_robotwin_square
from wall_x.robotwin.normalization import adapt_normalizers, resolve_normalization

REAL_LAYOUT = {
    "follow_left_ee_cartesian_pos": 3,
    "follow_left_ee_rotation": 6,
    "follow_left_gripper": 1,
    "follow_right_ee_cartesian_pos": 3,
    "follow_right_ee_rotation": 6,
    "follow_right_gripper": 1,
}


def encode_observation(observation, train_config, horizon):
    """Encode absolute 20D EEF state without an Euler round trip."""
    state = np.asarray(observation["state"], dtype=np.float32)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError("RoboTwin state must contain 20 finite EEF values")
    layouts = [resolve_agent_pos_config(train_config), train_config["dof_config"]]
    for index, layout in enumerate(layouts):
        # Accept the explicit _6D spelling used by some checkpoint schemas.
        real = [
            (key.replace("_6D", ""), value)
            for key, value in layout.items()
            if key != "action_padding"
        ]
        expected = [
            (
                key
                + ("_relative" if index == 1 and not key.endswith("gripper") else ""),
                value,
            )
            for key, value in REAL_LAYOUT.items()
        ]
        if (
            real != expected
            or layout.get("action_padding") != 6
            or list(layout)[-1] != "action_padding"
        ):
            raise ValueError(
                "RoboTwin requires left/right XYZ+rotation6D+gripper and six padding dimensions"
            )
    for camera in ("face_view", "left_wrist_view", "right_wrist_view"):
        image = np.asarray(observation[camera])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected uint8 RGB image from {camera}")
    padded = np.concatenate((state, np.zeros(6, dtype=np.float32)))
    mask = np.concatenate(
        (np.ones(20, dtype=np.float32), np.zeros(6, dtype=np.float32))
    )
    return {
        **{key: value for key, value in observation.items() if key != "state"},
        "proprioception": padded[None, None],
        "agent_pos_mask": mask[None, None],
        "dof_mask": np.broadcast_to(mask, (1, horizon, 26)).copy(),
    }


class RobotwinInferAdapter(Qwen2_5InferAdapter):
    """Only dataset-specific IO differs from the LIBERO adapter."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.normalization = resolve_normalization(self._train_config)
        self._normalizer_action, self._normalizer_propri = adapt_normalizers(
            self._train_config, self._normalizer_action, self._normalizer_propri
        )
        if not self.normalization.is_legacy:
            self._model.set_normalizer(
                copy.deepcopy(self._normalizer_action).to(self._device),
                copy.deepcopy(self._normalizer_propri).to(self._device),
            )

    def _encode_observation(self, observation):
        return encode_observation(observation, self._train_config, self._action_horizon)

    def _decode_action_chunk(self, actions):
        # generate_flow_action has already unnormalized the action. The simulator
        # applies its chunk-anchor-relative pose transform exactly once.
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if (
            actions.shape != (self._action_horizon, 26)
            or not np.isfinite(actions).all()
        ):
            raise ValueError(f"Invalid RoboTwin action chunk: {actions.shape}")
        return actions

    def _camera_label(self, cam_name):
        return resolve_camera_label(
            cam_name, self._train_config["data"].get("camera_name_mapping")
        )

    def _input_max_length(self):
        return resolve_max_length(self._train_config)

    def _resize_image(self, img, cam_name):
        adapter = self._train_config["data"].get("lerobot_config", {})
        if not adapter.get("pad_to_square", False):
            return super()._resize_image(img, cam_name)
        image = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        data = self._data_config
        image = resize_robotwin_square(
            image,
            data.resolution[cam_name],
            factor=data.image_factor,
            min_pixels=data.min_pixels,
            max_pixels=data.max_pixels,
        )
        return torch.from_numpy(np.array(image)).to(self._device)

    def _flow_batch(self, payloads: list[dict]) -> list[np.ndarray]:
        # 1) Encode each observation into proprioception, masks, and views.
        observations: list[dict] = []
        instructions: list[str] = []
        noises: list[np.ndarray | None] = []
        any_noise = False
        for p in payloads:
            observations.append(self._encode_observation(p["observation"]))
            instructions.append(p["instruction"])
            n = p.get("noise")
            if n is not None:
                any_noise = True
            noises.append(n)

        # 2) Stack noise when provided; all-None delegates sampling to the model.
        if any_noise:
            if any(n is None for n in noises):
                raise ValueError(
                    "payload noise must be provided for every payload or for none"
                )
            batch_noise = torch.stack(
                [torch.from_numpy(n).to(dtype=torch.float32) for n in noises], dim=0
            )
        else:
            batch_noise = None

        # 3) Prompts and model inputs.
        prefix_list, postfix_list = [], []
        for ins in instructions:
            prefix, postfix = self._get_flow_prompt(ins)
            prefix_list.append(prefix)
            postfix_list.append(postfix)
        batch_inputs = self._construct_model_input(
            observations, prefix_list, postfix_list
        )

        # 4) Caller fallback for inactive real dimensions. The model rebuilds
        # explicit virtual endpoints via build_flow_padding_action; these must
        # be normalized zero, while the flow starting point remains noise.
        padding = torch.zeros(
            (
                len(batch_inputs["dataset_names"]),
                1,
                self._action_dim,
            ),
            dtype=torch.float32,
        )
        padding_action = _normalize_real_prefix(
            self._normalizer_action, padding, batch_inputs["dataset_names"]
        ).to(batch_inputs["input_ids"].device)

        # 5) Flow forward.
        model_output = self._model.generate_flow_action(
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            num_inference_timesteps=self._num_inference_timesteps,
            padding_action=padding_action,
            noise=batch_noise,
            **batch_inputs,
        )

        # 6) Decode one action chunk per payload.
        predict_action = model_output["predict_action"]  # (B, H, D_action)
        if isinstance(predict_action, torch.Tensor):
            predict_action = predict_action.detach().cpu().numpy()
        return [
            self._decode_action_chunk(predict_action[i : i + 1])
            for i in range(len(payloads))
        ]

    def _get_flow_prompt(self, instruction: str) -> tuple[str, str]:
        """Build the flow-action prompt."""
        if self._train_config["data"].get("use_embodied_system_prompt_ratio", 0) > 0:
            robot_id = (
                self._robot_id if self._norm_key in ("x2_normal", "ex_normal") else 0
            )
            cam_name_mapping = {cn: cn for cn in self._cam_names}
            prologue = get_prologue_with_embodied_information(
                dataset_name=self._norm_key,
                cam_mapping=cam_name_mapping,
                robot_id=robot_id,
                uid="",
                config=self._data_config,
            )
        else:
            prologue = f"{_ROLE_START}system\nYou are a helpful assistant.{_ROLE_END}\n"
        user_request = f"{_ROLE_START}user\nObservation:"
        for cn in self._cam_names:
            user_request += (
                f" {self._camera_label(cn)}: "
                f"{_VISION_START}{_IMAGE_PAD}{_VISION_END}"
            )
        user_request += "\nInstruction:"
        text_prompt = (
            f"\nPredict the next action in robot action.\nProprioception: {_PROPRI}\n"
        )
        user_message = f"{user_request} {instruction}{text_prompt}{_ROLE_END}\n"
        assistant_message = f"{_ROLE_START}assistant\n"
        flow_action = _ACTION * self._action_horizon

        prefix_text = prologue + user_message + assistant_message
        postfix_text = flow_action
        return prefix_text, postfix_text

    def _construct_model_input(
        self,
        observations: list[dict],
        prefix_list: list[str],
        postfix_list: list[str],
    ) -> dict:
        """Build model inputs for the flow inference path."""
        batch_size = len(observations)
        dataset_names = [self._norm_key] * batch_size

        # Proprioception and masks are prepared as ndarrays in encode_proprio.
        agent_pos = torch.cat(
            [torch.from_numpy(o["proprioception"]) for o in observations], dim=0
        )
        agent_pos_mask = torch.cat(
            [torch.from_numpy(o["agent_pos_mask"]) for o in observations], dim=0
        )
        dof_mask = torch.cat(
            [torch.from_numpy(o["dof_mask"]) for o in observations], dim=0
        )
        agent_pos = _normalize_real_prefix(
            self._normalizer_propri, agent_pos, dataset_names
        )
        # LIBERO follows the training normalize-then-pad contract: explicit
        # virtual state values and masks must both be zero. Legacy padded stats
        # can map raw zero to -1; a zero mask alone does not isolate these inputs.
        # Resolve the layout: stats width cannot identify virtual channels.
        state_layout = resolve_agent_pos_config(self._train_config)
        agent_pos = zero_virtual_padding(agent_pos, state_layout)
        agent_pos_mask = zero_virtual_padding(agent_pos_mask, state_layout)

        # Resize images per sample.
        image_inputs: list[torch.Tensor] = []
        all_image_sizes: list[tuple[int, int]] = []
        for o in observations:
            for cn in self._cam_names:
                if cn not in o:
                    continue
                tensor = self._resize_image(o[cn], cn)
                image_inputs.append(tensor)
                # Tensor (H, W, C) to PIL-compatible (W, H).
                all_image_sizes.append((tensor.shape[1], tensor.shape[0]))

        inputs = preprocesser_call(
            processor=self._processor,
            prefix_text=prefix_list,
            postfix_text=postfix_list,
            images=image_inputs,
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self._input_max_length(),
            pad_to_128_multiple=False,
            pad_prefix_to_same_length=False,
            norm_state=(
                agent_pos
                if resolve_use_state_string_representation(self._train_config)
                else None
            ),
            agent_pos_mask=agent_pos_mask,
            state_augmentation_prob=0.0,
            state_drop_prob=0.0,
            state_augmentation_ratio=0.0,
            state_bins=resolve_state_bins(self._train_config),
            inference_mode=True,
        )

        action_token_id = self._processor.tokenizer.convert_tokens_to_ids(_ACTION)
        moe_token_types = inputs["input_ids"] == action_token_id

        extra = {
            "proprioception": agent_pos.detach(),
            "agent_pos_mask": agent_pos_mask,
            "dof_mask": dof_mask,
            "image_size": all_image_sizes,
            "moe_token_types": moe_token_types,
            "dataset_names": dataset_names,
        }
        inputs.update(extra)
        return _move_to_device(inputs, self._device)
