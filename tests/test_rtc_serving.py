"""Native serving preprocessing -> RTC condition, without model weights or GPU."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wall_x._vendor.harrix.serving._wallx_infer.joint_state import LAYOUT
from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import WallxModelWrapper
from wall_x._vendor.harrix.serving._wallx_infer.robot import DesktopRobotPreprocessor
from wall_x.data.backends.lerobot.utils import NormStats
from wall_x.model.core.action.normalizer import Normalizer


@pytest.mark.parametrize("prefix_steps", [20, 32])
@pytest.mark.parametrize("guidance_mode", [None, "lerobot", "pi_vjp"])
def test_real_preprocessor_mask_and_wrapper_guidance(
    monkeypatch, prefix_steps, guidance_mode
):
    cfg = SimpleNamespace(
        train_config={
            "agent_pos_config": dict(LAYOUT),
            "dof_config": dict(LAYOUT),
            "data": {},
        },
        data_config={},
        action_horizon=32,
        action_dim=26,
        cam_names=["front_view", "side_view"],
        camera_front_key="front_view",
        camera_left_key="left_wrist_view",
        camera_right_key="side_view",
        model_device="cpu",
        num_inference_timesteps=10,
    )
    wrapper = WallxModelWrapper.__new__(WallxModelWrapper)
    wrapper.config, wrapper.norm_key = cfg, "fixture"
    wrapper.logger = logging.getLogger("rtc_test")
    normalizer = Normalizer.from_lerobot_norm_stats(
        NormStats(min=torch.zeros(26), max=torch.ones(26), delta=torch.ones(26)),
        "fixture",
    )
    wrapper.normalizer_propri = wrapper.normalizer_action = normalizer
    wrapper.get_text_for_action = lambda _: ("prefix", "postfix")
    wrapper._resize_images = lambda _: []
    captured = []

    def generate(**kwargs):
        condition = kwargs["rtc_condition"]
        assert condition.guidance_mode == (guidance_mode or "pi_vjp")
        captured.append(condition)
        torch.testing.assert_close(condition.prefix[:, :20, :7], torch.ones(1, 20, 7))
        assert condition.prefix[:, 20:].count_nonzero() == 0
        assert condition.weights[:, 20:].count_nonzero() == 0
        assert condition.prefix[..., 7:].count_nonzero() == 0
        assert kwargs["dof_mask"].shape == (1, 32, 26)
        return {"predict_action": torch.zeros(1, 32, 26)}

    wrapper.model = SimpleNamespace(
        processor=SimpleNamespace(
            tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 7)
        ),
        generate_flow_action=generate,
    )
    module = "wall_x._vendor.harrix.serving._wallx_infer.model_wrapper"
    monkeypatch.setattr(module + ".ENABLE_FAST_PREPROCESS", False)
    monkeypatch.setattr(
        module + ".preprocesser_call",
        lambda *a, **k: {"input_ids": torch.tensor([[7]])},
    )
    views = {name: np.zeros((48, 64, 3), dtype=np.uint8) for name in cfg.cam_names}
    obs = DesktopRobotPreprocessor(cfg).get_observation(
        {"joint_state": np.zeros(7, dtype=np.float32)}, views
    )
    request = dict(
        prefix=np.ones((prefix_steps, 7)),
        committed_steps=14,
        guidance_horizon=20,
        max_guidance=5,
    )
    if guidance_mode is not None:
        request["guidance_mode"] = guidance_mode
    output = wrapper.infer_flow_action(obs, "test", rtc_request=request)
    assert output["predict_action"].shape == (1, 32, 26)
    assert len(captured) == 1
