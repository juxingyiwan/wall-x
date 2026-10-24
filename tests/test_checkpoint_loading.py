"""Exercise real weight loaders and serving metadata with tiny CPU tensors."""

import hashlib
import logging
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from deployment.cli import verify_checkpoint
from wall_x._vendor.harrix.serving._wallx_infer.joint_state import LAYOUT
from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import WallxModelWrapper
from wall_x._vendor.harrix.serving.policy.wall_x_policy import WallXPolicy
from wall_x._vendor.harrix.utils.ckpt_load import load_state_dict


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    @staticmethod
    def is_fused(state_dict):
        return True

    def set_normalizer(self, *normalizers):
        pass

    def to_bfloat16_for_selected_params(self):
        pass


@pytest.mark.parametrize("formats", ["safe", "fsdp", "both"])
def test_verified_loaded_and_advertised_weights_agree(tmp_path, formats):
    (tmp_path / "config.yml").write_text("{}")
    if formats in ("safe", "both"):
        save_file({"weight": torch.tensor([1.0])}, tmp_path / "model.safetensors")
    if formats in ("fsdp", "both"):
        torch.save(
            {"state_dict": {"weight": torch.tensor([2.0])}},
            tmp_path / "pytorch_model_fsdp.bin",
        )
    selected = tmp_path / (
        "model.safetensors" if formats == "safe" else "pytorch_model_fsdp.bin"
    )
    expected = 1.0 if formats == "safe" else 2.0
    expected_digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    verify_checkpoint(
        {
            "checkpoint_path": str(tmp_path),
            "model": {"checkpoint_sha256": expected_digest},
        }
    )

    config = SimpleNamespace(
        checkpoint_path=str(tmp_path),
        train_config={"agent_pos_config": dict(LAYOUT), "dof_config": dict(LAYOUT)},
        action_horizon=32,
        cam_names=["front_view", "side_view"],
        model_device="cpu",
    )
    wrapper = WallxModelWrapper.__new__(WallxModelWrapper)
    wrapper.config = config
    wrapper.logger = logging.getLogger("checkpoint_test")
    wrapper.ModelClass = TinyModel
    wrapper.model = TinyModel()
    wrapper.normalizer_action = wrapper.normalizer_propri = None
    wrapper.load_ckpt()
    assert wrapper.model.weight.item() == expected
    assert wrapper.checkpoint_weights == selected
    assert load_state_dict(str(tmp_path), TinyModel)["weight"].item() == expected

    # A newly added preferred file must not change metadata for an already loaded model.
    if formats == "safe":
        torch.save({"weight": torch.tensor([9.0])}, tmp_path / "pytorch_model_fsdp.bin")
    policy = WallXPolicy.__new__(WallXPolicy)
    policy.config = config
    policy.model_wrapper = wrapper
    policy.rtc_capability = {}
    assert policy.metadata["wallx_client"]["checkpoint_sha256"] == expected_digest
