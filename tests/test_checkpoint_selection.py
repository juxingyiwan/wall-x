import pytest
import torch
import yaml
from safetensors.torch import save_file

from wall_x._vendor.harrix.eval_config import autofill_from_checkpoint, load_eval_config
from wall_x._vendor.harrix.utils.ckpt_load import (
    load_state_dict,
    resolve_checkpoint_dir,
)
from wall_x.checkpoint_weights import resolve_checkpoint_weights


class TinyModel:
    @staticmethod
    def is_fused(state):
        return True


def test_explicit_file_wins_over_other_format(tmp_path):
    save_file({"marker": torch.tensor([1])}, str(tmp_path / "model.safetensors"))
    torch.save({"marker": torch.tensor([9])}, tmp_path / "pytorch_model_fsdp.bin")
    selected = resolve_checkpoint_weights(tmp_path / "model.safetensors")
    assert load_state_dict(str(selected), TinyModel)["marker"].item() == 1
    assert load_state_dict(str(tmp_path), TinyModel)["marker"].item() == 9
    with pytest.raises(ValueError, match="explicit weight file"):
        resolve_checkpoint_dir(str(selected))


def test_latest_root_resolves_metadata_from_selected_child(tmp_path):
    for step, horizon in [(2, 8), (9, 16)]:
        child = tmp_path / str(step)
        child.mkdir()
        save_file({"marker": torch.tensor([step])}, str(child / "model.safetensors"))
        torch.save({"global_step": step}, child / "global_step.pth")
        (child / "config.yml").write_text(
            yaml.safe_dump({"task": {"action_horizon_flow": horizon}})
        )
    (tmp_path / "config.yml").write_text("action_horizon_flow: 999\n")
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump({"model": {"checkpoint_path": str(tmp_path)}}))
    cfg = autofill_from_checkpoint(load_eval_config(str(config)))
    assert cfg.model.checkpoint_path == str(tmp_path / "9/model.safetensors")
    assert cfg.model.action_horizon == 16
    assert resolve_checkpoint_dir(str(tmp_path)) == str(tmp_path / "9")
