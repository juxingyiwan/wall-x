"""Prevent a syntactically accepted but ignored initialization configuration."""

from pathlib import Path

import pytest
import yaml

from wall_x.robotwin.preflight import audit_initialization

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    *sorted((ROOT / "workspace/example").glob("robotwin*.yml")),
]


@pytest.mark.parametrize(
    "clip_action,clip_state,step_limit",
    [(False, False, 20000), (True, True, 30000), (True, True, 25000)],
)
def test_initialization_with_training_variants(
    tmp_path, clip_action, clip_state, step_limit
):
    config = yaml.safe_load(CONFIGS[0].read_text())
    config["data"]["lerobot_config"]["normalization"] = {
        "clip_action_targets": clip_action,
        "clip_state_inputs": clip_state,
    }
    config["hyperparams"]["scheduler"]["num_training_steps"] = step_limit
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"native routing probe")
    config["checkpoint"]["resume_from"] = str(weights)
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    assert audit_initialization(path)["weights"] == str(weights)


@pytest.mark.parametrize("config_path", CONFIGS)
def test_robotwin_configs_select_native_weight_load(config_path, tmp_path):
    config = yaml.safe_load(config_path.read_text())
    # The actual server checkpoint is not needed for this control-flow regression.
    # Both pretrained files and full training checkpoints are valid starting points.
    assert config["checkpoint"]["resume_from"]
    assert not config["checkpoint"].get("resume_model")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"native routing probe")
    config["checkpoint"]["resume_from"] = str(weights)
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    assert audit_initialization(path)["weights"] == str(weights)


def test_reject_ignored_resume_model(tmp_path):
    config = yaml.safe_load(CONFIGS[0].read_text())
    config["checkpoint"]["resume_model"] = config["checkpoint"]["resume_from"]
    config["checkpoint"]["resume_from"] = None
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="ignores checkpoint.resume_model"):
        audit_initialization(path)


def test_full_checkpoint_selects_training_resume(tmp_path):
    from wall_x.robotwin.config import load_config
    from wall_x.trainer.fsdp_trainer.fsdp_trainer import FSDPTrainer

    config = yaml.safe_load(CONFIGS[0].read_text())
    config["checkpoint"]["resume_from"] = str(tmp_path)
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config))
    trainer = object.__new__(FSDPTrainer)
    trainer.cfg = load_config(path)
    assert trainer._resume_from_training_checkpoint()
    assert not trainer._resume_from_single_file()


@pytest.mark.parametrize(
    "config_path", sorted((ROOT / "workspace/example").glob("robotwin*.yml"))
)
def test_public_template_keeps_paths_out_of_normalizer_keys(config_path, tmp_path):
    import torch

    from wall_x.robotwin.norm_stats import parse_train_config

    config = yaml.safe_load(config_path.read_text())
    dataset = config["data"]["lerobot_config"]
    dataset["root"] = str(tmp_path / ".cache" / "robotwin")
    assert "." not in dataset["repo_id"]
    torch.nn.ParameterDict({dataset["repo_id"]: torch.nn.Parameter(torch.zeros(1))})
    context = parse_train_config(config)
    assert context.data_root == Path(dataset["root"])
