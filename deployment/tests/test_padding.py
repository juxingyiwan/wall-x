"""Both deployment routes carry the same explicit policy and reject stale peers."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from wallx_client.config import PolicyConfig
from wallx_client.contracts import ACTION_UNITS, STATE_UNITS, ProtocolError
from wallx_client.protocol import validate_metadata

from deployment import cli
from deployment.local_runtime import model_options

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", [None, "zero", "original"])
def test_profile_reaches_local_model_server_command_and_client(tmp_path, mode):
    raw = cli.read_yaml(cli.DEFAULT_CONFIG)
    raw["client_config"] = str(cli.DEFAULT_CONFIG.parent / raw["client_config"])
    if mode is not None:
        raw["padding"] = mode
    profile = tmp_path / "profile.yml"
    profile.write_text(yaml.safe_dump(raw))
    machine = tmp_path / "machine.yml"
    machine.write_text(yaml.safe_dump({"checkpoint_root": str(tmp_path)}))
    cfg = cli.load_deployment(profile, machine)
    effective = mode or "zero"
    assert cfg["padding"] == cfg["client"]["policy"]["padding"] == effective
    assert model_options(cfg)["padding"] == effective
    cmd, _, _ = cli.server_command(cfg)
    assert cmd[cmd.index("--model-config.padding") + 1] == effective


@pytest.mark.parametrize("expected", ["zero", "original"])
@pytest.mark.parametrize("actual", [None, "zero", "original", "bad"])
def test_both_backends_share_strict_mode_handshake(expected, actual):
    cfg = PolicyConfig(padding=expected)
    manifest = dict(
        protocol_version=1,
        state_schema="joint_state_v1",
        state_units=STATE_UNITS,
        action_units=ACTION_UNITS,
        state_dim=7,
        action_dim=26,
        action_horizon=32,
        camera_names=list(cfg.camera_names),
        checkpoint_id=cfg.checkpoint_id,
        request_id_echo=True,
    )
    if actual is not None:
        manifest["padding"] = actual
    if actual == expected:
        assert validate_metadata({"wallx_client": manifest}, cfg)["padding"] == actual
    else:
        with pytest.raises(ProtocolError, match="padding mismatch"):
            validate_metadata({"wallx_client": manifest}, cfg)


def test_config_validation_does_not_import_torch(tmp_path):
    code = "from wall_x.inference_padding import validate_padding; import sys; assert validate_padding('original') == 'original'; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


@pytest.mark.parametrize("mode", ["zero", "original"])
def test_command_override_reaches_both_deployments_without_changing_profile(
    tmp_path, monkeypatch, mode
):
    raw = cli.read_yaml(cli.DEFAULT_CONFIG)
    raw["padding"] = "original" if mode == "zero" else "zero"
    raw["client_config"] = str(cli.DEFAULT_CONFIG.parent / raw["client_config"])
    profile = tmp_path / "profile.yml"
    profile.write_text(yaml.safe_dump(raw))
    before = profile.read_bytes()
    machine = tmp_path / "machine.yml"
    machine.write_text(yaml.safe_dump({"checkpoint_root": str(tmp_path)}))
    monkeypatch.setenv("WALLX_PADDING", mode)
    cfg = cli.load_deployment(profile, machine)
    assert cfg["client"]["policy"]["padding"] == model_options(cfg)["padding"] == mode
    command, _, _ = cli.server_command(cfg)
    assert command[command.index("--model-config.padding") + 1] == mode
    monkeypatch.delenv("WALLX_PADDING")
    assert cli.load_deployment(profile, machine)["padding"] == raw["padding"]
    assert profile.read_bytes() == before


@pytest.mark.parametrize("value", ["", "aligned_zero", "-1"])
def test_invalid_command_override_fails_before_model_or_hardware_access(monkeypatch, value):
    monkeypatch.setenv("WALLX_PADDING", value)
    with pytest.raises(ValueError, match="WALLX_PADDING"):
        cli.load_deployment(cli.DEFAULT_CONFIG)
