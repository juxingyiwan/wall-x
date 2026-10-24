"""Regressions at the new configuration/process boundary; no robot or GPU use."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("unified_entry", ROOT / "deployment/cli.py")
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


@pytest.fixture
def machine(tmp_path):
    previous = entry.read_yaml(ROOT / "deployment/client/src/wallx_client/configs/b601.example.yml")
    return write_yaml(
        tmp_path / "machine.yml",
        {
            "checkpoint_root": str(tmp_path / "weights"),
            "client": {"robot": previous["robot"]},
        },
    )


def invoke(args, cwd):
    return subprocess.run(
        [sys.executable, str(ROOT / "deployment/cli.py"), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_effective_control_matches_packaged_example(machine, tmp_path):
    # Compare every default-expanded field, not a duplicate of individual constants.
    sys.path.insert(0, str(ROOT / "deployment/client/src"))
    from wallx_client.config import load_config

    previous = load_config(
        ROOT / "deployment/client/src/wallx_client/configs/b601.example.yml"
    ).to_dict()
    unified = entry.load_deployment(machine=machine)
    current = load_config(write_yaml(tmp_path / "effective.yml", unified["client"])).to_dict()
    previous["output_root"] = current["output_root"]
    assert current == previous
    assert "wallx-unified" in current["output_root"]


def test_client_uses_new_source_from_unrelated_cwd(machine, tmp_path):
    code = f"import runpy,sys; sys.argv=['wallx','--machine',{str(machine)!r},'client','doctor']; "
    code += f"\ntry: runpy.run_path({str(ROOT / 'deployment/cli.py')!r},run_name='__main__')\nexcept SystemExit as e: assert e.code == 0\n"
    code += f"assert sys.modules['wallx_client'].__file__.startswith({str(ROOT / 'deployment/client/src')!r})\n"
    code += "assert all(x not in sys.modules for x in ['torch','motorbridge','lerobot_robot_seeed_b601','can'])\n"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["hardware_opened"] is False


def test_native_dry_run_uses_shared_identity_and_assigned_gpu(machine, tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("SERIALIZE_ACTIONS", "1")
    monkeypatch.setenv("ENABLE_FAST_PREPROCESS", "true")
    monkeypatch.setenv("WALLX_VISION_ATTN_IMPLEMENTATION", "eager")
    monkeypatch.setenv("WALLX_SMOOTH_GRIPPER", "false")
    monkeypatch.setenv("ENABLE_EXPERIMENTAL_INFERENCE_ENGINE", "true")
    deployment = entry.load_deployment(machine=machine)
    command, env, values = entry.server_command(deployment)
    command.insert(2, "--dry-run")
    result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "12_16000" in result.stdout
    assert "--no-serialize-actions" in result.stdout
    assert "--model-config.norm-key" in result.stdout
    assert deployment["model"]["norm_key"] in result.stdout
    assert values["CUDA_ID"] == "3"
    assert values["ENABLE_FAST_PREPROCESS"] == "false"
    assert values["PYTHON_BIN"] == sys.executable
    assert env["PYTHONPATH"] == str(ROOT)
    # Exercise the inherited environment that reaches the native process, and
    # ensure the dry-run report exposes the same effective inference settings.
    for key, expected in {
        "WALLX_VISION_ATTN_IMPLEMENTATION": "flash_attention_2",
        "WALLX_SMOOTH_GRIPPER": "true",
        "ENABLE_EXPERIMENTAL_INFERENCE_ENGINE": "false",
    }.items():
        assert env[key] == values[key] == expected


@pytest.mark.parametrize("visible", [None, "", " "])
@pytest.mark.parametrize("override", [None, "0"])
def test_missing_slurm_gpu_explains_node_environment_and_cannot_be_overridden(
    machine, monkeypatch, visible, override
):
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    if override is not None:
        raw = entry.read_yaml(machine)
        raw["cuda_id"] = override
        write_yaml(machine, raw)
    with pytest.raises(ValueError, match="SLURM_JOB_ID.*CUDA_VISIBLE_DEVICES.*assigned"):
        entry.load_deployment(machine=machine)


def test_wrong_slurm_gpu_is_rejected(machine, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    raw = entry.read_yaml(machine)
    raw["cuda_id"] = "0"
    write_yaml(machine, raw)
    with pytest.raises(ValueError, match="Slurm"):
        entry.load_deployment(machine=machine)


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_id", "../12_16000"),
        ("epoch", 13),
        ("checkpoint_sha256", "bad"),
        ("horizon", 16),
        ("image_passing_mode", "numpy"),
        ("action_dims", [7, 19]),
    ],
)
def test_inconsistent_model_profile_is_rejected(tmp_path, field, value):
    raw = entry.read_yaml(entry.DEFAULT_CONFIG)
    raw["model"][field] = value
    with pytest.raises(ValueError):
        entry.load_deployment(write_yaml(tmp_path / "bad.yml", raw))


def test_machine_cannot_override_model(machine):
    raw = entry.read_yaml(machine)
    raw["client"]["policy"] = {"checkpoint_id": "other/8_12000"}
    write_yaml(machine, raw)
    with pytest.raises(ValueError, match="Unknown"):
        entry.load_deployment(machine=machine)


def zero_epoch_profile(tmp_path):
    raw = entry.read_yaml(entry.DEFAULT_CONFIG)
    raw["model"].update(epoch=0, step=2, checkpoint_id="experiment/0_2")
    raw["client_config"] = str(entry.DEFAULT_CONFIG.parent / raw["client_config"])
    return write_yaml(tmp_path / "profile.yml", raw)


def test_zero_epoch_checkpoint_reaches_server_and_client(machine, tmp_path):
    profile = zero_epoch_profile(tmp_path)
    deployment = entry.load_deployment(profile, machine)
    assert deployment["model"]["epoch"] == 0
    assert deployment["client"]["policy"]["checkpoint_id"] == "experiment/0_2"
    assert Path(deployment["checkpoint_path"]) == tmp_path / "weights/experiment/0_2"
    command, env, _ = entry.server_command(deployment)
    command.insert(2, "--dry-run")
    result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert deployment["checkpoint_path"] in result.stdout


@pytest.mark.parametrize("epoch", [-1, True, False, 0.0, 1.5, "0", None])
def test_epoch_requires_a_nonnegative_integer(tmp_path, epoch):
    profile = zero_epoch_profile(tmp_path)
    raw = entry.read_yaml(profile)
    raw["model"]["epoch"] = epoch
    with pytest.raises(ValueError, match="model.epoch must be a nonnegative integer"):
        entry.load_deployment(write_yaml(profile, raw))


@pytest.mark.parametrize("field", ["step", "horizon"])
@pytest.mark.parametrize("value", [0, True])
def test_other_model_counters_still_require_positive_integers(tmp_path, field, value):
    profile = zero_epoch_profile(tmp_path)
    raw = entry.read_yaml(profile)
    raw["model"][field] = value
    with pytest.raises(ValueError, match=rf"model\.{field} must be a positive integer"):
        entry.load_deployment(write_yaml(profile, raw))


def test_zero_epoch_still_requires_matching_checkpoint_identity(tmp_path):
    profile = zero_epoch_profile(tmp_path)
    raw = entry.read_yaml(profile)
    raw["model"]["checkpoint_id"] = "experiment/1_2"
    with pytest.raises(ValueError, match="checkpoint_id disagrees with epoch/step"):
        entry.load_deployment(write_yaml(profile, raw))


@pytest.mark.parametrize("flag", ["--config", "--config=other.yml", "--conf=other.yml"])
@pytest.mark.parametrize("mode", ["client", "local"])
def test_client_cannot_bypass_shared_config(tmp_path, flag, mode):
    prefix = ["local", "--"] if mode == "local" else ["client"]
    result = invoke([*prefix, flag, "doctor"], tmp_path)
    assert result.returncode == 2
    assert "overrides are disabled" in result.stderr


def test_local_run_dry_run_is_hardware_free_without_a_terminal(machine, tmp_path):
    result = invoke(
        [
            "--config",
            str(entry.DEFAULT_CONFIG),
            "--machine",
            str(machine),
            "local",
            "--dry-run",
            "--",
            "run",
        ],
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["hardware_opened"] is False
    assert report["weights_verified"] is False
    assert report["backend"] == "in_process"
    assert report["inference_port"] is None
    assert report["client"]["policy"]["uri"] == "local://wall-x"
    assert "server_command" not in report


def test_missing_machine_rejects_client_before_hardware(tmp_path):
    result = invoke(["client", "run"], tmp_path)
    assert result.returncode == 2
    assert "requires --machine" in result.stderr


def test_local_missing_machine_fails_before_model_or_hardware_import(tmp_path):
    code = "import runpy,sys; sys.argv=['wallx','--machine','missing.yml','local'];\n"
    code += f"try: runpy.run_path({str(ROOT / 'deployment/cli.py')!r},run_name='__main__')\nexcept SystemExit as e: assert e.code == 2\n"
    code += "assert all(x not in sys.modules for x in ['wallx_client','wall_x','torch','motorbridge','can'])\n"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "missing.yml" in result.stderr


def test_checkpoint_digest_is_enforced(machine):
    deployment = entry.load_deployment(machine=machine)
    deployment["model"]["checkpoint_sha256"] = "a" * 64
    checkpoint = Path(deployment["checkpoint_path"])
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.yml").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"wrong weights")
    with pytest.raises(ValueError, match="SHA256"):
        entry.verify_checkpoint(deployment)


def test_shell_entry_is_cwd_independent(tmp_path):
    env = dict(os.environ, WALLX_PYTHON=sys.executable)
    result = subprocess.run(
        ["bash", str(ROOT / "wallx"), "config"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    profile = entry.LOCAL_CONFIG if entry.LOCAL_CONFIG.exists() else entry.DEFAULT_CONFIG
    assert json.loads(result.stdout)["model"] == entry.read_yaml(profile)["model"]


def test_stock_server_rejected_before_model_import(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="joint_state_v1 prerequisite"):
        entry.verify_server_compatibility()


def test_placeholder_checkpoint_rejected_before_filesystem_access():
    deployment = entry.load_deployment()
    with pytest.raises(ValueError, match="Replace the example"):
        entry.verify_checkpoint(deployment)
