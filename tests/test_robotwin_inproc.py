"""Direct inference IO, LIBERO defaults, and chunk-driver regressions."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_native_flow_preserves_libero_io_and_prompt():
    import torch

    from wall_x._vendor.harrix.adapters.qwen_vlact import QwenVLActInferAdapter
    from wall_x._vendor.harrix.envs.libero_common import decode_chunk, encode_proprio

    adapter = object.__new__(QwenVLActInferAdapter)
    adapter._train_config = {
        "data": {},
        "dof_config": {
            "follow_right_ee_cartesian_pos": 3,
            "follow_right_ee_rotation": 3,
            "follow_right_gripper": 1,
        },
    }
    adapter._action_horizon = 4
    adapter._cam_names = ["face_view", "right_wrist_view"]
    obs = {
        "eef_pos": np.zeros(3),
        "eef_axisangle": np.zeros(3),
        "gripper": np.zeros(1),
        "face_view": np.zeros((28, 28, 3), np.uint8),
        "wrist_view": np.zeros((28, 28, 3), np.uint8),
    }
    expected = encode_proprio(obs, adapter._train_config, 4)
    actions = np.arange(28).reshape(1, 4, 7)
    captured = {}

    def make_inputs(observations, prefixes, postfixes):
        captured["observation"] = observations[0]
        return {
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "dataset_names": ["fixture"],
        }

    adapter._construct_model_input = make_inputs
    adapter._normalizer_action = None
    adapter._action_dim = 7
    adapter._num_inference_timesteps = 10
    adapter._model = SimpleNamespace(
        generate_flow_action=lambda **kwargs: {"predict_action": actions}
    )
    output = adapter._flow_batch([{"observation": obs, "instruction": "test"}])
    for key in expected:
        np.testing.assert_array_equal(captured["observation"][key], expected[key])
    np.testing.assert_array_equal(
        output[0], decode_chunk(actions, adapter._train_config)
    )
    prompt, suffix = adapter._get_flow_prompt("test")
    assert "front view:" in prompt and "right wrist view:" in prompt
    assert suffix == "<|action|>" * 4


def robotwin_config():
    state_layout = {
        "follow_left_ee_cartesian_pos": 3,
        "follow_left_ee_rotation_6D": 6,
        "follow_left_gripper": 1,
        "follow_right_ee_cartesian_pos": 3,
        "follow_right_ee_rotation_6D": 6,
        "follow_right_gripper": 1,
        "action_padding": 6,
    }
    action_layout = {
        key + ("_relative" if "cartesian" in key or "rotation" in key else ""): value
        for key, value in state_layout.items()
    }
    return {
        "data": {
            "max_length": 1024,
            "lerobot_config": {"adapter": "robotwin", "pad_to_square": True},
        },
        "agent_pos_config": state_layout,
        "dof_config": action_layout,
    }


def test_robotwin_state_images_and_prompt_match_training():
    from wall_x.robotwin.adapter import RobotwinInferAdapter

    config = robotwin_config()
    adapter = object.__new__(RobotwinInferAdapter)
    adapter._train_config = config
    adapter._action_horizon = 32
    adapter._cam_names = ["face_view", "left_wrist_view", "right_wrist_view"]
    adapter._device = "cpu"
    adapter._data_config = SimpleNamespace(
        resolution={key: 224 for key in adapter._cam_names},
        image_factor=28,
        min_pixels=28 * 28 * 4,
        max_pixels=1280 * 28 * 28,
    )
    state = np.arange(20, dtype=np.float32) / 20
    image = np.random.default_rng(8).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    payload = {"state": state, **{cam: image for cam in adapter._cam_names}}
    encoded = adapter._encode_observation(payload)
    np.testing.assert_array_equal(encoded["proprioception"][0, 0, :20], state)
    assert np.all(encoded["proprioception"][..., 20:] == 0)
    assert np.all(encoded["agent_pos_mask"][..., 20:] == 0)
    assert encoded["dof_mask"].shape == (1, 32, 26)
    assert np.all(encoded["dof_mask"][..., :20] == 1)
    assert np.all(encoded["dof_mask"][..., 20:] == 0)
    actual = adapter._resize_image(image, "face_view")
    assert actual.shape == (224, 224, 3)
    prompt, suffix = adapter._get_flow_prompt("Open the microwave.")
    assert " face_view:" in prompt and " front view:" not in prompt
    assert suffix.count("<|action|>") == 32
    assert adapter._input_max_length() == 1024
    actions = np.zeros((1, 32, 26), dtype=np.float32)
    np.testing.assert_array_equal(adapter._decode_action_chunk(actions), actions[0])
    actions[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid RoboTwin"):
        adapter._decode_action_chunk(actions)


class FakeAdapter:
    batches = []

    def __init__(self, cfg):
        import torch

        torch.rand(3)

    def predict_batch(self, payloads):
        import torch

        self.batches.append(len(payloads))
        return [torch.rand(32, 26).numpy() for _ in payloads]


class FakeWorker:
    reject_all = False

    def __init__(self, config, **unused):
        self.config = config
        self.result = None
        self.closed = False

    def submit(self, command, value=None):
        if command == "check_scene":
            self.result = {
                "valid": value["seed"] != 1 and not self.reject_all,
                "instruction": "fixed",
            }
        elif command == "reset_episode":
            self.scene = value
            self.executed = []
            self.result = {"obs": {}, "success": False}
        elif command == "execute_chunk":
            self.executed.append(np.asarray(value).tolist())
            self.result = {
                "obs": {},
                "done": True,
                "success": self.scene["seed"] % 2 == 0,
                "steps": len(value),
            }
        elif command == "finalize_episode":
            path = Path(self.config["output"]) / self.scene["video"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.executed))
            self.result = None

    def receive(self):
        return self.result

    def call(self, command, value=None):
        self.submit(command, value)
        return self.receive()

    def shutdown(self):
        self.closed = True


def args_for(tmp_path, name):
    from wall_x.robotwin.inproc import parse_args

    return parse_args(
        [
            "--checkpoint-path",
            "unused",
            "--robotwin-root",
            str(tmp_path),
            "--output",
            str(tmp_path / name),
            "--tasks",
            "fixture",
            "--episodes",
            "5",
            "--start-seed",
            "0",
        ]
    )


def test_direct_driver_resume_retains_failures_and_rng(tmp_path):
    from wall_x.robotwin.inproc import run

    cfg = SimpleNamespace(
        model=SimpleNamespace(action_horizon=32, norm_key="robotwin2_full27500")
    )
    identity = {"checkpoint": "a"}
    baseline = args_for(tmp_path, "baseline")
    run(baseline, cfg, identity, adapter_factory=FakeAdapter, worker_factory=FakeWorker)
    resumed = args_for(tmp_path, "resumed")
    resumed.scene_manifest = tmp_path / "panel.json"
    resumed.episodes = 2
    first = run(
        resumed, cfg, identity, adapter_factory=FakeAdapter, worker_factory=FakeWorker
    )
    first_rows = list(first["tasks"]["fixture"]["episodes"])
    resumed.episodes, resumed.resume = 5, True
    report = run(
        resumed, cfg, identity, adapter_factory=FakeAdapter, worker_factory=FakeWorker
    )
    rows = report["tasks"]["fixture"]["episodes"]
    assert rows[:2] == first_rows
    assert [row["seed"] for row in rows] == [0, 2, 3, 4, 5]
    assert [row["success"] for row in rows] == [True, True, False, True, False]
    assert report["protocol"]["action_horizon"] == 32
    assert report["protocol"]["execute_steps"] == 16
    assert all(row["steps"] == 16 for row in rows)
    for row in rows:
        assert (baseline.output / row["video"]).read_bytes() == (
            resumed.output / row["video"]
        ).read_bytes()
    assert report["overall"]["success_rate"] == 0.6
    with pytest.raises(ValueError, match="checkpoint or evaluation protocol"):
        run(
            resumed,
            cfg,
            {"checkpoint": "b"},
            adapter_factory=FakeAdapter,
            worker_factory=FakeWorker,
        )
    resumed.instruction_type = "unseen"
    with pytest.raises(ValueError, match="checkpoint or evaluation protocol"):
        run(
            resumed,
            cfg,
            identity,
            adapter_factory=FakeAdapter,
            worker_factory=FakeWorker,
        )


def test_batch_driver_and_explicit_execution_length(tmp_path):
    from wall_x.robotwin.inproc import run

    args = args_for(tmp_path, "batch")
    args.num_workers, args.execute_steps, args.episodes = 2, 16, 3
    cfg = SimpleNamespace(model=SimpleNamespace(action_horizon=32, norm_key="test"))
    FakeAdapter.batches = []
    report = run(args, cfg, {}, adapter_factory=FakeAdapter, worker_factory=FakeWorker)
    assert FakeAdapter.batches == [2, 1]
    assert all(row["steps"] == 16 for row in report["tasks"]["fixture"]["episodes"])


def test_default_16_step_protocol_cannot_resume_old_32_step_results(tmp_path):
    from wall_x.robotwin.inproc import run

    args = args_for(tmp_path, "old-32")
    cfg = SimpleNamespace(model=SimpleNamespace(action_horizon=32, norm_key="test"))
    args.execute_steps = 32
    run(args, cfg, {}, adapter_factory=FakeAdapter, worker_factory=FakeWorker)
    saved = (args.output / "results.json").read_bytes()
    args = args_for(tmp_path, "old-32")
    assert args.execute_steps == 16
    args.resume = True
    with pytest.raises(ValueError, match="checkpoint or evaluation protocol"):
        run(args, cfg, {}, adapter_factory=FakeAdapter, worker_factory=FakeWorker)
    assert (args.output / "results.json").read_bytes() == saved


def test_direct_driver_seed_controls_model_initialization(tmp_path):
    import random

    import torch

    from wall_x.robotwin.inproc import run

    samples = []

    class StopBeforeModelLoad(Exception):
        pass

    def capture(config):
        samples.append((random.random(), float(np.random.rand()), torch.rand(1).item()))
        raise StopBeforeModelLoad

    args = args_for(tmp_path, "seed-check")
    cfg = SimpleNamespace(model=SimpleNamespace(action_horizon=32, norm_key="test"))
    for seed in (42, 42, 43):
        args.seed = seed
        with pytest.raises(StopBeforeModelLoad):
            run(args, cfg, {}, adapter_factory=capture)
    assert samples[0] == samples[1]
    assert samples[0] != samples[2]


def test_32_predictions_execute_16_then_refresh_observation(tmp_path):
    from wall_x.robotwin.inproc import run_frame

    class Worker(FakeWorker):
        def submit(self, command, value=None):
            super().submit(command, value)
            if command == "reset_episode":
                self.result["obs"] = {"observed_step": 0}
            elif command == "execute_chunk":
                step = sum(len(chunk) for chunk in self.executed)
                self.result.update(
                    obs={"observed_step": step}, done=step >= 32, success=step >= 32
                )

    observed_steps = []

    def predict(payloads):
        observed_steps.append(payloads[0]["observation"]["observed_step"])
        return [np.repeat(np.arange(32)[:, None], 26, axis=1)]

    worker = Worker({"output": str(tmp_path)})
    scene = dict(task="fixture", seed=0, instruction="fixed", video="rollout.mp4")
    rows = run_frame(SimpleNamespace(predict_batch=predict), [worker], [scene], 16)
    assert observed_steps == [0, 16]
    assert rows[0]["steps"] == 32 and rows[0]["chunks"] == 2
    assert [np.asarray(chunk)[:, 0].tolist() for chunk in worker.executed] == [
        list(range(16))
    ] * 2


@pytest.mark.parametrize("separate_python", [False, True])
def test_real_pipe_worker_uses_no_torch_or_network(
    tmp_path, monkeypatch, separate_python
):
    import os

    from wall_x.robotwin.env import SubprocEnvHandle

    (tmp_path / "fixture_env.py").write_text(
        """import os, sys
class Environment:
    def __init__(self, config): pass
    def check_scene(self, request):
        if request.get("fail"): raise RuntimeError("fixture failure")
        return {"pid": os.getpid(), "torch": "torch" in sys.modules, "prefix": sys.prefix}
    def shutdown(self): pass
"""
    )
    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    python = os.environ.get("ROBOTWIN_TEST_SIM_PYTHON") if separate_python else None
    if separate_python and not python:
        pytest.skip(
            "Set ROBOTWIN_TEST_SIM_PYTHON to test a separate simulator environment"
        )
    worker = SubprocEnvHandle(
        {}, python=python, env_factory="fixture_env:Environment", timeout=30
    )
    try:
        result = worker.call("check_scene", {})
        assert result["pid"] != os.getpid()
        assert result["torch"] is False
        if python:
            assert result["prefix"] == str(Path(python).parent.parent)
        with pytest.raises(RuntimeError, match="fixture failure"):
            worker.call("check_scene", {"fail": True})
    finally:
        worker.shutdown()
    assert worker.process.poll() is not None


def test_robotwin_uses_native_model_forward_without_second_unnormalization():
    import torch

    from wall_x.robotwin.adapter import RobotwinInferAdapter

    adapter = object.__new__(RobotwinInferAdapter)
    adapter._train_config = robotwin_config()
    adapter._action_mode = "flow"
    adapter._action_horizon, adapter._action_dim, adapter._num_inference_timesteps = (
        32,
        26,
        10,
    )
    adapter._cam_names = ["face_view", "left_wrist_view", "right_wrist_view"]
    adapter._normalizer_action = None
    captured = {}

    def inputs(observations, prefixes, postfixes):
        captured["observations"] = observations
        return {
            "input_ids": torch.zeros((2, 1), dtype=torch.int64),
            "dataset_names": ["fixture"] * 2,
        }

    def forward(**kwargs):
        captured.update(kwargs)
        return {"predict_action": torch.full((2, 32, 26), 0.25)}

    adapter._construct_model_input = inputs
    adapter._model = SimpleNamespace(generate_flow_action=forward)
    obs = {
        "state": np.zeros(20),
        **{cam: np.zeros((28, 28, 3), dtype=np.uint8) for cam in adapter._cam_names},
    }
    output = adapter.predict_batch([{"observation": obs, "instruction": "test"}] * 2)
    assert len(output) == 2 and all(np.all(chunk == 0.25) for chunk in output)
    assert captured["num_inference_timesteps"] == 10
    assert captured["noise"] is None
    assert captured["padding_action"].shape == (2, 1, 26)


def test_simulator_executes_chunk_anchor_once_and_stops_on_success():
    from wall_x.robotwin.env import RobotwinEnv

    commands = []

    class Simulator:
        take_action_cnt, step_lim, eval_success = 0, 20, False

        def get_obs(self):
            return {
                "observation": {
                    "head_camera": {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}
                }
            }

        def take_action(self, action, action_type):
            assert action_type == "ee"
            commands.append(action.copy())
            self.take_action_cnt += 1
            self.eval_success = self.take_action_cnt == 3

    env = RobotwinEnv({})
    env.env = Simulator()
    env.video = SimpleNamespace(append_data=lambda image: None)
    env.anchor = np.array([2, 0, 0, 1, 0, 0, 0, 1, 0, 1] * 2, dtype=np.float32)
    actions = np.zeros((32, 26))
    for offset in (0, 10):
        actions[:, offset] = np.arange(32) * 0.01
        actions[:, offset + 3] = 1
        actions[:, offset + 7] = 1
    result = env.execute_chunk(actions)
    assert result == {"obs": None, "done": True, "success": True, "steps": 3}
    np.testing.assert_allclose(np.array(commands)[:, 0], [2, 2.01, 2.02])


def test_prevalidated_panel_does_not_repeat_stochastic_expert(tmp_path):
    from wall_x.robotwin.inproc import run
    from wall_x.robotwin.panel import observation_digest

    obs = {
        "state": np.zeros(20, np.float32),
        **{
            k: np.zeros((2, 2, 3), np.uint8)
            for k in ("face_view", "left_wrist_view", "right_wrist_view")
        },
    }

    class FrozenWorker(FakeWorker):
        changed = False

        def submit(self, command, value=None):
            if command == "check_scene":
                raise AssertionError(
                    "Expert should not run on an already validated panel"
                )
            super().submit(command, value)
            if command == "reset_episode":
                self.result["obs"] = {k: v.copy() for k, v in obs.items()}
                if self.changed:
                    self.result["obs"]["face_view"][0, 0, 0] = 1

    args = args_for(tmp_path, "frozen")
    args.episodes = 1
    args.scene_manifest = tmp_path / "frozen.json"
    row = dict(
        seed=0,
        instruction="fixed",
        expert_validated=True,
        initial_observation_sha256=observation_digest(obs),
    )
    args.scene_manifest.write_text(
        json.dumps(
            dict(
                task="fixture",
                task_config=args.task_config,
                start_seed=0,
                instruction_type=args.instruction_type,
                instruction_count=args.instruction_count,
                scenes=[row],
            )
        )
    )
    cfg = SimpleNamespace(model=SimpleNamespace(action_horizon=32, norm_key="test"))
    report = run(
        args, cfg, {}, adapter_factory=FakeAdapter, worker_factory=FrozenWorker
    )
    assert (
        report["tasks"]["fixture"]["episodes"][0]["initial_observation_sha256"]
        == row["initial_observation_sha256"]
    )
    args.resume = True
    run(args, cfg, {}, adapter_factory=FakeAdapter, worker_factory=FrozenWorker)
    args.resume, args.output = False, tmp_path / "changed"
    FrozenWorker.changed = True
    with pytest.raises(RuntimeError, match="Initial observation changed"):
        run(args, cfg, {}, adapter_factory=FakeAdapter, worker_factory=FrozenWorker)


def test_incomplete_panel_proof_is_rejected():
    from wall_x.robotwin.panel import is_prevalidated_scene

    for row in [
        {"expert_validated": True},
        {"initial_observation_sha256": "a" * 64},
        {"expert_validated": False, "initial_observation_sha256": "a" * 64},
    ]:
        with pytest.raises(ValueError, match="Incomplete expert validation proof"):
            is_prevalidated_scene(row)


@pytest.mark.parametrize("explicit_file", [False, True])
def test_checkpoint_identity_preserves_autofilled_weight_selection(
    tmp_path, explicit_file, capsys
):
    from wall_x._vendor.harrix.eval_config import EvalConfig, autofill_from_checkpoint
    from wall_x.robotwin.inproc import checkpoint_identity

    native = tmp_path / "pytorch_model_fsdp.bin"
    exported = tmp_path / "model.safetensors"
    native.write_bytes(b"native")
    exported.write_bytes(b"exported")
    import yaml

    config = robotwin_config()
    config["model_type"] = "qwen2_5"
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    stats = {"q01": [-1.0] * 26, "q99": [1.0] * 26}
    (tmp_path / "norm_stats.json").write_text(
        json.dumps({"norm_stats": {"action": stats, "observation.state": stats}})
    )
    cfg = EvalConfig()
    cfg.model.checkpoint_path = str(exported if explicit_file else tmp_path)
    cfg = autofill_from_checkpoint(cfg)
    identity = checkpoint_identity(
        cfg.model.checkpoint_path, cfg.model.train_config_path
    )
    assert identity["directory"] == str(tmp_path)
    assert identity["weights"] == (exported if explicit_file else native).name
    assert identity["weights_size"] == (8 if explicit_file else 6)
    assert "norm_stats.json" in identity["normalization_sha256"]

    # Exercise the public CLI preflight too: the weights remain a selected file,
    # while config overlays and normalizers are read from its parent directory.
    from wall_x.robotwin.inproc import main

    main(
        [
            "--checkpoint-path",
            str(exported if explicit_file else tmp_path),
            "--robotwin-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
            "--norm-key",
            "fixture",
            "--dry-run",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert report["checkpoint"] == identity
