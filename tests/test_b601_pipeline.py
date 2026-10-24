"""Physical-unit regression tests across real training and serving entry points."""

import copy
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wall_x.data.b601 import (
    ABSOLUTE,
    DELTA,
    LAYOUT,
    SCALE,
    B601Codec,
    absolute_window,
    configure_normalizers,
    fit_statistics,
    read_split,
    save_manifest,
    settings,
    validate_checkpoint,
    validate_statistics,
)
from wall_x.data.b601 import load_b601_norm_stats as load_norm_stats
from wall_x.data.b601_prepare import make_stats
from wall_x.data.b601_split import stratified_split
from wall_x.model.core.action.affine import scale, unscale
from wall_x.model.core.action.normalizer import (
    create_normalizers_from_lerobot_norm_stats,
)


def config(representation=DELTA, split_path="split.json"):
    return {
        "task": {
            "dof_config": dict(LAYOUT),
            "agent_pos_config": dict(LAYOUT),
            "action_horizon": 32,
            "use_state_string_representation": False,
        },
        "data": {
            "lerobot_config": {
                "repo_id": "fixture",
                "b601": {
                    "representation": representation,
                    "target_policy": "recorded_commands_v1",
                    "command_to_physical_scale": SCALE,
                    "fps": 30,
                    "clip_action_targets": False,
                    "clip_state_inputs": False,
                    "split_manifest": str(split_path),
                },
            },
            "key_mappings": {
                "state": "observation.state",
                "action": "action",
                "camera": {
                    "observation.images.front": "front_view",
                    "observation.images.side": "side_view",
                },
            },
            "resolution": {"front_view": 256, "side_view": 256},
        },
    }


def episodes():
    result = {}
    for ep in range(6):
        actions = np.tile(
            np.array([100, 80, -90, -30, -20, 10, 25], np.float32), (4, 1)
        )
        actions[:, :6] += np.arange(4, dtype=np.float32)[:, None]
        state = (actions - np.array([1, 2, -3, -4, -5, 6, 0], np.float32)) * SCALE
        result[ep] = {
            "action": actions,
            "observation.state": state.astype(np.float32),
            "task_index": np.full(4, ep // 2),
        }
    return result


def artifact(tmp_path, representation=DELTA):
    ep = episodes()
    split = stratified_split(ep)
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split))
    cfg = config(representation, path)
    stats = make_stats(cfg, ep, split, [])
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(stats))
    cfg["data"]["norm_stats_path"] = str(path)
    return cfg, stats, ep, split


def test_one_validation_episode_per_task_is_reproducible_and_disjoint():
    groups = {
        episode: {"task_index": np.array([task])}
        for task, ids in enumerate((range(50), range(50, 150), range(150, 200)))
        for episode in ids
    }
    split = stratified_split(groups, validation_episodes_per_task=1)
    assert split == stratified_split(groups, validation_episodes_per_task=1)
    assert len(split["train_episodes"]) == 197
    assert len(split["validation_episodes"]) == 3
    assert set(split["train_episodes"]).isdisjoint(split["validation_episodes"])
    assert set(split["train_episodes"] + split["validation_episodes"]) == set(groups)
    for task in range(3):
        assert (
            sum(
                int(groups[e]["task_index"][0]) == task
                for e in split["validation_episodes"]
            )
            == 1
        )


@pytest.mark.parametrize("count", [0, -1, 0.01, True, 2])
def test_invalid_per_task_validation_count_is_rejected(count):
    with pytest.raises(ValueError):
        stratified_split(episodes(), validation_episodes_per_task=count)


class IntegerSubclass(int):
    pass


@pytest.mark.parametrize("value", [True, False, 1.0, IntegerSubclass(1)])
def test_episode_indices_require_exact_integers(value):
    with pytest.raises(ValueError, match="integer episode lists"):
        read_split({"train_episodes": [value], "validation_episodes": []})


@pytest.mark.parametrize("representation", [ABSOLUTE, DELTA])
def test_codec_physical_roundtrip_and_snapshot(representation):
    codec = B601Codec(representation)
    state = np.array([100, 80, 90, 30, 20, 10, 150], np.float32)
    anchor = codec.anchor_from_state(state)
    raw = np.tile([101, 82, -93, -34, -25, 16, 30], (32, 1)).astype(np.float32)
    encoded = codec.encode_absolute_commands(raw, anchor)
    if representation == DELTA:
        np.testing.assert_array_equal(encoded[0], [1, 2, -3, -4, -5, 6, 30])
    state[:] = 999  # A later feedback sample must not move the saved anchor.
    np.testing.assert_array_equal(codec.decode_model_commands(encoded, anchor), raw)
    assert not anchor.flags.writeable
    padded = torch.nn.functional.pad(torch.from_numpy(encoded)[None], (0, 19))
    decoded = codec.decode_model_commands(padded, anchor)
    np.testing.assert_array_equal(decoded[0, :, :7], raw)
    assert decoded[..., 7:].count_nonzero() == 0


def test_rtc_rebase_preserves_gripper_and_absolute_path():
    codec = B601Codec(DELTA)
    raw = np.tile([105, 80, -90, -30, -20, 10, 25], (20, 1)).astype(np.float32)
    for pos, expected in [(100, 5), (104, 1), (108, -3)]:
        anchor = codec.anchor_from_state([pos, 80, 90, 30, 20, 10, 150])
        prefix = codec.encode_absolute_commands(raw, anchor)
        assert prefix[0, 0] == expected and prefix[0, 6] == 25
        np.testing.assert_array_equal(codec.decode_model_commands(prefix, anchor), raw)


def test_affine_outliers_and_degenerate_dimensions():
    values = np.zeros((1000, 7), np.float32)
    values[-1, 0] = 100  # q01=q99=0, but not a constant dimension.
    values[:, 1] = 7
    stats = fit_statistics(values)
    assert stats["lower"][0] == 0 and stats["span"][0] == 100
    assert stats["lower"][1] == 7 and stats["span"][1] == 1
    x = torch.tensor([0.0, 10.0, 20.0, 30.0, 35.0])
    z = scale(x, 10, 20, clip=False)
    torch.testing.assert_close(z, torch.tensor([-2.0, -1.0, 0.0, 1.0, 1.5]))
    torch.testing.assert_close(unscale(z, 10, 20), x)
    y = torch.from_numpy(values)
    torch.testing.assert_close(
        unscale(
            scale(y, stats["lower"], stats["span"], clip=False),
            stats["lower"],
            stats["span"],
        ),
        y,
    )


def test_train_only_whole_chunk_statistics_and_manifest(tmp_path):
    cfg, stats, ep, split = artifact(tmp_path)
    train_count = sum(len(ep[i]["action"]) for i in split["train_episodes"])
    assert stats["norm_stats"]["action"]["count"] == train_count * 32
    assert stats["norm_stats"]["observation.state"]["count"] == train_count
    assert set(split["train_episodes"]).isdisjoint(split["validation_episodes"])
    for i in split["validation_episodes"]:
        ep[i]["action"][:] = 1e6
    assert make_stats(cfg, ep, split, []) == stats
    validate_statistics(
        cfg, tmp_path / "norm_stats.json", split_path=tmp_path / "split.json"
    )
    save_manifest(cfg, tmp_path)
    validate_checkpoint(cfg, tmp_path)
    wrong = copy.deepcopy(cfg)
    wrong["data"]["lerobot_config"]["b601"]["representation"] = ABSOLUTE
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_checkpoint(wrong, tmp_path)
    stats["norm_stats"]["action"]["lower"][0] += 1
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="checksum"):
        validate_checkpoint(cfg, tmp_path)


def test_stats_normalizers_collator_and_serving_parity(tmp_path, monkeypatch):
    from wall_x.data.b601 import build_serving_normalizers as build_normalizers
    from wall_x.data.backends.lerobot.b601_adapter import (
        B601DataCollator as DataCollator,
    )

    cfg, _, ep, split = artifact(tmp_path)
    save_manifest(cfg, tmp_path)
    stats = load_norm_stats(tmp_path / "norm_stats.json", cfg["data"]["key_mappings"])
    action, state = create_normalizers_from_lerobot_norm_stats(stats, "fixture", 26, 26)
    configure_normalizers(action, state, cfg)
    serve_a, serve_s, _ = build_normalizers(str(tmp_path), cfg, "fixture")
    monkeypatch.setattr(DataCollator, "load_processor", lambda self: None)
    collator = DataCollator(cfg, cfg["data"], stats, cfg["data"]["lerobot_config"])
    another = DataCollator(cfg, cfg["data"], stats, cfg["data"]["lerobot_config"])
    torch.testing.assert_close(
        collator.sample_time(16, "cpu", torch.float32),
        another.sample_time(16, "cpu", torch.float32),
    )
    raw = torch.from_numpy(ep[split["train_episodes"][0]]["observation.state"])[None]
    raw[:, :, 0] += 100  # Deliberately outside quantiles.
    expected = collator._normalize_field(
        raw, stats["state"].min, stats["state"].delta, "clip_state_inputs"
    )
    padded = torch.nn.functional.pad(raw, (0, 19))
    torch.testing.assert_close(
        serve_s.normalize_data(padded, ["fixture"])[..., :7], expected
    )
    torch.testing.assert_close(
        state.normalize_data(padded, ["fixture"])[..., :7], expected
    )
    labels = torch.randn(1, 32, 7) * 100
    labels = torch.nn.functional.pad(labels, (0, 19))
    normalized = serve_a.normalize_data(labels, ["fixture"])
    torch.testing.assert_close(
        serve_a.unnormalize_data(normalized, ["fixture"]), labels, atol=1e-4, rtol=0
    )


def serving_wrapper(cfg, tmp_path, monkeypatch):
    from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import (
        WallxModelWrapper,
    )
    from wall_x._vendor.harrix.serving._wallx_infer.robot import (
        DesktopRobotPreprocessor,
    )
    from wall_x.data.b601 import build_serving_normalizers as build_normalizers

    save_manifest(cfg, tmp_path)
    config_obj = SimpleNamespace(
        train_config={**cfg, "dof_config": LAYOUT, "agent_pos_config": LAYOUT},
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
    wrapper.config, wrapper.norm_key, wrapper.logger = (
        config_obj,
        "fixture",
        logging.getLogger("b601_test"),
    )
    wrapper.normalizer_action, wrapper.normalizer_propri, _ = build_normalizers(
        str(tmp_path), cfg, "fixture"
    )
    wrapper.get_text_for_action = lambda _: ("prefix", "postfix")
    wrapper._resize_images = lambda _: []
    module = "wall_x._vendor.harrix.serving._wallx_infer.model_wrapper"
    monkeypatch.setattr(
        module + ".preprocesser_call",
        lambda *a, **k: {"input_ids": torch.tensor([[7]] * len(k["prefix_text"]))},
    )

    def observation(pan):
        views = {
            name: np.zeros((48, 64, 3), dtype=np.uint8) for name in config_obj.cam_names
        }
        return DesktopRobotPreprocessor(config_obj).get_observation(
            {"joint_state": np.array([pan, 80, 90, 30, 20, 10, 150], np.float32)}, views
        )

    wrapper.model = SimpleNamespace(
        processor=SimpleNamespace(
            tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 7)
        )
    )
    return wrapper, observation


@pytest.mark.parametrize("representation", [ABSOLUTE, DELTA])
def test_real_serving_prefix_decode_and_later_feedback(
    tmp_path, monkeypatch, representation
):
    cfg, _, _, _ = artifact(tmp_path, representation)
    wrapper, observation = serving_wrapper(cfg, tmp_path, monkeypatch)
    obs = observation(100)
    target = np.tile([101, 82, -93, -34, -25, 16, 30], (32, 1)).astype(np.float32)
    codec = B601Codec(representation)
    anchor = codec.anchor_from_state([100, 80, 90, 30, 20, 10, 150])
    labels = codec.encode_absolute_commands(target, anchor)
    prefix = target[:20].copy()

    def generate(**kwargs):
        expected = scale(
            torch.from_numpy(codec.encode_absolute_commands(prefix, anchor)),
            wrapper.normalizer_action.min["fixture"][:7],
            wrapper.normalizer_action.delta["fixture"][:7],
            clip=False,
        )
        torch.testing.assert_close(kwargs["rtc_condition"].prefix[0, :20, :7], expected)
        assert kwargs["rtc_condition"].prefix[..., 7:].count_nonzero() == 0
        assert kwargs["proprioception"][..., 7:].count_nonzero() == 0
        # Simulate feedback mutation during model execution.
        obs["robot_state_action_data"].data["state_right_arm_joint_pos"][:] = 200
        return {
            "predict_action": torch.nn.functional.pad(
                torch.from_numpy(labels)[None], (0, 19)
            )
        }

    wrapper.model.generate_flow_action = generate
    request = {
        "prefix": prefix,
        "committed_steps": 14,
        "guidance_horizon": 20,
        "max_guidance": 10,
        "guidance_mode": "lerobot",
    }
    output = wrapper.infer_flow_action(obs, "test", rtc_request=request)
    np.testing.assert_allclose(output["predict_action"][0, :, :7], target, atol=1e-4)
    np.testing.assert_array_equal(request["prefix"], target[:20])


def test_batch_has_independent_anchors(tmp_path, monkeypatch):
    cfg, _, _, _ = artifact(tmp_path)
    wrapper, observation = serving_wrapper(cfg, tmp_path, monkeypatch)

    def generate(**kwargs):
        result = torch.zeros(2, 32, 26)
        result[:, :, 0] = 1
        result[:, :, 6] = 30
        return {"predict_action": result}

    wrapper.model.generate_flow_action = generate
    result = wrapper.infer_flow_action_batch(
        [observation(100), observation(120)], ["a", "b"]
    )
    assert result[0]["predict_action"][0, 0, 0] == 101
    assert result[1]["predict_action"][0, 0, 0] == 121
    assert all(x["predict_action"][0, 0, 6] == 30 for x in result)


def test_loader_episode_split_and_shared_tail_window(tmp_path, monkeypatch):
    from wall_x.data.backends.lerobot.b601_adapter import (
        B601PreprocessedDataset as PreprocessedDataset,
    )
    from wall_x.data.backends.lerobot.loader import get_data_configs

    cfg, _, ep, split = artifact(tmp_path)
    rows = []
    for e, values in ep.items():
        for f in range(len(values["action"])):
            rows.append(
                {
                    "episode_index": torch.tensor(e),
                    "frame_index": torch.tensor(f),
                    "task_index": torch.tensor(e // 2),
                    "task": "test",
                    "action": torch.from_numpy(values["action"][f]),
                    "observation.state": torch.from_numpy(
                        values["observation.state"][f]
                    ),
                }
            )

    class FakeDataset:
        meta = SimpleNamespace(camera_keys=[])
        hf_dataset = {
            key: np.stack([row[key] for row in rows])
            for key in ("episode_index", "frame_index", "action", "task_index")
        }

        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return rows[i]

    data = get_data_configs(
        cfg["data"]
        | {
            "action_horizon": 32,
            "resolution": {},
            "key_mappings": cfg["data"]["key_mappings"],
        }
    )
    monkeypatch.setattr(
        PreprocessedDataset,
        "_vision_preprocess",
        lambda self, row: ([], 48, 64, 48, 64),
    )

    def reject_implicit_split(*args, **kwargs):
        raise AssertionError("B601 must not enter native random_split")

    monkeypatch.setattr(
        "wall_x.data.backends.lerobot.loader.random_split", reject_implicit_split
    )
    dataset = PreprocessedDataset(
        FakeDataset(), cfg, {}, data, cfg["data"]["lerobot_config"]
    )
    assert len(dataset.train_dataset) == len(split["train_episodes"]) * 4
    assert len(dataset.val_dataset) == len(split["validation_episodes"]) * 4
    for index in range(len(dataset)):
        row = rows[dataset.train_dataset.indices[index]]
        item = dataset[index]
        target = absolute_window(
            ep[int(row["episode_index"])]["action"], int(row["frame_index"]), 32
        )
        codec = B601Codec(DELTA)
        decoded = codec.decode_model_commands(
            item["action"], codec.anchor_from_state(item["agent_pos"])
        )
        np.testing.assert_array_equal(decoded, target)


def test_invalid_and_legacy_configs(tmp_path):
    assert settings({}) is None
    cfg = config()
    cfg["data"]["lerobot_config"]["b601"]["command_to_physical_scale"] = [1] * 7
    with pytest.raises(ValueError, match="mapping"):
        settings(cfg)
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train_episodes": [0, 1], "validation_episodes": [1]}))
    with pytest.raises(ValueError, match="overlapping"):
        read_split(path)


def test_checkpoint_cannot_silently_drop_representation(tmp_path):
    cfg, _, _, _ = artifact(tmp_path)
    save_manifest(cfg, tmp_path)
    cfg["data"]["lerobot_config"].pop("b601")
    with pytest.raises(ValueError, match="explicit"):
        validate_checkpoint(cfg, tmp_path)


def test_optimizer_cap_preserves_schedule_and_final_save(monkeypatch):
    from wall_x.config.hyperparams_config import CosineSchedulerConfig, TrainHyperParams
    from wall_x.trainer.fsdp_trainer.base_trainer import DistributedTrainer

    hp = TrainHyperParams(
        max_optimizer_steps=16000,
        scheduler=CosineSchedulerConfig(num_training_steps=25000),
    )
    assert hp.scheduler.num_training_steps == 25000
    saves, epochs = [], []
    fake = SimpleNamespace(
        cfg=SimpleNamespace(
            hyperparams=hp,
            logging=SimpleNamespace(epoch_save_interval=10),
            checkpoint=SimpleNamespace(validate_first=False),
            debug=SimpleNamespace(profile=False),
        ),
        nvtx=False,
        start_epoch=0,
        num_epoch=20,
        global_step=15999,
        num_training_steps=16000,
    )

    def train_loop(epoch, profiler):
        epochs.append(epoch)
        fake.global_step += 1

    fake.train_loop = train_loop
    fake.save_checkpoint = lambda epoch, step: saves.append((epoch, step))
    monkeypatch.setattr(
        "wall_x.trainer.fsdp_trainer.base_trainer.barrier", lambda: None
    )
    DistributedTrainer.fit(fake)
    assert epochs == [0] and saves == [(0, 16000)]
    DistributedTrainer.fit(fake)
    assert epochs == [0]  # A resumed completed run performs no extra optimizer update.


def test_parquet_reader_keeps_episodes_separate(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from wall_x.data.b601_prepare import load_episodes

    directory = tmp_path / "data"
    directory.mkdir()
    raw = np.arange(4 * 7, dtype=np.float32).reshape(4, 7)
    table = pa.table(
        {
            "episode_index": [1, 0, 1, 0],
            "frame_index": [1, 0, 0, 1],
            "task_index": [2, 1, 2, 1],
            "timestamp": [1 / 30, 0.0, 0.0, 1 / 30],
            "action": raw.tolist(),
            "observation.state": (raw * SCALE).tolist(),
        }
    )
    pq.write_table(table, directory / "all.parquet")
    result, sources = load_episodes(tmp_path)
    np.testing.assert_array_equal(result[0]["action"], raw[[1, 3]])
    np.testing.assert_array_equal(result[1]["action"], raw[[2, 0]])
    assert len(sources) == 1


@pytest.mark.parametrize("scope", ["train_episodes", "full_dataset"])
def test_native_checkpoint_metadata_is_portable_and_checked(tmp_path, scope):
    import yaml

    from wall_x._vendor.harrix.utils.train_config import _canonicalize_task_layouts
    from wall_x.data.b601 import build_serving_normalizers as build_normalizers
    from wall_x.trainer.fsdp_trainer.checkpoint_io import (
        _save_training_checkpoint_metadata,
    )

    cfg, _, ep, split = artifact(tmp_path)
    cfg["data"]["lerobot_config"]["b601"]["statistics_scope"] = scope
    (tmp_path / "norm_stats.json").write_text(
        json.dumps(make_stats(cfg, ep, split, []))
    )
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "preprocessor_config.json").write_text('{"fixture": true}')
    cfg["model"] = {"processor_path": str(processor)}
    stats = load_norm_stats(tmp_path / "norm_stats.json", cfg["data"]["key_mappings"])
    action, state = create_normalizers_from_lerobot_norm_stats(stats, "fixture", 26, 26)
    destination = tmp_path / "checkpoint"
    destination.mkdir()
    _save_training_checkpoint_metadata(
        ckpt_path=str(destination),
        config=cfg,
        epoch=1,
        global_step=15999,
        seed=10222,
        world_size=8,
        normalizer_action=action,
        normalizer_propri=state,
        log_fn=lambda *a, **k: None,
    )
    restored = yaml.safe_load((destination / "config.yml").read_text())
    _canonicalize_task_layouts(restored)
    # Deployment must use checkpoint-local files, even if original paths are gone.
    restored["data"]["norm_stats_path"] = "/unavailable/norm_stats.json"
    restored["data"]["lerobot_config"]["b601"][
        "split_manifest"
    ] = "/unavailable/split.json"
    built_a, built_s, _ = build_normalizers(str(destination), restored, "fixture")
    assert built_a.clip is False and built_s.clip is False
    (destination / "preprocessor_config.json").write_text('{"fixture": false}')
    with pytest.raises(ValueError, match="manifest mismatch"):
        build_normalizers(str(destination), restored, "fixture")


def test_full_statistics_include_validation_and_ignore_split(tmp_path):
    cfg, _, ep, split = artifact(tmp_path)
    cfg["data"]["lerobot_config"]["b601"]["statistics_scope"] = "full_dataset"
    # Distinct held-out values make accidentally using training-only data visible.
    for e in split["validation_episodes"]:
        ep[e]["action"] += 100
        ep[e]["observation.state"] += 20
    full = make_stats(cfg, ep, split, [])
    swapped = {
        **split,
        "train_episodes": split["validation_episodes"],
        "validation_episodes": split["train_episodes"],
    }
    assert make_stats(cfg, ep, swapped, []) == full
    assert make_stats(cfg, ep, None, []) == full
    stats = full["norm_stats"]
    assert stats["observation.state"]["count"] == 24
    assert stats["action"]["count"] == 24 * 32
    assert stats["observation.state"] == fit_statistics(
        np.concatenate([e["observation.state"] for e in ep.values()])
    )
    codec = B601Codec(DELTA)
    expected = np.concatenate(
        [
            codec.encode_absolute_commands(
                absolute_window(e["action"], i, 32), codec.anchor_from_state(s)
            )
            for e in ep.values()
            for i, s in enumerate(e["observation.state"])
        ]
    )
    assert stats["action"] == fit_statistics(expected)
    assert "split_sha256" not in full["b601_contract"]
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(full))
    save_manifest(cfg, tmp_path)
    validate_checkpoint(cfg, tmp_path)
    (tmp_path / "episode_split.json").write_text(json.dumps(swapped))
    validate_statistics(cfg, path, split_path=tmp_path / "episode_split.json")
    # Checkpoints freeze the actual training split even though the scale is shared.
    with pytest.raises(ValueError, match="manifest mismatch"):
        validate_checkpoint(cfg, tmp_path)
    swapped["train_episodes"].append(999)
    (tmp_path / "episode_split.json").write_text(json.dumps(swapped))
    with pytest.raises(ValueError, match="outside fixed statistics"):
        validate_statistics(cfg, path, split_path=tmp_path / "episode_split.json")


def test_frozen_statistics_reuse_and_resplit_never_refit(tmp_path, monkeypatch):
    from wall_x.data import b601_prepare as prepare

    cfg = config()
    cfg["data"]["lerobot_config"].update(root="fixture")
    cfg["data"]["lerobot_config"]["b601"].update(
        statistics_scope="full_dataset", validation_episodes_per_task=1
    )
    cfg["hyperparams"] = {"seed": 10222}
    ep, sources = episodes(), [{"path": "fixture.parquet", "sha256": "fixture"}]
    monkeypatch.setattr(prepare, "load_episodes", lambda root: (ep, sources))
    path = tmp_path / "norm_stats.json"
    cfg["data"]["norm_stats_path"] = str(path)
    assert prepare.freeze_statistics(cfg, path)["reused"] is False
    original, mtime = path.read_bytes(), path.stat().st_mtime_ns

    def no_refit(*args, **kwargs):
        raise AssertionError("Reusing statistics or resplitting must not fit again")

    monkeypatch.setattr(prepare, "make_stats", no_refit)
    assert prepare.freeze_statistics(cfg, path)["reused"] is True
    for seed in (10222, 7):
        cfg["hyperparams"]["seed"] = seed
        split_path = tmp_path / f"split_{seed}.json"
        assert prepare.prepare_split(cfg, split_path)["statistics_reused"]
        validate_statistics(cfg, path, split_path=split_path)
    assert path.read_bytes() == original and path.stat().st_mtime_ns == mtime
    sources[0]["sha256"] = "changed-dataset"
    with pytest.raises(ValueError, match="different dataset version"):
        prepare.freeze_statistics(cfg, path)
    cfg["data"]["lerobot_config"]["b601"]["representation"] = ABSOLUTE
    with pytest.raises(ValueError, match="contract mismatch"):
        prepare.freeze_statistics(cfg, path)


def test_full_statistics_tamper_and_legacy_scope_rejected(tmp_path):
    cfg, _, ep, split = artifact(tmp_path)
    cfg["data"]["lerobot_config"]["b601"]["statistics_scope"] = "full_dataset"
    path = tmp_path / "norm_stats.json"
    with pytest.raises(ValueError, match="scope mismatch"):
        validate_statistics(cfg, path)
    full = make_stats(cfg, ep, None, [])
    path.write_text(json.dumps(full))
    validate_statistics(cfg, path)
    full["norm_stats"]["action"]["lower"][0] += 1
    path.write_text(json.dumps(full))
    with pytest.raises(ValueError, match="checksum"):
        validate_statistics(cfg, path)
