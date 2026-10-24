"""CPU-only real-data parity check. No model weights, network server or hardware."""

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import WallxModelWrapper
from wall_x._vendor.harrix.serving._wallx_infer.robot import DesktopRobotPreprocessor
from wall_x.config.loader import load_config
from wall_x.data.b601 import (
    B601Codec,
    absolute_window,
    validate_statistics,
)
from wall_x.data.b601_split import resolve_split
from wall_x.data.backends.lerobot.bridge import build_dataset_config
from wall_x.data.backends.lerobot.build import (
    load_lerobot_normalizers,
    load_lerobot_v2,
)
from wall_x.data.backends.lerobot.config import LerobotConfig


def validate(directory=None, samples=12, *, config_path=None, output_path=None):
    if config_path is not None:
        paths = [Path(config_path)]
        directory = paths[0].parent
    else:
        directory = Path(directory)
        paths = [directory / f"{name}.yml" for name in ("absolute", "delta")]
    reports = []
    for path in paths:
        raw = yaml.safe_load(path.read_text())
        cfg = load_config(str(path))
        flat, lerobot_cfg = build_dataset_config(cfg)
        opts = lerobot_cfg["b601"]
        name = opts["representation"]
        split = resolve_split(raw)
        validate_statistics(raw, raw["data"]["norm_stats_path"], split_path=split)
        print(f"Loading {name} actual LeRobot dataset", flush=True)
        data_wrapper, train_loader, _ = load_lerobot_v2(cfg)
        dataset = data_wrapper._inner
        collator = train_loader.collate_fn
        normal_a, normal_s, _, _ = load_lerobot_normalizers(cfg)
        codec = B601Codec(opts["representation"])
        wrapper = WallxModelWrapper.__new__(WallxModelWrapper)
        wrapper.config = SimpleNamespace(
            train_config=flat,
            data_config=LerobotConfig.from_yaml_dict(raw),
            action_horizon=32,
            action_dim=26,
            cam_names=list(raw["data"]["key_mappings"]["camera"].values()),
            camera_front_key="front_view",
            camera_right_key="side_view",
            camera_left_key="left_wrist_view",
            model_device="cpu",
        )
        wrapper.cam_names = wrapper.config.cam_names
        for attr, value in {
            "role_start_symbol": "<|im_start|>",
            "role_end_symbol": "<|im_end|>",
            "vision_start_symbol": "<|vision_start|>",
            "vision_end_symbol": "<|vision_end|>",
            "image_pad_symbol": "<|image_pad|>",
            "propri_symbol": "<|propri|>",
            "action_symbol": "<|action|>",
        }.items():
            setattr(wrapper, attr, value)
        wrapper.norm_key = lerobot_cfg["repo_id"]
        wrapper.normalizer_action, wrapper.normalizer_propri = normal_a, normal_s
        wrapper.model = SimpleNamespace(processor=collator.processor)
        wrapper.logger = logging.getLogger("b601_validate")
        max_error = 0.0
        checks = []
        # Samples span the actual training subset; include episode tails explicitly.
        indices = set(np.linspace(0, len(dataset) - 1, samples, dtype=int).tolist())
        train_rows = dataset.train_dataset.indices
        ep_col = dataset.b601_episode_ids
        for episode in split["train_episodes"][
            :: max(1, len(split["train_episodes"]) // samples)
        ]:
            candidates = [
                i for i, row in enumerate(train_rows) if ep_col[row] == episode
            ]
            indices.add(candidates[-1])
        for index in sorted(indices):
            item = dataset[index]
            recorded = dataset.train_dataset[index]
            batch = collator([item])
            reconstructed = normal_a.unnormalize_data(
                batch["action_chunk"], [wrapper.norm_key]
            )
            target = codec.decode_model_commands(
                reconstructed, codec.anchor_from_state(item["agent_pos"])
            )[0, :, :7].numpy()
            expected = absolute_window(
                dataset.b601_actions[int(recorded["episode_index"])],
                int(recorded["frame_index"]),
                32,
            )
            max_error = max(max_error, float(np.max(np.abs(target - expected))))
            np.testing.assert_allclose(target, expected, atol=1e-4, rtol=0)
            views = {
                destination: (recorded[source].permute(1, 2, 0).numpy() * 255).astype(
                    np.uint8
                )
                for source, destination in raw["data"]["key_mappings"]["camera"].items()
            }
            observation = DesktopRobotPreprocessor(wrapper.config).get_observation(
                {"joint_state": item["agent_pos"].numpy()}, views
            )
            prefix, postfix = wrapper.get_text_for_action(recorded["task"])
            inference = wrapper.construct_model_input(
                [observation], [prefix], [postfix]
            )
            for key in (
                "proprioception",
                "agent_pos_mask",
                "dof_mask",
                "pixel_values",
                "image_grid_thw",
            ):
                torch.testing.assert_close(
                    inference[key], batch[key], atol=0, rtol=0, msg=key
                )
            training_ids = batch["input_ids"][batch["attention_mask"].bool()]
            inference_ids = inference["input_ids"][inference["attention_mask"].bool()]
            torch.testing.assert_close(training_ids, inference_ids, atol=0, rtol=0)
            checks.append(
                {
                    "episode": int(recorded["episode_index"]),
                    "frame": int(recorded["frame_index"]),
                }
            )
        reports.append(
            {
                "mode": name,
                "train_frames": len(dataset.train_dataset),
                "validation_frames": len(dataset.val_dataset),
                "samples": checks,
                "max_command_roundtrip_error_deg": max_error,
                "input_parity": "exact: state, masks, token IDs, image pixels and grids",
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    result = {"checks": reports, "robot_motion": False, "model_weights_loaded": False}
    destination = Path(output_path) if output_path else directory / "verification.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--directory")
    source.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    validate(
        args.directory, args.samples, config_path=args.config, output_path=args.output
    )
