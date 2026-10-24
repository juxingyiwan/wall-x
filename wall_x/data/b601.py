"""Versioned B601 command codec, shared by labels, statistics and serving.

State is measured motor degrees; commands precede driver sign/gripper scaling.
Every horizon uses one observation anchor. Gripper commands are always absolute.
"""

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np
import torch

from wall_x.utils.validation import is_builtin_int

LAYOUT = {
    "follow_right_arm_joint_pos": 6,
    "follow_right_gripper_pos": 1,
    "action_padding": 19,
}
JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]
SCALE = [1, 1, -1, -1, -1, 1, 6]
ABSOLUTE = "b601_joint_absolute_v1"
DELTA = "b601_joint_delta_v1"
FULL_DATASET = "full_dataset"
TRAIN_EPISODES = "train_episodes"


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def settings(config):
    """Opt in explicitly; absence retains all legacy checkpoint behavior."""
    data = config.get("data", {})
    opts = (data.get("lerobot_config") or config.get("lerobot_config") or {}).get(
        "b601"
    )
    if opts is None:
        return None
    opts = dict(opts)
    if opts.get("representation") not in (ABSOLUTE, DELTA):
        raise ValueError("B601 requires an explicit versioned action representation")
    if opts.get("target_policy") != "recorded_commands_v1":
        raise ValueError(
            "B601 v1 preserves recorded commands; historical limits are not inferred"
        )
    if opts.get("command_to_physical_scale") != SCALE:
        raise ValueError("B601 command/state direction and scale mapping mismatch")
    if (
        type(opts.get("fps")) not in (int, float)
        or not np.isfinite(opts["fps"])
        or opts["fps"] <= 0
    ):
        raise ValueError("B601 requires a finite positive data frequency")
    for key in ("clip_action_targets", "clip_state_inputs"):
        if not isinstance(opts.get(key), bool):
            raise ValueError(f"B601 requires explicit boolean {key}")
    task = config.get("task", {})
    for key in ("dof_config", "agent_pos_config"):
        layout = config.get(key) or task.get(key)
        if list((layout or {}).items()) != list(LAYOUT.items()):
            raise ValueError(f"B601 requires ordered 6+1+19 {key}")
    if config.get("use_fast_tokenizer", data.get("use_fast_tokenizer", False)):
        raise ValueError("B601 codec requires continuous flow actions, without FAST")
    if config.get(
        "use_state_string_representation",
        task.get("use_state_string_representation", False),
    ):
        raise ValueError("B601 codec requires continuous absolute state inputs")
    # The native normalizer bridge accepts an optional split_path. Selection
    # itself is resolved by the B601 adapter, without changing native files.
    opts.setdefault("split_manifest", None)
    return opts


def semantics(config):
    opts = settings(config)
    if opts is None:
        return None
    task, data = config.get("task", {}), config.get("data", {})
    horizon = config.get(
        "action_horizon", task.get("action_horizon", data.get("action_horizon"))
    )
    if not is_builtin_int(horizon) or horizon < 1:
        raise ValueError("B601 requires a positive action_horizon")
    return {
        "version": 1,
        "representation": opts["representation"],
        "joint_order": JOINTS,
        "command_to_physical_scale": SCALE,
        "state_units": "physical_motor_degrees",
        "wire_action_representation": ABSOLUTE,
        "target_policy": opts["target_policy"],
        "horizon": horizon,
        "fps": opts["fps"],
        "virtual_padding": 19,
        "clip_action_targets": opts["clip_action_targets"],
        "clip_state_inputs": opts["clip_state_inputs"],
        "window_rule": "same_episode_repeat_last_target_before_encode",
        "anchor_rule": "observation_state_in_command_coordinates",
    }


def statistics_scope(config):
    scope = settings(config).get("statistics_scope", TRAIN_EPISODES)
    if scope not in (FULL_DATASET, TRAIN_EPISODES):
        raise ValueError("Unknown B601 statistics_scope")
    return scope


@dataclass(frozen=True)
class B601Codec:
    representation: str

    def __post_init__(self):
        if self.representation not in (ABSOLUTE, DELTA):
            raise ValueError("Unknown B601 action representation")

    @staticmethod
    def anchor_from_state(state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError(
                "B601 anchor requires seven finite raw physical motor angles"
            )
        anchor = state[:6] / np.asarray(SCALE[:6], dtype=np.float32)
        anchor.setflags(write=False)
        return anchor

    def _convert(self, commands, anchor, sign):
        is_tensor = isinstance(commands, torch.Tensor)
        result = (
            commands.float().clone()
            if is_tensor
            else np.array(commands, dtype=np.float32, copy=True)
        )
        if result.ndim < 2 or result.shape[-1] not in (7, 26):
            raise ValueError("B601 commands must have horizon and 7 or 26 channels")
        if not (
            torch.isfinite(result).all() if is_tensor else np.isfinite(result).all()
        ):
            raise ValueError("B601 commands must be finite")
        anchor = np.asarray(anchor, dtype=np.float32)
        if anchor.shape != (6,) or not np.isfinite(anchor).all():
            raise ValueError("B601 conversion requires one six-joint request anchor")
        # A batch may only enter after being split into request-scoped samples.
        if result.ndim > 3 or (result.ndim == 3 and result.shape[0] != 1):
            raise ValueError("B601 codec accepts one request at a time")
        if self.representation == DELTA:
            offset = (
                torch.tensor(anchor.copy(), device=result.device)
                if is_tensor
                else anchor
            )
            result[..., :6] += sign * offset
        if result.shape[-1] == 26:
            result[..., 7:] = 0
        return result

    def encode_absolute_commands(self, commands, anchor):
        return self._convert(commands, anchor, -1)

    def decode_model_commands(self, commands, anchor):
        """Input is already denormalized; never apply action statistics here."""
        return self._convert(commands, anchor, 1)


def codec_from_config(config):
    opts = settings(config)
    return None if opts is None else B601Codec(opts["representation"])


def absolute_window(actions, frame, horizon):
    actions = np.asarray(actions, dtype=np.float32)
    if (
        actions.ndim != 2
        or actions.shape[1] != 7
        or not 0 <= frame < len(actions)
        or horizon < 1
    ):
        raise ValueError("Invalid B601 episode/window")
    return actions[
        np.minimum(np.arange(frame, frame + horizon), len(actions) - 1)
    ].copy()


def fit_statistics(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1, 7)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Statistics require finite, nonempty B601 data")
    low, high = np.quantile(values, [0.01, 0.99], axis=0).astype(np.float32)
    minimum, maximum = values.min(0), values.max(0)
    lower = np.where(high > low, low, minimum)
    span = np.where(high > low, high - low, maximum - minimum)
    span = np.where(span > 0, span, np.float32(1))
    return {
        "mean": values.mean(0, dtype=np.float64).tolist(),
        "std": values.std(0, dtype=np.float64).tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "q01": low.tolist(),
        "q99": high.tolist(),
        "lower": lower.tolist(),
        "span": span.tolist(),
        "count": len(values),
    }


def read_split(path):
    split = (
        copy.deepcopy(path)
        if isinstance(path, dict)
        else json.loads(Path(path).read_text())
    )
    train, val = split["train_episodes"], split["validation_episodes"]
    if not train or any(not is_builtin_int(x) or x < 0 for x in train + val):
        raise ValueError(
            "B601 split requires training episodes and integer episode lists"
        )
    if len(set(train + val)) != len(train + val):
        raise ValueError("B601 split has duplicate or overlapping episodes")
    return split


def validate_statistics(config, path, *, split_path=None):
    expected = semantics(config)
    stats = json.loads(Path(path).read_text())
    if expected is None:
        if "b601_contract" in stats:
            raise ValueError("B601 statistics require an explicit B601 configuration")
        return None
    contract = stats.get("b601_contract", {})
    if contract.get("semantics") != expected:
        raise ValueError("B601 statistics/action representation contract mismatch")
    if contract.get("statistics_sha256") != digest(stats["norm_stats"]):
        raise ValueError("B601 statistics checksum mismatch")
    scope = contract.get("statistics_scope", TRAIN_EPISODES)
    if scope != statistics_scope(config):
        raise ValueError("B601 statistics scope mismatch")
    if scope == FULL_DATASET:
        ids = contract.get("statistics_episodes", [])
        if (
            not ids
            or any(not is_builtin_int(x) or x < 0 for x in ids)
            or ids != sorted(set(ids))
            or "split_sha256" in contract
        ):
            raise ValueError("Invalid B601 full-dataset statistics provenance")
        if contract.get("dataset_files_sha256") != digest(contract.get("source_files")):
            raise ValueError("B601 statistics source checksum mismatch")
        count = stats["norm_stats"]["observation.state"].get("count")
        if (
            not is_builtin_int(count)
            or count < 1
            or stats["norm_stats"]["action"].get("count") != count * expected["horizon"]
        ):
            raise ValueError("B601 full-dataset statistics counts mismatch")
        if split_path:
            split = read_split(split_path)
            if not set(split["train_episodes"] + split["validation_episodes"]).issubset(
                ids
            ):
                raise ValueError(
                    "B601 split contains episodes outside fixed statistics"
                )
    elif split_path and contract.get("split_sha256") != digest(read_split(split_path)):
        raise ValueError("B601 statistics/episode split mismatch")
    for key in ("action", "observation.state"):
        for field in ("lower", "span"):
            value = np.asarray(stats["norm_stats"][key][field])
            if (
                value.shape != (7,)
                or not np.isfinite(value).all()
                or (field == "span" and (value <= 0).any())
            ):
                raise ValueError("Invalid persisted B601 affine parameters")
    return contract


def configure_normalizers(action, state, config):
    opts = settings(config)
    if opts is not None:
        for normalizer, key in (
            (action, "clip_action_targets"),
            (state, "clip_state_inputs"),
        ):
            normalizer.affine_fp32 = True
            normalizer.clip = opts[key]
    return action, state


def load_b601_norm_stats(path, key_mappings):
    """Adapt persisted affine parameters to the native NormStats interface."""
    from wall_x.data.backends.lerobot.utils import NormStats

    artifact = json.loads(Path(path).read_text())
    if "b601_contract" not in artifact:
        raise ValueError("B601 adapter requires versioned statistics")
    result = {}
    for name, key in (
        ("action", key_mappings["action"]),
        ("state", key_mappings["state"]),
    ):
        stat = artifact["norm_stats"][key]
        lower = torch.tensor(stat["lower"], dtype=torch.float32)
        span = torch.tensor(stat["span"], dtype=torch.float32)
        result[name] = NormStats(min=lower, max=lower + span, delta=span)
    return result


def build_serving_normalizers(checkpoint_path, train_config, norm_key):
    """Select B601 adaptation while preserving the native serving interface."""
    from wall_x._vendor.harrix.utils.normalizer import build_normalizers
    from wall_x._vendor.harrix.utils.train_config import (
        resolve_agent_pos_config,
        resolve_dof_config,
    )
    from wall_x.model.core.action.normalizer import (
        create_normalizers_from_lerobot_norm_stats,
    )

    validate_checkpoint(train_config, checkpoint_path)
    if settings(train_config) is None:
        return build_normalizers(checkpoint_path, train_config, norm_key)
    stats = load_b601_norm_stats(
        Path(checkpoint_path) / "norm_stats.json",
        train_config["data"]["key_mappings"],
    )
    action, state = create_normalizers_from_lerobot_norm_stats(
        stats,
        norm_key,
        sum(resolve_dof_config(train_config).values()),
        sum(resolve_agent_pos_config(train_config).values()),
    )
    configure_normalizers(action, state, train_config)
    return action, state, norm_key


def checkpoint_config(config):
    """Keep backend-only YAML fields while giving typed/CLI values precedence."""
    current = asdict(config) if is_dataclass(config) else copy.deepcopy(config)

    result = {
        key: copy.deepcopy(value)
        for key, value in (current.get("_raw_yaml", {}) or {}).items()
        if not key.startswith("_")
    }
    # The bridge reads backend-only fields from _raw_data, not stale _raw_yaml.
    if "_raw_data" in current:
        result["data"] = copy.deepcopy(current["_raw_data"] or {})
    sections = {
        "task",
        "model",
        "data",
        "hyperparams",
        "distributed",
        "logging",
        "checkpoint",
        "debug",
    }
    for key, value in current.items():
        if key.startswith("_"):
            continue
        # Merge dataclass sections only. Dict-valued fields such as an empty
        # noise_scheduler replace old values instead of resurrecting raw keys.
        if key in sections and isinstance(value, dict):
            result[key] = {**result.get(key, {}), **copy.deepcopy(value)}
        else:
            result[key] = copy.deepcopy(value)
    return result


def save_manifest(config, checkpoint_path):
    config = checkpoint_config(config)
    if settings(config) is None:
        return
    directory = Path(checkpoint_path)
    from wall_x.data.b601_split import resolve_split

    split_path = resolve_split(config)
    contract = validate_statistics(
        config, directory / "norm_stats.json", split_path=split_path
    )
    manifest = {
        **contract,
        "preprocessing": preprocessing_contract(config),
        "processor_files": processor_checksums(directory),
    }
    if contract.get("statistics_scope") == FULL_DATASET:
        # Experiment provenance remains frozen independently of the shared scale.
        manifest["training_split_sha256"] = digest(read_split(split_path))
    (directory / "b601_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "episode_split.json").write_text(
        json.dumps(read_split(split_path), indent=2) + "\n"
    )
    # Native checkpoint YAML sorts mappings. Preserve B601's ordered layout in
    # this adapter-owned save hook, including the real training reload path.
    import yaml

    from wall_x.config.loader import _sanitize_for_yaml

    saved_config = copy.deepcopy(config)
    # Make checkpoint configuration self-contained even if the original
    # experiment has no split file or its dataset is unavailable during serving.
    saved_config["data"]["lerobot_config"]["b601"]["split_manifest"] = split_path
    for key in ("dof_config", "agent_pos_config", "ar_dof_config"):
        for section in (saved_config, saved_config.get("task", {})):
            if key in section:
                if section[key] is None and key == "ar_dof_config":
                    section[key] = dict(LAYOUT)
                if dict(section[key]) != LAYOUT:
                    raise ValueError(f"B601 checkpoint has invalid {key}")
                section[key] = dict(LAYOUT)
    (directory / "config.yml").write_text(
        yaml.safe_dump(_sanitize_for_yaml(saved_config), sort_keys=False)
    )


def preprocessing_contract(config):
    from wall_x.data.backends.lerobot.config import LerobotConfig

    config = checkpoint_config(config)
    data = config.get("data", {})
    vision = LerobotConfig.from_yaml_dict(config)
    return {
        "key_mappings": data.get("key_mappings"),
        "camera_order": list(data.get("key_mappings", {}).get("camera", {}).values()),
        "resolution": data.get("resolution"),
        "camera_name_mapping": data.get("camera_name_mapping"),
        "priority_order": data.get("priority_order"),
        "color_order": "RGB",
        "use_state_string_representation": False,
        "use_fast_tokenizer": False,
        "resize_implementation": "PIL_then_qwen_smart_resize_v1",
        "min_pixels": vision.min_pixels,
        "max_pixels": vision.max_pixels,
        "image_factor": vision.image_factor,
    }


def processor_checksums(directory):
    names = (
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    )
    return {
        name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
        for name in names
        if (Path(directory) / name).exists()
    }


def validate_checkpoint(config, checkpoint_path):
    if settings(config) is None:
        directory = Path(checkpoint_path)
        if (directory / "b601_manifest.json").exists():
            raise ValueError("B601 checkpoint requires an explicit B601 configuration")
        if (directory / "norm_stats.json").exists():
            validate_statistics(config, directory / "norm_stats.json")
        return None
    directory = Path(checkpoint_path)
    manifest = json.loads((directory / "b601_manifest.json").read_text())
    contract = validate_statistics(
        config,
        directory / "norm_stats.json",
        split_path=directory / "episode_split.json",
    )
    expected_manifest = {
        **contract,
        "preprocessing": preprocessing_contract(config),
        "processor_files": processor_checksums(directory),
    }
    if contract.get("statistics_scope") == FULL_DATASET:
        expected_manifest["training_split_sha256"] = digest(
            read_split(directory / "episode_split.json")
        )
    if manifest != expected_manifest:
        raise ValueError("B601 checkpoint preprocessing manifest mismatch")
    return manifest
