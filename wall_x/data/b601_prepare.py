"""B601 data preparation: episode loading, frozen statistics and splits."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from wall_x.data.b601 import (
    ABSOLUTE,
    DELTA,
    FULL_DATASET,
    SCALE,
    B601Codec,
    absolute_window,
    digest,
    fit_statistics,
    semantics,
    statistics_scope,
    validate_statistics,
)
from wall_x.data.b601_split import stratified_split


def load_episodes(root):
    episodes, files = {}, []
    for path in sorted((Path(root) / "data").rglob("*.parquet")):
        table = pq.read_table(
            path,
            columns=[
                "episode_index",
                "frame_index",
                "task_index",
                "timestamp",
                "observation.state",
                "action",
            ],
        )
        cols = table.to_pydict()
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        arrays = {
            key: np.asarray(
                value,
                dtype=(
                    np.float32
                    if key in ("action", "observation.state", "timestamp")
                    else np.int64
                ),
            )
            for key, value in cols.items()
        }
        ep = arrays["episode_index"]
        for episode in np.unique(ep):
            rows = np.flatnonzero(ep == episode)
            part = {key: value[rows] for key, value in arrays.items()}
            episodes.setdefault(int(episode), []).append(part)
    result = {}
    for episode, parts in episodes.items():
        merged = {
            key: np.concatenate([part[key] for part in parts]) for key in parts[0]
        }
        order = np.argsort(merged["frame_index"])
        merged = {key: value[order] for key, value in merged.items()}
        if not np.array_equal(merged["frame_index"], np.arange(len(order))):
            raise ValueError(f"Episode {episode}: duplicate or missing frames")
        if len(np.unique(merged["task_index"])) != 1 or np.any(
            np.diff(merged["timestamp"]) <= 0
        ):
            raise ValueError(f"Episode {episode}: inconsistent task or timestamps")
        for key in ("action", "observation.state"):
            if (
                merged[key].shape != (len(order), 7)
                or not np.isfinite(merged[key]).all()
            ):
                raise ValueError(f"Episode {episode}: invalid {key}")
        result[episode] = merged
    if not result:
        raise ValueError("No B601 episodes found")
    return result, files


def make_stats(config, episodes, split, source_files):
    spec = semantics(config)
    scope = statistics_scope(config)
    ids = sorted(episodes) if scope == FULL_DATASET else split["train_episodes"]
    codec = B601Codec(spec["representation"])
    state_rows, action_rows = [], []
    for episode in ids:
        values = episodes[episode]
        states, actions = values["observation.state"], values["action"]
        state_rows.append(states)
        for frame, state in enumerate(states):
            action_rows.append(
                codec.encode_absolute_commands(
                    absolute_window(actions, frame, spec["horizon"]),
                    codec.anchor_from_state(state),
                )
            )
    stats = {
        "observation.state": fit_statistics(np.concatenate(state_rows)),
        "action": fit_statistics(np.concatenate(action_rows)),
    }
    provenance = (
        {"statistics_scope": FULL_DATASET, "statistics_episodes": ids}
        if scope == FULL_DATASET
        else {"split_sha256": digest(split)}
    )
    return {
        "norm_stats": stats,
        "b601_contract": {
            "semantics": spec,
            "statistics_sha256": digest(stats),
            **provenance,
            "dataset_files_sha256": digest(source_files),
            "source_files": source_files,
            "recording_driver_limits": "unverified; recorded targets preserved without rewriting",
        },
    }


def check_full_statistics_source(contract, episodes, source_files):
    if contract.get("statistics_scope") != FULL_DATASET:
        raise ValueError("Split-independent reuse requires full_dataset statistics")
    if contract["dataset_files_sha256"] != digest(source_files) or contract[
        "statistics_episodes"
    ] != sorted(episodes):
        raise ValueError("Frozen B601 statistics belong to a different dataset version")


def freeze_statistics(base, output):
    """Create once, otherwise verify and reuse without refitting or rewriting."""
    if statistics_scope(base) != FULL_DATASET:
        raise ValueError("Set b601.statistics_scope: full_dataset before freezing")
    episodes, sources = load_episodes(base["data"]["lerobot_config"]["root"])
    output = Path(output).resolve()
    reused = output.exists()
    if reused:
        contract = validate_statistics(base, output)
        check_full_statistics_source(contract, episodes, sources)
    else:
        artifact = make_stats(base, episodes, None, sources)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(artifact, indent=2) + "\n")
        contract = validate_statistics(base, output)
    return {
        "statistics_path": str(output),
        "reused": reused,
        "episodes": len(episodes),
        "frames": sum(len(e["action"]) for e in episodes.values()),
        "statistics_sha256": contract["statistics_sha256"],
    }


def prepare_split(base, output):
    """Change episode assignments while keeping a full-dataset scale untouched."""
    opts = base["data"]["lerobot_config"]["b601"]
    episodes, sources = load_episodes(base["data"]["lerobot_config"]["root"])
    contract = validate_statistics(base, base["data"]["norm_stats_path"])
    check_full_statistics_source(contract, episodes, sources)
    split = stratified_split(
        episodes,
        seed=base["hyperparams"]["seed"],
        validation_episodes_per_task=opts.get("validation_episodes_per_task"),
        validation_ratio=opts.get("validation_ratio"),
    )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(split, indent=2) + "\n")
    return {
        "split_path": str(output),
        "train_episodes": len(split["train_episodes"]),
        "validation_episodes": len(split["validation_episodes"]),
        "statistics_sha256": contract["statistics_sha256"],
        "statistics_reused": True,
    }


def prepare(base, output):
    output = Path(output).resolve()
    # No partial overwrite of a previously frozen experiment.
    output.mkdir(parents=True, exist_ok=False)
    root = Path(base["data"]["lerobot_config"]["root"])
    info = json.loads((root / "meta/info.json").read_text())
    episodes, sources = load_episodes(root)
    base_opts = base["data"]["lerobot_config"].get("b601", {})
    validation_count = base_opts.get("validation_episodes_per_task")
    validation_ratio = base_opts.get("validation_ratio")
    split = stratified_split(
        episodes,
        seed=base["hyperparams"]["seed"],
        validation_episodes_per_task=validation_count,
        validation_ratio=validation_ratio,
    )
    (output / "episode_split.json").write_text(json.dumps(split, indent=2) + "\n")
    reports = []
    for name, representation in (("absolute", ABSOLUTE), ("delta", DELTA)):
        config = copy.deepcopy(base)
        data = config["data"]
        data["lerobot_config"]["b601"] = {
            "representation": representation,
            "target_policy": "recorded_commands_v1",
            "command_to_physical_scale": SCALE,
            "fps": info["fps"],
            "clip_action_targets": False,
            "clip_state_inputs": False,
            "split_manifest": str(output / "episode_split.json"),
        }
        base_opts = base["data"]["lerobot_config"].get("b601", {})
        if "statistics_scope" in base_opts:
            data["lerobot_config"]["b601"]["statistics_scope"] = base_opts[
                "statistics_scope"
            ]
        data["lerobot_config"].pop("episodes", None)
        if validation_count is not None:
            data["lerobot_config"]["b601"][
                "validation_episodes_per_task"
            ] = validation_count
        if validation_ratio is not None:
            data["lerobot_config"]["b601"]["validation_ratio"] = validation_ratio
        # The manifest is authoritative for B601; omit the unused legacy ratio.
        data.pop("train_test_split", None)
        data["use_fast_tokenizer"] = False
        config["task"]["use_state_string_representation"] = False
        config["task"]["action_horizon"] = config["task"]["action_horizon_flow"] = 32
        config["hyperparams"]["max_optimizer_steps"] = 16000
        data["norm_stats_path"] = str(output / f"{name}_norm_stats.json")
        run_name = f"b601_{name}_s10222_16k_noclip"
        config["logging"]["log_name"] = run_name
        config["checkpoint"]["save_path"] = str(output / "checkpoints" / run_name)
        if (
            base_opts.get("statistics_scope") == FULL_DATASET
            and base_opts.get("representation") == representation
        ):
            source_path = base["data"]["norm_stats_path"]
            contract = validate_statistics(config, source_path)
            check_full_statistics_source(contract, episodes, sources)
            stats = json.loads(Path(source_path).read_text())
        else:
            stats = make_stats(config, episodes, split, sources)
        (output / f"{name}_norm_stats.json").write_text(
            json.dumps(stats, indent=2) + "\n"
        )
        (output / f"{name}.yml").write_text(yaml.safe_dump(config, sort_keys=False))
        reports.append(
            {
                "representation": representation,
                "statistics_sha256": stats["b601_contract"]["statistics_sha256"],
            }
        )
    report = {
        "episodes": len(episodes),
        "train_episodes": len(split["train_episodes"]),
        "validation_episodes": len(split["validation_episodes"]),
        "split_sha256": digest(split),
        "artifacts": reports,
    }
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
