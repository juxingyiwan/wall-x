"""Convert official RoboTwin XPolicyLab v1.0 HDF5 episodes to LeRobot 0.4.4."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from wall_x.robotwin.codec import CAMERAS, pose_to_state


def read_vectors(episode, group):
    """Read explicit states/targets; never shift an already aligned action field."""
    arms = []
    for side in ("left", "right"):
        pose = np.asarray(episode[f"{group}/{side}_ee_poses"], dtype=np.float32)
        grip = np.asarray(episode[f"{group}/{side}_ee_joint_states"], dtype=np.float32)
        if pose.ndim != 2 or pose.shape[1] != 7 or grip.shape != (len(pose), 1):
            raise ValueError(f"Invalid {group}/{side} pose or gripper shape")
        if np.any((grip < -1e-5) | (grip > 1 + 1e-5)):
            raise ValueError("Expected normalized gripper in [0,1]")
        arms.append(np.concatenate((pose, grip), axis=-1))
    return pose_to_state(np.concatenate(arms, axis=-1))


def episode_metadata(episode):
    version = episode["data_format_version"][()]
    if isinstance(version, bytes):
        version = version.decode()
    if version != "v1.0":
        raise ValueError(
            f"Unsupported data format {version!r}; expected XPolicyLab v1.0"
        )
    frequency = float(episode["additional_info/frequency"][()])
    if frequency <= 0 or frequency != int(frequency):
        raise ValueError(f"Invalid frequency {frequency}")
    instructions = json.loads(episode["instructions"][()])
    if (
        not isinstance(instructions, list)
        or not instructions
        or not isinstance(instructions[0], str)
    ):
        raise ValueError("Expected a nonempty list of language instructions")
    # One deterministic instruction per episode. Record this choice in provenance.
    return int(frequency), instructions[0]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(raw_root, output, tasks, episodes_per_task, validation_per_task, seed):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from XPolicyLab.utils.process_data import decode_image_bit

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    if not 0 < validation_per_task < episodes_per_task or len(set(tasks)) != len(tasks):
        raise ValueError(
            "Need unique tasks and 0 < validation episodes < episodes per task"
        )
    sources = []
    for task in tasks:
        folder = raw_root / task / task / "aloha_agilex"
        files = sorted((folder / "data").glob("episode_*.hdf5"))[:episodes_per_task]
        if len(files) != episodes_per_task:
            raise ValueError(
                f"Found {len(files)} episodes for {task}; requested {episodes_per_task}"
            )
        seeds = [int(value) for value in (folder / "seed.txt").read_text().split()]
        for file in files:
            source_id = int(file.stem.rsplit("_", 1)[1])
            sources.append((task, file, source_id, seeds[source_id]))
    with h5py.File(sources[0][1], "r") as episode:
        fps, _ = episode_metadata(episode)
        shapes = {
            key: decode_image_bit(episode[f"vision/{key}/colors"][0]).shape
            for key in CAMERAS
        }
    features = {
        "observation.state": {"dtype": "float32", "shape": (20,), "names": None},
        "action": {"dtype": "float32", "shape": (20,), "names": None},
        **{
            f"observation.images.{model_key}": {
                "dtype": "video",
                "shape": shapes[key],
                "names": ["height", "width", "channels"],
            }
            for key, (_, model_key) in CAMERAS.items()
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=f"local/{output.name}",
        root=output,
        fps=fps,
        features=features,
        robot_type="aloha_agilex",
        vcodec="h264",
        image_writer_threads=3,
    )
    provenance = []
    for index, (task, file, source_id, source_seed) in enumerate(sources):
        with h5py.File(file, "r") as episode:
            frequency, instruction = episode_metadata(episode)
            if frequency != fps:
                raise ValueError("Episodes have inconsistent frame rates")
            states, actions = read_vectors(episode, "state"), read_vectors(
                episode, "action"
            )
            if states.shape != actions.shape or len(states) == 0:
                raise ValueError(f"Unaligned or empty state/action arrays: {file}")
            for key in CAMERAS:
                if len(episode[f"vision/{key}/colors"]) != len(states):
                    raise ValueError(f"Camera {key} is not aligned with state/action")
            for frame in range(len(states)):
                record = {
                    "observation.state": states[frame],
                    "action": actions[frame],
                    "task": instruction,
                }
                for key, (_, model_key) in CAMERAS.items():
                    rgb = decode_image_bit(episode[f"vision/{key}/colors"][frame])
                    if rgb.shape != shapes[key] or rgb.dtype != np.uint8:
                        raise ValueError(f"Unexpected RGB shape/dtype for {key}")
                    record[f"observation.images.{model_key}"] = rgb
                dataset.add_frame(record)
            dataset.save_episode(parallel_encoding=False)
        provenance.append(
            {
                "episode_index": index,
                "task": task,
                "source_episode": source_id,
                "seed": source_seed,
                "frames": len(states),
                "instruction": instruction,
                "source": file.relative_to(raw_root).as_posix(),
                "sha256": sha256(file),
            }
        )
        print(f"Converted {task}/{source_id}: {len(states)} frames", flush=True)
    dataset.finalize()
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for task in tasks:
        ids = [row["episode_index"] for row in provenance if row["task"] == task]
        shuffled = rng.permutation(ids).tolist()
        validation.extend(shuffled[:validation_per_task])
        train.extend(shuffled[validation_per_task:])
    (output / "meta/episode_split.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "train": sorted(train),
                "validation": sorted(validation),
            },
            indent=2,
        )
        + "\n"
    )
    (output / "meta/robotwin_provenance.json").write_text(
        json.dumps(
            {
                "format": "XPolicyLab-v1.0",
                "action_contract": "absolute_xyz_row6d_gripper_dual_arm",
                "fps": fps,
                "instruction_selection": "first",
                "episodes": provenance,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Dataset ready: {output} ({len(train)} train / {len(validation)} validation)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["adjust_bottle", "click_bell"])
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--validation-per-task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10222)
    args = parser.parse_args()
    convert(
        args.raw_root.resolve(),
        args.output.resolve(),
        args.tasks,
        args.episodes_per_task,
        args.validation_per_task,
        args.seed,
    )


if __name__ == "__main__":
    main()
