"""Convert pinned RoboTwin LeRobot v2.1 EEF datasets into one Wall-X v3 dataset.

Videos are stored as regular files without re-encoding. Absolute WXYZ state and action vectors
are converted independently to Wall-X row-6D rotations; actions are not shifted.
All source episodes belong to training; independently collected dev data is extra.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from wall_x.robotwin.codec import pose_to_state, state_to_pose
from wall_x.robotwin.storage import RelocatedVideoIndex, store_video

CAMERAS = {
    "cam_high": "face_view",
    "cam_left_wrist": "left_wrist_view",
    "cam_right_wrist": "right_wrist_view",
}


def convert_vectors(states, actions):
    states, actions = np.asarray(states, dtype=np.float32), np.asarray(
        actions, dtype=np.float32
    )
    if (
        states.shape != actions.shape
        or states.ndim != 2
        or states.shape[1] != 16
        or len(states) == 0
    ):
        raise ValueError("Expected aligned nonempty [frames,16] state/action arrays")
    for name, array in [("state", states), ("action", actions)]:
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite {name}")
        if np.any((array[:, [7, 15]] < -1e-5) | (array[:, [7, 15]] > 1 + 1e-5)):
            raise ValueError(f"{name} grippers must be in [0,1]")
        for offset in [3, 11]:
            if (
                np.max(
                    np.abs(np.linalg.norm(array[:, offset : offset + 4], axis=1) - 1)
                )
                > 1e-3
            ):
                raise ValueError(f"{name} quaternion is not unit WXYZ")
    converted = [pose_to_state(x) for x in [states, actions]]
    for original, result in zip([states, actions], converted):
        reconstructed = state_to_pose(result)
        if not np.allclose(
            original[:, [0, 1, 2, 7, 8, 9, 10, 15]],
            reconstructed[:, [0, 1, 2, 7, 8, 9, 10, 15]],
            atol=1e-6,
        ):
            raise ValueError("Position/gripper round trip failed")
        for offset in [3, 11]:
            dot = np.sum(
                original[:, offset : offset + 4]
                * reconstructed[:, offset : offset + 4],
                axis=1,
            )
            if np.max(np.abs(np.abs(dot) - 1)) > 1e-4:
                raise ValueError("Quaternion/row-6D round trip failed")
    return converted


class DatasetWriter:
    def __init__(self, output, fps=50, video_storage="copy"):
        if video_storage not in {"copy", "move"}:
            raise ValueError(video_storage)
        self.video_storage = video_storage
        self.output = Path(output).resolve()
        if self.output.exists():
            raise FileExistsError(f"Refusing to overwrite {self.output}")
        self.output.mkdir(parents=True)
        self.fps = fps
        self.frames = 0
        self.episodes = []
        self.task_ids = {}
        self.buffer = []
        self.file_index = 0
        self.train = []
        self.validation = []
        self.source_features = None
        self.provenance = []
        self.video_records = []

    def append(
        self,
        states,
        actions,
        instructions,
        videos,
        split,
        provenance,
        video_sources=None,
    ):
        states, actions = convert_vectors(states, actions)
        count = len(states)
        if len(instructions) != count:
            raise ValueError("Instructions are not frame aligned")
        task_indices = []
        for instruction in instructions:
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Empty instruction")
            task_indices.append(
                self.task_ids.setdefault(instruction, len(self.task_ids))
            )
        index = len(self.episodes)
        table = pa.table(
            {
                "observation.state": pa.FixedSizeListArray.from_arrays(
                    pa.array(states.ravel()), 20
                ),
                "action": pa.FixedSizeListArray.from_arrays(
                    pa.array(actions.ravel()), 20
                ),
                "timestamp": pa.array(np.arange(count, dtype=np.float32) / self.fps),
                "frame_index": pa.array(np.arange(count, dtype=np.int64)),
                "episode_index": pa.array(np.full(count, index, dtype=np.int64)),
                "index": pa.array(
                    np.arange(self.frames, self.frames + count, dtype=np.int64)
                ),
                "task_index": pa.array(task_indices, type=pa.int64()),
            }
        )
        self.buffer.append(table)
        row = {
            "episode_index": index,
            "tasks": list(dict.fromkeys(instructions)),
            "length": count,
            "data/chunk_index": self.file_index // 1000,
            "data/file_index": self.file_index % 1000,
            "dataset_from_index": self.frames,
            "dataset_to_index": self.frames + count,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        import av

        for key, source in videos.items():
            source = Path(source).resolve(strict=True)
            with av.open(str(source)) as container:
                stream = container.streams.video[0]
                if stream.width != 640 or stream.height != 480:
                    raise ValueError(f"Unexpected camera resolution in {source}")
                if stream.frames and stream.frames != count:
                    raise ValueError(
                        f"Video/frame count mismatch {source}: {stream.frames} != {count}"
                    )
                if abs(float(stream.average_rate) - self.fps) > 1e-5:
                    raise ValueError(f"Unexpected video FPS in {source}")
            destination = (
                self.output
                / f"videos/{key}/chunk-{index//1000:03d}/file-{index%1000:03d}.mp4"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.video_records.append(
                {
                    "source_relative": (video_sources or {}).get(key, str(source)),
                    "path": str(destination.relative_to(self.output)),
                    "size": source.stat().st_size,
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
            store_video(source, destination, move=self.video_storage == "move")
            row.update(
                {
                    f"videos/{key}/chunk_index": index // 1000,
                    f"videos/{key}/file_index": index % 1000,
                    f"videos/{key}/from_timestamp": 0.0,
                    f"videos/{key}/to_timestamp": count / self.fps,
                }
            )
        if set(videos) != {f"observation.images.{x}" for x in CAMERAS.values()}:
            raise ValueError("Need exactly three named camera streams")
        self.episodes.append(row)
        self.frames += count
        {"train": self.train, "validation": self.validation}[split].append(index)
        self.provenance.append(
            {"episode_index": index, "split": split, "frames": count, **provenance}
        )
        if len(self.buffer) >= 128:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        path = (
            self.output
            / f"data/chunk-{self.file_index//1000:03d}/file-{self.file_index%1000:03d}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.concat_tables(self.buffer), path, compression="zstd")
        self.buffer.clear()
        self.file_index += 1

    def finalize(self):
        self.flush()
        if not self.train or not self.validation:
            raise ValueError("Need nonempty train and independent validation splits")
        meta = self.output / "meta"
        (meta / "episodes/chunk-000").mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(self.episodes),
            meta / "episodes/chunk-000/file-000.parquet",
        )
        pd.DataFrame(
            {"task_index": list(self.task_ids.values())},
            index=list(self.task_ids.keys()),
        ).to_parquet(meta / "tasks.parquet")
        features = {
            name: {"dtype": "float32", "shape": [20], "names": None}
            for name in ["observation.state", "action"]
        }
        for key in CAMERAS.values():
            features[f"observation.images.{key}"] = {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.fps": self.fps,
                    "video.channels": 3,
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        for name in [
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ]:
            features[name] = {
                "dtype": "float32" if name == "timestamp" else "int64",
                "shape": [1],
                "names": None,
            }
        info = {
            "codebase_version": "v3.0",
            "robot_type": "aloha_agilex",
            "total_episodes": len(self.episodes),
            "total_frames": self.frames,
            "total_tasks": len(self.task_ids),
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": self.fps,
            "splits": {"train": f"0:{len(self.episodes)}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
        }
        for filename, value in [
            ("info.json", info),
            (
                "episode_split.json",
                {"train": self.train, "validation": self.validation, "seed": 10222},
            ),
            (
                "robotwin_provenance.json",
                {
                    "action_contract": "absolute_xyz_row6d_gripper_dual_arm",
                    "source_quaternion_order": "WXYZ",
                    "action_shift_applied": False,
                    "fps": self.fps,
                    "videos": f"regular files ({self.video_storage}); no temporal resampling or re-encoding",
                    "episodes": self.provenance,
                },
            ),
        ]:
            (meta / filename).write_text(json.dumps(value, indent=2) + "\n")
        (meta / "video_sources.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "storage": "regular_files",
                    "video_count": len(self.video_records),
                    "videos": self.video_records,
                },
                indent=2,
            )
            + "\n"
        )
        return info


def convert(
    source_root,
    audit_path,
    dev_root,
    output,
    max_per_folder=None,
    folders=None,
    video_storage="copy",
    video_index=None,
):
    audit = json.loads(Path(audit_path).read_text())
    selected = [r for r in audit["rows"] if folders is None or r["folder"] in folders]
    writer = DatasetWriter(output, video_storage=video_storage)
    relocated = RelocatedVideoIndex(video_index)
    expected_train = expected_frames = 0
    for row in selected:
        source = Path(source_root) / row["folder"]
        info = json.loads((source / "meta/info.json").read_text())
        if info["fps"] != 50 or info["features"]["observation.state"]["shape"] != [16]:
            raise ValueError(f"Unexpected source format: {source}")
        tasks = {
            r["task_index"]: r["task"]
            for r in map(
                json.loads, (source / "meta/tasks.jsonl").read_text().splitlines()
            )
        }
        episodes = [
            json.loads(line)
            for line in (source / "meta/episodes.jsonl").read_text().splitlines()
        ]
        if max_per_folder is not None:
            episodes = episodes[:max_per_folder]
        for episode in episodes:
            index = episode["episode_index"]
            path = source / info["data_path"].format(
                episode_chunk=index // info["chunks_size"], episode_index=index
            )
            table = pq.read_table(path)
            if len(table) != episode["length"]:
                raise ValueError(f"Source length mismatch: {path}")
            if table["frame_index"].to_pylist() != list(range(len(table))):
                raise ValueError(f"Noncontiguous source frames: {path}")
            if not np.allclose(
                table["timestamp"].to_numpy(), np.arange(len(table)) / 50, atol=1e-5
            ):
                raise ValueError(f"Unexpected source timebase: {path}")
            videos = {
                f"observation.images.{destination}": source
                / info["video_path"].format(
                    episode_chunk=index // info["chunks_size"],
                    episode_index=index,
                    video_key=f"observation.images.{camera}",
                )
                for camera, destination in CAMERAS.items()
            }
            video_sources = {
                key: str(path.relative_to(Path(source_root)))
                for key, path in videos.items()
            }
            videos = {
                key: relocated.resolve(path, Path(source_root))
                for key, path in videos.items()
            }
            writer.append(
                table["observation.state"].to_pylist(),
                table["action"].to_pylist(),
                [tasks[i] for i in table["task_index"].to_pylist()],
                videos,
                "train",
                {
                    "source_repo": audit["repo"],
                    "source_revision": audit["revision"],
                    "source_folder": row["folder"],
                    "source_episode": index,
                    "task": row["task"],
                    "source_parquet_sha256": hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest(),
                },
                video_sources=video_sources,
            )
            expected_train += 1
            expected_frames += len(table)
        print(
            json.dumps(
                {
                    "source_folder": row["folder"],
                    "converted_train_episodes": expected_train,
                    "train_frames": expected_frames,
                }
            ),
            flush=True,
        )
    if max_per_folder is None and folders is None:
        if expected_train != 27500 or expected_frames != 6077247:
            raise ValueError("Full training counts differ from the pinned audit")
    for task in sorted({r["task"] for r in selected}):
        folder = Path(dev_root) / task
        record = json.loads((folder / "complete.json").read_text())
        vectors = np.load(folder / "vectors.npz", allow_pickle=False)
        writer.append(
            vectors["state"],
            vectors["action"],
            [record["instruction"]] * record["frames"],
            {
                f"observation.images.{key}": relocated.resolve(
                    folder / f"{key}.mp4", Path(dev_root).parent
                )
                for key in CAMERAS.values()
            },
            "validation",
            {
                "task": task,
                "seed": record["seed"],
                "independent_expert": True,
                "task_config": record["task_config"],
            },
            video_sources={
                f"observation.images.{key}": str(
                    (folder / f"{key}.mp4").relative_to(Path(dev_root).parent)
                )
                for key in CAMERAS.values()
            },
        )
    info = writer.finalize()
    print(
        json.dumps(
            {
                "completed": True,
                "output": str(writer.output),
                "train_episodes": len(writer.train),
                "validation_episodes": len(writer.validation),
                "train_frames": expected_frames,
                "total_frames": info["total_frames"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--dev-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-per-folder", type=int)
    parser.add_argument("--folders", nargs="+")
    parser.add_argument("--video-storage", choices=["copy", "move"], default="copy")
    parser.add_argument(
        "--video-index",
        type=Path,
        help="Existing prepared dataset meta/video_sources.json",
    )
    args = parser.parse_args()
    convert(
        args.source_root,
        args.audit,
        args.dev_root,
        args.output,
        args.max_per_folder,
        args.folders,
        args.video_storage,
        args.video_index,
    )


if __name__ == "__main__":
    main()
