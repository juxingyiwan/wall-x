"""Build the converter manifest from the pinned public LeRobot metadata."""

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = "robbyant/robotwin-clean-and-aug-lerobot"
REVISION = "595d4acc7e026aa57afba218f4170c6d79e39103"
SPLITS = {
    "lerobot_robotwin_eef_clean_50": ("-demo_clean_collect_200-50", 50),
    "lerobot_robotwin_eef_aug_500": ("-aloha-agilex_randomized_500-1000", 500),
}


def audit_source(source_root, tasks=None):
    root = Path(source_root)
    requested = set(tasks) if tasks is not None else None
    if requested == set():
        raise ValueError("Select at least one task")
    rows, task_sets = [], []
    for directory, (_, expected_episodes) in SPLITS.items():
        found = set()
        for source in sorted((root / directory).glob("*")):
            if not source.is_dir():
                continue
            task = source.name.split("-", 1)[0]
            if not re.fullmatch(r"[a-z][a-z0-9_]*", task):
                raise ValueError(f"Unexpected source folder name: {source}")
            if requested is not None and task not in requested:
                continue
            if task in found:
                raise ValueError(f"Duplicate source task: {task}")
            info_path = source / "meta/info.json"
            episodes_path = source / "meta/episodes.jsonl"
            info = json.loads(info_path.read_text())
            episodes = [
                json.loads(line) for line in episodes_path.read_text().splitlines()
            ]
            if (
                len(episodes) != expected_episodes
                or [e["episode_index"] for e in episodes]
                != list(range(expected_episodes))
                or any(
                    not isinstance(e["length"], int)
                    or isinstance(e["length"], bool)
                    or e["length"] <= 0
                    for e in episodes
                )
                or info["total_episodes"] != expected_episodes
                or info["total_frames"] != sum(e["length"] for e in episodes)
                or info["fps"] != 50
                or any(
                    info["features"][key]["shape"] != [16]
                    for key in ("observation.state", "action")
                )
            ):
                raise ValueError(f"Unexpected source metadata: {source}")
            found.add(task)
            rows.append(
                {
                    "folder": source.relative_to(root).as_posix(),
                    "task": task,
                    "episodes": expected_episodes,
                    "frames": info["total_frames"],
                    "fps": 50,
                    "sha256": {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (info_path, episodes_path)
                    },
                }
            )
        task_sets.append(found)
    if not task_sets[0] or task_sets[0] != task_sets[1]:
        raise ValueError(
            "Clean and randomized folders must contain the same nonempty task set"
        )
    if requested is not None and task_sets[0] != requested:
        raise ValueError(f"Missing requested tasks: {sorted(requested - task_sets[0])}")
    counts = {key: sum(row[key] for row in rows) for key in ("episodes", "frames")}
    if requested is None and (
        len(task_sets[0]) != 50 or counts != {"episodes": 27500, "frames": 6077247}
    ):
        raise ValueError("Full metadata counts differ from the pinned 50-task dataset")
    return {"repo": REPO, "revision": REVISION, "counts": counts, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    result = audit_source(args.source_root, args.tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
