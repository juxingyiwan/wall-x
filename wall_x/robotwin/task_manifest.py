"""Resolve known folder-label defects in the pinned RoboTwin EEF source."""

from collections import Counter

SOURCE_REPO = "robbyant/robotwin-clean-and-aug-lerobot"
SOURCE_REVISION = "595d4acc7e026aa57afba218f4170c6d79e39103"
SWAPPED_AUG_TASKS = {
    "put_object_cabinet": "blocks_ranking_size",
    "blocks_ranking_size": "put_object_cabinet",
}


def semantic_task(episode):
    task = episode["task"]
    if (
        episode.get("source_repo") == SOURCE_REPO
        and episode.get("source_revision") == SOURCE_REVISION
        and episode.get("source_folder", "").startswith("lerobot_robotwin_eef_aug_500/")
        and task in SWAPPED_AUG_TASKS
    ):
        return SWAPPED_AUG_TASKS[task]
    return task


def select_episodes(provenance, tasks, split="train"):
    wanted = set(tasks)
    return sorted(
        row["episode_index"]
        for row in provenance
        if row["split"] == split and semantic_task(row) in wanted
    )


def validate_task_selection(provenance, selected, expected_tasks):
    selected = set(selected)
    rows = [row for row in provenance if row["episode_index"] in selected]
    if len(rows) != len(selected):
        raise ValueError("Selected episodes are missing from RoboTwin provenance")
    counts = Counter(semantic_task(row) for row in rows)
    if set(counts) != set(expected_tasks):
        raise ValueError(
            f"RoboTwin task selection differs from expected tasks: {dict(counts)}"
        )
    return dict(counts)
