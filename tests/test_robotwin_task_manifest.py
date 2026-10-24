import pytest

from wall_x.robotwin.task_manifest import (
    SOURCE_REPO,
    SOURCE_REVISION,
    select_episodes,
    semantic_task,
    validate_task_selection,
)


def test_pinned_aug_folder_swap_and_subset_guard():
    rows = []
    for i, (task, kind) in enumerate(
        [
            ("put_object_cabinet", "aug"),
            ("blocks_ranking_size", "aug"),
            ("put_object_cabinet", "clean"),
        ]
    ):
        rows.append(
            dict(
                task=task,
                source_repo=SOURCE_REPO,
                source_revision=SOURCE_REVISION,
                source_folder=f"lerobot_robotwin_eef_{kind}_500/{task}",
                split="train",
                episode_index=i,
            )
        )
    assert [semantic_task(r) for r in rows] == [
        "blocks_ranking_size",
        "put_object_cabinet",
        "put_object_cabinet",
    ]
    assert select_episodes(rows, ["put_object_cabinet"]) == [1, 2]
    assert validate_task_selection(rows, [1, 2], ["put_object_cabinet"]) == {
        "put_object_cabinet": 2
    }
    with pytest.raises(ValueError, match="differs from expected"):
        validate_task_selection(rows, [0, 2], ["put_object_cabinet"])
    assert semantic_task(dict(rows[0], source_revision="other")) == "put_object_cabinet"
    assert (
        semantic_task(dict(rows[0], split="validation", source_folder=""))
        == "put_object_cabinet"
    )
