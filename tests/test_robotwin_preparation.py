import json

import numpy as np
import pytest

from wall_x.robotwin.audit_source import SPLITS, audit_source
from wall_x.robotwin.collect_development import CAMERAS, EpisodeRecorder
from wall_x.robotwin.convert_lerobot import convert_vectors


def source_metadata(root, task="click_bell"):
    for directory, (suffix, count) in SPLITS.items():
        meta = root / directory / (task + suffix) / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text(
            json.dumps(
                {
                    "fps": 50,
                    "total_episodes": count,
                    "total_frames": count * 2,
                    "features": {
                        key: {"shape": [16]} for key in ("observation.state", "action")
                    },
                }
            )
        )
        (meta / "episodes.jsonl").write_text(
            "\n".join(
                json.dumps({"episode_index": i, "length": 2}) for i in range(count)
            )
        )


def test_metadata_audit_counts_pairing_and_missing_tasks(tmp_path):
    source_metadata(tmp_path)
    # The pinned source uses a bare task name for some randomized folders.
    directory, (suffix, _) = list(SPLITS.items())[1]
    (tmp_path / directory / ("click_bell" + suffix)).rename(
        tmp_path / directory / "click_bell"
    )
    directory, (suffix, _) = list(SPLITS.items())[0]
    (tmp_path / directory / ("click_bell" + suffix)).rename(
        tmp_path / directory / "click_bell-piper_clean_50-50"
    )
    audit = audit_source(tmp_path, ["click_bell"])
    assert audit["counts"] == {"episodes": 550, "frames": 1100}
    assert len(audit["rows"]) == 2
    assert all(len(row["sha256"]["info.json"]) == 64 for row in audit["rows"])
    with pytest.raises(ValueError, match="Missing requested tasks"):
        audit_source(tmp_path, ["click_bell", "open_microwave"])
    with pytest.raises(ValueError, match="50-task"):
        audit_source(tmp_path)
    next(tmp_path.glob("*/click*/meta/episodes.jsonl")).write_text(
        json.dumps({"episode_index": 1, "length": 2})
    )
    with pytest.raises(ValueError, match="Unexpected source metadata"):
        audit_source(tmp_path, ["click_bell"])


def observation(x):
    return {
        "endpose": {
            "left_endpose": [x, 0, 0, 1, 0, 0, 0],
            "left_gripper": 0.5,
            "right_endpose": [0, 0, 0, 1, 0, 0, 0],
            "right_gripper": 1,
        },
        "observation": {
            camera: {"rgb": np.full((480, 640, 3), x, dtype=np.uint8)}
            for camera in CAMERAS
        },
    }


def test_expert_targets_and_images_stay_aligned_through_conversion():
    class Writer:
        def __init__(self):
            self.values = []

        def append_data(self, rgb):
            self.values.append(int(rgb[0, 0, 0]))

    writers = {camera: Writer() for camera in CAMERAS.values()}
    recorder = EpisodeRecorder(writers)
    for x in (1, 2, 3):
        recorder.append(observation(x))
    states, actions = convert_vectors(recorder.states, recorder.actions)
    np.testing.assert_array_equal(states[:, 0], [1, 2])
    np.testing.assert_array_equal(actions[:, 0], [2, 3])
    assert all(writer.values == [1, 2] for writer in writers.values())


@pytest.mark.parametrize("kind", ["quaternion", "gripper", "image"])
def test_invalid_observations_are_rejected_before_writing(kind):
    sample = observation(1)
    if kind == "quaternion":
        sample["endpose"]["left_endpose"][3] = 0
    elif kind == "gripper":
        sample["endpose"]["left_gripper"] = 2
    else:
        sample["observation"]["head_camera"]["rgb"] = np.zeros(
            (3, 3, 3), dtype=np.uint8
        )
    recorder = EpisodeRecorder({})
    with pytest.raises(ValueError):
        recorder.append(sample)
    assert recorder.pending is None
