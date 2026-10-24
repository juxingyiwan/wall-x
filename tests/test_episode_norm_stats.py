import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location(
    "episode_stats",
    Path(__file__).resolve().parents[1] / "scripts/compute_norm_stats.py",
)
stats = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stats
spec.loader.exec_module(stats)


def test_held_out_outliers_do_not_change_stats_and_root_can_move(tmp_path):
    root = tmp_path / ".cache/data.v1"
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text('{"total_episodes":4}')
    values = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [999.0], [999.0]]
    table = pa.table(
        {
            "episode_index": [0, 0, 1, 1, 2, 2, 3, 3],
            "frame_index": [0, 1, 0, 1, 0, 1, 0, 1],
            "observation.state": values,
            "action": values,
        }
    )
    pq.write_table(table, root / "data/chunk-000/file-000.parquet")
    config = {
        "task": {
            "dof_config": {"joint": 1},
            "agent_pos_config": {"joint": 1},
            "action_horizon": 2,
        },
        "data": {
            "lerobot_config": {"repo_id": "libero_all", "root": str(root)},
            "train_test_split": 0.75,
            "key_mappings": {"state": "observation.state", "action": "action"},
        },
        "norm_stats_path": str(tmp_path / "stats.json"),
    }
    ctx = stats.parse_train_config(config)
    assert ctx.data_root == root
    first = stats.compute_norm_stats(root, ctx.output_path, train_ctx=ctx)
    assert first["action"]["mean"] == [3.5]
    payload = json.loads(ctx.output_path.read_text())
    assert payload["episode_split"]["train_episodes"] == [0, 1, 2]
    moved = tmp_path / "moved.v2"
    root.rename(moved)
    second = stats.compute_norm_stats(moved, ctx.output_path, train_ctx=ctx)
    assert first == second


def test_relative_statistics_do_not_cross_episodes():
    states = np.array([[0], [1], [1000], [1001]], dtype=np.float32)
    result = stats.collect_episode_dof_vectors(
        states,
        states + 10,
        np.array([0, 0, 1, 1]),
        {"joint": [0, 1]},
        {"joint_relative": [0, 1]},
        3,
    )
    np.testing.assert_array_equal(
        result["action"]["joint_relative"].ravel(), [10, 10, 11, 10, 10, 11]
    )
