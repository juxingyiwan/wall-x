import json
from types import SimpleNamespace

import pytest

from wall_x.data.backends.lerobot.build import load_lerobot_normalizers


@pytest.mark.parametrize("repo_id", ["libero_all", "/legacy/data/libero"])
def test_physical_root_is_not_used_as_parameter_key(tmp_path, repo_id):
    path = tmp_path / "stats.json"
    vector = {"mean": [0], "std": [1], "q01": [-1], "q99": [1]}
    path.write_text(
        json.dumps({"norm_stats": {"action": vector, "observation.state": vector}})
    )
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            norm_stats_path=str(path),
            key_mappings={"action": "action", "state": "observation.state"},
            lerobot_config={
                "repo_id": repo_id,
                "root": str(tmp_path / ".cache/data.v2"),
            },
        ),
        task=SimpleNamespace(dof_config={"joint": 1}, agent_pos_config={"joint": 1}),
        action_dim=1,
        propri_dim=1,
    )
    action, propri, _, name = load_lerobot_normalizers(cfg)
    assert name == repo_id
    assert repo_id in action.delta and repo_id in propri.delta
    cfg.data.lerobot_config["repo_id"] = str(tmp_path / ".cache/data.v2")
    with pytest.raises(ValueError, match="normalizer key"):
        load_lerobot_normalizers(cfg)
