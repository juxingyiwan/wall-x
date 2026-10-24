"""Native Wall-X clipping must remain exactly as in the ReBot baseline."""

import torch

from wall_x.config.data_config import LeRobotDataConfig
from wall_x.data.backends.lerobot.loader import DataCollator


def test_native_collator_and_schema_are_not_extended_for_robotwin():
    actual = DataCollator._normalize(
        torch.tensor([-6.0, 6.0]), torch.tensor([-2.0, -2.0]), torch.tensor([4.0, 4.0])
    )
    torch.testing.assert_close(actual, torch.tensor([-1.0, 1.0]), atol=0, rtol=0)
    assert not hasattr(LeRobotDataConfig(), "clip_action_targets")
