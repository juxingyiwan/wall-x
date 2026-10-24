"""Common inference adapter abstraction.

The adapter owns all architecture-specific model setup and exposes a single
``predict_batch`` entry point. Environment drivers pass payloads through without
interpreting the env-adapter schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from wall_x._vendor.harrix.eval_config import EvalConfig


class BaseInferAdapter(ABC):
    @abstractmethod
    def __init__(self, cfg: EvalConfig) -> None:
        """Load processors, model config, checkpoints, and normalizers.

        Subclasses should reject unsupported ``cfg.model.action_mode`` values
        during construction.
        """

    @property
    @abstractmethod
    def chunk_horizon(self) -> int:
        """Number of action steps returned by each ``predict_batch`` call."""

    @property
    @abstractmethod
    def action_mode(self) -> str:
        """Configured inference algorithm, fixed at construction time."""

    def model_rtc_capability(self):
        """Model support only; excludes environment units, dimensions and rates."""
        from wall_x.model.core.action.rtc_contract import ModelRTCCapability

        return ModelRTCCapability()

    def supports_rtc(self) -> bool:
        return self.model_rtc_capability().supported

    @abstractmethod
    def predict_batch(self, payloads: list[dict]) -> list[np.ndarray]:
        """Run one batched inference call and return one chunk per payload.

        ``payloads[i]`` schema is defined by the env-adapter pair:
            {
                "observation": dict[str, np.ndarray],
                    # Env-defined ndarray bundle, for example LIBERO:
                    # {"eef_pos":(3,), "eef_axisangle":(3,), "gripper":(1,),
                    #  "face_view":(H,W,3), "wrist_view":(H,W,3)}
                "instruction": str,
                "noise":       np.ndarray | None,
                    # Flow may pass (chunk_horizon, action_dim); other modes
                    # may leave this as None.
                "rtc_condition": RTCCondition | None,
                    # Optional capability: normalized model-space [1,H,D]
                    # prefix and [1,H,1] weights, not a serving wire request.
                    # Use model.adapters.rtc_preprocessing for raw actions.
            }
        """
