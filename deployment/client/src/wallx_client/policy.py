"""One policy contract for in-process and WebSocket inference.

Backends only own connection/model resources and payload exchange. Ordinary
and RTC execution use the same request association, validation and result clock.
This module never transforms trajectories or owns robot hardware.
"""

import copy
import math
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from .contracts import InferenceTiming, Observation, Prediction, ProtocolError
from .protocol import build_request, decode_actions, validate_metadata


class PolicyConnection(ABC):
    def __init__(self, config, *, rtc=None):
        self.config, self.rtc = config, rtc
        self.metadata = None
        self._busy = threading.Lock()

    def _metadata(self, metadata):
        checked = validate_metadata(metadata, self.config)
        if self.rtc is not None and self.rtc.enabled:
            from .rtc import validate_capability

            try:
                validate_capability(metadata.get("rtc"), self.config, self.rtc)
            except (TypeError, ValueError) as exc:
                raise ProtocolError("Malformed RTC capability metadata") from exc
            checked = {**checked, "rtc": metadata["rtc"]}
        self.metadata = copy.deepcopy(checked)
        return self.metadata

    def _rtc_request(self, value, generation):
        enabled = self.rtc is not None and self.rtc.enabled
        if value is None:
            if not enabled:
                return None
            from .rtc import request

            return request(self.rtc, generation, 0)
        if not enabled:
            raise ProtocolError("RTC request supplied while RTC is disabled")
        fields = {
            "version",
            "enabled",
            "epoch",
            "plan_id",
            "model_dt_s",
            "prefix",
            "committed_steps",
            "guidance_horizon",
            "max_guidance",
            "guidance_mode",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ProtocolError("Invalid RTC v1 request fields")
        if (
            value["version"] != 1
            or value["enabled"] is not True
            or any(
                type(value[k]) is not int or value[k] < 0
                for k in ("epoch", "plan_id", "committed_steps", "guidance_horizon")
            )
        ):
            raise ProtocolError("Invalid RTC version, enabled flag or action indices")
        if value["epoch"] != generation:
            raise ProtocolError("RTC request epoch must match the controller generation")
        if value["guidance_mode"] != self.rtc.guidance_mode:
            raise ProtocolError("RTC request guidance_mode must match the negotiated configuration")
        dt = value["model_dt_s"]
        weight = value["max_guidance"]
        if type(dt) not in (int, float) or not math.isclose(dt, self.rtc.model_dt_s, abs_tol=1e-8):
            raise ProtocolError("RTC request model timebase mismatch")
        if type(weight) not in (int, float) or not math.isfinite(weight) or not 0 < weight <= 20:
            raise ProtocolError("RTC max_guidance must be in (0,20]")
        committed, horizon = value["committed_steps"], value["guidance_horizon"]
        if value["prefix"] is None:
            if committed or horizon:
                raise ProtocolError("RTC bootstrap cannot commit a prefix")
        else:
            try:
                prefix = np.asarray(value["prefix"])
                if prefix.dtype.kind not in "biuf" or prefix.dtype.hasobject:
                    raise ProtocolError("Unsafe or unsupported RTC prefix dtype")
                prefix = prefix.astype(np.float32, copy=False)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ProtocolError("RTC prefix must contain numeric values") from exc
            # Emit guidance-length prefixes; still accept full-horizon v1
            # requests from existing callers while both endpoints are updated.
            if (
                prefix.shape not in ((horizon, 7), (self.config.horizon, 7))
                or not np.isfinite(prefix).all()
            ):
                raise ProtocolError("RTC prefix shape/values invalid")
            if not 1 <= committed < horizon < self.config.horizon:
                raise ProtocolError("RTC needs committed, tapered and free actions")
        # The model and caller must not be able to rewrite the ACK expectation.
        checked = copy.deepcopy(value)
        if isinstance(checked["prefix"], np.ndarray):
            checked["prefix"] = checked["prefix"].tolist()
        return checked

    def predict(
        self, obs: Observation, instruction: str, generation: int = 0, *, rtc_request=None
    ) -> Prediction:
        if not self._busy.acquire(blocking=False):
            raise ProtocolError("Only one in-flight request is allowed")
        try:
            if type(generation) is not int or generation < 0:
                raise ProtocolError("Controller generation must be a nonnegative integer")
            if not isinstance(instruction, str):
                raise ProtocolError("Policy instruction must be a string")
            obs.validate_payload(self.config.camera_names)
            sent_rtc = self._rtc_request(rtc_request, generation)
            self._ensure_open()
            prepared = time.monotonic()
            request_id, body = build_request(obs, instruction, self.config, copy.deepcopy(sent_rtc))
            payload = self._encode(body)
            started = time.monotonic()
            response = self._exchange(payload, request_id)
            actions = decode_actions(response, request_id, self.config)
            if sent_rtc is not None:
                from .rtc import validate_ack

                validate_ack(response.get("rtc"), sent_rtc)
            elif response.get("rtc") is not None:
                raise ProtocolError("Unexpected RTC acknowledgement for an ordinary request")
            server_timing = response.get("server_timing", {})
            if not isinstance(server_timing, dict):
                raise ProtocolError("Policy server_timing must be a mapping")
            server_timing = copy.deepcopy(server_timing)
            ack = copy.deepcopy(response.get("rtc"))
            received = time.monotonic()
            timing = InferenceTiming(prepared, started, received)
            return Prediction(
                request_id=request_id,
                generation=generation,
                actions=actions,
                observation=obs,
                received_monotonic=received,
                roundtrip_s=timing.call_s,
                server_timing=server_timing,
                rtc=ack,
                timing=timing,
            )
        finally:
            self._busy.release()

    @abstractmethod
    def open(self) -> dict:
        """Open resources and validate peer metadata before hardware is enabled."""

    @abstractmethod
    def _ensure_open(self):
        """Reuse the current session or perform the backend's existing open logic."""

    def _encode(self, body):
        return body

    @abstractmethod
    def _exchange(self, payload, request_id):
        """Return a decoded mapping, or fail without returning a late prediction."""

    @abstractmethod
    def close(self) -> None:
        """Close backend resources; hardware shutdown belongs to the controller."""

    def close_with_status(self, timeout_s):
        """Backend resource cleanup; local models report their inner worker.

        The controller bounds this call even for third-party implementations.
        Socket closure does not imply a remote model service has terminated.
        """
        self.close()
        return {
            "kind": "connection",
            "owner": "policy_transport",
            "status": "closed",
            "timed_out": False,
        }

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
