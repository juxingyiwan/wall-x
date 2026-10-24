"""Optional read-only rollout traces, without changing third-party simulator code."""

import json

import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class RolloutTrace:
    """Capture decoded commands, planner outcomes, and bell contact checks.

    Hooks call each original simulator method exactly once and return its
    unchanged result. Contact summaries cover physics substeps; endpoint poses
    alone cannot establish whether a transient contact happened.
    """

    def __init__(self, env, path):
        self.env, self.path = env, path
        self.restores = []
        self.current = None

    def write(self, record):
        self.file.write(
            json.dumps(record, default=_json_default, allow_nan=False) + "\n"
        )
        self.file.flush()

    def pose(self):
        return np.concatenate(
            (
                self.env.get_arm_pose("left"),
                [self.env.robot.get_left_gripper_val()],
                self.env.get_arm_pose("right"),
                [self.env.robot.get_right_gripper_val()],
            )
        )

    def _hook(self, obj, name, replacement):
        present, previous = name in vars(obj), vars(obj).get(name)
        self.restores.append((obj, name, present, previous))
        setattr(obj, name, replacement)

    def __enter__(self):
        self.file = self.path.open("w")
        try:
            for arm in ("left", "right"):
                original = getattr(self.env.robot, f"{arm}_plan_path")

                def plan(*args, _original=original, _arm=arm, **kwargs):
                    result = _original(*args, **kwargs)
                    if self.current is not None:
                        self.current["planner"][_arm] = {
                            "status": str(result["status"]),
                            "path_steps": len(result.get("position", [])),
                        }
                    return result

                self._hook(self.env.robot, f"{arm}_plan_path", plan)
            initial = {"kind": "initial", "pose": self.pose()}
            if hasattr(self.env, "bell"):
                initial.update(
                    bell_target=self.env.bell.get_contact_point(0)[:3],
                    bell_id=int(self.env.bell_id),
                    required_arm=(
                        "left" if self.env.bell.get_pose().p[0] < 0 else "right"
                    ),
                )
                original_check = self.env.check_success

                def check():
                    result = original_check()
                    if self.current is not None:
                        self.current["physics_checks"] += 1
                        closed = bool(self.env.check_arm_function())
                        self.current["closed_checks"] += int(closed)
                        target = np.asarray(self.env.bell.get_contact_point(0)[:3])
                        for position in self.env.get_gripper_actor_contact_position(
                            "050_bell"
                        ):
                            delta = np.asarray(position) - target
                            self.current["contact_points"] += 1
                            if (
                                self.current["nearest_contact"] is None
                                or np.linalg.norm(delta)
                                < self.current["nearest_contact"]["distance"]
                            ):
                                self.current["nearest_contact"] = {
                                    "delta": delta,
                                    "distance": float(np.linalg.norm(delta)),
                                    "required_gripper_closed": closed,
                                }
                            in_target = bool(
                                np.all(np.abs(delta) < [0.025, 0.025, 0.03])
                            )
                            self.current["target_contacts"] += int(in_target)
                            self.current["closed_target_contacts"] += int(
                                in_target and closed
                            )
                    return result

                self._hook(self.env, "check_success", check)
            self.write(initial)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def record_chunk(self, raw, anchor, decoded, seed):
        self.write(
            {
                "kind": "chunk",
                "step": int(self.env.take_action_cnt),
                "raw": raw,
                "anchor": anchor,
                "decoded": decoded,
                "inference_seed": seed,
            }
        )

    def take_action(self, action):
        self.current = {
            "kind": "action",
            "step": int(self.env.take_action_cnt),
            "command": np.asarray(action).copy(),
            "before": self.pose(),
            "planner": {},
            "physics_checks": 0,
            "closed_checks": 0,
            "contact_points": 0,
            "target_contacts": 0,
            "closed_target_contacts": 0,
            "nearest_contact": None,
        }
        try:
            self.env.take_action(action, action_type="ee")
        finally:
            self.current.update(after=self.pose(), success=bool(self.env.eval_success))
            self.write(self.current)
            self.current = None

    def __exit__(self, *args):
        for obj, name, present, previous in reversed(self.restores):
            if present:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self.restores.clear()
        self.file.close()
