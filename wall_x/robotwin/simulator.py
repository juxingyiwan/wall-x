"""Simulator configuration, isolated from model and serving dependencies."""

import importlib
import os
import sys
from pathlib import Path

import yaml


def prepare_simulator(root, task, task_config, *, full_cameras=False):
    root = Path(root).resolve()
    os.chdir(root)
    for path in (root, root / "description/utils"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    config = yaml.safe_load((root / f"task_config/{task_config}.yml").read_text())
    embodiments = yaml.safe_load(
        (root / "task_config/_embodiment_config.yml").read_text()
    )
    cameras = yaml.safe_load((root / "task_config/_camera_config.yml").read_text())
    if config["embodiment"] != ["aloha-agilex"]:
        raise ValueError(
            "RoboTwin adapter requires the dual-arm aloha-agilex embodiment"
        )
    robot_file = embodiments["aloha-agilex"]["file_path"]
    robot_config = yaml.safe_load((Path(robot_file) / "config.yml").read_text())
    if full_cameras:
        config["camera"].update(
            head_camera_type="Large_D435", wrist_camera_type="Large_D435"
        )
    head = cameras[config["camera"]["head_camera_type"]]
    config.update(
        task_name=task,
        task_config=task_config,
        ckpt_setting="wallx",
        eval_mode=True,
        collect_data=False,
        eval_video_log=False,
        render_freq=0,
        left_robot_file=robot_file,
        right_robot_file=robot_file,
        dual_arm_embodied=True,
        left_embodiment_config=robot_config,
        right_embodiment_config=robot_config,
        head_camera_h=head["h"],
        head_camera_w=head["w"],
    )
    env = getattr(importlib.import_module(f"envs.{task}"), task)()
    if full_cameras:
        from envs.utils.create_actor import UnStableError

        from wall_x.robotwin.expert import (
            install_failed_grasp_guard,
            install_unreachable_pose_guard,
        )

        install_failed_grasp_guard(env, task, UnStableError)
        install_unreachable_pose_guard(env, task, UnStableError)
        original_play = env.play_once

        def checked_play(*args, **kwargs):
            try:
                return original_play(*args, **kwargs)
            except AssertionError as error:
                if str(error) != "target_pose cannot be None for move action.":
                    raise
                raise UnStableError(str(error)) from error

        env.play_once = checked_play
    return env, config
