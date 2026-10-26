"""Collect one successful independent expert episode for the full-data route."""

import argparse
import json
import random
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

CAMERAS = {
    "head_camera": "face_view",
    "left_camera": "left_wrist_view",
    "right_camera": "right_wrist_view",
}


def observation_frame(observation):
    pose = observation["endpose"]
    vector = np.concatenate(
        (
            pose["left_endpose"],
            [pose["left_gripper"]],
            pose["right_endpose"],
            [pose["right_gripper"]],
        )
    ).astype(np.float32)
    if vector.shape != (16,) or not np.isfinite(vector).all():
        raise ValueError("Expected a finite 16D EEF observation")
    for start in (0, 8):
        if abs(np.linalg.norm(vector[start + 3 : start + 7]) - 1) > 1e-3:
            raise ValueError("Expected normalized WXYZ quaternions")
        if not 0 <= vector[start + 7] <= 1:
            raise ValueError("Expected grippers in [0, 1]")
    views = {}
    for source, target in CAMERAS.items():
        rgb = np.asarray(observation["observation"][source]["rgb"])
        if rgb.dtype != np.uint8 or rgb.shape != (480, 640, 3):
            raise ValueError("Expected three 640x480 uint8 RGB views")
        views[target] = rgb.copy()
    return vector, views


class EpisodeRecorder:
    """Pair each observation with the next sampled pose, retaining its RGB."""

    def __init__(self, writers):
        self.writers = writers
        self.pending = None
        self.states, self.actions = [], []

    def append(self, observation):
        current = observation_frame(observation)
        if self.pending is not None:
            state, views = self.pending
            self.states.append(state)
            self.actions.append(current[0])
            for camera, writer in self.writers.items():
                writer.append_data(views[camera])
        self.pending = current


def collect(robotwin_root, task, seed, output, max_attempts=100):
    # Imports stay here so --help and CPU tests do not initialize SAPIEN/CUDA.
    import imageio.v2 as imageio

    from wall_x.robotwin.simulator import prepare_simulator

    if seed < 0 or max_attempts <= 0:
        raise ValueError("Use a nonnegative seed and a positive attempt limit")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new episode directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    env, config = prepare_simulator(
        robotwin_root, task, "demo_clean", full_cameras=True
    )
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    config["save_freq"] = 15
    rejected = []
    for candidate in range(seed, seed + max_attempts):
        print(f"Expert seed attempt {candidate}", flush=True)
        with tempfile.TemporaryDirectory(
            prefix=".wallx-dev-", dir=output.parent
        ) as tmp:
            episode = Path(tmp) / "episode"
            episode.mkdir()
            with ExitStack() as stack:
                writers = {
                    camera: stack.enter_context(
                        imageio.get_writer(
                            episode / f"{camera}.mp4",
                            fps=50,
                            codec="libx264",
                            ffmpeg_params=["-threads", "2"],
                        )
                    )
                    for camera in CAMERAS.values()
                }
                recorder = EpisodeRecorder(writers)

                def capture():
                    recorder.append(env.get_obs())
                    env.FRAME_IDX += 1

                env._take_picture = capture
                try:
                    env.setup_demo(now_ep_num=0, seed=candidate, is_test=True, **config)
                    info = env.play_once()
                    valid = bool(env.plan_success and env.check_success())
                except UnStableError as error:
                    print(f"Rejected seed {candidate}: {error}", flush=True)
                    valid = False
                finally:
                    env.close_env()
            if not valid or not recorder.states:
                rejected.append(candidate)
                continue
            random.seed(candidate)
            descriptions = generate_episode_descriptions(task, [info["info"]], 10)
            instruction = str(
                np.random.default_rng(candidate).choice(descriptions[0]["seen"])
            )
            np.savez(
                episode / "vectors.npz", state=recorder.states, action=recorder.actions
            )
            record = {
                "task": task,
                "seed": candidate,
                "rejected_seeds": rejected,
                "frames": len(recorder.states),
                "fps": 50,
                "instruction": instruction,
                "task_config": "demo_clean",
                "expert_success": True,
                "is_test": True,
                "save_freq_simulation_steps": 15,
                "contract": "state[t], action[t]=next sampled pose; WXYZ; RGB from state[t]; 50 Hz file timebase",
            }
            (episode / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
            episode.rename(output)
            return record
    raise RuntimeError(
        f"No valid development episode in {max_attempts} attempts: {rejected}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(collect(**vars(args))), flush=True)


if __name__ == "__main__":
    main()
