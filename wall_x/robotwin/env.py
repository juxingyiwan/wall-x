"""RoboTwin chunk environment and spawned pipe worker, matching LIBERO's lifecycle.

No model, torch, websocket or listening socket is needed in the worker.
"""

import importlib
import os
import random
import subprocess
import sys
import traceback
from multiprocessing.connection import Connection
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from wall_x.robotwin.codec import observation_to_payload, relative_to_pose
from wall_x.robotwin.panel import observation_digest
from wall_x.robotwin.simulator import prepare_simulator


class RobotwinEnv:
    def __init__(self, config):
        self.config = config
        self.env = None
        self.task = None
        self.opened = False
        self.video = None
        self.anchor = None

    def _select_task(self, task):
        self._close_episode()
        if task != self.task:
            self.env, self.sim_config = prepare_simulator(
                self.config["robotwin_root"],
                task,
                self.config["task_config"],
                full_cameras=True,
            )
            self.task = task

    def _setup(self, seed, index):
        self.opened = True
        self.env.setup_demo(
            now_ep_num=index, seed=seed, is_test=True, **self.sim_config
        )
        if self.task == "put_object_cabinet":
            # Upstream play_once initializes success-check state before moving.
            # Frozen panels skip expert replay, so initialize it on every reset.
            from envs.utils import ArmTag

            pose = self.env.object.get_pose().p
            self.env.arm_tag = ArmTag("right" if pose[0] > 0 else "left")
            self.env.origin_z = float(pose[2])

    def check_scene(self, request):
        self._select_task(request["task"])
        from envs.utils.create_actor import UnStableError
        from generate_episode_instructions import generate_episode_descriptions

        try:
            self._setup(request["seed"], request["index"])
            initial_digest = observation_digest(self._observation())
            info = self.env.play_once()
            valid = bool(self.env.plan_success and self.env.check_success())
        except UnStableError:
            return {"valid": False}
        finally:
            self._close_episode()
        if not valid:
            return {"valid": False}
        state = random.getstate()
        try:
            random.seed(request["seed"])
            descriptions = generate_episode_descriptions(
                request["task"],
                [info["info"]],
                self.config["instruction_count"],
            )
        finally:
            random.setstate(state)
        instruction = str(
            np.random.default_rng(request["seed"]).choice(
                descriptions[0][self.config["instruction_type"]],
            )
        )
        return {
            "valid": True,
            "instruction": instruction,
            "expert_validated": True,
            "initial_observation_sha256": initial_digest,
        }

    def _observation(self):
        observation = self.env.get_obs()
        payload = observation_to_payload(observation)
        self.anchor = payload["state"].copy()
        return payload

    def reset_episode(self, request):
        self._select_task(request["task"])
        self._setup(request["seed"], request["index"])
        self.env.set_instruction(instruction=request["instruction"])
        output = Path(self.config["output"])
        output.mkdir(parents=True, exist_ok=True)
        self.video = imageio.get_writer(output / request["video"], fps=15)
        observation = self._observation()
        self.video.append_data(observation["face_view"])
        return {"obs": observation, "success": bool(self.env.eval_success)}

    def execute_chunk(self, actions):
        # All poses use the observation at this chunk boundary as their anchor.
        chunk = relative_to_pose(actions, self.anchor)
        start = self.env.take_action_cnt
        for action in chunk:
            if self.env.eval_success or self.env.take_action_cnt >= self.env.step_lim:
                break
            self.video.append_data(
                self.env.get_obs()["observation"]["head_camera"]["rgb"]
            )
            self.env.take_action(action, action_type="ee")
        steps = int(self.env.take_action_cnt - start)
        success = bool(self.env.eval_success)
        done = success or self.env.take_action_cnt >= self.env.step_lim
        if steps <= 0 and not done:
            raise RuntimeError(
                "RoboTwin simulator made no progress while executing actions"
            )
        return {
            "obs": None if done else self._observation(),
            "done": done,
            "success": success,
            "steps": steps,
        }

    def _close_episode(self):
        try:
            if self.video is not None:
                self.video.close()
        finally:
            self.video = None
            if self.opened:
                self.opened = False
                self.env.close_env()

    def finalize_episode(self, unused=None):
        self._close_episode()

    def shutdown(self):
        self._close_episode()


def _worker_main(reader, writer, config, env_factory):
    env = None
    try:
        env = env_factory(config)
        writer.send(("ok", None))
        while True:
            try:
                command, value = reader.recv()
            except EOFError:
                break
            if command == "shutdown":
                env.shutdown()
                env = None
                writer.send(("ok", None))
                break
            if command not in {
                "check_scene",
                "reset_episode",
                "execute_chunk",
                "finalize_episode",
            }:
                raise ValueError(f"Unknown simulator command: {command}")
            writer.send(("ok", getattr(env, command)(value)))
    except Exception:
        try:
            writer.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if env is not None:
            try:
                env.shutdown()
            except Exception:
                pass
        reader.close()
        writer.close()


class SubprocEnvHandle:
    """Model-parent-controlled worker; optional Python executable isolates dependencies."""

    def __init__(self, config, *, python=None, timeout=600, env_factory=None):
        # Fresh interpreter avoids both inherited CUDA and multiprocessing.spawn's
        # copied sys.path, which otherwise leaks model-venv packages into the sim.
        child_read, parent_write = os.pipe()
        parent_read, child_write = os.pipe()
        self.reader = Connection(parent_read, writable=False)
        self.writer = Connection(parent_write, readable=False)
        self.timeout = timeout
        self.pending = True
        environment = dict(os.environ)
        # Load only this Wall-X package in the selected interpreter. Adding its
        # parent directory would expose the entire model site-packages when
        # running an installed wheel, overriding the simulator's dependencies.
        command = [
            str(python or sys.executable),
            str(Path(__file__).resolve().with_name("_worker.py")),
            "--read-fd",
            str(child_read),
            "--write-fd",
            str(child_write),
        ]
        if env_factory:
            command.extend(["--factory", env_factory])
        try:
            self.process = subprocess.Popen(
                command,
                pass_fds=(child_read, child_write),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=sys.stderr,
            )
        except Exception:
            self.reader.close()
            self.writer.close()
            raise
        finally:
            os.close(child_read)
            os.close(child_write)
        try:
            self.writer.send(config)
            self.receive()
        except Exception:
            self.shutdown()
            raise

    def submit(self, command, value=None):
        if self.pending:
            raise RuntimeError("A simulator command is already pending")
        self.writer.send((command, value))
        self.pending = True

    def receive(self):
        if not self.pending:
            raise RuntimeError("No simulator command is pending")
        try:
            if not self.reader.poll(self.timeout):
                raise TimeoutError("RoboTwin simulator worker timed out")
            status, value = self.reader.recv()
        except EOFError as error:
            raise RuntimeError(
                f"RoboTwin worker exited (exitcode={self.process.poll()})"
            ) from error
        finally:
            self.pending = False
        if status != "ok":
            raise RuntimeError(f"RoboTwin worker failed:\n{value}")
        return value

    def call(self, command, value=None):
        self.submit(command, value)
        return self.receive()

    def shutdown(self):
        try:
            if self.process.poll() is None and not self.pending:
                self.submit("shutdown")
                if self.reader.poll(5):
                    self.receive()
        except (OSError, RuntimeError, EOFError):
            pass
        finally:
            if self.process.poll() is None:
                try:
                    self.process.wait(5)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
            self.reader.close()
            self.writer.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-fd", type=int, required=True)
    parser.add_argument("--write-fd", type=int, required=True)
    parser.add_argument("--factory", default="wall_x.robotwin.env:RobotwinEnv")
    args = parser.parse_args()
    reader = Connection(args.read_fd, writable=False)
    writer = Connection(args.write_fd, readable=False)
    module, name = args.factory.split(":")
    _worker_main(
        reader, writer, reader.recv(), getattr(importlib.import_module(module), name)
    )


if __name__ == "__main__":
    main()
