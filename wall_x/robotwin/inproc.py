"""Direct model inference with locally controlled RoboTwin simulator workers."""

import argparse
import base64
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np

from wall_x.robotwin.env import SubprocEnvHandle
from wall_x.robotwin.panel import (
    is_prevalidated_scene,
    observation_digest,
    scene_identity,
)

FIVE_TASKS = [
    "open_microwave",
    "click_bell",
    "stack_blocks_three",
    "place_shoe",
    "put_object_cabinet",
]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def capture_rng():
    import torch

    def encode(value):
        return base64.b64encode(value.cpu().numpy().tobytes()).decode("ascii")

    numpy_state = np.random.get_state()
    return {
        "torch": encode(torch.get_rng_state()),
        "cuda": (
            [encode(state) for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }


def restore_rng(state):
    import torch

    def decode(value):
        return torch.from_numpy(
            np.frombuffer(base64.b64decode(value), dtype=np.uint8).copy()
        )

    def tuples(value):
        return (
            tuple(tuples(item) for item in value) if isinstance(value, list) else value
        )

    if len(state["cuda"]) != torch.cuda.device_count():
        raise ValueError("Resume requires the same number of visible CUDA devices")
    torch.set_rng_state(decode(state["torch"]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all([decode(value) for value in state["cuda"]])
    random.setstate(tuples(state["python"]))
    algorithm, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state(
        (algorithm, np.asarray(keys, dtype=np.uint32), position, gaussian, cached)
    )


def checkpoint_identity(directory, train_config):
    from wall_x.checkpoint_weights import resolve_checkpoint_weights

    weights = resolve_checkpoint_weights(directory)
    directory = weights.parent
    stats = [
        directory / name
        for name in [
            "norm_stats.json",
            "normalizer_action.pth",
            "normalizer_propri.pth",
        ]
        if (directory / name).is_file()
    ]
    if not stats:
        raise FileNotFoundError(
            "Evaluation requires checkpoint-local normalization files"
        )
    stat = weights.stat()
    return {
        "directory": str(directory),
        "weights": weights.name,
        "weights_size": stat.st_size,
        "weights_mtime_ns": stat.st_mtime_ns,
        "train_config_sha256": hashlib.sha256(
            Path(train_config).read_bytes()
        ).hexdigest(),
        "normalization_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stats
        },
    }


def run_frame(adapter, workers, scenes, execute_steps):
    """Same reset -> batch predict -> execute chunk cycle as the LIBERO driver."""
    started = time.monotonic()
    for worker, scene in zip(workers, scenes):
        worker.submit("reset_episode", scene)
    initial = [worker.receive() for worker in workers[: len(scenes)]]
    observations = [row["obs"] for row in initial]
    for scene, observation in zip(scenes, observations):
        if (
            is_prevalidated_scene(scene)
            and observation_digest(observation) != scene["initial_observation_sha256"]
        ):
            raise RuntimeError(
                f"Initial observation changed for fixed scene {scene['task']}/{scene['seed']}"
            )
    results = [
        {
            "success": bool(row.get("success", False)),
            "done": bool(row.get("success", False)),
            "steps": 0,
            "chunks": 0,
        }
        for row in initial
    ]
    while True:
        active = [i for i, row in enumerate(results) if not row["done"]]
        if not active:
            break
        payloads = [
            {
                "observation": observations[i],
                "instruction": scenes[i]["instruction"],
                "noise": None,
            }
            for i in active
        ]
        chunks = adapter.predict_batch(payloads)
        if len(chunks) != len(active):
            raise ValueError("Model returned the wrong number of action chunks")
        for i, chunk in zip(active, chunks):
            workers[i].submit("execute_chunk", chunk[:execute_steps])
        for i in active:
            row = workers[i].receive()
            if row["steps"] <= 0 and not row["done"]:
                raise RuntimeError("Simulator made no progress")
            observations[i] = row["obs"]
            results[i]["steps"] += row["steps"]
            results[i]["chunks"] += 1
            results[i].update(done=bool(row["done"]), success=bool(row["success"]))
    for worker in workers[: len(scenes)]:
        worker.submit("finalize_episode")
    for worker in workers[: len(scenes)]:
        worker.receive()
    return [
        {
            **scene_identity(scene),
            "video": scene["video"],
            "success": result["success"],
            "steps": result["steps"],
            "chunks": result["chunks"],
            "seconds": (time.monotonic() - started) / len(scenes),
        }
        for scene, result in zip(scenes, results)
    ]


def summarize(report):
    rates, successes, attempted = [], 0, 0
    for task in report["tasks"].values():
        rows = task["episodes"]
        count = sum(row["success"] for row in rows)
        task["success_rate"] = count / len(rows) if rows else None
        if rows:
            rates.append(task["success_rate"])
        successes += count
        attempted += len(rows)
    report["overall"] = {
        "attempted": attempted,
        "successes": successes,
        "success_rate": successes / attempted if attempted else None,
        "macro_success_rate": float(np.mean(rates)) if rates else None,
    }


def run(args, cfg, identity, *, adapter_factory=None, worker_factory=SubprocEnvHandle):
    args.output = Path(args.output).resolve()
    result_path = args.output / "results.json"
    protocol = {
        "driver": "in_process",
        "tasks": args.tasks,
        "task_config": args.task_config,
        "start_seed": args.start_seed,
        "seed": args.seed,
        "instruction_type": args.instruction_type,
        "instruction_count": args.instruction_count,
        "action_horizon": cfg.model.action_horizon,
        "execute_steps": args.execute_steps,
        "num_workers": args.num_workers,
        "flow_steps": 10,
        "norm_key": cfg.model.norm_key,
    }
    if not 1 <= protocol["execute_steps"] <= cfg.model.action_horizon:
        raise ValueError(
            "execute-steps must be between 1 and the checkpoint action horizon"
        )
    if args.resume and result_path.exists():
        report = json.loads(result_path.read_text())
        if report["checkpoint"] != identity or report["protocol"] != protocol:
            raise ValueError("Resume checkpoint or evaluation protocol differs")
    elif result_path.exists():
        raise FileExistsError(
            "Output contains results.json; use --resume or a new output directory"
        )
    else:
        report = {
            "checkpoint": identity,
            "protocol": protocol,
            "tasks": {},
            "rng_state": None,
        }
    for task in report["tasks"].values():
        if len(task["episodes"]) > args.episodes:
            raise ValueError("Cannot reduce an existing evaluation denominator")
        for row in task["episodes"]:
            if not (args.output / row["video"]).is_file():
                raise FileNotFoundError("Missing resumed rollout video")
    manifest = None
    if args.scene_manifest:
        if len(args.tasks) != 1:
            raise ValueError("scene-manifest requires exactly one task")
        fields = {
            "task": args.tasks[0],
            "task_config": args.task_config,
            "start_seed": args.start_seed,
            "instruction_type": args.instruction_type,
            "instruction_count": args.instruction_count,
        }
        manifest = (
            json.loads(args.scene_manifest.read_text())
            if args.scene_manifest.exists()
            else {**fields, "scenes": []}
        )
        if any(manifest[key] != value for key, value in fields.items()):
            raise ValueError("Scene manifest protocol differs")
        completed = report["tasks"].get(args.tasks[0], {}).get("episodes", [])
        if [scene_identity(row) for row in completed] != [
            scene_identity(row) for row in manifest["scenes"][: len(completed)]
        ]:
            raise ValueError("Resumed episodes differ from scene manifest")
    from wall_x._vendor.harrix.utils.seed import set_seed_everywhere

    set_seed_everywhere(args.seed)
    if adapter_factory is None:
        from wall_x.robotwin.adapter import RobotwinInferAdapter

        adapter_factory = RobotwinInferAdapter
    adapter = adapter_factory(cfg)
    if report["rng_state"] is not None:
        restore_rng(report["rng_state"])
    else:
        report["rng_state"] = capture_rng()
    workers = []
    worker_config = {
        "robotwin_root": str(args.robotwin_root.resolve()),
        "output": str(args.output),
        "task_config": args.task_config,
        "instruction_count": args.instruction_count,
        "instruction_type": args.instruction_type,
    }
    atomic_json(result_path, report)
    try:
        for _ in range(args.num_workers):
            workers.append(
                worker_factory(
                    worker_config, python=args.sim_python, timeout=args.worker_timeout
                )
            )
        for task in args.tasks:
            record = report["tasks"].setdefault(
                task, {"episodes": [], "expert_rejected_seeds": []}
            )
            episodes, rejected = record["episodes"], record["expert_rejected_seeds"]
            seed = (
                max(
                    [args.start_seed - 1, *[row["seed"] for row in episodes], *rejected]
                )
                + 1
            )
            while len(episodes) < args.episodes:
                scenes = []
                count = min(args.num_workers, args.episodes - len(episodes))
                for index in range(count):
                    episode_index = len(episodes) + index
                    expected = (
                        manifest["scenes"][episode_index]
                        if manifest and episode_index < len(manifest["scenes"])
                        else None
                    )
                    if expected:
                        seed = expected["seed"]
                    if expected and is_prevalidated_scene(expected):
                        # Expert validity was established before policy execution.
                        # Re-running a stochastic planner here changes the panel
                        # according to previous policy trajectories. Instead,
                        # verify the frozen initial observation in run_frame.
                        scene = {
                            "task": task,
                            "index": episode_index,
                            **scene_identity(expected),
                            "video": f"{task}-{seed}.mp4",
                        }
                        scenes.append(scene)
                        seed += 1
                        continue
                    while (
                        expected is not None
                        or seed < args.start_seed + args.episodes * 10
                    ):
                        scene = {"task": task, "seed": seed, "index": episode_index}
                        checked = workers[index].call("check_scene", scene)
                        if checked["valid"]:
                            break
                        if expected:
                            raise RuntimeError(
                                f"Fixed scene {task}/{seed} is no longer expert-valid"
                            )
                        rejected.append(seed)
                        seed += 1
                    else:
                        raise RuntimeError(
                            f"Insufficient expert-valid seeds for {task}"
                        )
                    panel_row = {"seed": seed, "instruction": checked["instruction"]}
                    if expected and panel_row != {
                        key: expected[key] for key in ("seed", "instruction")
                    }:
                        raise RuntimeError("Fixed scene instruction changed")
                    if expected is None and is_prevalidated_scene(checked):
                        panel_row.update(
                            expert_validated=True,
                            initial_observation_sha256=checked[
                                "initial_observation_sha256"
                            ],
                        )
                    if manifest is not None and expected is None:
                        manifest["scenes"].append(panel_row)
                        atomic_json(args.scene_manifest, manifest)
                    scene.update(**panel_row, video=f"{task}-{seed}.mp4")
                    scenes.append(scene)
                    seed += 1
                rows = run_frame(adapter, workers, scenes, protocol["execute_steps"])
                episodes.extend(rows)
                report["rng_state"] = capture_rng()
                summarize(report)
                atomic_json(result_path, report)
                for row in rows:
                    print(json.dumps({"task": task, **row}), flush=True)
    finally:
        for worker in workers:
            worker.shutdown()
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--train-config-path")
    parser.add_argument("--norm-key", default="robotwin2_full27500")
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=FIVE_TASKS)
    parser.add_argument(
        "--task-config", choices=["demo_clean", "demo_randomized"], default="demo_clean"
    )
    parser.add_argument("--episodes", "--num-trials-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10222)
    parser.add_argument("--start-seed", type=int, default=100000)
    parser.add_argument(
        "--instruction-type", choices=["seen", "unseen"], default="seen"
    )
    parser.add_argument("--instruction-count", type=int, default=10)
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=16,
        help="Execute the first N actions before observing again (default: 16 of 32)",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--sim-python",
        type=Path,
        help="Python for simulator subprocesses; defaults to the current interpreter",
    )
    parser.add_argument("--worker-timeout", type=float, default=600)
    parser.add_argument("--scene-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint/config without loading model or simulator",
    )
    args = parser.parse_args(argv)
    if (
        min(
            args.episodes, args.instruction_count, args.num_workers, args.worker_timeout
        )
        <= 0
    ):
        parser.error(
            "episodes, instruction-count, num-workers and worker-timeout must be positive"
        )
    if args.execute_steps is not None and args.execute_steps <= 0:
        parser.error("execute-steps must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("tasks must not contain duplicates")
    args.robotwin_root = args.robotwin_root.resolve()
    args.output = args.output.resolve()
    if args.scene_manifest:
        args.scene_manifest = args.scene_manifest.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    from wall_x._vendor.harrix.eval_config import EvalConfig, autofill_from_checkpoint

    cfg = EvalConfig()
    cfg.model.checkpoint_path = args.checkpoint_path
    cfg.model.train_config_path = args.train_config_path
    cfg.model.norm_key = args.norm_key
    cfg.model.cam_names = ["face_view", "left_wrist_view", "right_wrist_view"]
    cfg = autofill_from_checkpoint(cfg)
    identity = checkpoint_identity(
        cfg.model.checkpoint_path, cfg.model.train_config_path
    )
    # Fail incompatible checkpoints before loading GPU weights or spawning sims.
    from wall_x._vendor.harrix.utils.normalizer import build_normalizers
    from wall_x._vendor.harrix.utils.train_config import (
        load_train_config_with_ckpt_overlay,
        normalize_train_config_for_inference,
    )
    from wall_x.robotwin.adapter import encode_observation

    train = normalize_train_config_for_inference(
        load_train_config_with_ckpt_overlay(
            cfg.model.train_config_path, identity["directory"]
        ),
        cfg.model.train_config_path,
    )
    if (
        train.get("model_type") != "qwen2_5"
        or train["data"].get("lerobot_config", {}).get("adapter") != "robotwin"
    ):
        raise ValueError(
            "Use a RoboTwin Qwen2.5 checkpoint and its matching training config"
        )
    encode_observation(
        {
            "state": np.zeros(20, dtype=np.float32),
            **{cam: np.zeros((1, 1, 3), dtype=np.uint8) for cam in cfg.model.cam_names},
        },
        train,
        cfg.model.action_horizon,
    )
    _, _, resolved_norm_key = build_normalizers(
        identity["directory"], train, cfg.model.norm_key
    )
    cfg.model.norm_key = resolved_norm_key
    if not args.robotwin_root.is_dir():
        raise FileNotFoundError(args.robotwin_root)
    if args.sim_python and not args.sim_python.is_file():
        raise FileNotFoundError(args.sim_python)
    if args.execute_steps and args.execute_steps > cfg.model.action_horizon:
        raise ValueError("execute-steps exceeds checkpoint action horizon")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "checkpoint": identity,
                    "action_horizon": cfg.model.action_horizon,
                    "execute_steps": args.execute_steps,
                    "tasks": args.tasks,
                    "driver": "in_process",
                },
                indent=2,
            )
        )
        return
    logging.basicConfig(level=logging.INFO)
    run(args, cfg, identity)


if __name__ == "__main__":
    main()
