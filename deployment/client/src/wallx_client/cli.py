"""Command-line entry points; importing this module never touches hardware."""

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import (
    check_legacy_continuous,
    load_config,
    require_execution_ready,
    resolve_execution,
    validate_config,
)
from .contracts import ClientError, Observation
from .lifecycle import protect_shutdown
from .recording import Recorder, write_json
from .transport import WallXConnection


def doctor(cfg):
    packages = {}
    for name in (
        "wallx-rebot-client",
        "numpy",
        "websockets",
        "msgpack",
        "PyYAML",
        "opencv-python-headless",
        "lerobot",
        "lerobot_robot_seeed_b601",
        "motorbridge",
        "python-can",
        "rerun-sdk",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    print(
        json.dumps(
            {
                "python": sys.executable,
                "python_version": platform.python_version(),
                "packages": packages,
                "server_uri": cfg.policy.uri,
                "camera_devices_exist": {
                    k: Path(v).exists() for k, v in cfg.robot.camera_devices.items()
                },
                "can_device_exists": Path("/sys/class/net", cfg.robot.can_channel).exists(),
                "camera_mapping_verified": cfg.robot.camera_mapping_verified,
                "checkpoint_sha256_pinned": bool(cfg.policy.checkpoint_sha256),
                "hardware_opened": False,
            },
            indent=2,
        )
    )


def replay(cfg, source, recorder, connection):
    with np.load(source, allow_pickle=False) as data:
        views = {k: data[k].copy() for k in cfg.policy.camera_names}
        state = data["state"].copy()
    now = time.monotonic()
    obs = Observation(state, views, now, time.time(), now)
    obs.validate(cfg.policy.camera_names, float("inf"))
    write_json(recorder.path / "server.json", connection.open())
    prediction = connection.predict(obs, cfg.task)
    recorder.sample(obs)
    recorder.prediction(prediction)
    return {
        "outcome": "replay_complete",
        "roundtrip_s": prediction.roundtrip_s,
        "action_shape": list(prediction.actions.shape),
        "hardware_opened": False,
        "source": str(Path(source).resolve()),
    }


def hardware_session(cfg, mode, recorder, connection):
    from .controller import control_loop, operator_terminal, preflight
    from .hardware.b601 import B601
    from .hardware.homing import HomingTimeout
    from .locks import robot_locks

    if mode == "run":
        require_execution_ready(cfg)
        if not sys.stdin.isatty():
            raise ClientError("run needs an interactive on-site operator terminal")
    import logging

    logging.getLogger("wallx_client.hardware.b601").setLevel(logging.INFO)
    hw = B601(cfg.robot, recorder.event, shutdown=cfg.shutdown)
    report = {"outcome": "running", "policy_commands": 0}
    with robot_locks(cfg.robot.lock_paths):
        try:
            hw.open_observation()
            write_json(recorder.path / "hardware.json", hw.manifest())
            if mode == "capture":
                obs, _ = hw.observe(disabled=True)
                obs.validate(cfg.policy.camera_names, cfg.robot.max_camera_age_s)
                recorder.observation(obs, "observation")
                report.update(outcome="capture_complete", motors_enabled=False)
            else:
                preflight(hw, connection, cfg, recorder)
                if mode == "preflight":
                    report.update(outcome="preflight_complete", motors_enabled=False)
                else:
                    with operator_terminal() as read_key:
                        if cfg.control.startup_home:
                            print(
                                "Startup homing: waiting for arm and gripper arrival.", flush=True
                            )
                        hw.enable(cfg.control)
                        if cfg.control.startup_home:
                            print("Startup homing verified; starting policy control.", flush=True)
                        if cfg.rtc.enabled:
                            from .rtc_queue_controller import control_loop as rtc_loop

                            report = rtc_loop(hw, connection, cfg, recorder, read_key)
                        else:
                            report = control_loop(hw, connection, cfg, recorder, read_key)
        except KeyboardInterrupt:
            report.update(outcome="aborted")
        except (Exception, HomingTimeout) as exc:
            report.update(outcome="fault", error=f"{type(exc).__name__}: {exc}")
            if hasattr(exc, "homing_context"):
                report["homing_diagnostic"] = exc.homing_context
        finally:
            report.setdefault("stop_outcome", report["outcome"])
            with protect_shutdown():
                try:
                    hw.close()
                except BaseException as exc:
                    report.update(outcome="fault", shutdown_error=f"{type(exc).__name__}: {exc}")
                report["shutdown"] = dict(hw.shutdown_report)
    return report


def parse_bool(value):
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")


class _RetiredContinuous(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        try:
            check_legacy_continuous(value, option_string)
        except ValueError as exc:
            parser.error(str(exc))


def build_parser():
    parser = argparse.ArgumentParser(description="Wall-X B601 client (joint_state_v1)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs/b601.example.yml",
    )
    parser.add_argument("--task", help="Override with a full training task instruction")
    parser.add_argument("--output", help="New run directory; existing paths are never reused")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("doctor", help="Inspect configuration/dependencies without opening hardware")
    preview = sub.add_parser("config", help="Print effective configuration")
    sub.add_parser("capture", help="Read cameras/CAN once, save PNG/NPZ; no server or motor enable")
    camera = sub.add_parser(
        "camera-test", help="Test cameras only; no CAN, policy, or motor access"
    )
    camera.add_argument("--duration", type=float, default=60.0, help="Camera test seconds")
    offline = sub.add_parser("replay", help="Infer from an NPZ snapshot without touching hardware")
    offline.add_argument("--observation", required=True)
    sub.add_parser("preflight", help="Read cameras/CAN and infer once; never enable motors")
    run = sub.add_parser("run", help="Enable motors and run an attended episode")
    run.add_argument("--duration", type=float, help="Episode seconds after startup home")
    for command in (run, preview):
        command.add_argument(
            "--max-policy-chunks", type=int, help="Maximum activated action plans per episode"
        )
    for command in (parser, *sub.choices.values()):
        command.add_argument("--executor", choices=("sync", "rtc_fifo"), default=argparse.SUPPRESS)
        command.add_argument(
            "--guidance-mode", choices=("lerobot", "pi_vjp"), default=argparse.SUPPRESS
        )
        command.add_argument(
            "--continuous",
            type=parse_bool,
            action=_RetiredContinuous,
            default=argparse.SUPPRESS,
            help=argparse.SUPPRESS,
        )
        command.add_argument(
            "--rtc",
            type=parse_bool,
            default=argparse.SUPPRESS,
            help="Select guided RTC per request (true/false)",
        )
        for name in ("display_data", "save_rollout"):
            command.add_argument(
                "--" + name.replace("_", "-"),
                "--" + name,
                type=parse_bool,
                nargs="?",
                const=True,
                default=argparse.SUPPRESS,
            )
        command.add_argument("--display-ip", "--display_ip", default=argparse.SUPPRESS)
        command.add_argument(
            "--display-port", "--display_port", type=int, default=argparse.SUPPRESS
        )
    return parser


def apply_cli_overrides(cfg, args):
    """One effective configuration for local and network entry points."""
    if hasattr(args, "executor"):
        if hasattr(args, "rtc") and args.rtc != (args.executor != "sync"):
            raise ValueError("--executor conflicts with --rtc")
        cfg = replace(cfg, control=replace(cfg.control, executor=args.executor))
    if hasattr(args, "guidance_mode"):
        cfg = replace(cfg, rtc=replace(cfg.rtc, guidance_mode=args.guidance_mode))
    if hasattr(args, "rtc"):
        if cfg.control.executor is not None and args.rtc != (cfg.control.executor != "sync"):
            raise ValueError("--rtc conflicts with control.executor; select --executor explicitly")
        cfg = replace(cfg, rtc=replace(cfg.rtc, enabled=args.rtc))
    presentation = {
        name: getattr(args, name)
        for name in ("display_data", "save_rollout", "display_ip", "display_port")
        if hasattr(args, name)
    }
    if presentation:
        cfg = replace(cfg, **presentation)
    if args.task is not None:
        cfg = replace(cfg, task=args.task)
    if args.mode == "run" and args.duration is not None:
        cfg = replace(cfg, control=replace(cfg.control, duration_s=args.duration))
    if getattr(args, "max_policy_chunks", None) is not None:
        cfg = replace(cfg, control=replace(cfg.control, max_policy_chunks=args.max_policy_chunks))
    cfg = resolve_execution(cfg)
    validate_config(cfg)
    return cfg


def main(argv=None, *, connection_factory=None):
    args = build_parser().parse_args(argv)
    recorder = connection = None
    report = {"outcome": "fault"}
    code = 0
    try:
        cfg = load_config(args.config, validate=False)
        cfg = apply_cli_overrides(cfg, args)
        if args.mode == "config":
            print(
                json.dumps(
                    {**cfg.to_dict(), "execution_mode": cfg.execution_mode},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.mode == "doctor":
            doctor(cfg)
            return 0
        if args.mode == "run":
            require_execution_ready(cfg)
            schedule = cfg.execution_mode
            print(f"Execution schedule: {schedule}; motor commands {cfg.control.hz} Hz", flush=True)
        recorder = Recorder(cfg, args.mode, args.output)
        recorder.event(
            "execution_mode_resolved",
            executor=cfg.execution_mode,
            selection="explicit" if cfg.control.executor else "rtc_flag",
            guidance_mode=cfg.rtc.guidance_mode if cfg.rtc.enabled else None,
        )
        print(f"Run directory: {recorder.path}", flush=True)
        if args.mode == "camera-test":
            from .camera_diagnostics import camera_test

            report = camera_test(cfg, args.duration, recorder)
        else:
            recorder.start_telemetry()
            factory = connection_factory or WallXConnection
            connection = factory(cfg.policy, rtc=cfg.rtc)
            if args.mode == "replay":
                report = replay(cfg, args.observation, recorder, connection)
            else:
                report = hardware_session(cfg, args.mode, recorder, connection)
        code = 1 if report["outcome"] == "fault" else 0
    except KeyboardInterrupt:
        report = {"outcome": "aborted"}
        code = 130
    except Exception as exc:
        report = {"outcome": "fault", "error": f"{type(exc).__name__}: {exc}"}
        print(report["error"], file=sys.stderr)
        code = 1
    finally:
        if connection and not report.get("transport_close_attempted"):
            try:
                connection.close()
            except Exception as exc:
                report.setdefault("stop_outcome", report["outcome"])
                report.update(outcome="fault", transport_close_error=str(exc))
                code = 1
        if recorder:
            try:
                recorder.close(report)
            except Exception as exc:
                print(f"Log finalization failed: {exc}", file=sys.stderr)
                code = 1
    print(json.dumps(report, ensure_ascii=False))
    return code
