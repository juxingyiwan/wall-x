import logging
import sys
from types import SimpleNamespace

import pytest

from scripts import train_b601 as cli


def test_native_log_flag_and_existing_overrides():
    args = cli.parse_args(
        [
            "--config",
            "config.yml",
            "--log_to_file",
            "--save-path",
            "custom",
            "--max-updates",
            "16000",
        ]
    )
    assert args.log_to_file and args.save_path == "custom" and args.max_updates == 16000
    assert not cli.parse_args(["--config", "config.yml"]).log_to_file
    with pytest.raises(SystemExit):
        cli.parse_args(["--config", "config.yml", "--max-updates", "0"])


def test_rank_logs_keep_stdout_stderr_and_logging_separate(
    tmp_path, monkeypatch, capsys
):
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    original_stdout, original_stderr = sys.stdout, sys.stderr
    for rank in [0, 1]:
        monkeypatch.setenv("RANK", str(rank))
        with cli.training_output(tmp_path, True):
            print(f"out-{rank}")
            print(f"err-{rank}", file=sys.stderr)
            logging.info("info-%s", rank)
        assert sys.stdout is original_stdout and sys.stderr is original_stderr
        assert root.handlers == handlers and root.level == level
    files = list(tmp_path.glob("*.log"))
    assert len(files) == 4
    for rank in [0, 1]:
        stdout = next(
            p
            for p in files
            if f"_rank{rank}_" in p.name and not p.name.endswith("_stderr.log")
        )
        stderr = next(
            p
            for p in files
            if f"_rank{rank}_" in p.name and p.name.endswith("_stderr.log")
        )
        assert (
            f"out-{rank}" in stdout.read_text() and f"info-{rank}" in stdout.read_text()
        )
        assert f"err-{rank}" not in stdout.read_text()
        assert stderr.read_text() == f"err-{rank}\n"
    terminal = capsys.readouterr()
    assert "out-0" in terminal.out and "info-1" in terminal.out
    assert terminal.err == "err-0\nerr-1\n"


def test_failure_keeps_traceback_and_restores_streams(tmp_path):
    before = sys.stdout, sys.stderr
    with pytest.raises(RuntimeError, match="injected failure"):
        with cli.training_output(tmp_path, True):
            print("before-failure")
            raise RuntimeError("injected failure")
    assert (sys.stdout, sys.stderr) == before
    assert (
        "RuntimeError: injected failure"
        in next(tmp_path.glob("*_stderr.log")).read_text()
    )


def test_disabled_logging_creates_no_directory(tmp_path, capsys):
    path = tmp_path / "no-logs"
    with cli.training_output(path, False):
        print("terminal-only")
    assert not path.exists()
    assert "terminal-only" in capsys.readouterr().out


@pytest.mark.parametrize("fail_init", [False, True])
def test_main_uses_effective_output_path_and_captures_startup(
    tmp_path, monkeypatch, fail_init
):
    target = tmp_path / "override"
    cfg = SimpleNamespace(
        checkpoint=SimpleNamespace(save_path=str(tmp_path / "original")),
        hyperparams=SimpleNamespace(max_optimizer_steps=8),
        logging=SimpleNamespace(use_wandb=True),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_b601",
            "--config",
            "unused.yml",
            "--log_to_file",
            "--save-path",
            str(target),
            "--no-wandb",
        ],
    )
    monkeypatch.setattr(cli, "load_config", lambda _: cfg)
    state = {"initialized": False, "cleaned": False, "trained": False}
    monkeypatch.setattr(cli.torch.cuda, "set_device", lambda _: None)

    def init(*args, **kwargs):
        print("initializing")
        if fail_init:
            raise RuntimeError("init failed")
        state["initialized"] = True

    monkeypatch.setattr(cli.torch.distributed, "init_process_group", init)
    monkeypatch.setattr(
        cli.torch.distributed, "is_initialized", lambda: state["initialized"]
    )
    monkeypatch.setattr(cli, "cleanup_distributed", lambda: state.update(cleaned=True))
    monkeypatch.setattr(cli, "setup_logger", lambda config: None)

    class Trainer:
        def __init__(self, config, wandb_run):
            assert config.checkpoint.save_path == str(target)
            assert config.logging.use_wandb is False

        def fit(self):
            state["trained"] = True
            print("trained")

    monkeypatch.setattr(cli, "B601Trainer", Trainer)
    if fail_init:
        with pytest.raises(RuntimeError, match="init failed"):
            cli.main()
        assert not state["cleaned"] and not state["trained"]
        assert "init failed" in next(target.glob("*_stderr.log")).read_text()
    else:
        cli.main()
        assert state["cleaned"] and state["trained"]
    assert len(list(target.glob("*.log"))) == 2
    assert not (tmp_path / "original").exists()
