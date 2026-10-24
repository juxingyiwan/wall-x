"""Entry logging must not duplicate native records or cross rank boundaries."""

import logging
import os
import subprocess
import sys
import warnings

import pytest
import yaml

from wall_x.robotwin.logging_utils import training_logging
from wall_x.utils.logger import DistributedLogger


def config_file(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump({"checkpoint": {"save_path": str(tmp_path)}}))
    return config


def test_native_metrics_written_once_and_restarts_append(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    config = config_file(tmp_path)
    stdout, stderr = sys.stdout, sys.stderr
    for session in range(2):
        with training_logging(config, enabled=True):
            assert sys.stdout is stdout and sys.stderr is stderr
            logging.getLogger("robotwin.entry").info("entry-%s", session)
            native = DistributedLogger(
                name="robotwin.logging.test", save_path=str(tmp_path)
            )
            try:
                native.info("unique-native-metric-%s", session)
            finally:
                for handler in native._logger.handlers[:]:
                    native._logger.removeHandler(handler)
                    handler.close()
    text = (tmp_path / "logs/rank_0.log").read_text()
    for session in range(2):
        assert text.count(f"unique-native-metric-{session}") == 1
        assert text.count(f"entry-{session}") == 1
    assert text.count("Training session started:") == 2
    assert text.count("Training session completed:") == 2
    assert not list(tmp_path.glob("training_log*"))


def test_warning_and_failure_saved_and_logging_restored(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    config = config_file(tmp_path)
    root = logging.getLogger()
    handlers, level, showwarning = root.handlers[:], root.level, warnings.showwarning
    with pytest.raises(RuntimeError, match="startup-failure"):
        with training_logging(config, enabled=True):
            warnings.warn("unique-training-warning", RuntimeWarning)
            raise RuntimeError("startup-failure")
    assert root.handlers == handlers
    assert root.level == level
    assert warnings.showwarning is showwarning
    text = (tmp_path / "logs/rank_3.log").read_text()
    assert "unique-training-warning" in text
    assert "RuntimeError: startup-failure" in text
    assert "Training session failed:" in text
    assert "Training session completed:" not in text


def test_disabled_file_logging_does_not_create_files(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    with training_logging(tmp_path / "not-read.yml", enabled=False) as path:
        assert path is None
        logging.getLogger("robotwin.entry").info("console only")
    assert list(tmp_path.iterdir()) == []


def test_native_checkpoint_tuple_config_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    config = tmp_path / "checkpoint-config.yml"
    config.write_text(
        yaml.dump(
            {
                "checkpoint": {"save_path": str(tmp_path)},
                "hyperparams": {"optimizer": {"betas": (0.9, 0.95)}},
            }
        )
    )
    assert "!!python/tuple" in config.read_text()
    with training_logging(config, enabled=True):
        logging.getLogger("robotwin.entry").info("checkpoint-config-loaded")
    assert "checkpoint-config-loaded" in (tmp_path / "logs/rank_0.log").read_text()


def test_concurrent_ranks_use_global_rank_and_append(tmp_path):
    config = config_file(tmp_path)
    code = """
import logging, sys
from wall_x.robotwin.logging_utils import training_logging
for session in range(2):
    with training_logging(sys.argv[1], enabled=True):
        for index in range(40):
            logging.getLogger('entry').info('marker-%s-%s', session, index)
"""
    ranks = (0, 1, 8, 9)
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(config)],
            env={**os.environ, "RANK": str(rank), "LOCAL_RANK": str(rank % 8)},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for rank in ranks
    ]
    try:
        for process in processes:
            _, error = process.communicate(timeout=30)
            assert process.returncode == 0, error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert {p.name for p in (tmp_path / "logs").iterdir()} == {
        f"rank_{rank}.log" for rank in ranks
    }
    for rank in ranks:
        lines = (tmp_path / "logs" / f"rank_{rank}.log").read_text().splitlines()
        assert all(f"[rank{rank}]" in line for line in lines)
        assert len([line for line in lines if "marker-" in line]) == 80
        assert len([line for line in lines if "Training session started:" in line]) == 2
