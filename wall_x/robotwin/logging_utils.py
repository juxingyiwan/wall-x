"""RoboTwin entry logging in the native trainer's per-rank log files."""

import logging
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def training_logging(config_path, *, enabled=False, debug=False):
    """Append entry logs and warnings without teeing native trainer output.

    DistributedLogger already writes to logs/rank_N.log and echoes rank zero
    to stdout. Keeping stdout untouched avoids saving those records twice.
    RANK is available before process-group initialization and is global across
    nodes; LOCAL_RANK must not be used to name these files.
    """
    rank = int(os.environ.get("RANK", "0"))
    formatter = logging.Formatter(
        f"%(asctime)s - [rank{rank}] - %(levelname)s - %(message)s"
    )
    console = logging.StreamHandler(sys.stdout)
    handlers = [console]
    log_path = None
    if enabled:
        import yaml

        from wall_x.config.loader import _TrainConfigSafeLoader

        with open(config_path, encoding="utf-8") as source:
            # Native checkpoint YAML may contain optimizer betas as a tuple.
            config = yaml.load(source, Loader=_TrainConfigSafeLoader)
        save_path = config.get("save_path") or config.get("checkpoint", {}).get(
            "save_path", "./ckpt"
        )
        if debug:
            save_path = "./ckpt/debug"
        log_path = Path(save_path) / "logs" / f"rank_{rank}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))

    root = logging.getLogger()
    previous_handlers, previous_level = root.handlers[:], root.level
    previous_showwarning = warnings.showwarning
    for handler in previous_handlers:
        root.removeHandler(handler)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(logging.INFO)

    def log_warning(message, category, filename, lineno, file=None, line=None):
        if file is not None:
            return previous_showwarning(message, category, filename, lineno, file, line)
        logging.getLogger(__name__).warning(
            "%s",
            warnings.formatwarning(message, category, filename, lineno, line).rstrip(),
        )

    warnings.showwarning = log_warning
    logger = logging.getLogger(__name__)
    try:
        logger.info(
            "Training session started: pid=%s rank=%s config=%s log=%s",
            os.getpid(),
            rank,
            Path(config_path).resolve(),
            log_path or "console",
        )
        yield log_path
    except BaseException:
        logger.exception("Training session failed: pid=%s rank=%s", os.getpid(), rank)
        raise
    else:
        logger.info("Training session completed: pid=%s rank=%s", os.getpid(), rank)
    finally:
        warnings.showwarning = previous_showwarning
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()
        for handler in previous_handlers:
            root.addHandler(handler)
        root.setLevel(previous_level)
