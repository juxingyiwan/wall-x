"""B601 adapter entry: torchrun --nproc_per_node=8 -m scripts.train_b601 --config CONFIG.yml."""

import argparse
import contextlib
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

from wall_x.config import load_config
from wall_x.trainer.adapters.b601_trainer import B601Trainer
from wall_x.trainer.fsdp_trainer.base_trainer import cleanup_distributed
from wall_x.trainer.fsdp_trainer.train_fsdp import setup_logger


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume", help="Exact resume from a complete B601 checkpoint directory"
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        help="Override completed-update cap, not scheduler horizon",
    )
    parser.add_argument("--save-path", help="Override checkpoint output directory")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--log_to_file",
        action="store_true",
        help="Write stdout/stderr to separate per-rank files in checkpoint.save_path.",
    )
    args = parser.parse_args(argv)
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates must be positive")
    return args


class _TeeOutput:
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextlib.contextmanager
def training_output(save_path, enabled):
    """Keep terminal output, with independent files for every global rank."""
    with contextlib.ExitStack() as stack:
        stderr_file = None
        paths = None
        if enabled:
            directory = Path(save_path)
            directory.mkdir(parents=True, exist_ok=True)
            rank = int(os.environ.get("RANK", "0"))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stem = f"training_log_{timestamp}_rank{rank}_pid{os.getpid()}"
            paths = [directory / f"{stem}.log", directory / f"{stem}_stderr.log"]
            # Exclusive creation prevents silently truncating an earlier log.
            stdout_file = stack.enter_context(paths[0].open("x", encoding="utf-8"))
            stderr_file = stack.enter_context(paths[1].open("x", encoding="utf-8"))
            stack.enter_context(
                contextlib.redirect_stdout(_TeeOutput(sys.stdout, stdout_file))
            )
            stack.enter_context(
                contextlib.redirect_stderr(_TeeOutput(sys.stderr, stderr_file))
            )
        root = logging.getLogger()
        previous_handlers, previous_level = root.handlers[:], root.level
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        try:
            if paths:
                logging.info("Training logs: stdout=%s stderr=%s", *paths)
            yield
        except BaseException:
            # The interpreter prints the terminal traceback after streams are
            # restored; retain the same failure in this rank's stderr file.
            if stderr_file is not None:
                traceback.print_exc(file=stderr_file)
            raise
        finally:
            handler.flush()
            root.handlers, root.level = previous_handlers, previous_level
            handler.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.resume:
        cfg.checkpoint.resume_from = args.resume
    if args.max_updates is not None:
        cfg.hyperparams.max_optimizer_steps = args.max_updates
    if args.save_path:
        cfg.checkpoint.save_path = args.save_path
    if args.no_wandb:
        cfg.logging.use_wandb = False
    with training_output(cfg.checkpoint.save_path, args.log_to_file):
        run = None
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            torch.distributed.init_process_group(
                "nccl", device_id=torch.device(f"cuda:{local_rank}")
            )
            run = setup_logger(cfg)
            B601Trainer(cfg, wandb_run=run).fit()
        finally:
            if torch.distributed.is_initialized():
                cleanup_distributed()
            if run is not None:
                run.finish()


if __name__ == "__main__":
    main()
