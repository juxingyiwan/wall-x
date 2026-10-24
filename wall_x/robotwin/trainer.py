"""RoboTwin orchestration around the unchanged native FSDP optimizer/model."""

import contextlib
import gc
import logging
import os
from pathlib import Path

import torch

from wall_x.robotwin.build import load_robotwin_v2
from wall_x.robotwin.normalization import adapt_normalizers, resolve_normalization
from wall_x.robotwin.resume import (
    atomic_json,
    capture_rng,
    legacy_cursor,
    read_state,
    restore_rng,
    resume_protocol,
    validate_legacy_config,
)
from wall_x.trainer.fsdp_trainer import checkpoint_io
from wall_x.trainer.fsdp_trainer.base_trainer import barrier
from wall_x.trainer.fsdp_trainer.fsdp_trainer import FSDPTrainer


class RobotwinFSDPTrainer(FSDPTrainer):
    def __init__(self, train_config, wandb_run=None):
        resolve_normalization(train_config._raw_yaml)
        self._resume_state = None
        self._pending_rng = None
        self._resume_batch = 0
        self._pending_validation = False
        self._pending_epoch_validation = False
        self._inside_train_loop = False
        super().__init__(train_config, wandb_run)
        self.num_training_steps = self.cfg.hyperparams.training_step_limit

    def load_normalizer(self):
        super().load_normalizer()
        self.normalizer_action, self.normalizer_propri = adapt_normalizers(
            self.cfg._raw_yaml, self.normalizer_action, self.normalizer_propri
        )

    def resume_from_checkpoint(self):
        path = Path(self.cfg.checkpoint.resume_from)
        for name in (
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pt",
        ):
            if not (path / name).is_file():
                raise ValueError(f"RoboTwin training resume requires {name}")
        self._resume_state = read_state(path, self.rank)

        def checked_log(message, **kwargs):
            if any(
                word in message
                for word in ("cold-start", "keeping fresh state", "DROPPED")
            ):
                raise RuntimeError(f"Incomplete RoboTwin training restore: {message}")
            self.log(message, **kwargs)

        checkpoint_io.resume_from_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            resume_config={"ckpt": str(path)},
            rank=self.rank,
            grad_scaler=self.grad_scaler,
            model_class=self._checkpoint_model_class(),
            log_fn=checked_log,
        )
        # Dataset construction below reseeds Torch. Defer final restoration
        # until all initialization is complete, using this rank's own state.
        self._pending_rng = (
            self._resume_state[1] if self._resume_state else capture_rng()
        )
        barrier()

    def load_dataset(self):
        torch.cuda.empty_cache()
        barrier()
        self.dataset, self.train_dataloader, self.train_num = load_robotwin_v2(self.cfg)
        self.train_dataloader.configure(accumulation=self.grad_accum_steps)
        if not len(self.train_dataloader):
            raise ValueError("RoboTwin dataset has no complete optimizer update")
        self._resume_protocol = resume_protocol(
            self.cfg, self.train_dataloader, self.world_size
        )
        path = self.cfg.checkpoint.resume_from
        if not path or not os.path.isdir(path):
            return
        if self._resume_state:
            metadata, _ = self._resume_state
            if metadata["protocol"] != self._resume_protocol:
                changed = [
                    k
                    for k, v in self._resume_protocol.items()
                    if metadata["protocol"].get(k) != v
                ]
                raise ValueError(f"RoboTwin resume protocol changed: {changed}")
            epoch, cursor = metadata["epoch"], metadata["next_batch"]
            self.global_step = metadata["completed_updates"]
            self._pending_validation = metadata["pending_validation"]
            self._pending_epoch_validation = metadata["pending_epoch_validation"]
        else:
            validate_legacy_config(path, self.cfg)
            epoch, cursor, self.global_step = legacy_cursor(
                path, self.train_dataloader, self.grad_accum_steps, self.world_size
            )
            log = self.cfg.logging
            previous_step = self.global_step - 1
            self._pending_validation = (
                log.val_interval > 0
                and previous_step >= log.ignore_until_interval
                and previous_step > 0
                and previous_step % log.val_interval == 0
            )
            self._pending_epoch_validation = cursor == len(self.train_dataloader)
            self.log(
                "Legacy RoboTwin checkpoint: recovered the batch cursor; original "
                "per-rank/worker RNG states were not saved, so past uninterrupted "
                "numerical equivalence cannot be recovered.",
                level=logging.WARNING,
            )
        self.train_dataloader.configure(cursor, self.grad_accum_steps)
        if cursor == len(self.train_dataloader):
            epoch, cursor = epoch + 1, 0
        self.start_epoch, self._resume_batch = epoch, cursor
        self.log(
            f"RoboTwin resume: completed_updates={self.global_step}, epoch={epoch}, "
            f"next_batch={cursor}/{len(self.train_dataloader)}"
        )

    def fit(self):
        barrier()
        if self._pending_rng is not None:
            restore_rng(self._pending_rng)
            self._pending_rng = None
        # Native step checkpoints are written before validation; replay that
        # pending operation before consuming the next training batch.
        if self._pending_validation:
            self.val_loop()
        if self._pending_epoch_validation and not self._at_limit():
            self.val_loop()
        if (
            not self._resume_from_training_checkpoint()
            and self.cfg.checkpoint.validate_first
        ):
            self.val_loop()
            barrier()
        if self._at_limit():
            return
        if self.nvtx:
            torch.cuda.cudart().cudaProfilerStart()
        profiler = (
            self._setup_profiler()
            if self.cfg.debug.profile
            else contextlib.nullcontext()
        )
        for epoch in range(self.start_epoch, self.num_epoch):
            before = self.global_step
            self.train_loop(epoch, profiler=profiler)
            barrier()
            terminal = self._at_limit() or epoch + 1 == self.num_epoch
            interval = self.cfg.logging.epoch_save_interval
            due = interval > 0 and (epoch + 1) % interval == 0
            if (
                self.global_step > before
                and (terminal or due)
                and getattr(self, "_last_saved_step", None) != self.global_step
            ):
                self.save_checkpoint(epoch, self.global_step)
            if self._at_limit():
                break
            self.val_loop()
            barrier()
            gc.collect()
            torch.cuda.empty_cache()
        if self.nvtx:
            torch.cuda.cudart().cudaProfilerStop()

    def _at_limit(self):
        return (
            self.num_training_steps > 0 and self.global_step >= self.num_training_steps
        )

    def train_loop(self, epoch, profiler=contextlib.nullcontext()):
        cursor = self._resume_batch if epoch == self.start_epoch else 0
        self.train_dataloader.configure(cursor, self.grad_accum_steps)
        self.initial_step = cursor
        self._inside_train_loop = True
        try:
            super().train_loop(epoch, profiler=profiler)
        finally:
            self._inside_train_loop = False
            self.initial_step = 0
            self._resume_batch = 0

    def save_checkpoint(self, epoch, step=0):
        rng = capture_rng()
        path = Path(self.cfg.checkpoint.save_path) / (
            f"{epoch}_{step}" if step else str(epoch)
        )
        directory = path / "robotwin_resume"
        directory.mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            (directory / "metadata.json").unlink(missing_ok=True)
        barrier()
        super().save_checkpoint(epoch, step)
        temporary = directory / f"rank_{self.rank:05d}.tmp"
        torch.save(rng, temporary)
        temporary.replace(directory / f"rank_{self.rank:05d}.pt")
        barrier()
        if self.rank == 0:
            cursor = self.train_dataloader.next_batch
            atomic_json(
                directory / "metadata.json",
                {
                    "version": 1,
                    "epoch": epoch,
                    "next_batch": cursor,
                    "completed_updates": self.global_step,
                    "pending_validation": self._inside_train_loop
                    and self._should_validate(),
                    "pending_epoch_validation": cursor == len(self.train_dataloader),
                    "protocol": self._resume_protocol,
                },
            )
        barrier()
        restore_rng(rng)

    def _should_save_checkpoint(self):
        log = self.cfg.logging
        # Preserve the configured save cadence with a completed-update counter.
        step = self.global_step if log.save_on_completed_steps else self.global_step - 1
        return (
            log.save_interval > 0
            and step >= log.ignore_until_interval
            and step % log.save_interval == 0
            and step > 0
        )

    def _should_validate(self):
        # Historical RoboTwin validation ran before incrementing the counter.
        log = self.cfg.logging
        step = self.global_step - 1
        return (
            log.val_interval > 0
            and step >= log.ignore_until_interval
            and step > 0
            and step % log.val_interval == 0
        )
