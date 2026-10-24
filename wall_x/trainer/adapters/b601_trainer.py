"""B601 training orchestration around WALL-X's unchanged model/FSDP machinery."""

import contextlib
import dataclasses
import gc
import json
import logging
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from wall_x.data.b601 import validate_checkpoint
from wall_x.trainer.adapters.b601_state import (
    canonicalize_config,
    capture_rng,
    code_checksums,
    file_sha256,
    load_runtime,
    load_scaler_state,
    model_config_file,
    restore_rng,
    resumable_loader,
    runtime_contract,
    validate_dataset_sources,
)
from wall_x.trainer.fsdp_trainer import checkpoint_io
from wall_x.trainer.fsdp_trainer.base_trainer import barrier, is_main_process
from wall_x.trainer.fsdp_trainer.fsdp_trainer import FSDPTrainer
from wall_x.trainer.utils import move_batch_to_device


def all_ranks_require(condition, message, device, exception=ValueError):
    if hasattr(condition, "to_local"):
        condition = condition.to_local()
    failed = torch.as_tensor(
        ~condition if torch.is_tensor(condition) else not condition,
        dtype=torch.int32,
        device=device,
    ).clone()
    if dist.is_initialized():
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise exception(message)


class B601Trainer(FSDPTrainer):
    """Counters mean completed optimizer updates, including saved metadata."""

    def __init__(self, train_config, wandb_run=None):
        canonicalize_config(train_config)
        if train_config.data.num_workers != 0:
            raise ValueError("Exact B601 resume requires data.num_workers: 0")
        self._resume_state = None
        self.next_epoch = 0
        self.next_batch = 0
        self._last_saved_step = -1
        super().__init__(train_config, wandb_run)
        # The native constructor restores RNG before loading data. Restore our
        # per-rank snapshot last, after every component has been initialized.
        if self._resume_state is not None:
            restore_rng(self._resume_state["rng"])

    def load_normalizer(self):
        validate_dataset_sources(self.cfg)
        super().load_normalizer()

    def load_dataset(self):
        self.dataset, loader, _ = self.adapter.load_dataset(
            self.data_config,
            self.processor,
            self.rank,
            self.world_size,
            model_config=self.model_config,
            tokenizer_mixin=self.tokenizer_mixin,
            normalizer_action=self.normalizer_action,
            normalizer_propri=self.normalizer_propri,
        )
        self.train_dataloader, self.batch_sampler = resumable_loader(
            loader, self.grad_accum_steps, self.seed + self.rank
        )
        self.train_num = self.batch_sampler.stop
        self._training_contract = runtime_contract(
            self.cfg, self.world_size, self.train_num
        )
        state = self._resume_state
        if state is not None:
            if state["contract"] != self._training_contract:
                raise ValueError(
                    "B601 resume data/optimizer/scheduler contract mismatch"
                )
            self.global_step = state["completed_updates"]
            self.next_epoch = self.start_epoch = state["next_epoch"]
            self.next_batch = state["next_batch"]
            if (
                not 0 <= self.next_batch < self.train_num
                or self.next_batch % self.grad_accum_steps
            ):
                raise ValueError("Invalid B601 checkpoint batch cursor")
            self.train_dataloader.collate_fn.load_state_dict(state["collator"])
        self.log(
            f"B601 progress: completed_updates={self.global_step}, "
            f"next_epoch={self.next_epoch}, next_batch={self.next_batch}"
        )

    def resume_from_checkpoint(self):
        path = self.cfg.checkpoint.resume_from
        error = None
        try:
            self._resume_state = load_runtime(path, self.rank, self.world_size)
            validate_checkpoint(dataclasses.asdict(self.cfg), path)
            scaler_state = load_scaler_state(
                path, self.grad_scaler, self._resume_state["contract"]
            )
        except (ValueError, OSError, KeyError, RuntimeError) as exc:
            error = str(exc)
        all_ranks_require(
            error is None,
            f"B601 resume preflight failed: {error or 'another rank'}",
            self.device,
        )
        warnings = []

        def resume_log(message, **kwargs):
            # Informational load results and user paths can contain words such
            # as "incompatible" or "failed" without indicating a failed restore.
            if kwargs.get("level", logging.INFO) >= logging.WARNING:
                warnings.append(message.lower())
            self.log(message, **kwargs)

        model = self.model.module if isinstance(self.model, DDP) else self.model
        normalizer_prefixes = (
            "action_preprocessor.normalizer",
            "action_processor.normalizer",
        )
        # Native loading deliberately keeps current-run normalizers. Their
        # statistics are bound separately by the B601 checkpoint contract.
        allowed_missing = {
            name
            for name, param in model.named_parameters()
            if not param.requires_grad and name.startswith(normalizer_prefixes)
        }
        allowed_missing.update(
            name
            for name, _ in model.named_buffers()
            if name.startswith(normalizer_prefixes)
        )
        model_mismatches = []

        def record_model_status(_module, incompatible_keys):
            model_mismatches.extend(
                key
                for key in incompatible_keys.missing_keys
                if key not in allowed_missing
            )
            model_mismatches.extend(incompatible_keys.unexpected_keys)

        kernel_options = [
            {key: group[key] for key in ("fused", "foreach") if key in group}
            for group in self.optimizer.param_groups
        ]
        # Observe PyTorch's structured result without changing it or the native
        # loader. Remove the hook even if loading raises an exception.
        with model.register_load_state_dict_post_hook(record_model_status):
            checkpoint_io.resume_from_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                resume_config={"ckpt": path},
                rank=self.rank,
                # The B601 preflight validates this separately; native loading
                # may silently retain a fresh scaler when the file is absent.
                grad_scaler=None,
                model_class=self._checkpoint_model_class(),
                log_fn=resume_log,
            )
        failed = bool(model_mismatches) or any(
            any(
                word in m
                for word in ("cold-start", "failed", "incompatible", "keeping fresh")
            )
            for m in warnings
        )
        all_ranks_require(
            not failed,
            "B601 exact resume refused a partial model/optimizer/scheduler restore"
            f"; incompatible model keys: {model_mismatches}",
            self.device,
        )
        # Native compatibility loading strips these for legacy checkpoints.
        # Exact B601 resume binds optimizer config, so retain its execution path.
        all_ranks_require(
            len(kernel_options) == len(self.optimizer.param_groups),
            "B601 optimizer parameter groups changed during resume",
            self.device,
        )
        for group, options in zip(self.optimizer.param_groups, kernel_options):
            group.update(options)
        if scaler_state is not None:
            self.grad_scaler.load_state_dict(scaler_state)
        barrier()

    def _require_finite(self, value, field):
        all_ranks_require(
            torch.isfinite(torch.as_tensor(value, device=self.device)).all(),
            f"Non-finite {field} on at least one rank; synchronized stop before optimizer update "
            f"{self.global_step + 1}. Resume the last complete checkpoint.",
            self.device,
            FloatingPointError,
        )

    def train_loop(self, epoch, profiler=None):
        self.dataset.set_epoch(epoch)
        self.model.train()
        self.batch_sampler.start = self.next_batch
        start = self.next_batch
        total = self.batch_sampler.stop
        was_gc_enabled = gc.isenabled()
        gc.disable()
        timer = time.perf_counter()
        try:
            for index, batch in enumerate(self.train_dataloader, start):
                batch = move_batch_to_device(batch, self.device)
                sync = self.sync_gradients()
                context = (
                    contextlib.nullcontext()
                    if sync
                    else self.strategy.no_sync(self.model)
                )
                # DDP no_sync must cover both forward and backward.
                with context:
                    with self.autocast_context():
                        outputs = self.adapter.forward(
                            self.model, batch, global_step=self.global_step
                        )
                    loss = self.adapter.extract_loss(outputs)
                    self._require_finite(loss, "loss")
                    self.backward(loss / self.grad_accum_steps)
                self.next_epoch, self.next_batch = epoch, index + 1
                if not sync:
                    self.micro_step += 1
                    continue

                self._component_grad_norms = {}
                self.adapter.collect_grad_norms(
                    self.model,
                    self._component_grad_norms,
                    device=self.device,
                    reduce_tensor_fn=self.reduce_tensor,
                    params_sharded=self.strategy.params_sharded,
                )
                if self.cfg.hyperparams.optimizer.enable_grad_clip:
                    total_norm = self.clip_grad_norm(self.max_grad_norm)
                    self._require_finite(total_norm, "gradient norm")
                else:
                    if self.grad_scaler is not None:
                        self.grad_scaler.unscale_(self.optimizer)
                    finite = []
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            grad = parameter.grad
                            grad = (
                                grad.to_local() if hasattr(grad, "to_local") else grad
                            )
                            finite.append(torch.isfinite(grad).all())
                    all_ranks_require(
                        torch.stack(finite).all(),
                        "Non-finite gradients; synchronized stop",
                        self.device,
                        FloatingPointError,
                    )
                    total_norm = 0.0
                self.optimizer_step()
                self.optimizer_zero_grad()
                self.lr_scheduler_step()
                self.global_step += 1
                self.micro_step = 0
                if self.next_batch == total:
                    self.next_epoch, self.next_batch = epoch + 1, 0
                self._log_training_metrics(
                    epoch, index, total, loss, total_norm, outputs
                )
                elapsed = time.perf_counter() - timer
                self.training_log(
                    epoch,
                    self.num_epoch,
                    index,
                    total,
                    loss,
                    self.get_lr(),
                    elapsed,
                    False,
                )
                timer = time.perf_counter()
                if self._should_validate():
                    self.val_loop()
                if self._should_save_checkpoint():
                    self.save_checkpoint(epoch)
                if profiler is not None:
                    profiler.step()
                if (
                    self.cfg.logging.gc_interval_steps > 0
                    and self.global_step % self.cfg.logging.gc_interval_steps == 0
                ):
                    gc.collect()
                if (
                    self.num_training_steps > 0
                    and self.global_step >= self.num_training_steps
                ):
                    break
        finally:
            if was_gc_enabled:
                gc.enable()

    def _should_save_checkpoint(self):
        log = self.cfg.logging
        return (
            self.global_step >= log.ignore_until_interval
            and log.save_interval > 0
            and self.global_step % log.save_interval == 0
            and (
                self.num_training_steps <= 0
                or self.global_step < self.num_training_steps
            )
        )

    def _should_validate(self):
        log = self.cfg.logging
        return (
            log.val_interval > 0
            and self.global_step >= log.ignore_until_interval
            and self.global_step % log.val_interval == 0
        )

    def val_loop(self):
        rng, training = capture_rng(), self.model.training
        try:
            super().val_loop()
        finally:
            self.dataset.get_train_dataloader()
            self.model.train(training)
            restore_rng(rng)

    def fit(self):
        barrier()
        initial_updates = self.global_step
        if self.cfg.checkpoint.validate_first:
            self.val_loop()
        profiler = (
            self._setup_profiler()
            if self.cfg.debug.profile
            else contextlib.nullcontext()
        )
        with profiler as active_profiler:
            while self.next_epoch < self.num_epoch:
                if (
                    self.num_training_steps > 0
                    and self.global_step >= self.num_training_steps
                ):
                    break
                epoch = self.next_epoch
                self.train_loop(epoch, profiler=active_profiler)
                barrier()
                if (
                    self.num_training_steps > 0
                    and self.global_step >= self.num_training_steps
                ):
                    self.save_checkpoint(epoch)
                    break
                self.val_loop()
                if (epoch + 1) % self.cfg.logging.epoch_save_interval == 0:
                    self.save_checkpoint(epoch)
            # A normal epoch-budget exit needs the same final-save guarantee as
            # an update-cap exit. Never save here after a raised training error,
            # and do not resave a resumed run that performed no new updates.
            if (
                self.global_step > initial_updates
                and self._last_saved_step != self.global_step
            ):
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch, step=None):
        if getattr(self, "_last_saved_step", -1) == self.global_step:
            return
        if self.micro_step:
            raise ValueError("B601 saves only at completed optimizer updates")
        path = Path(self.cfg.checkpoint.save_path) / f"{epoch}_{self.global_step}"
        # Never replace an existing complete/partial checkpoint in-place.
        all_ranks_require(
            not path.exists(), f"B601 checkpoint already exists: {path}", self.device
        )
        rng = capture_rng()
        state = {
            "rank": self.rank,
            "completed_updates": self.global_step,
            "next_epoch": self.next_epoch,
            "next_batch": self.next_batch,
            "micro_step": 0,
            "contract": self._training_contract,
            "rng": rng,
            "collator": self.train_dataloader.collate_fn.state_dict(),
        }
        config = dataclasses.asdict(self.cfg)
        if self.cfg.model.config_path:
            # Native metadata copying expects a file. Persist the same resolved
            # JSON used by our fingerprint, including directory/version inputs.
            config["model"]["config_path"] = str(
                model_config_file(self.cfg.model.config_path)
            )
        checkpoint_io.save_checkpoint(
            ckpt_path=str(path),
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config=config,
            rank=self.rank,
            is_main=is_main_process(),
            epoch=epoch,
            global_step=self.global_step,
            seed=self.seed,
            normalizer_action=self.normalizer_action,
            normalizer_propri=self.normalizer_propri,
            grad_scaler=self.grad_scaler,
            log_fn=self.log,
            frozen_prefixes=self._frozen_prefixes,
        )
        torch.save(state, path / f"b601_training_rank{self.rank}.pt")
        checkpoint_io.finalize_save()
        if is_main_process():
            metadata = {
                k: state[k] for k in ("completed_updates", "next_epoch", "next_batch")
            }
            metadata.update(
                version=1,
                world_size=self.world_size,
                rank_files={
                    f"b601_training_rank{r}.pt": file_sha256(
                        path / f"b601_training_rank{r}.pt"
                    )
                    for r in range(self.world_size)
                },
            )
            metadata["code_files"] = code_checksums()
            metadata["torch_version"] = str(torch.__version__)
            metadata["metadata_files"] = {
                name: file_sha256(path / name)
                for name in (
                    "config.yml",
                    "norm_stats.json",
                    "episode_split.json",
                    "b601_manifest.json",
                )
            }
            for name in ("config.json", "grad_scaler.pt"):
                if (path / name).is_file():
                    metadata["metadata_files"][name] = file_sha256(path / name)
            (path / "b601_training_state.json").write_text(
                json.dumps(metadata, indent=2) + "\n"
            )
        barrier()
        restore_rng(rng)
        self._last_saved_step = self.global_step
        self.log(f"Saved B601 checkpoint {path}: completed_updates={self.global_step}")
