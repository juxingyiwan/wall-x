"""Small real-optimizer harness shared by CPU and eight-GPU integration tests."""

import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import yaml
from test_b601_pipeline import config, episodes
from torch.utils.data import DataLoader, Dataset

from wall_x.config import load_config
from wall_x.data.b601_prepare import make_stats
from wall_x.data.b601_split import stratified_split
from wall_x.data.backends.lerobot.b601_adapter import B601DataCollator
from wall_x.data.sampling import DistributedSquareRootSampler
from wall_x.trainer.adapters.b601_state import canonicalize_config, restore_rng
from wall_x.trainer.adapters.b601_trainer import B601Trainer
from wall_x.trainer.fsdp_trainer.distribution_strategy import FSDPStrategy
from wall_x.trainer.optimizer.utils import get_adamw_optimizer
from wall_x.trainer.trainer_utils import seed_all


def fixture_config(root, accumulation=1, updates=8):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    split = stratified_split(episodes())
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split))
    cfg = config(split_path=split_path)
    cfg["task"]["ar_dof_config"] = cfg["task"]["dof_config"].copy()
    cfg["data"]["lerobot_config"]["root"] = str(root)
    cfg["data"]["lerobot_config"]["b601"]["statistics_scope"] = "full_dataset"
    cfg["data"].update(
        num_workers=0,
        task_sampling="sqrt",
        norm_stats_path=str(root / "norm_stats.json"),
    )
    (root / "norm_stats.json").write_text(
        json.dumps(make_stats(cfg, episodes(), None, []))
    )
    (root / "model.json").write_text("{}")
    (root / "processor").mkdir(exist_ok=True)
    cfg.update(
        model_type="qwen2_5",
        model={
            "config_path": str(root / "model.json"),
            "processor_path": str(root / "processor"),
        },
        hyperparams={
            "seed": 10222,
            "num_epoch": 6,
            "max_optimizer_steps": updates,
            "batch_size_per_gpu": 2,
            "gradient_accumulation_steps": accumulation,
            "optimizer": {"learning_rate": 0.001},
            "scheduler": {"num_training_steps": 100, "num_warmup_steps": 2},
        },
        logging={
            "use_wandb": False,
            "save_interval": 0,
            "val_interval": 0,
            "epoch_save_interval": 99,
        },
        checkpoint={"save_path": str(root / "ckpts")},
    )
    (root / "config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return load_config(str(root / "config.yml"))


class Rows(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


class Collator(B601DataCollator):
    def __init__(self, seed):
        self.dataload_config = {"seed": seed}
        self.np_rng = np.random.default_rng()
        self._b601_rng_context = None
        self.beta_alpha, self.beta_beta, self.time_shift, self.s = 1.5, 1.0, 1.0, 0.999

    def __call__(self, batch):
        return {
            "ids": torch.tensor(batch),
            "time": self.sample_time(len(batch), "cpu", torch.float32),
        }


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        return self.dropout(self.layer(x))


class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, grad):
        return grad * float("inf")


class Adapter:
    def __init__(self, trainer, failure=None):
        self.trainer = trainer
        self.failure = failure

    def load_dataset(self, *args, **kwargs):
        # Exercise the old failure mode: data initialization consumes/reseeds RNG.
        torch.manual_seed(991)
        np.random.seed(991)
        random.seed(991)
        t = self.trainer
        rows = Rows(t.world_size * 12)
        sampler = DistributedSquareRootSampler(
            [i % 3 for i in range(len(rows))],
            rank=t.rank,
            num_replicas=t.world_size,
            seed=t.seed,
        )
        loader = DataLoader(
            rows,
            batch_size=2,
            sampler=sampler,
            drop_last=True,
            collate_fn=Collator(t.seed),
        )
        wrapper = SimpleNamespace(
            set_epoch=sampler.set_epoch,
            get_train_dataloader=lambda: loader,
            get_val_dataloader=lambda: None,
        )
        return wrapper, loader, len(loader)

    def forward(self, model, batch, **kwargs):
        noise = torch.randn((len(batch["ids"]), 4), device=batch["time"].device)
        extra = random.random() + float(np.random.rand())
        x = noise + batch["time"][:, None] + batch["ids"][:, None] / 100 + extra
        loss = model(x).float().square().mean()
        if self.trainer.rank == 0:
            if self.failure in ("nan", "inf"):
                loss = loss * float(self.failure)
            elif self.failure == "gradient":
                loss = BadGradient.apply(loss)
        self.trainer.trace.append(
            {
                "ids": batch["ids"].tolist(),
                "time": batch["time"].tolist(),
                "noise": noise.tolist(),
                "extra": extra,
                "loss": float(loss.detach()),
            }
        )
        return {"loss": loss}

    @staticmethod
    def extract_loss(outputs):
        return outputs["loss"]

    @staticmethod
    def collect_grad_norms(*args, **kwargs):
        pass


class Harness(B601Trainer):
    def __init__(self, cfg, device="cpu", resume=None, failure=None):
        canonicalize_config(cfg)
        self.cfg = cfg
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device(device)
        self.seed = cfg.hyperparams.seed
        seed_all(self.seed)
        self.logger = logging.getLogger("b601_integration")
        self.log = lambda *a, **kw: None
        self.global_step = self.start_epoch = self.micro_step = 0
        self.next_epoch = self.next_batch = 0
        self._last_saved_step = -1
        self._resume_state = None
        self.trace = []
        self.grad_accum_steps = cfg.hyperparams.gradient_accumulation_steps
        self.num_training_steps = cfg.hyperparams.max_optimizer_steps
        self.num_epoch = cfg.hyperparams.num_epoch
        self.max_grad_norm = 1.0
        self.grad_scaler = None
        self.use_amp = False
        self._frozen_prefixes = None
        self.model = Tiny().to(self.device)
        if self.device.type == "cuda":
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

            fully_shard(
                self.model,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
            )
            self.strategy = FSDPStrategy({})
        else:
            self.strategy = SimpleNamespace(
                params_sharded=False,
                last_grad_clip_stats=None,
                clip_grad_norm=lambda model, maximum, **kw: (
                    torch.nn.utils.clip_grad_norm_(model.parameters(), maximum)
                ),
                no_sync=lambda model: __import__("contextlib").nullcontext(),
            )
        cfg.hyperparams.optimizer.fused = self.device.type == "cuda"
        self.optimizer = get_adamw_optimizer(
            self.model, opt_cfg=cfg.hyperparams.optimizer
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.8
        )
        self.normalizer_action = self.normalizer_propri = torch.nn.Identity()
        self.data_config = self.processor = self.model_config = self.tokenizer_mixin = (
            None
        )
        self.adapter = Adapter(self, failure)
        if resume:
            self.cfg.checkpoint.resume_from = str(resume)
            self.resume_from_checkpoint()
        self.load_dataset()
        if self._resume_state:
            restore_rng(self._resume_state["rng"])
        assert all(
            group["fused"] == cfg.hyperparams.optimizer.fused
            and group["foreach"] == cfg.hyperparams.optimizer.foreach
            for group in self.optimizer.param_groups
        )

    def _log_training_metrics(self, *args):
        pass

    def training_log(self, *args):
        pass

    def parameters_snapshot(self):
        # A synchronized pre-backward abort can leave FSDP's temporary gathered
        # BF16 parameters installed. Inspect the unchanged FP32 master shards.
        if hasattr(self.model, "reshard"):
            self.model.reshard()
        return [
            (
                p.detach().to_local().cpu().clone()
                if hasattr(p, "to_local")
                else p.detach().cpu().clone()
            )
            for p in self.model.parameters()
        ]
