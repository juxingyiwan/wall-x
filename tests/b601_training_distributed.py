"""Run via torchrun on the existing allocation; this script never owns/releases it."""

import copy
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from b601_training_fixture import Harness, fixture_config

from wall_x.config import load_config
from wall_x.trainer.adapters.b601_state import load_runtime


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        timeout=timedelta(minutes=4),
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(sys.argv[1])
    results = []
    try:
        for accumulation in (1, 2):
            folder = root / f"accum_{accumulation}"
            if rank == 0:
                fixture_config(folder, accumulation=accumulation)
            dist.barrier()
            cfg = load_config(str(folder / "config.yml"))
            continuous = Harness(copy.deepcopy(cfg), device=f"cuda:{local_rank}")
            continuous.cfg.checkpoint.save_path = str(folder / "continuous")
            continuous.fit()
            reference = continuous.parameters_snapshot()
            trace = continuous.trace
            scheduler = continuous.lr_scheduler.state_dict()
            del continuous
            torch.cuda.empty_cache()
            before = Harness(copy.deepcopy(cfg), device=f"cuda:{local_rank}")
            before.cfg.checkpoint.save_path = str(folder / "interrupted")
            before.num_training_steps = 2
            before.fit()
            first_trace = before.trace
            del before
            torch.cuda.empty_cache()
            checkpoint = folder / "interrupted/0_2"
            state = load_runtime(checkpoint, rank, world)
            assert (
                state["next_batch"] == 2 * accumulation
                and state["completed_updates"] == 2
            )
            resumed = Harness(
                copy.deepcopy(cfg), device=f"cuda:{local_rank}", resume=checkpoint
            )
            resumed.cfg.checkpoint.save_path = str(folder / "resumed")
            resumed.fit()
            assert resumed.global_step == 8
            assert (
                trace == first_trace + resumed.trace
            ), "Sample/time/noise/loss continuity mismatch"
            assert scheduler == resumed.lr_scheduler.state_dict()
            for left, right in zip(reference, resumed.parameters_snapshot()):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            results.append(
                {
                    "accumulation": accumulation,
                    "completed_updates": 8,
                    "exact_trace": True,
                    "exact_parameters": True,
                }
            )
            del resumed
            torch.cuda.empty_cache()
        for failure in ("nan", "inf", "gradient"):
            trainer = Harness(
                copy.deepcopy(cfg), device=f"cuda:{local_rank}", failure=failure
            )
            original = trainer.parameters_snapshot()
            stopped = False
            try:
                trainer.fit()
            except FloatingPointError:
                stopped = True
            assert stopped and trainer.global_step == 0 and not trainer.optimizer.state
            for left, right in zip(original, trainer.parameters_snapshot()):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            dist.barrier()
            results.append(
                {
                    "injection_rank": 0,
                    "failure": failure,
                    "all_ranks_stopped": True,
                    "updates": 0,
                }
            )
            del trainer
            torch.cuda.empty_cache()
        (root / f"rank{rank}.json").write_text(
            json.dumps({"rank": rank, "world_size": world, "checks": results}, indent=2)
        )
        dist.barrier()
        if rank == 0:
            print(
                json.dumps({"passed": True, "world_size": world, "checks": results}),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
