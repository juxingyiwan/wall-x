"""Compatibility imports and CLI for square-root task sampling.

Runtime implementations live in wall_x.data.sampling.
"""

import argparse

from wall_x.data.sampling import (
    DistributedSquareRootSampler,
    lerobot_train_task_ids,
    square_root_probabilities,
)

__all__ = [
    "DistributedSquareRootSampler",
    "lerobot_train_task_ids",
    "square_root_probabilities",
    "main",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate square-root task sampling ratios"
    )
    parser.add_argument("counts", nargs="+", metavar="TASK=FRAMES")
    args = parser.parse_args()
    counts = {}
    for item in args.counts:
        name, separator, value = item.rpartition("=")
        if not separator or not name or name in counts:
            parser.error(f"Expected unique TASK=FRAMES, got {item!r}")
        try:
            counts[name] = int(value)
        except ValueError:
            parser.error(f"Frame count must be an integer: {item!r}")
    try:
        probabilities = square_root_probabilities(counts)
    except ValueError as exc:
        parser.error(str(exc))
    for name, probability in probabilities.items():
        print(f"{name}\t{counts[name]}\t{probability:.4%}")


if __name__ == "__main__":
    main()
