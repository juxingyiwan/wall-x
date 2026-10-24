"""Prepare B601 statistics or episode splits; never starts training.

Run with: python -m scripts.prepare_b601 --config BASE.yml --output-dir DIR
Freeze/reuse full statistics: --config B601.yml --statistics-output FILE.json
Resplit without fitting statistics: --config B601.yml --split-output FILE.json
The original recorded commands are preserved, including historical saturation.
"""

import argparse
import json
from pathlib import Path

import yaml

from wall_x.data.b601_prepare import (
    check_full_statistics_source,
    freeze_statistics,
    load_episodes,
    make_stats,
    prepare,
    prepare_split,
)
from wall_x.data.b601_split import stratified_split

# Retain the historical Python imports as well as the command-line entry point.
__all__ = [
    "check_full_statistics_source",
    "freeze_statistics",
    "load_episodes",
    "make_stats",
    "prepare",
    "prepare_split",
    "stratified_split",
    "main",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--output-dir", help="Prepare separate absolute/delta A/B artifacts"
    )
    modes.add_argument(
        "--statistics-output", help="Freeze/reuse one full-dataset statistics file"
    )
    modes.add_argument(
        "--split-output", help="Write a new split using existing full statistics"
    )
    args = parser.parse_args()
    base = yaml.safe_load(Path(args.config).read_text())
    if args.statistics_output:
        report = freeze_statistics(base, args.statistics_output)
    elif args.split_output:
        report = prepare_split(base, args.split_output)
    else:
        report = prepare(base, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
