"""Copy the installed extension into this checkout for source-based launching."""

import os
import shutil
import sys
from importlib import metadata
from pathlib import Path


def main() -> None:
    prefix = Path(sys.prefix).resolve()
    root = Path(__file__).resolve().parents[2]
    expected = os.environ.get("WALLX_BUILD_ENV", "wallx-unified")
    if prefix.name != expected:
        raise SystemExit(
            f"Activate {expected} first; set WALLX_BUILD_ENV to select a dedicated environment."
        )
    dist = metadata.distribution("wall_x")
    candidates = [
        entry
        for entry in dist.files or []
        if str(entry).startswith("wall_x/model/core/ops/_cuda_ext_bin")
        and str(entry).endswith(".so")
    ]
    if len(candidates) != 1:
        raise SystemExit(f"Expected one installed CUDA extension, found {len(candidates)}")
    source = Path(dist.locate_file(candidates[0])).resolve()
    if not source.is_relative_to(prefix):
        raise SystemExit(f"Extension must reside in the selected environment: {source}")
    target = root / "wall_x/model/core/ops" / source.name
    shutil.copy2(source, target)
    print(target)


if __name__ == "__main__":
    main()
