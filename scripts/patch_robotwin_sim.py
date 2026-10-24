#!/usr/bin/env python3
"""Apply RoboTwin's SAPIEN/MPLib compatibility fixes to the active sim env only."""

import sys
import sysconfig
from importlib.metadata import version
from pathlib import Path


def main():
    if version("sapien") != "3.0.0b1" or version("mplib") != "0.2.1":
        raise RuntimeError("This patch requires sapien==3.0.0b1 and mplib==0.2.1")
    site = Path(sysconfig.get_paths()["purelib"]).resolve()
    if not site.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Package location is outside the active environment")
    changes = {
        site
        / "sapien/wrapper/urdf_loader.py": [
            ('open(urdf_file, "r")', 'open(urdf_file, "r", encoding="utf-8")'),
            ('open(srdf_file, "r")', 'open(srdf_file, "r", encoding="utf-8")'),
            ('urdf_file[:-4] + "srdf"', 'urdf_file[:-4] + ".srdf"'),
        ],
        site
        / "mplib/planner.py": [
            (
                "if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:",
                "if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:",
            ),
        ],
    }
    for path, replacements in changes.items():
        original = path.read_text()
        text = original
        for before, after in replacements:
            if before not in text and after not in text:
                raise RuntimeError(f"Unrecognized upstream source in {path}")
            text = text.replace(before, after)
        if text != original:
            path.write_text(text)
        print(f"Compatible: {path}")


if __name__ == "__main__":
    main()
