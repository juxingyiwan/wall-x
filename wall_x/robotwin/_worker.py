"""Bootstrap a simulator worker without exposing the model's site-packages."""

import importlib.util
import sys
from pathlib import Path


def main():
    package = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "wall_x", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["wall_x"] = module
    spec.loader.exec_module(module)

    from wall_x.robotwin.env import main as run_worker

    run_worker()


if __name__ == "__main__":
    main()
