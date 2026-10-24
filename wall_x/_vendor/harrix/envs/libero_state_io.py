"""Load LIBERO's numeric NumPy initial states with PyTorch's restricted loader."""

import numpy as np


def load_task_initial_states(task_suite, task_id):
    import torch

    core = getattr(np, "_core", None)
    if core is None:
        core = np.core
    reconstruct = core.multiarray._reconstruct
    # Old LIBERO files use numpy.core; NumPy 2 saves numpy._core instead.
    allowed = [
        np.ndarray,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        (reconstruct, "numpy._core.multiarray._reconstruct"),
    ]
    # Nested callers may already allow some entries. PyTorch removes every
    # entry supplied to this context on exit, including previously allowed ones.
    existing = set(torch.serialization.get_safe_globals())
    additions = [entry for entry in allowed if entry not in existing]
    with torch.serialization.safe_globals(additions):
        return task_suite.get_task_init_states(task_id)
