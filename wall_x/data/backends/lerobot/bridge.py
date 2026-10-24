"""Public bridge for scenario extensions of the LeRobot backend.

Conversion and trainer wrapper semantics remain owned by the native backend.
Scenario code can adapt the returned configuration or choose its own loader
without importing the backend's private implementation names.
"""

from typing import Any, Dict, Tuple

from wall_x.data.backends.lerobot.build import (
    _build_flat_config,
    _LerobotDatasetWrapper,
)


def build_dataset_config(cfg: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return the native (flat configuration, LeRobot options) pair unchanged.

    This retains field defaults, layout order and normalizer keys used by saved
    training configurations. Scenario-specific overrides belong to the caller.
    """
    return _build_flat_config(cfg)


def wrap_dataset(
    inner,
    train_dataloader,
    train_sampler,
    train_num: int,
    *,
    val_dataloader=None,
    val_num: int = 0,
):
    """Expose an already-built dataset through the native trainer contract.

    Loaders must be constructed by the caller, with validation before training
    where required by the backend. This function does not rebuild loaders,
    advance a sampler, consume RNG or choose a split. It retains the native
    wrapper's epoch seeding and separate train/validation views.
    """
    return _LerobotDatasetWrapper(
        inner,
        train_dataloader,
        train_sampler,
        train_num,
        val_dataloader=val_dataloader,
        val_num=val_num,
    )
