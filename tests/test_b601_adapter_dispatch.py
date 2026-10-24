"""Regression checks for native/adapter routing and independent dataset views."""

import importlib
import pickle
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("b601", [False, True])
def test_backend_uses_only_the_selected_loader(monkeypatch, b601):
    from wall_x.data.backends.lerobot import b601_adapter, loader

    build = importlib.import_module("wall_x.data.backends.lerobot.build")

    config = SimpleNamespace(
        hyperparams=SimpleNamespace(seed=10222),
        checkpoint=SimpleNamespace(resume_from=None),
        data=SimpleNamespace(train_test_split=0.95),
    )
    lerobot = {"repo_id": "fixture"}
    if b601:
        lerobot["b601"] = {"representation": "b601_joint_delta_v1"}
    flat = {"batch_size_per_gpu": 8, "num_workers": 0}
    monkeypatch.setattr(build, "_build_flat_config", lambda cfg: (flat, lerobot))
    calls = []
    inner = SimpleNamespace(
        get_val_dataloader=lambda: ([0], None),
        get_train_dataloader=lambda: ([0, 1], None),
    )

    def selected(*args, **kwargs):
        calls.append((args, kwargs))
        return inner, 2

    def forbidden(*args, **kwargs):
        raise AssertionError("The other dataset loader must not run")

    monkeypatch.setattr(loader, "load_lerobot_data", forbidden if b601 else selected)
    monkeypatch.setattr(b601_adapter, "load_b601_data", selected if b601 else forbidden)
    wrapper, train_loader, train_num = build.load_lerobot_v2(config)
    assert wrapper._inner is inner
    assert train_loader == [0, 1] and train_num == 2
    assert calls == [((flat, lerobot), {"rank": 0, "world_size": 1, "seed": 10222})]


def test_camera_order_view_preserves_source_and_can_be_pickled():
    from wall_x.data.backends.lerobot.b601_adapter import _CameraOrderDataset

    raw = SimpleNamespace(meta=SimpleNamespace(camera_keys=["side", "front"], fps=30))
    view = _CameraOrderDataset(raw, ["front", "side"])
    restored = pickle.loads(pickle.dumps(view))
    assert view.meta.camera_keys == restored.meta.camera_keys == ["front", "side"]
    assert restored.meta.fps == 30
    assert raw.meta.camera_keys == ["side", "front"]
