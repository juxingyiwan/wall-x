"""LIBERO state-file compatibility and evaluation resource lifecycle."""

import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from wall_x._vendor.harrix.envs.libero_state_io import load_task_initial_states


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_initial_states_remain_restricted_and_preserve_values(tmp_path, dtype):
    path = tmp_path / "task.init"
    expected = np.arange(24, dtype=dtype).reshape(3, 8)
    torch.save(expected, path)
    suite = SimpleNamespace(
        get_task_init_states=lambda task_id: torch.load(path, weights_only=True)
    )
    existing = set(torch.serialization.get_safe_globals())
    additions = [] if np.ndarray in existing else [np.ndarray]
    with torch.serialization.safe_globals(additions):
        before = set(torch.serialization.get_safe_globals())
        actual = load_task_initial_states(suite, 0)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert set(torch.serialization.get_safe_globals()) == before


class UnsupportedState:
    def __reduce__(self):
        return (eval, ("1 + 1",))


def test_initial_state_allowlist_does_not_enable_arbitrary_pickle(tmp_path):
    path = tmp_path / "unsupported.init"
    torch.save(UnsupportedState(), path)
    suite = SimpleNamespace(
        get_task_init_states=lambda task_id: torch.load(path, weights_only=True)
    )
    with pytest.raises(pickle.UnpicklingError):
        load_task_initial_states(suite, 0)


@pytest.mark.parametrize("failure", ["worker", "episodes", "reset"])
def test_driver_closes_started_workers_when_evaluation_fails(
    monkeypatch, tmp_path, failure
):
    from wall_x._vendor.harrix.drivers import inproc
    from wall_x._vendor.harrix.envs import registry
    from wall_x._vendor.harrix.utils import seed

    closed = []
    cfg = SimpleNamespace(
        env=SimpleNamespace(seed=7, type="fixture"),
        runtime=SimpleNamespace(num_workers=2, log_dir=str(tmp_path)),
    )
    monkeypatch.setattr(seed, "set_seed_everywhere", lambda _: None)
    monkeypatch.setattr(
        inproc,
        "DirectModelHandle",
        lambda _: SimpleNamespace(shutdown=lambda: closed.append("model")),
    )

    def worker(_cfg, worker_id):
        if failure == "worker" and worker_id == 1:
            raise RuntimeError("worker failed")
        return SimpleNamespace(shutdown=lambda: closed.append(worker_id))

    def episodes(_cfg):
        if failure == "episodes":
            raise RuntimeError("episodes failed")
        return [("fixture", 0, 0)]

    def frame(*_args):
        raise RuntimeError("reset failed")

    monkeypatch.setattr(inproc, "SubprocEnvHandle", worker)
    monkeypatch.setattr(registry, "enumerate_episodes_for", episodes)
    monkeypatch.setattr(inproc, "_run_frame", frame)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        inproc.run(cfg)
    assert closed == ([0, "model"] if failure == "worker" else [1, 0, "model"])


def test_shutdown_terminates_an_unresponsive_worker_without_waiting_for_reply():
    from wall_x._vendor.harrix.drivers.inproc.env_handle import SubprocEnvHandle

    handle = object.__new__(SubprocEnvHandle)
    handle._parent_conn = Mock()
    handle._parent_conn.poll.return_value = False
    handle._proc = Mock()
    handle._proc.is_alive.side_effect = [True, True, False]
    handle._has_pending = False
    handle.shutdown()
    handle._parent_conn.recv.assert_not_called()
    handle._parent_conn.close.assert_called_once()
    handle._proc.terminate.assert_called_once()
    assert all(
        call.kwargs.get("timeout") == 2 for call in handle._proc.join.call_args_list
    )
