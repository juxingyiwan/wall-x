"""Use the SAME advisory lock files as the existing Cosmos client."""

from contextlib import ExitStack, contextmanager
from pathlib import Path

from .contracts import HardwareError


@contextmanager
def robot_locks(paths):
    import fcntl

    with ExitStack() as stack:
        for name in paths:
            path = Path(name).expanduser()
            # Never unlink/recreate a lock: that would create a different lock inode.
            if not path.parent.is_dir():
                raise HardwareError(f"Shared lock directory missing: {path.parent}")
            handle = stack.enter_context(path.open("a"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise HardwareError(f"Robot is already reserved: {path}") from exc
        yield
