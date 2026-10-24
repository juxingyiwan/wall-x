"""Present the local Rerun viewer without touching other viewers or robot devices.

Rerun's spawn reuses a listening port but does not activate that desktop window.
Keep X11 calls in a bounded helper process: a display server must never stall
robot startup or become a dependency of the control loop.
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path


def present_local_viewer(port, recording_id):
    if sys.platform != "linux" or not os.environ.get("DISPLAY"):
        return {"status": "unsupported_desktop"}
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(port), recording_id],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        logging.getLogger(__name__).warning(
            "Rerun display is connected on port %s, but its window could not be activated: %s",
            port,
            exc,
        )
        return {"status": "activation_failed", "error": str(exc)}


def _is_viewer(pid, port, *, proc=Path("/proc")):
    """Match the native viewer's owner, executable and exact configured port."""
    path = proc / str(pid)
    try:
        if path.stat().st_uid != os.getuid() or (path / "exe").resolve().name != "rerun":
            return False
        argv = (path / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
    except (OSError, UnicodeError):
        return False  # The viewer may close while the desktop list is being read.
    for index, arg in enumerate(argv[1:], 1):
        if arg.startswith("--port="):
            return arg == f"--port={port}"
        if arg == "--port":
            return index + 1 < len(argv) and argv[index + 1] == str(port)
    return False


def _present(port, recording_id):
    from Xlib import X, display, protocol

    connection = display.Display()
    try:
        root = connection.screen().root
        clients_atom = connection.intern_atom("_NET_CLIENT_LIST_STACKING")
        pid_atom = connection.intern_atom("_NET_WM_PID")
        active_atom = connection.intern_atom("_NET_ACTIVE_WINDOW")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            clients = root.get_full_property(clients_atom, X.AnyPropertyType)
            for wid in () if clients is None else reversed(clients.value):
                window = connection.create_resource_object("window", int(wid))
                owner = window.get_full_property(pid_atom, X.AnyPropertyType)
                if owner is None or not owner.value or not _is_viewer(int(owner.value[0]), port):
                    continue
                title = f"Wall-X - Rerun ({port}) - {recording_id}"
                window.set_wm_name(title)
                window.change_property(
                    connection.intern_atom("_NET_WM_NAME"),
                    connection.intern_atom("UTF8_STRING"),
                    8,
                    title.encode(),
                )
                root.send_event(
                    protocol.event.ClientMessage(
                        window=window,
                        client_type=active_atom,
                        data=(32, [2, X.CurrentTime, 0, 0, 0]),
                    ),
                    event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask,
                )
                connection.flush()
                while time.monotonic() < deadline:
                    active = root.get_full_property(active_atom, X.AnyPropertyType)
                    if active is not None and len(active.value) and int(active.value[0]) == wid:
                        return {"status": "activated", "pid": int(owner.value[0]), "title": title}
                    time.sleep(0.05)
                raise RuntimeError("Desktop did not activate the Rerun window")
            time.sleep(0.05)
        raise RuntimeError(f"No local Rerun window found for port {port}")
    finally:
        connection.close()


if __name__ == "__main__":
    print(json.dumps(_present(int(sys.argv[1]), sys.argv[2])))
