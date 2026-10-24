"""Episode-owned workers with bounded teardown and no motor access.

Python cannot kill a running thread. Inference therefore runs on a daemon owned
here (not concurrent.futures' interpreter-joined threads). A timed-out thread is
reported as still running; closing its transport is the cooperative interrupt.
"""

import queue
import threading
from concurrent.futures import Future


class WorkerClosed(RuntimeError):
    pass


class DaemonWorker:
    def __init__(self, max_workers=1, thread_name_prefix="wallx-worker"):
        if max_workers != 1:
            raise ValueError("An episode worker has exactly one owner thread")
        self._jobs = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=thread_name_prefix, daemon=True)
        self._thread.start()

    def submit(self, fn, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise WorkerClosed("Worker already closed")
            future = Future()
            self._jobs.put((future, fn, args, kwargs))
            return future

    def _run(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            future, fn, args, kwargs = job
            with self._lock:
                if self._closed:
                    future.cancel()
                run = future.set_running_or_notify_cancel()
            if run:
                try:
                    future.set_result(fn(*args, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)

    def _stop(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            while True:
                try:
                    job = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    job[0].cancel()
            self._jobs.put(None)

    def close(self, timeout):
        self._stop()
        self._thread.join(max(0, timeout))
        alive = self._thread.is_alive()
        return {
            "kind": "thread",
            "status": "running_daemon" if alive else "joined",
            "timed_out": alive,
        }

    def shutdown(self, wait=True, cancel_futures=True):
        return self.close(2.0 if wait else 0.0)


def bounded_call(fn, timeout):
    """Also bound third-party transport close; never claim thread cancellation."""
    worker = DaemonWorker(thread_name_prefix="wallx-cleanup")
    future = worker.submit(fn)
    try:
        result = future.result(timeout=max(0, timeout))
        return result
    finally:
        worker.close(0)
