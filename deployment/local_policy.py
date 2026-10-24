"""Native Wall-X calls in the controller process, without sockets or MessagePack.

A single inference thread retains the existing request deadline: a late result
is discarded and the failed session cannot issue another request. Hardware is
owned exclusively by the existing controller, never by this worker.
"""

import queue
import threading
from concurrent.futures import Future, TimeoutError

from wallx_client.contracts import ProtocolError, TransportError
from wallx_client.policy import PolicyConnection


class _InferenceWorker:
    def __init__(self, policy):
        self.jobs = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self.thread = threading.Thread(
            target=self._run, args=(policy, self.jobs), name="wallx-local-model", daemon=True
        )
        self.thread.start()

    @staticmethod
    def _run(policy, jobs):
        while (job := jobs.get()) is not None:
            future, body = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(policy.infer(body))
            except BaseException as exc:
                future.set_exception(exc)

    def submit(self, body):
        with self._lock:
            if self._closed:
                raise TransportError("Local model worker is closed")
            future = Future()
            self.jobs.put((future, body))
            return future

    def close(self, timeout_s=0):
        # CUDA kernels cannot be cancelled safely. No further work is accepted;
        # the daemon releases its model reference after the pending call returns.
        with self._lock:
            if not self._closed:
                self._closed = True
                while True:
                    try:
                        self.jobs.get_nowait()[0].cancel()
                    except queue.Empty:
                        break
                self.jobs.put(None)
        self.thread.join(max(0, timeout_s))
        alive = self.thread.is_alive()
        return {
            "kind": "thread",
            "owner": "local_model",
            "status": "running_daemon" if alive else "joined",
            "timed_out": alive,
            "thread_alive": alive,
            "model_reference_held_by_worker": alive,
            "session_reusable": False,
        }


class LocalPolicyConnection(PolicyConnection):
    """The same controller interface as WallXConnection, backed by a Python object."""

    def __init__(self, config, policy, *, rtc=None):
        super().__init__(config, rtc=rtc)
        self.policy = policy
        self._state = threading.Lock()
        self._worker = None
        self._closed = False

    def open(self):
        with self._state:
            if self._closed:
                raise TransportError("Local inference session is closed; start a new session")
            return self._metadata(self.policy.metadata)

    def _ensure_open(self):
        self.open()

    def _exchange(self, body, request_id):
        with self._state:
            if self._closed:
                raise TransportError("Local inference session is closed")
            if self._worker is None:
                self._worker = _InferenceWorker(self.policy)
            future = self._worker.submit(body)
        try:
            response = future.result(timeout=self.config.request_timeout_s)
        except TimeoutError as exc:
            self.close()
            raise TransportError(
                "Local policy request timed out; pending action discarded"
            ) from exc
        except Exception as exc:
            self.close()
            raise ProtocolError(f"Local policy inference failed: {exc}") from exc
        with self._state:
            if self._closed:
                raise TransportError("Local inference closed; pending action discarded")
        if not isinstance(response, dict):
            raise ProtocolError("Policy response must be a mapping")
        # The native infer() does not add the server's request envelope. Add it
        # only when absent: an explicit mismatched ID must fail the shared check.
        return {"request_id": request_id, **response}

    def close(self):
        self.close_with_status(0)

    def close_with_status(self, timeout_s):
        """Close ownership now; wait only within the caller's remaining budget.

        Repeated calls re-observe the worker even after request timeout closed
        the connection. Model references owned by external callers are outside
        this report; thread completion does not assert CUDA memory was freed.
        """
        with self._state:
            self._closed = True
            worker = self._worker
            self.policy = None
        if worker is not None:
            return worker.close(timeout_s)
        return {
            "kind": "thread",
            "owner": "local_model",
            "status": "not_started",
            "timed_out": False,
            "thread_alive": False,
            "model_reference_held_by_worker": False,
            "session_reusable": False,
        }
