"""One request in flight per persistent WebSocket; no blind request retries."""

import msgpack
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from .config import PolicyConfig
from .contracts import ProtocolError, TransportError
from .policy import PolicyConnection
from .protocol import unpack


class WallXConnection(PolicyConnection):
    def __init__(self, config: PolicyConfig, rtc=None):
        super().__init__(config, rtc=rtc)
        self.socket = None

    def open(self) -> dict:
        self.close()
        try:
            self.socket = connect(
                self.config.uri,
                open_timeout=self.config.connect_timeout_s,
                close_timeout=1,
                max_size=self.config.max_message_bytes,
                max_queue=1,
                compression=None,
                ping_interval=10,
                ping_timeout=10,
            )
            metadata = unpack(self.socket.recv(timeout=self.config.connect_timeout_s))
            return self._metadata(metadata)
        except ProtocolError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise TransportError(f"Cannot connect to {self.config.uri}: {exc}") from exc

    def _ensure_open(self):
        if self.socket is None:
            self.open()

    def _encode(self, body):
        return msgpack.packb(body, use_bin_type=True)

    def _exchange(self, payload, request_id):
        try:
            socket = self.socket
            if socket is None:
                raise TransportError("Policy connection closed; pending action discarded")
            socket.send(payload)
            response = unpack(socket.recv(timeout=self.config.request_timeout_s))
            if self.socket is not socket:
                raise TransportError("Policy connection changed; pending action discarded")
            return response
        except (TimeoutError, ConnectionClosed, OSError) as exc:
            self.close()
            raise TransportError(f"Policy request failed; pending action discarded: {exc}") from exc

    def close(self) -> None:
        socket, self.socket = self.socket, None
        if socket is not None:
            socket.close()
