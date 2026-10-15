"""Reverse tunnel: an agent dials out to the coordinator, requests come back through it.

On a LAN the coordinator connects to each worker's URL. A Mac behind a home
router (NAT) cannot accept that connection, so with ``swarm agent --tunnel``
the agent opens a WebSocket to ``/internal/tunnel/{node_id}`` instead and the
coordinator sends requests down it. The worker then listens on 127.0.0.1 only.

Frames are JSON text messages; ``id`` multiplexes concurrent requests:

    coordinator -> agent   {"t": "req", "id", "method", "path", "body"}
                           {"t": "cancel", "id"}           (client went away)
    agent -> coordinator   {"t": "start", "id", "status", "content_type"}
                           {"t": "data", "id", "data"}     (response text, in order)
                           {"t": "end", "id"} | {"t": "error", "id", "error"}

WebSocket pings (both sides, every 20 s) keep NAT mappings open and detect a
dead link. ``Tunnel`` (coordinator, asyncio) hands the proxy a response object
with the parts of ``httpx.Response`` it uses, and raises ``TunnelError`` (an
``httpx.TransportError``) so the proxy's failover treats a broken tunnel like
an unreachable worker. ``TunnelClient`` (agent, threads) does the other half.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import threading
import time
from urllib.parse import urlparse

import httpx

log = logging.getLogger("swarm.tunnel")

SCHEME = "tunnel://"  # node URL of a tunnelled worker: tunnel://<node_id>
START_TIMEOUT_S = 600.0  # non-streaming responses start only when generation is done


class TunnelError(httpx.TransportError):
    def __init__(self, message: str):
        super().__init__(message)


# -- coordinator side ----------------------------------------------------------
class Tunnel:
    def __init__(self, ws, node_id: str, remote: str | None = None):
        self.ws = ws
        self.node_id = node_id
        self.remote = remote
        self.pending: dict[int, asyncio.Queue] = {}
        self.ids = itertools.count(1)
        self.closed = False
        self.since = time.time()

    async def run(self) -> None:
        """Dispatch frames to the requests waiting for them, until the socket closes."""
        try:
            while True:
                frame = json.loads(await self.ws.receive_text())
                queue = self.pending.get(frame.get("id"))
                if queue is not None:
                    queue.put_nowait(frame)
        except Exception as e:  # WebSocketDisconnect, bad JSON, ...
            log.info("tunnel from %s closed: %r", self.node_id, e)
        finally:
            self.closed = True
            for queue in self.pending.values():
                queue.put_nowait({"t": "error", "error": "tunnel closed"})

    async def close(self) -> None:
        self.closed = True
        try:
            await self.ws.close()
        except Exception:
            pass

    async def request(self, method: str, path: str, body) -> "TunnelResponse":
        if self.closed:
            raise TunnelError(f"tunnel to {self.node_id} is closed")
        rid = next(self.ids)
        queue: asyncio.Queue = asyncio.Queue()
        self.pending[rid] = queue
        try:
            await self.ws.send_text(json.dumps({"t": "req", "id": rid, "method": method, "path": path,
                                                "body": body}))
            first = await asyncio.wait_for(queue.get(), START_TIMEOUT_S)
        except Exception as e:
            self.pending.pop(rid, None)
            raise TunnelError(f"tunnel to {self.node_id}: {e!r}") from e
        if first.get("t") != "start":
            self.pending.pop(rid, None)
            raise TunnelError(f"tunnel to {self.node_id}: {first.get('error', 'no response')}")
        return TunnelResponse(self, rid, queue, int(first["status"]),
                              first.get("content_type") or "application/json")


class TunnelResponse:
    def __init__(self, tunnel: Tunnel, rid: int, queue: asyncio.Queue, status: int, content_type: str):
        self.tunnel, self.rid, self.queue = tunnel, rid, queue
        self.status_code = status
        self.headers = {"content-type": content_type}
        self.done = False

    async def aiter_text(self):
        while not self.done:
            frame = await self.queue.get()
            kind = frame.get("t")
            if kind == "data":
                yield frame["data"]
            elif kind == "end":
                self.done = True
            else:
                self.done = True
                raise TunnelError(f"tunnel to {self.tunnel.node_id}: {frame.get('error', kind)}")

    async def aiter_lines(self):
        buf = ""
        async for text in self.aiter_text():
            buf += text
            *lines, buf = buf.split("\n")
            for line in lines:
                yield line.rstrip("\r")
        if buf:
            yield buf

    async def aread(self) -> bytes:
        return "".join([t async for t in self.aiter_text()]).encode()

    async def aclose(self) -> None:
        self.tunnel.pending.pop(self.rid, None)
        if not self.done and not self.tunnel.closed:
            self.done = True
            try:  # stop the generation on the worker
                await self.tunnel.ws.send_text(json.dumps({"t": "cancel", "id": self.rid}))
            except Exception:
                pass


# -- agent side ----------------------------------------------------------------
def tunnel_url(coordinator: str, node_id: str) -> str:
    u = urlparse(coordinator)
    scheme = "wss" if u.scheme == "https" else "ws"
    return f"{scheme}://{u.netloc}{u.path.rstrip('/')}/internal/tunnel/{node_id}"


class TunnelClient:
    """Keeps a tunnel to the coordinator open and serves its requests from the local worker."""

    def __init__(self, coordinator: str, node_id: str, local_url: str, headers: dict,
                 timeout: float = 900.0):
        self.coordinator = coordinator
        self.node_id = node_id
        self.local_url = local_url
        self.headers = headers
        self.timeout = timeout
        self.stopping = threading.Event()
        self.connected = threading.Event()
        self.ws = None
        self.send_lock = threading.Lock()
        self.cancelled: dict[int, threading.Event] = {}
        self.http = httpx.Client(timeout=httpx.Timeout(connect=5.0, read=timeout, write=30.0, pool=5.0))

    def start(self) -> None:
        threading.Thread(target=self.loop, name="tunnel", daemon=True).start()

    def stop(self) -> None:
        self.stopping.set()
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass

    def loop(self) -> None:
        from websockets.sync.client import connect

        backoff = 1.0
        while not self.stopping.is_set():
            url = tunnel_url(self.coordinator, self.node_id)
            try:
                with connect(url, additional_headers=self.headers, open_timeout=10,
                             ping_interval=20, ping_timeout=20, max_size=2**24) as ws:
                    self.ws = ws
                    self.connected.set()
                    backoff = 1.0
                    log.info("tunnel open to %s", url)
                    for message in ws:
                        frame = json.loads(message)
                        if frame.get("t") == "req":
                            self.cancelled[frame["id"]] = threading.Event()
                            threading.Thread(target=self.forward, args=(frame,), daemon=True).start()
                        elif frame.get("t") == "cancel" and frame.get("id") in self.cancelled:
                            self.cancelled[frame["id"]].set()
            except Exception as e:
                if not self.stopping.is_set():
                    log.warning("tunnel to %s failed: %r; retrying in %.0fs", url, e, backoff)
            finally:
                self.ws = None
                self.connected.clear()
                for event in self.cancelled.values():
                    event.set()
            self.stopping.wait(backoff)
            backoff = min(backoff * 2, 30.0)

    def stop_socket(self) -> None:
        """Drop the current connection; the loop reconnects (e.g. to a new coordinator URL)."""
        ws = self.ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def send(self, frame: dict) -> None:
        ws = self.ws
        if ws is None:
            raise ConnectionError("tunnel is down")
        with self.send_lock:
            ws.send(json.dumps(frame))

    def forward(self, frame: dict) -> None:
        rid = frame["id"]
        cancelled = self.cancelled[rid]
        started = False
        try:
            with self.http.stream(frame.get("method", "POST"), self.local_url + frame["path"],
                                  json=frame.get("body")) as r:
                self.send({"t": "start", "id": rid, "status": r.status_code,
                           "content_type": r.headers.get("content-type")})
                started = True
                for text in r.iter_text():
                    if cancelled.is_set():
                        break  # closing the stream stops the worker's generation
                    if text:
                        self.send({"t": "data", "id": rid, "data": text})
            if not cancelled.is_set():
                self.send({"t": "end", "id": rid})
        except Exception as e:
            log.warning("tunnelled request %s failed%s: %r", rid, " mid-stream" if started else "", e)
            try:
                self.send({"t": "error", "id": rid, "error": repr(e)})
            except Exception:
                pass
        finally:
            self.cancelled.pop(rid, None)
