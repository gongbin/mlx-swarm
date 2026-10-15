"""Node registry and request routing for the coordinator (no I/O, easy to test).

Routing picks, among healthy nodes serving the requested model, the one with
the lowest expected wait: (in-flight requests + 1) / node speed. A session
(``X-Session-Id`` header or the OpenAI ``user`` field) sticks to the node that
served it last, so mlx-lm's prompt cache gets reused, unless that node is
clearly busier than the best alternative.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

HEARTBEAT_TIMEOUT_S = 15.0
FAILURE_COOLDOWN_S = 10.0
AFFINITY_SLACK = 2  # extra in-flight requests tolerated to keep a session on its node
MIN_FREE_MEMORY_PCT = 10


@dataclass
class Node:
    node_id: str
    url: str  # worker base URL, e.g. http://169.254.101.6:8081
    model: str  # model name clients use, e.g. "Qwen3-8B-4bit"
    speed: float = 1.0  # relative decode speed (e.g. measured memory bandwidth)
    info: dict = field(default_factory=dict)
    last_seen: float = 0.0
    memory_free_pct: float | None = None
    inflight: int = 0
    served: int = 0
    failures: int = 0
    cooldown_until: float = 0.0
    draining: bool = False  # set from the dashboard: no new requests or batch tasks
    stats: dict = field(default_factory=dict)  # from heartbeats: disk, load, ...
    registered_at: float = 0.0
    token_id: str | None = None  # credential that registered it ("admin", a node token id, or None: open)
    connected: bool = True  # tunnelled workers: whether the agent's tunnel is up
    remote: str | None = None  # address the agent connects from

    def healthy(self, now: float) -> bool:
        return (
            self.connected
            and now - self.last_seen < HEARTBEAT_TIMEOUT_S
            and now >= self.cooldown_until
            and (self.memory_free_pct is None or self.memory_free_pct >= MIN_FREE_MEMORY_PCT)
        )

    def load(self) -> float:
        return (self.inflight + 1) / max(self.speed, 1e-6)

    def to_dict(self, now: float) -> dict:
        return {
            "node_id": self.node_id,
            "url": self.url,
            "model": self.model,
            "speed": self.speed,
            "healthy": self.healthy(now),
            "last_seen_s": round(now - self.last_seen, 1),
            "memory_free_pct": self.memory_free_pct,
            "inflight": self.inflight,
            "served": self.served,
            "failures": self.failures,
            "draining": self.draining,
            "tunnel": self.url.startswith("tunnel://"),
            "connected": self.connected,
            "remote": self.remote,
            "token_id": self.token_id,
            "up_s": round(now - self.registered_at, 1),
            "stats": self.stats,
            "info": self.info,
        }


class NoNodeAvailable(LookupError):
    pass


class Registry:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.nodes: dict[str, Node] = {}
        self.sessions: dict[str, str] = {}  # session key -> node_id

    def register(self, node_id: str, url: str, model: str, speed: float = 1.0, info=None) -> Node:
        node = self.nodes.get(node_id)
        if node is None or node.url != url or node.model != model:
            node = Node(node_id, url, model, registered_at=self.clock())
            self.nodes[node_id] = node
        node.speed = speed
        node.info = info or {}
        node.last_seen = self.clock()
        node.cooldown_until = 0.0
        return node

    def heartbeat(self, node_id: str, memory_free_pct: float | None = None, stats: dict | None = None) -> bool:
        node = self.nodes.get(node_id)
        if node is None:
            return False
        node.last_seen = self.clock()
        node.memory_free_pct = memory_free_pct
        if stats is not None:
            node.stats = stats
        return True

    def deregister(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)

    def models(self) -> list[str]:
        now = self.clock()
        return sorted({n.model for n in self.nodes.values() if n.healthy(now)})

    def pick(self, model: str | None, session: str | None = None, exclude=()) -> Node:
        now = self.clock()
        candidates = [
            n
            for n in self.nodes.values()
            if n.healthy(now) and not n.draining and n.node_id not in exclude
            and (model is None or n.model == model)
        ]
        if model is None and len({n.model for n in candidates}) > 1:
            raise NoNodeAvailable("several models are served; set the 'model' field")
        if not candidates:
            raise NoNodeAvailable(
                f"no healthy node serves model {model!r}; available: {self.models()}"
            )
        best = min(candidates, key=Node.load)
        if session:
            sticky = self.nodes.get(self.sessions.get(session, ""))
            if sticky in candidates and sticky.inflight <= best.inflight + AFFINITY_SLACK:
                best = sticky
            self.sessions[session] = best.node_id
        return best

    def failed(self, node: Node) -> None:
        node.failures += 1
        node.cooldown_until = self.clock() + FAILURE_COOLDOWN_S
