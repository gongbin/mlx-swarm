"""Rolling request metrics for the dashboard (in memory; a restart clears them).

Each finished request is one event: node, API key, token counts, duration and
time to first token. ``summary`` turns the last few minutes into per-node
throughput and a time series in which each request's output tokens are spread
evenly over the time it was generating.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

WINDOW_S = 600.0
BUCKET_S = 5.0
RATE_WINDOW_S = 60.0


@dataclass
class Event:
    end: float
    node_id: str
    model: str
    kind: str  # "chat" | "completion" | "batch"
    key: str | None
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    ttft_s: float | None
    ok: bool


class Metrics:
    def __init__(self, clock: Callable[[], float] = time.time, maxlen: int = 5000):
        self.clock = clock
        self.events: deque[Event] = deque(maxlen=maxlen)
        self.started = clock()
        self.totals = {"requests": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def record(self, node_id: str, model: str, kind: str, key: str | None = None,
               prompt_tokens: int = 0, completion_tokens: int = 0, seconds: float = 0.0,
               ttft_s: float | None = None, ok: bool = True) -> None:
        self.events.append(Event(self.clock(), node_id, model, kind, key, prompt_tokens,
                                 completion_tokens, seconds, ttft_s, ok))
        self.totals["requests"] += 1
        self.totals["errors"] += not ok
        self.totals["prompt_tokens"] += prompt_tokens
        self.totals["completion_tokens"] += completion_tokens

    def summary(self, recent: int = 20) -> dict:
        now = self.clock()
        while self.events and now - self.events[0].end > WINDOW_S:
            self.events.popleft()
        n_buckets = int(WINDOW_S / BUCKET_S)
        origin = now - n_buckets * BUCKET_S
        series: dict[str, list[float]] = {}
        nodes: dict[str, dict] = {}
        for e in self.events:
            # Spread the output tokens over [end - seconds, end].
            start = e.end - max(e.seconds, 1e-3)
            per_s = e.completion_tokens / max(e.seconds, 1e-3)
            row = series.setdefault(e.node_id, [0.0] * n_buckets)
            b = max(0, int((start - origin) // BUCKET_S))
            while b < n_buckets:
                lo, hi = origin + b * BUCKET_S, origin + (b + 1) * BUCKET_S
                overlap = min(hi, e.end) - max(lo, start)
                if overlap <= 0 and lo >= e.end:
                    break
                if overlap > 0:
                    row[b] += per_s * overlap / BUCKET_S
                b += 1
            if now - e.end <= RATE_WINDOW_S:
                n = nodes.setdefault(e.node_id, {"requests": 0, "errors": 0, "completion_tokens": 0,
                                                 "ttft": []})
                n["requests"] += 1
                n["errors"] += not e.ok
                n["completion_tokens"] += e.completion_tokens
                if e.ttft_s is not None:
                    n["ttft"].append(e.ttft_s)
        for node_id, row in series.items():
            # The last full minute of the series, as tokens per second.
            tail = row[-int(RATE_WINDOW_S / BUCKET_S):]
            n = nodes.setdefault(node_id, {"requests": 0, "errors": 0, "completion_tokens": 0, "ttft": []})
            n["tok_s"] = round(sum(tail) / len(tail), 2)
        for n in nodes.values():
            ttft = sorted(n.pop("ttft"))
            n["ttft_p50_s"] = round(ttft[len(ttft) // 2], 3) if ttft else None
            n.setdefault("tok_s", 0.0)
        return {
            "now": now,
            "uptime_s": round(now - self.started, 1),
            "totals": dict(self.totals),
            "bucket_s": BUCKET_S,
            "series": {k: [round(v, 2) for v in row] for k, row in series.items()},
            "nodes": nodes,
            "recent": [asdict(e) for e in list(self.events)[-recent:][::-1]],
        }
