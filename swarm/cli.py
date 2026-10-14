"""swarm: run an MLX inference pool across Apple Silicon Macs.

    swarm coordinator [--port 8000]                    API gateway (run once)
    swarm agent --coordinator URL --model PATH ...     on every Mac that serves
    swarm worker --model PATH ...                      a bare guarded worker
    swarm nodes [--coordinator URL]                    list registered nodes
    swarm chat "prompt" [--model NAME]                 one streaming request
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_COORDINATOR = os.environ.get("SWARM_COORDINATOR", "http://127.0.0.1:8000")


def nodes(argv: list[str]) -> None:
    import httpx

    parser = argparse.ArgumentParser(prog="swarm nodes")
    parser.add_argument("--coordinator", default=DEFAULT_COORDINATOR)
    args = parser.parse_args(argv)
    data = httpx.get(args.coordinator.rstrip("/") + "/nodes", timeout=5).json()["nodes"]
    if not data:
        print("no nodes registered")
        return
    rows = [("NODE", "MODEL", "HEALTHY", "MEM FREE", "INFLIGHT", "SERVED", "URL")]
    for n in data:
        mem = f"{n['memory_free_pct']:.0f}%" if n["memory_free_pct"] is not None else "-"
        rows.append(
            (n["node_id"], n["model"], "yes" if n["healthy"] else "no", mem,
             str(n["inflight"]), str(n["served"]), n["url"])
        )
    widths = [max(len(r[i]) for r in rows) + 2 for i in range(len(rows[0]) - 1)]
    for r in rows:
        print("".join(c.ljust(w) for c, w in zip(r, widths)) + r[-1])


def chat(argv: list[str]) -> None:
    import httpx

    parser = argparse.ArgumentParser(prog="swarm chat")
    parser.add_argument("prompt")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--session", default=None)
    parser.add_argument("--coordinator", default=DEFAULT_COORDINATOR)
    args = parser.parse_args(argv)
    body = {
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
    }
    if args.model:
        body["model"] = args.model
    headers = {"x-session-id": args.session} if args.session else {}
    url = args.coordinator.rstrip("/") + "/v1/chat/completions"
    with httpx.stream("POST", url, json=body, headers=headers, timeout=600) as r:
        if r.status_code != 200:
            r.read()
            sys.exit(f"error {r.status_code}: {r.text}")
        print(f"[{r.headers.get('x-swarm-node')}] ", end="", flush=True, file=sys.stderr)
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            print(delta.get("content") or "", end="", flush=True)
    print()


def main() -> None:
    commands = {
        "coordinator": lambda argv: __import__("swarm.coordinator").coordinator.main(argv),
        "agent": lambda argv: __import__("swarm.agent").agent.main(argv),
        "worker": lambda argv: __import__("swarm.worker").worker.main(argv),
        "nodes": nodes,
        "chat": chat,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
