"""swarm: run an MLX inference pool across Apple Silicon Macs.

    swarm coordinator [--port 8000]                    API gateway (run once)
    swarm agent --coordinator URL --model PATH ...     on every Mac that serves
    swarm worker --model PATH ...                      a bare guarded worker
    swarm nodes [--coordinator URL]                    list registered nodes
    swarm chat "prompt" [--model NAME]                 one streaming request
    swarm jobs submit FILE.jsonl [--model M] [--wait]  batch job (OpenAI Batch input format)
    swarm jobs list | status ID | results ID [-o F] | cancel ID
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


def read_tasks(path: str) -> list[dict]:
    """OpenAI Batch input lines ({custom_id, url, body}), or bare prompts as {"prompt": ...}."""
    tasks = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "body" not in row:
                if "prompt" not in row:
                    sys.exit(f"{path}:{n}: need either body or prompt")
                row = {"custom_id": row.get("custom_id"), "body": {
                    "messages": [{"role": "user", "content": row["prompt"]}],
                    **{k: v for k, v in row.items() if k not in ("custom_id", "prompt")},
                }}
            tasks.append(row)
    return tasks


def print_job(job: dict) -> None:
    c = job["request_counts"]
    print(f"{job['id']}  {job['status']}  done {c['completed']}/{c['total']}  failed {c['failed']}"
          f"  running {c['running']}  queued {c['queued']}  cancelled {c['cancelled']}")


def jobs(argv: list[str]) -> None:
    import time

    import httpx

    parser = argparse.ArgumentParser(prog="swarm jobs")
    parser.add_argument("--coordinator", default=DEFAULT_COORDINATOR)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("submit")
    p.add_argument("file")
    p.add_argument("--model", default=None, help="overrides each task's body.model")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--key", default=None, help="idempotency key: resubmitting returns the same job")
    p.add_argument("--wait", action="store_true", help="poll until done, then print results path")
    p.add_argument("-o", "--output", default=None, help="with --wait: write results here")
    sub.add_parser("list")
    for name in ("status", "cancel"):
        sub.add_parser(name).add_argument("id")
    p = sub.add_parser("results")
    p.add_argument("id")
    p.add_argument("-o", "--output", default=None)
    args = parser.parse_args(argv)
    base = args.coordinator.rstrip("/") + "/v1/jobs"

    def call(method, url, **kw):
        r = httpx.request(method, url, timeout=60, **kw)
        if r.status_code >= 400:
            sys.exit(f"error {r.status_code}: {r.text}")
        return r

    def save_results(job_id, output):
        text = call("GET", f"{base}/{job_id}/results").text
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"results written to {output}", file=sys.stderr)
        else:
            sys.stdout.write(text)

    if args.cmd == "submit":
        body = {"tasks": read_tasks(args.file), "max_attempts": args.max_attempts,
                "metadata": {"file": args.file}}
        if args.model:
            body["model"] = args.model
        if args.key:
            body["idempotency_key"] = args.key
        r = call("POST", base, json=body)
        job = r.json()
        if r.status_code == 200:
            print("(existing job for this idempotency key)", file=sys.stderr)
        print_job(job)
        if args.wait:
            while job["status"] != "completed":
                time.sleep(2)
                job = call("GET", f"{base}/{job['id']}").json()
                print_job(job)
            save_results(job["id"], args.output)
    elif args.cmd == "list":
        for job in call("GET", base).json()["data"]:
            print_job(job)
    elif args.cmd == "status":
        print_job(call("GET", f"{base}/{args.id}").json())
    elif args.cmd == "results":
        save_results(args.id, args.output)
    elif args.cmd == "cancel":
        print_job(call("POST", f"{base}/{args.id}/cancel").json())


def main() -> None:
    commands = {
        "coordinator": lambda argv: __import__("swarm.coordinator").coordinator.main(argv),
        "agent": lambda argv: __import__("swarm.agent").agent.main(argv),
        "worker": lambda argv: __import__("swarm.worker").worker.main(argv),
        "nodes": nodes,
        "chat": chat,
        "jobs": jobs,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
