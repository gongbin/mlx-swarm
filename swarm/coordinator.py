"""Coordinator: OpenAI-compatible gateway in front of the swarm's workers.

Agents register their worker and send heartbeats; clients call the usual
OpenAI endpoints and are routed to a healthy worker serving the requested
model (see ``swarm.router``). If a worker cannot be reached, or fails before
sending a response, the request is retried once on another node.

Batch jobs (``/v1/jobs``) go into a SQLite queue (see ``swarm.jobs``); agents
lease tasks for the model they serve and post results back.

    swarm coordinator --host 0.0.0.0 --port 8000 --db jobs.sqlite3
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from swarm.jobs import JobError, JobStore
from swarm.router import NoNodeAvailable, Registry

log = logging.getLogger("swarm.coordinator")

MAX_ATTEMPTS = 2
UPSTREAM_TIMEOUT = httpx.Timeout(connect=3.0, read=600.0, write=30.0, pool=5.0)
WORKER_MODEL = "default_model"  # the name a worker's pinned model answers to


def error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def create_app(registry: Registry | None = None, client: httpx.AsyncClient | None = None,
               store: JobStore | None = None) -> FastAPI:
    reg = registry or Registry()
    http = client or httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    jobs = store or JobStore()

    @asynccontextmanager
    async def lifespan(app):
        yield
        await http.aclose()

    app = FastAPI(title="mlx-swarm coordinator", lifespan=lifespan)
    app.state.registry = reg
    app.state.jobs = jobs

    @app.post("/internal/nodes/register")
    async def register(body: dict):
        node = reg.register(
            body["node_id"], body["url"], body["model"], float(body.get("speed", 1.0)), body.get("info")
        )
        log.info("registered %s at %s serving %s", node.node_id, node.url, node.model)
        return {"ok": True}

    @app.post("/internal/nodes/{node_id}/heartbeat")
    async def heartbeat(node_id: str, body: dict):
        if not reg.heartbeat(node_id, body.get("memory_free_pct")):
            return error(404, f"unknown node {node_id}; register first")
        return {"ok": True}

    @app.post("/internal/nodes/{node_id}/deregister")
    async def deregister(node_id: str):
        reg.deregister(node_id)
        log.info("deregistered %s", node_id)
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"status": "ok", "models": reg.models()}

    @app.get("/nodes")
    async def nodes():
        now = reg.clock()
        return {"nodes": [n.to_dict(now) for n in reg.nodes.values()]}

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": "mlx-swarm"} for m in reg.models()],
        }

    async def proxy(request: Request, path: str):
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return error(400, "request body must be JSON")
        model = body.get("model")
        session = request.headers.get("x-session-id") or body.get("user")
        stream = bool(body.get("stream"))
        # Workers only serve their pinned model; name it explicitly so a stock
        # mlx_lm.server would not try to load the client's model name either.
        upstream = {**body, "model": WORKER_MODEL}
        pinned = re.compile(rb'"model":\s*"%s"' % re.escape(WORKER_MODEL.encode()))
        client_model = b'"model": ' + json.dumps(model).encode() if model else None

        def restore(data: bytes) -> bytes:
            return pinned.sub(lambda _: client_model, data) if client_model else data

        tried: set[str] = set()
        last_problem = ""
        for _ in range(MAX_ATTEMPTS):
            try:
                node = reg.pick(model, session, exclude=tried)
            except NoNodeAvailable as e:
                if tried:
                    return error(503, f"all workers failed ({last_problem})", "service_unavailable")
                return error(404 if model else 400, str(e))
            tried.add(node.node_id)
            node.inflight += 1
            started = time.perf_counter()
            try:
                resp = await http.send(
                    http.build_request("POST", node.url + path, json=upstream), stream=True
                )
            except httpx.TransportError as e:
                node.inflight -= 1
                reg.failed(node)
                last_problem = f"{node.node_id}: {e!r}"
                log.warning("worker %s unreachable: %r", node.node_id, e)
                continue
            if resp.status_code >= 500:
                await resp.aclose()
                node.inflight -= 1
                reg.failed(node)
                last_problem = f"{node.node_id}: HTTP {resp.status_code}"
                log.warning("worker %s returned %s", node.node_id, resp.status_code)
                continue

            headers = {"x-swarm-node": node.node_id}
            if not stream:
                try:
                    data = await resp.aread()
                finally:
                    await resp.aclose()
                    node.inflight -= 1
                node.served += 1
                log.info("%s -> %s %.2fs", path, node.node_id, time.perf_counter() - started)
                return Response(
                    restore(data),
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"),
                    headers=headers,
                )

            async def relay(resp=resp, node=node, started=started):
                try:
                    # Line by line, so the model-name rewrite never straddles chunks.
                    async for line in resp.aiter_lines():
                        yield restore(line.encode()) + b"\n"
                    node.served += 1
                except httpx.TransportError as e:
                    reg.failed(node)
                    log.warning("stream from %s broke: %r", node.node_id, e)
                finally:
                    await resp.aclose()
                    node.inflight -= 1
                    log.info("%s -> %s %.2fs (stream)", path, node.node_id, time.perf_counter() - started)

            return StreamingResponse(
                relay(),
                status_code=resp.status_code,
                media_type="text/event-stream",
                headers=headers,
            )
        return error(503, f"all workers failed ({last_problem})", "service_unavailable")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await proxy(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await proxy(request, "/v1/completions")

    # -- batch jobs: clients ------------------------------------------------
    @app.post("/v1/jobs")
    async def submit_job(body: dict):
        try:
            job_id, created = jobs.submit(
                body.get("tasks") or [],
                model=body.get("model"),
                max_attempts=int(body.get("max_attempts", 3)),
                idempotency_key=body.get("idempotency_key"),
                metadata=body.get("metadata"),
            )
        except (JobError, TypeError) as e:
            return error(400, str(e))
        log.info("job %s %s", job_id, "created" if created else "already exists (idempotency key)")
        return JSONResponse(jobs.job(job_id), status_code=201 if created else 200)

    @app.get("/v1/jobs")
    async def list_jobs(limit: int = 20):
        return {"object": "list", "data": jobs.jobs(limit)}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        return jobs.job(job_id) or error(404, f"no job {job_id}")

    @app.get("/v1/jobs/{job_id}/results")
    async def job_results(job_id: str):
        if jobs.job(job_id) is None:
            return error(404, f"no job {job_id}")
        lines = (json.dumps(r, ensure_ascii=False) + "\n" for r in jobs.results(job_id))
        return PlainTextResponse("".join(lines), media_type="application/jsonl")

    @app.post("/v1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        if jobs.job(job_id) is None:
            return error(404, f"no job {job_id}")
        jobs.cancel(job_id)
        return jobs.job(job_id)

    # -- batch jobs: agents -------------------------------------------------
    @app.post("/internal/tasks/lease")
    async def lease_tasks(body: dict):
        node = reg.nodes.get(body["node_id"])
        if node is None:
            return error(404, f"unknown node {body['node_id']}; register first")
        if not node.healthy(reg.clock()):
            return {"tasks": []}
        return {"tasks": jobs.lease(node.node_id, node.model, int(body.get("max", 1)))}

    @app.post("/internal/tasks/renew")
    async def renew_tasks(body: dict):
        return {"lost": jobs.renew(body["node_id"], body.get("leases") or [])}

    @app.post("/internal/tasks/{task_id}/complete")
    async def complete_task(task_id: int, body: dict):
        result = body["result"]
        if isinstance(result, dict) and result.get("model") == WORKER_MODEL:
            result["model"] = body.get("model") or result["model"]
        if not jobs.complete(task_id, body["token"], body["node_id"], result):
            return error(409, "stale fencing token; result discarded", "conflict")
        return {"ok": True}

    @app.post("/internal/tasks/{task_id}/fail")
    async def fail_task(task_id: int, body: dict):
        if not jobs.fail(task_id, body["token"], body["node_id"], body.get("error", ""),
                         bool(body.get("retryable", True))):
            return error(409, "stale fencing token", "conflict")
        return {"ok": True}

    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="swarm coordinator", description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default="jobs.sqlite3", help="batch job queue (SQLite file)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(name)s %(message)s")
    app = create_app(store=JobStore(args.db))
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
