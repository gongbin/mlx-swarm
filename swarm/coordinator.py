"""Coordinator: OpenAI-compatible gateway in front of the swarm's workers.

Agents register their worker and send heartbeats; clients call the usual
OpenAI endpoints and are routed to a healthy worker serving the requested
model (see ``swarm.router``). If a worker cannot be reached, or fails before
sending a response, the request is retried once on another node.

Batch jobs (``/v1/jobs``) go into a SQLite queue (see ``swarm.jobs``); agents
lease tasks for the model they serve and post results back.

The coordinator advertises itself on mDNS (``_mlxswarm._tcp``), so agents and
the CLI find it without an address (see ``swarm.discovery``).

``/`` serves the dashboard (``swarm/ui.html``): nodes, live throughput, a chat
playground, batch jobs and API keys. Access control:

- ``/v1/*`` is open until the first API key is created (see ``swarm.keys``);
  from then on it needs ``Authorization: Bearer <key>``, or an administrator;
- ``/admin/*`` (dashboard management) needs an administrator: a request from
  this Mac itself, or one carrying the ``--admin-token``;
- ``/internal/*`` (agents) is open until the first node token is created; then
  each agent needs one (``swarm join``), and may only act for the nodes it
  registered itself;
- ``/health`` and ``/nodes`` stay open.

Agents behind NAT dial out and receive requests over a WebSocket tunnel (see
``swarm.tunnel``). For the internet, serve TLS (``--ssl-certfile``/``--ssl-keyfile``,
or a reverse proxy in front).

    swarm coordinator --host 0.0.0.0 --port 8000 --db jobs.sqlite3 [--admin-token T]
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import socket
import time
from contextlib import asynccontextmanager
from importlib import resources

import httpx
from fastapi import FastAPI, Request, WebSocket
from starlette.requests import HTTPConnection
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse

from swarm.jobs import JobError, JobStore
from swarm.keys import NODE_TOKENS, KeyStore
from swarm.metrics import Metrics
from swarm.router import NoNodeAvailable, Registry
from swarm.tunnel import SCHEME as TUNNEL, Tunnel, TunnelError

log = logging.getLogger("swarm.coordinator")

MAX_ATTEMPTS = 2
UPSTREAM_TIMEOUT = httpx.Timeout(connect=3.0, read=600.0, write=30.0, pool=5.0)
WORKER_MODEL = "default_model"  # the name a worker's pinned model answers to


def error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def bearer(request: HTTPConnection) -> str | None:
    auth = request.headers.get("authorization", "")
    return auth[7:].strip() if auth[:7].lower() == "bearer " else None


def usage_of(payload: dict) -> tuple[int, int]:
    u = payload.get("usage") or {}
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


class Guard:
    """Who is calling: administrator, API key, node token."""

    def __init__(self, keys: KeyStore, node_tokens: KeyStore, admin_token: str | None, trust_local: bool):
        self.keys = keys
        self.node_tokens = node_tokens
        self.admin_token = admin_token
        self.trust_local = trust_local

    def is_admin(self, conn: HTTPConnection) -> bool:
        token = bearer(conn)
        if self.admin_token and token and hmac.compare_digest(token, self.admin_token):
            return True
        if not self.trust_local or conn.client is None:
            return False
        client = conn.client.host
        server = (conn.scope.get("server") or ("",))[0]
        return client in LOOPBACK or client == server  # e.g. a browser or agent on this Mac

    def node_credential(self, conn: HTTPConnection) -> tuple[bool, str | None]:
        """(allowed, credential) for /internal/*: a node token id, "admin", or None while open."""
        token = self.node_tokens.verify(bearer(conn))
        if token:
            return True, token
        if self.is_admin(conn):
            return True, "admin"
        return not self.node_tokens.required, None


def owns(credential: str | None, node) -> bool:
    """An agent may only act for nodes registered with its own credential."""
    return credential == "admin" or credential == node.token_id


class AccessControl:
    """ASGI middleware: API keys on /v1/*, node tokens on /internal/*, administrators on /admin/*.

    Sets ``scope["state"]``: ``admin``; ``key``, the caller's API key id ("admin"
    for an administrator, None on an open API) for usage accounting; and
    ``node``, the caller's node credential. WebSockets check for themselves.
    """

    def __init__(self, app, guard: Guard):
        self.app = app
        self.guard = guard
        self.keys = guard.keys

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        path = scope["path"]
        state = scope.setdefault("state", {})
        state["admin"] = self.guard.is_admin(request)
        denied = None
        if path.startswith("/admin/") and path != "/admin/session" and not state["admin"]:
            denied = error(403, "administrator only: use this Mac or the coordinator's --admin-token",
                           "permission_error")
        elif path.startswith("/v1/"):
            key = self.keys.verify(bearer(request))
            if key:
                state["key"] = key
            elif state["admin"]:
                state["key"] = "admin"
            elif self.keys.required:
                denied = error(401, "missing or invalid API key (Authorization: Bearer sk-swarm-...)",
                               "authentication_error")
        elif path.startswith("/internal/"):
            allowed, state["node"] = self.guard.node_credential(request)
            if not allowed:
                denied = error(401, "missing or invalid node token (see `swarm join`)", "authentication_error")
        if denied is not None:
            return await denied(scope, receive, send)
        await self.app(scope, receive, send)


def create_app(registry: Registry | None = None, client: httpx.AsyncClient | None = None,
               store: JobStore | None = None, keys: KeyStore | None = None,
               node_tokens: KeyStore | None = None,
               admin_token: str | None = None, trust_local: bool = True) -> FastAPI:
    reg = registry or Registry()
    http = client or httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    jobs = store or JobStore()
    keys = keys or KeyStore()
    node_tokens = node_tokens or KeyStore(**NODE_TOKENS)
    guard = Guard(keys, node_tokens, admin_token, trust_local)
    metrics = Metrics()
    tunnels: dict[str, Tunnel] = {}

    @asynccontextmanager
    async def lifespan(app):
        yield
        await http.aclose()

    app = FastAPI(title="mlx-swarm coordinator", lifespan=lifespan)
    app.state.registry = reg
    app.state.jobs = jobs
    app.state.keys = keys
    app.state.metrics = metrics
    app.state.node_tokens = node_tokens
    app.state.tunnels = tunnels
    app.add_middleware(AccessControl, guard=guard)

    def account(request: Request, node, model, kind, payload: dict | None, seconds, ttft=None, ok=True):
        p, c = usage_of(payload or {})
        key = request.state.key if hasattr(request.state, "key") else None
        metrics.record(node.node_id, node.model, kind, key, p, c, seconds, ttft, ok)
        if key and key != "admin":
            keys.record(key, p, c)

    def not_yours(node_id: str):
        return error(403, f"node {node_id} was registered with another credential", "permission_error")

    @app.post("/internal/nodes/register")
    async def register(request: Request, body: dict):
        credential = request.state.node
        node_id = body["node_id"]
        existing = reg.nodes.get(node_id)
        if existing is not None and not owns(credential, existing):
            return not_yours(node_id)
        url = body["url"]
        if url.startswith(TUNNEL):
            url = TUNNEL + node_id
        node = reg.register(node_id, url, body["model"], float(body.get("speed", 1.0)), body.get("info"))
        node.token_id = credential
        node.remote = request.client.host if request.client else None
        node.connected = not url.startswith(TUNNEL) or node_id in tunnels
        if credential and credential != "admin":
            node_tokens.record(credential)
        log.info("registered %s at %s serving %s", node.node_id, node.url, node.model)
        return {"ok": True}

    @app.post("/internal/nodes/{node_id}/heartbeat")
    async def heartbeat(request: Request, node_id: str, body: dict):
        node = reg.nodes.get(node_id)
        if node is None:
            return error(404, f"unknown node {node_id}; register first")
        if not owns(request.state.node, node):
            return not_yours(node_id)
        reg.heartbeat(node_id, body.get("memory_free_pct"), body.get("stats"))
        return {"ok": True}

    @app.post("/internal/nodes/{node_id}/deregister")
    async def deregister(request: Request, node_id: str):
        node = reg.nodes.get(node_id)
        if node is not None and not owns(request.state.node, node):
            return not_yours(node_id)
        reg.deregister(node_id)
        log.info("deregistered %s", node_id)
        return {"ok": True}

    @app.websocket("/internal/tunnel/{node_id}")
    async def tunnel(ws: WebSocket, node_id: str):
        allowed, credential = guard.node_credential(ws)
        node = reg.nodes.get(node_id)
        if not allowed:
            return await ws.close(code=4401, reason="missing or invalid node token")
        if node is None:
            return await ws.close(code=4404, reason="register first")
        if not owns(credential, node):
            return await ws.close(code=4403, reason="registered with another credential")
        await ws.accept()
        t = Tunnel(ws, node_id, ws.client.host if ws.client else None)
        old = tunnels.pop(node_id, None)
        if old is not None:
            await old.close()
        tunnels[node_id] = t
        node.connected = True
        node.remote = t.remote
        log.info("tunnel up for %s from %s", node_id, t.remote)
        try:
            await t.run()
        finally:
            if tunnels.get(node_id) is t:
                del tunnels[node_id]
                current = reg.nodes.get(node_id)
                if current is not None and current.url.startswith(TUNNEL):
                    current.connected = False
            log.info("tunnel down for %s", node_id)

    async def open_upstream(node, path: str, body: dict):
        """A streaming response from the node's worker: over HTTP, or through its tunnel."""
        if node.url.startswith(TUNNEL):
            t = tunnels.get(node.node_id)
            if t is None:
                raise TunnelError(f"no tunnel from {node.node_id}")
            return await t.request("POST", path, body)
        return await http.send(http.build_request("POST", node.url + path, json=body), stream=True)

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
        kind = "chat" if path.endswith("chat/completions") else "completion"
        # Workers only serve their pinned model; name it explicitly so a stock
        # mlx_lm.server would not try to load the client's model name either.
        upstream = {**body, "model": WORKER_MODEL}
        # Always ask for the usage chunk when streaming (for accounting); it is
        # passed on only if the client asked for it too.
        client_options = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
        wants_usage = bool(client_options.get("include_usage"))
        if stream:
            upstream["stream_options"] = {**client_options, "include_usage": True}
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
                resp = await open_upstream(node, path, upstream)
            except httpx.TransportError as e:
                node.inflight -= 1
                reg.failed(node)
                account(request, node, model, kind, None, time.perf_counter() - started, ok=False)
                last_problem = f"{node.node_id}: {e!r}"
                log.warning("worker %s unreachable: %r", node.node_id, e)
                continue
            if resp.status_code >= 500:
                await resp.aclose()
                node.inflight -= 1
                reg.failed(node)
                account(request, node, model, kind, None, time.perf_counter() - started, ok=False)
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
                elapsed = time.perf_counter() - started
                try:
                    payload = json.loads(data)
                except ValueError:
                    payload = None
                account(request, node, model, kind, payload if isinstance(payload, dict) else None,
                        elapsed, ok=resp.status_code < 400)
                log.info("%s -> %s %.2fs", path, node.node_id, elapsed)
                return Response(
                    restore(data),
                    status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"),
                    headers=headers,
                )

            async def relay(resp=resp, node=node, started=started):
                usage, ttft, ok, skip_blank = None, None, False, False
                client_gone = False  # the client hung up: not the worker's fault
                try:
                    # Line by line, so the model-name rewrite never straddles chunks.
                    async for line in resp.aiter_lines():
                        if skip_blank and not line:
                            skip_blank = False
                            continue
                        if line.startswith("data: {"):
                            if ttft is None:
                                ttft = time.perf_counter() - started
                            if '"usage"' in line:
                                try:
                                    chunk = json.loads(line[6:])
                                except ValueError:
                                    chunk = {}
                                if chunk.get("usage") and not chunk.get("choices"):
                                    usage = chunk
                                    if not wants_usage:
                                        skip_blank = True
                                        continue
                        yield restore(line.encode()) + b"\n"
                    node.served += 1
                    ok = True
                except httpx.TransportError as e:
                    reg.failed(node)
                    log.warning("stream from %s broke: %r", node.node_id, e)
                except BaseException:  # cancelled when the client disconnects
                    client_gone = True
                    raise
                finally:
                    await resp.aclose()
                    node.inflight -= 1
                    elapsed = time.perf_counter() - started
                    account(request, node, model, kind, usage, elapsed, ttft,
                            (ok or client_gone) and resp.status_code < 400)
                    log.info("%s -> %s %.2fs (stream)", path, node.node_id, elapsed)

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
    async def lease_tasks(request: Request, body: dict):
        node = reg.nodes.get(body["node_id"])
        if node is None:
            return error(404, f"unknown node {body['node_id']}; register first")
        if not owns(request.state.node, node):
            return not_yours(node.node_id)
        if node.draining or not node.healthy(reg.clock()):
            return {"tasks": []}
        return {"tasks": jobs.lease(node.node_id, node.model, int(body.get("max", 1)))}

    @app.post("/internal/tasks/renew")
    async def renew_tasks(request: Request, body: dict):
        node = reg.nodes.get(body["node_id"])
        if node is not None and not owns(request.state.node, node):
            return not_yours(node.node_id)
        return {"lost": jobs.renew(body["node_id"], body.get("leases") or [])}

    @app.post("/internal/tasks/{task_id}/complete")
    async def complete_task(request: Request, task_id: int, body: dict):
        # Ownership is checked while the node is registered; otherwise (e.g. right
        # after a coordinator restart) the fencing token and owner still must match.
        node = reg.nodes.get(body["node_id"])
        if node is not None and not owns(request.state.node, node):
            return not_yours(node.node_id)
        result = body["result"]
        if isinstance(result, dict) and result.get("model") == WORKER_MODEL:
            result["model"] = body.get("model") or result["model"]
        if not jobs.complete(task_id, body["token"], body["node_id"], result):
            return error(409, "stale fencing token; result discarded", "conflict")
        node = reg.nodes.get(body["node_id"])
        if node is not None:
            p, c = usage_of(result if isinstance(result, dict) else {})
            metrics.record(node.node_id, node.model, "batch", None, p, c, float(body.get("seconds") or 0))
        return {"ok": True}

    @app.post("/internal/tasks/{task_id}/fail")
    async def fail_task(request: Request, task_id: int, body: dict):
        node = reg.nodes.get(body["node_id"])
        if node is not None and not owns(request.state.node, node):
            return not_yours(node.node_id)
        if not jobs.fail(task_id, body["token"], body["node_id"], body.get("error", ""),
                         bool(body.get("retryable", True))):
            return error(409, "stale fencing token", "conflict")
        node = reg.nodes.get(body["node_id"])
        if node is not None:
            metrics.record(node.node_id, node.model, "batch", ok=False,
                           seconds=float(body.get("seconds") or 0))
        return {"ok": True}

    # -- dashboard ----------------------------------------------------------
    ui = resources.files("swarm").joinpath("ui.html")

    @app.get("/", include_in_schema=False)
    async def dashboard():
        return HTMLResponse(ui.read_text(encoding="utf-8"), headers={"cache-control": "no-cache"})

    @app.get("/admin/session")
    async def session(request: Request):
        """What the caller may do: lets the dashboard ask for a token when needed."""
        return {"admin": request.state.admin, "keys_required": keys.required,
                "key": keys.verify(bearer(request)) is not None,
                "node_tokens_required": node_tokens.required}

    @app.get("/admin/overview")
    async def overview():
        now = reg.clock()
        return {
            "coordinator": {"host": socket.gethostname(), "pid": os.getpid(),
                            "keys_required": keys.required, "node_tokens_required": node_tokens.required},
            "nodes": [n.to_dict(now) for n in reg.nodes.values()],
            "models": reg.models(),
            "backlog": jobs.model_backlog(),
            "metrics": metrics.summary(),
        }

    @app.post("/admin/nodes/{node_id}/drain")
    async def drain(node_id: str, body: dict):
        node = reg.nodes.get(node_id)
        if node is None:
            return error(404, f"no node {node_id}")
        node.draining = bool(body.get("draining", True))
        log.info("node %s %s", node_id, "draining" if node.draining else "back in service")
        return node.to_dict(reg.clock())

    @app.delete("/admin/nodes/{node_id}")
    async def remove_node(node_id: str):
        """Forget a node (e.g. a Mac that is gone); a live agent re-registers on its next heartbeat."""
        if node_id not in reg.nodes:
            return error(404, f"no node {node_id}")
        reg.deregister(node_id)
        t = tunnels.pop(node_id, None)
        if t is not None:
            await t.close()
        log.info("removed node %s", node_id)
        return {"ok": True}

    @app.get("/admin/keys")
    async def list_keys():
        return {"data": keys.list(), "required": keys.required}

    @app.post("/admin/keys")
    async def create_key(body: dict):
        record, secret = keys.create(str(body.get("name") or ""))
        log.info("created API key %s (%s)", record["id"], record["name"])
        return JSONResponse({**record, "key": secret}, status_code=201)

    @app.delete("/admin/keys/{key_id}")
    async def revoke_key(key_id: str):
        if not keys.revoke(key_id):
            return error(404, f"no active key {key_id}")
        log.info("revoked API key %s", key_id)
        return {"ok": True, "required": keys.required}

    @app.get("/admin/node-tokens")
    async def list_node_tokens():
        now = reg.clock()
        data = node_tokens.list()
        for t in data:
            t["nodes"] = [n.node_id for n in reg.nodes.values() if n.token_id == t["id"] and n.healthy(now)]
        return {"data": data, "required": node_tokens.required}

    @app.post("/admin/node-tokens")
    async def create_node_token(body: dict):
        record, secret = node_tokens.create(str(body.get("name") or ""))
        log.info("created node token %s (%s)", record["id"], record["name"])
        return JSONResponse({**record, "token": secret}, status_code=201)

    @app.delete("/admin/node-tokens/{token_id}")
    async def revoke_node_token(token_id: str):
        if not node_tokens.revoke(token_id):
            return error(404, f"no active node token {token_id}")
        gone = [n.node_id for n in reg.nodes.values() if n.token_id == token_id]
        for node_id in gone:  # disconnect the nodes that joined with it
            reg.deregister(node_id)
            t = tunnels.pop(node_id, None)
            if t is not None:
                await t.close()
        log.info("revoked node token %s; removed %s", token_id, gone or "no nodes")
        return {"ok": True, "removed": gone, "required": node_tokens.required}

    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="swarm coordinator", description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default="jobs.sqlite3", help="batch job queue (SQLite file)")
    parser.add_argument("--name", default=None, help="mDNS service name; default: short hostname")
    parser.add_argument("--no-mdns", action="store_true", help="do not advertise on the local network")
    parser.add_argument("--admin-token", default=os.environ.get("SWARM_ADMIN_TOKEN"),
                        help="lets other machines manage the cluster from the dashboard "
                             "(default: $SWARM_ADMIN_TOKEN; without it, only this Mac can)")
    parser.add_argument("--ssl-certfile", default=None, help="serve HTTPS with this certificate (PEM)")
    parser.add_argument("--ssl-keyfile", default=None, help="private key for --ssl-certfile")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(name)s %(message)s")
    app = create_app(store=JobStore(args.db), keys=KeyStore(args.db), node_tokens=KeyStore(args.db, **NODE_TOKENS),
                     admin_token=args.admin_token)
    advertiser = None
    if not args.no_mdns:
        from swarm.discovery import Advertiser

        try:
            advertiser = Advertiser(args.port, args.name)
        except OSError as e:  # e.g. macOS local-network permission
            log.warning("mDNS advertising failed (%r); clients need --coordinator", e)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level,
                    ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)
    finally:
        if advertiser:
            advertiser.close()


if __name__ == "__main__":
    main()
