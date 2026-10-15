"""End to end: a coordinator on a real port, a fake worker, and an agent-side tunnel."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import uvicorn

from swarm.coordinator import create_app
from swarm.tunnel import TunnelClient

ADMIN = {"authorization": "Bearer adm"}
CHAT = {"model": "Qwen3-8B-4bit", "messages": [{"role": "user", "content": "hi"}]}


ABANDONED = threading.Event()  # the slow stream's client went away


class FakeWorker(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if body.get("slow"):  # a long generation, until the reader disappears
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            try:
                for _ in range(200):
                    self.wfile.write(b'data: {"choices": [{"delta": {"content": "x"}}]}\n\n')
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                ABANDONED.set()
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for word in ["one", "two", "three"]:
                chunk = {"model": body["model"], "choices": [{"delta": {"content": word}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.02)
            usage = {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        else:
            data = json.dumps({"model": body["model"], "choices": [{"message": {"content": "tunnelled"}}],
                               "usage": {"prompt_tokens": 4, "completion_tokens": 1}}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


@pytest.fixture()
def cluster():
    worker = ThreadingHTTPServer(("127.0.0.1", 0), FakeWorker)
    threading.Thread(target=worker.serve_forever, daemon=True).start()
    # trust_local=False: on 127.0.0.1 everyone would otherwise be an administrator.
    app = create_app(admin_token="adm", trust_local=False)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    clients = []

    def tunnel(node_id, token):
        c = TunnelClient(base, node_id, f"http://127.0.0.1:{worker.server_port}",
                         {"authorization": f"Bearer {token}"})
        c.start()
        clients.append(c)
        return c

    yield base, app, tunnel
    for c in clients:
        c.stop()
    server.should_exit = True
    worker.shutdown()


def wait(cond, timeout=5.0):
    deadline = time.time() + timeout
    while not cond():
        assert time.time() < deadline, "timed out"
        time.sleep(0.05)


def register(base, node_id, token):
    headers = {"authorization": f"Bearer {token}"} if token else {}
    return httpx.post(f"{base}/internal/nodes/register", headers=headers,
                      json={"node_id": node_id, "url": "tunnel://", "model": "Qwen3-8B-4bit"})


def test_requests_reach_a_worker_behind_nat_through_its_tunnel(cluster):
    base, app, tunnel = cluster
    token = httpx.post(f"{base}/admin/node-tokens", headers=ADMIN, json={"name": "home mac"}).json()["token"]
    assert register(base, "home", token).status_code == 200
    node = app.state.registry.nodes["home"]
    assert node.url == "tunnel://home" and not node.connected  # no tunnel yet: not routable
    client = tunnel("home", token)
    wait(lambda: app.state.registry.nodes["home"].connected)

    r = httpx.post(f"{base}/v1/chat/completions", json=CHAT)
    assert r.status_code == 200 and r.headers["x-swarm-node"] == "home"
    assert r.json()["choices"][0]["message"]["content"] == "tunnelled"
    assert r.json()["model"] == "Qwen3-8B-4bit"  # pinned name restored, as over HTTP

    with httpx.stream("POST", f"{base}/v1/chat/completions", json={**CHAT, "stream": True}) as r:
        text = "".join(r.iter_text())
    assert [json.loads(l[6:])["choices"][0]["delta"]["content"]
            for l in text.splitlines() if l.startswith("data: {")] == ["one", "two", "three"]
    assert text.rstrip().endswith("data: [DONE]")
    events = app.state.metrics.summary()["recent"]
    assert events[0]["completion_tokens"] == 3 and all(e["ok"] for e in events)

    client.stop()  # the Mac goes offline: the node stops being routable
    wait(lambda: not app.state.registry.nodes["home"].connected)
    assert httpx.post(f"{base}/v1/chat/completions", json=CHAT).status_code == 404


def test_node_tokens_guard_the_agent_endpoints(cluster):
    base, app, tunnel = cluster
    # Open until the first token exists (the LAN setup keeps working).
    assert register(base, "lan", "").status_code == 200
    a = httpx.post(f"{base}/admin/node-tokens", headers=ADMIN, json={"name": "a"}).json()
    b = httpx.post(f"{base}/admin/node-tokens", headers=ADMIN, json={"name": "b"}).json()
    assert a["token"].startswith("nt-swarm-") and "hash" not in a
    assert register(base, "x", "").status_code == 401
    assert register(base, "x", "nt-swarm-forged").status_code == 401
    assert register(base, "x", a["token"]).status_code == 200
    # Another token cannot take over, heartbeat for, or lease work as node x.
    assert register(base, "x", b["token"]).status_code == 403
    hb = httpx.post(f"{base}/internal/nodes/x/heartbeat", json={},
                    headers={"authorization": f"Bearer {b['token']}"})
    assert hb.status_code == 403
    lease = httpx.post(f"{base}/internal/tasks/lease", json={"node_id": "x"},
                       headers={"authorization": f"Bearer {b['token']}"})
    assert lease.status_code == 403
    # Neither can it open node x's tunnel.
    thief = tunnel("x", b["token"])
    time.sleep(0.5)
    assert not thief.connected.is_set() and not app.state.registry.nodes["x"].connected
    thief.stop()
    # Revoking a token disconnects and removes the nodes that joined with it.
    owner = tunnel("x", a["token"])
    wait(lambda: app.state.registry.nodes["x"].connected)
    listed = httpx.get(f"{base}/admin/node-tokens", headers=ADMIN).json()["data"]
    assert {t["id"]: t["nodes"] for t in listed}[a["id"]] == ["x"]
    r = httpx.delete(f"{base}/admin/node-tokens/{a['id']}", headers=ADMIN).json()
    assert r["removed"] == ["x"] and "x" not in app.state.registry.nodes
    wait(lambda: not owner.connected.is_set())
    assert register(base, "x", a["token"]).status_code == 401
    # Without the admin token, management is refused.
    assert httpx.get(f"{base}/admin/node-tokens").status_code == 403


def test_client_disconnect_stops_the_generation_on_the_worker(cluster):
    base, app, tunnel = cluster
    assert register(base, "home", "").status_code == 200
    tunnel("home", "")
    wait(lambda: app.state.registry.nodes["home"].connected)
    ABANDONED.clear()
    with httpx.stream("POST", f"{base}/v1/chat/completions", json={**CHAT, "stream": True, "slow": True}) as r:
        for line in r.iter_lines():
            if line.startswith("data: {"):
                break  # read one token, then hang up
    wait(ABANDONED.is_set, timeout=5)  # far less than the 10 s the stream would take
    wait(lambda: app.state.registry.nodes["home"].inflight == 0)
