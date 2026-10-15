import json

import httpx
import pytest
from fastapi.testclient import TestClient

from swarm.coordinator import WORKER_MODEL, create_app


def fake_workers(dead=(), seen=None):
    """Upstream transport: each host is a worker; hosts in ``dead`` refuse connections."""

    def handle(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host in dead:
            raise httpx.ConnectError("refused", request=request)
        body = json.loads(request.content)
        if seen is not None:
            seen.append((host, body))
        if body.get("stream"):
            sse = f'data: {{"model": "{WORKER_MODEL}", "choices": [{{"delta": {{"content": "hi"}}}}]}}\n\n'
            if (body.get("stream_options") or {}).get("include_usage"):
                sse += ('data: {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, '
                        '"total_tokens": 10}}\n\n')
            sse += "data: [DONE]\n\n"
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={"model": WORKER_MODEL, "host": host, "choices": [{"message": {"content": "2"}}],
                  "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}},
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def make_client(client_host="testclient", admin_token=None, **kw):
    app = create_app(client=fake_workers(**kw), admin_token=admin_token)
    client = TestClient(app, client=(client_host, 50000))
    for node_id, host, speed in [("m5", "a", 122), ("mini", "b", 101)]:
        r = client.post(
            "/internal/nodes/register",
            json={"node_id": node_id, "url": f"http://{host}:8081", "model": "Qwen3-8B-4bit", "speed": speed},
        )
        assert r.status_code == 200
    return client


CHAT = {"model": "Qwen3-8B-4bit", "messages": [{"role": "user", "content": "1+1"}]}


def test_routes_and_restores_model_name():
    seen = []
    client = make_client(seen=seen)
    r = client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 200
    assert r.headers["x-swarm-node"] == "m5"
    assert r.json()["model"] == "Qwen3-8B-4bit"
    # The worker is asked for its pinned model, never the client's name.
    assert seen == [("a", {**CHAT, "model": WORKER_MODEL})]
    events = client.app.state.metrics.summary()["recent"]
    assert [(e["node_id"], e["kind"], e["prompt_tokens"], e["completion_tokens"]) for e in events] == [
        ("m5", "chat", 5, 1)]


def test_models_and_nodes_listing():
    client = make_client()
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["Qwen3-8B-4bit"]
    nodes = {n["node_id"]: n for n in client.get("/nodes").json()["nodes"]}
    assert set(nodes) == {"m5", "mini"} and all(n["healthy"] for n in nodes.values())


def test_unknown_model_404():
    client = make_client()
    r = client.post("/v1/chat/completions", json={**CHAT, "model": "nope"})
    assert r.status_code == 404
    assert "nope" in r.json()["error"]["message"]


def test_fails_over_to_another_node_and_cools_down_dead_one():
    client = make_client(dead={"a"})
    r = client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 200 and r.headers["x-swarm-node"] == "mini"
    nodes = {n["node_id"]: n for n in client.get("/nodes").json()["nodes"]}
    assert nodes["m5"]["failures"] == 1 and not nodes["m5"]["healthy"]
    assert nodes["mini"]["inflight"] == 0 and nodes["mini"]["served"] == 1


def test_all_nodes_down_is_503():
    client = make_client(dead={"a", "b"})
    r = client.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 503


def test_streaming_is_relayed_with_model_name_restored():
    client = make_client()
    with client.stream("POST", "/v1/chat/completions", json={**CHAT, "stream": True}) as r:
        assert r.status_code == 200
        text = "".join(r.iter_text())
    assert '"model": "Qwen3-8B-4bit"' in text and WORKER_MODEL not in text
    assert text.rstrip().endswith("data: [DONE]")
    # Usage was requested upstream for accounting, but not passed on unasked.
    assert '"usage"' not in text
    (event,) = client.app.state.metrics.summary()["recent"]
    assert event["completion_tokens"] == 3 and event["ttft_s"] is not None and event["ok"]
    nodes = {n["node_id"]: n for n in client.get("/nodes").json()["nodes"]}
    assert nodes["m5"]["inflight"] == 0 and nodes["m5"]["served"] == 1


def test_heartbeat_from_unknown_node_asks_to_register():
    client = make_client()
    assert client.post("/internal/nodes/ghost/heartbeat", json={}).status_code == 404
    assert client.post("/internal/nodes/m5/heartbeat", json={"memory_free_pct": 50}).status_code == 200


@pytest.mark.parametrize("session_header", [True, False])
def test_session_affinity(session_header):
    seen = []
    client = make_client(seen=seen)
    kw = {"headers": {"x-session-id": "s1"}} if session_header else {}
    body = CHAT if session_header else {**CHAT, "user": "s1"}
    client.post("/v1/chat/completions", json=body, **kw)
    # Make m5 slightly busier than mini; the session should still stay on m5.
    client.app.state.registry.nodes["m5"].inflight = 1
    client.post("/v1/chat/completions", json=body, **kw)
    assert [h for h, _ in seen] == ["a", "a"]


def test_stream_usage_is_passed_on_when_asked():
    client = make_client()
    body = {**CHAT, "stream": True, "stream_options": {"include_usage": True}}
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        text = "".join(r.iter_text())
    assert '"completion_tokens": 3' in text


def test_dashboard_is_served():
    r = make_client().get("/")
    assert r.status_code == 200 and "mlx-swarm" in r.text and r.headers["content-type"].startswith("text/html")


def test_admin_endpoints_need_this_mac_or_the_admin_token():
    remote = make_client(admin_token="s3cret")
    assert remote.get("/admin/overview").status_code == 403
    assert remote.get("/admin/session").json()["admin"] is False
    auth = {"authorization": "Bearer s3cret"}
    assert remote.get("/admin/overview", headers=auth).status_code == 200
    assert remote.get("/admin/session", headers=auth).json()["admin"] is True
    local = make_client(client_host="127.0.0.1")
    overview = local.get("/admin/overview").json()
    assert {n["node_id"] for n in overview["nodes"]} == {"m5", "mini"}


def test_api_keys_lock_the_api_and_count_usage():
    admin = make_client(client_host="127.0.0.1")
    remote = TestClient(admin.app, client=("10.0.0.9", 50000))
    assert remote.post("/v1/chat/completions", json=CHAT).status_code == 200  # open: no keys yet
    assert remote.post("/admin/keys", json={"name": "x"}).status_code == 403
    created = admin.post("/admin/keys", json={"name": "notebook"}).json()
    secret = created["key"]
    assert secret.startswith("sk-swarm-") and "hash" not in created
    assert remote.post("/v1/chat/completions", json=CHAT).status_code == 401
    assert remote.get("/v1/models").status_code == 401
    assert remote.get("/health").status_code == 200 and remote.get("/nodes").status_code == 200
    bad = {"authorization": "Bearer sk-swarm-nope"}
    assert remote.post("/v1/chat/completions", json=CHAT, headers=bad).status_code == 401
    good = {"authorization": f"Bearer {secret}"}
    assert remote.post("/v1/chat/completions", json=CHAT, headers=good).status_code == 200
    assert admin.post("/v1/chat/completions", json=CHAT).status_code == 200  # this Mac needs no key
    (key,) = admin.get("/admin/keys").json()["data"]
    assert (key["requests"], key["prompt_tokens"], key["completion_tokens"]) == (1, 5, 1)
    assert admin.delete(f"/admin/keys/{key['id']}").json() == {"ok": True, "required": False}
    assert remote.post("/v1/chat/completions", json=CHAT, headers=good).status_code == 200  # open again


def test_draining_node_gets_no_requests_or_tasks():
    seen = []
    client = make_client(client_host="127.0.0.1", seen=seen)
    assert client.post("/admin/nodes/m5/drain", json={"draining": True}).json()["draining"]
    for _ in range(3):
        client.post("/v1/chat/completions", json=CHAT)
    assert {h for h, _ in seen} == {"b"}
    client.post("/v1/jobs", json={"tasks": [{"body": {"messages": []}}], "model": "Qwen3-8B-4bit"})
    assert client.post("/internal/tasks/lease", json={"node_id": "m5"}).json() == {"tasks": []}
    assert len(client.post("/internal/tasks/lease", json={"node_id": "mini"}).json()["tasks"]) == 1
    assert client.delete("/admin/nodes/m5").status_code == 200
    assert "m5" not in {n["node_id"] for n in client.get("/nodes").json()["nodes"]}
