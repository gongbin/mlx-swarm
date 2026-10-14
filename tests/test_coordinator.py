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
            sse = (
                f'data: {{"model": "{WORKER_MODEL}", "choices": [{{"delta": {{"content": "hi"}}}}]}}\n\n'
                "data: [DONE]\n\n"
            )
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={"model": WORKER_MODEL, "host": host, "choices": [{"message": {"content": "2"}}]},
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handle))


def make_client(**kw):
    client = TestClient(create_app(client=fake_workers(**kw)))
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
