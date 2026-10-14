import pytest
from fastapi.testclient import TestClient

from swarm.coordinator import WORKER_MODEL, create_app
from swarm.jobs import BACKOFF_BASE_S, JobError, JobStore


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def task(i, model="m"):
    return {"custom_id": f"c{i}", "body": {"model": model, "messages": [{"role": "user", "content": str(i)}]}}


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(clock):
    return JobStore(clock=clock, lease_s=30)


def test_lease_complete_and_results_in_order(store):
    job, created = store.submit([task(i) for i in range(3)])
    assert created
    leases = store.lease("a", "m", limit=2)
    assert [l["task_id"] for l in leases] == [1, 2]
    assert leases[0]["body"]["stream"] is False
    assert store.lease("b", "other-model") == []
    assert store.complete(leases[1]["task_id"], leases[1]["token"], "a", {"x": 2})
    status = store.job(job)
    assert status["status"] == "in_progress"
    assert status["request_counts"] == {"total": 3, "completed": 1, "failed": 0, "cancelled": 0,
                                        "queued": 1, "running": 1}
    res = store.results(job)
    assert [r["custom_id"] for r in res] == ["c0", "c1", "c2"]
    assert res[1]["response"]["body"] == {"x": 2} and res[1]["node"] == "a"


def test_expired_lease_is_requeued_and_stale_result_rejected(store, clock):
    store.submit([task(0)])
    [old] = store.lease("a", "m")
    clock.t += 31  # node a went silent
    assert store.lease("b", "m") == []  # requeued, but backing off
    clock.t += BACKOFF_BASE_S
    [new] = store.lease("b", "m")
    assert new["task_id"] == old["task_id"] and new["token"] > old["token"]
    # a comes back and reports late: fencing drops it; b's result is the one kept.
    assert not store.complete(old["task_id"], old["token"], "a", {"from": "a"})
    assert store.renew("a", [old]) == [old["task_id"]]
    assert store.complete(new["task_id"], new["token"], "b", {"from": "b"})
    assert not store.complete(new["task_id"], new["token"], "b", {"again": True})  # exactly once
    [r] = store.results(store.jobs()[0]["id"])
    assert r["response"]["body"] == {"from": "b"} and r["attempts"] == 2


def test_renew_keeps_lease_alive(store, clock):
    store.submit([task(0)])
    [l] = store.lease("a", "m")
    for _ in range(5):
        clock.t += 20
        assert store.renew("a", [l]) == []
    assert store.lease("b", "m") == []
    assert store.complete(l["task_id"], l["token"], "a", {})


def test_retries_until_max_attempts_then_fails(store, clock):
    job, _ = store.submit([task(0)], max_attempts=2)
    [l] = store.lease("a", "m")
    assert store.fail(l["task_id"], l["token"], "a", "boom")
    clock.t += 60
    [l] = store.lease("a", "m")
    assert store.fail(l["task_id"], l["token"], "a", "boom again")
    clock.t += 60
    assert store.lease("a", "m") == []
    [r] = store.results(job)
    assert r["status"] == "failed" and r["error"]["message"] == "boom again"
    assert store.job(job)["status"] == "completed"


def test_non_retryable_failure_is_final(store):
    job, _ = store.submit([task(0)])
    [l] = store.lease("a", "m")
    store.fail(l["task_id"], l["token"], "a", "HTTP 400", retryable=False)
    assert store.results(job)[0]["status"] == "failed"


def test_idempotency_key_and_validation(store):
    a, created_a = store.submit([task(0)], idempotency_key="k")
    b, created_b = store.submit([task(0), task(1)], idempotency_key="k")
    assert a == b and created_a and not created_b
    with pytest.raises(JobError, match="duplicate"):
        store.submit([task(0), task(0)])
    with pytest.raises(JobError, match="model"):
        store.submit([{"body": {"messages": []}}])
    with pytest.raises(JobError, match="url"):
        store.submit([{**task(0), "url": "/v1/embeddings"}])
    with pytest.raises(JobError):
        store.submit([])


def test_cancel_drops_running_result(store):
    job, _ = store.submit([task(0), task(1)])
    [l] = store.lease("a", "m")
    assert store.cancel(job) == 2
    assert not store.complete(l["task_id"], l["token"], "a", {})
    assert store.job(job)["request_counts"]["cancelled"] == 2
    assert store.lease("a", "m") == []


def test_survives_restart(tmp_path, clock):
    path = str(tmp_path / "jobs.sqlite3")
    s1 = JobStore(path, clock=clock)
    job, _ = s1.submit([task(0)])
    [l] = s1.lease("a", "m")
    s1.db.close()
    s2 = JobStore(path, clock=clock)  # coordinator restarted; the agent still holds its lease
    assert s2.renew("a", [l]) == []
    assert s2.complete(l["task_id"], l["token"], "a", {"ok": 1})
    assert s2.job(job)["status"] == "completed"


def test_http_flow(clock):
    client = TestClient(create_app(store=JobStore(clock=clock)))
    client.post("/internal/nodes/register",
                json={"node_id": "n1", "url": "http://a:8081", "model": "Qwen3-8B-4bit"})
    r = client.post("/v1/jobs", json={"model": "Qwen3-8B-4bit", "tasks": [task(0, None)]})
    assert r.status_code == 201
    job = r.json()["id"]
    assert client.post("/v1/jobs", json={"tasks": []}).status_code == 400
    assert client.post("/internal/tasks/lease", json={"node_id": "ghost"}).status_code == 404
    [t] = client.post("/internal/tasks/lease", json={"node_id": "n1", "max": 4}).json()["tasks"]
    assert t["body"]["model"] == "Qwen3-8B-4bit"
    done = {"node_id": "n1", "token": t["token"], "model": "Qwen3-8B-4bit",
            "result": {"model": WORKER_MODEL, "choices": []}}
    assert client.post(f"/internal/tasks/{t['task_id']}/complete", json=done).status_code == 200
    assert client.post(f"/internal/tasks/{t['task_id']}/complete", json=done).status_code == 409
    assert client.get(f"/v1/jobs/{job}").json()["status"] == "completed"
    lines = client.get(f"/v1/jobs/{job}/results").text.splitlines()
    assert len(lines) == 1 and '"model": "Qwen3-8B-4bit"' in lines[0]
    assert client.get("/v1/jobs/nope").status_code == 404
    assert [j["id"] for j in client.get("/v1/jobs").json()["data"]] == [job]
