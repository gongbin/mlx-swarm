import pytest

from swarm.router import (
    FAILURE_COOLDOWN_S,
    HEARTBEAT_TIMEOUT_S,
    NoNodeAvailable,
    Registry,
)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def reg():
    r = Registry(clock=Clock())
    r.register("m5", "http://a:8081", "Qwen3-8B-4bit", speed=122)
    r.register("mini", "http://b:8081", "Qwen3-8B-4bit", speed=101)
    return r


def test_picks_least_loaded_relative_to_speed(reg):
    assert reg.pick("Qwen3-8B-4bit").node_id == "m5"
    reg.nodes["m5"].inflight = 1
    # m5: 2/122 > mini: 1/101
    assert reg.pick("Qwen3-8B-4bit").node_id == "mini"


def test_unknown_model_is_an_error(reg):
    with pytest.raises(NoNodeAvailable):
        reg.pick("Qwen3-32B-4bit")


def test_model_may_be_omitted_when_only_one_is_served(reg):
    assert reg.pick(None).model == "Qwen3-8B-4bit"
    reg.register("x", "http://c:8081", "Other")
    with pytest.raises(NoNodeAvailable):
        reg.pick(None)


def test_stale_nodes_are_skipped(reg):
    reg.clock.t += HEARTBEAT_TIMEOUT_S + 1
    reg.heartbeat("mini")
    assert reg.pick("Qwen3-8B-4bit").node_id == "mini"


def test_failed_node_cools_down_then_returns(reg):
    reg.failed(reg.nodes["m5"])
    assert reg.pick("Qwen3-8B-4bit").node_id == "mini"
    reg.clock.t += FAILURE_COOLDOWN_S + 1
    reg.heartbeat("m5")
    reg.heartbeat("mini")
    assert reg.pick("Qwen3-8B-4bit").node_id == "m5"


def test_low_memory_node_is_skipped(reg):
    reg.heartbeat("m5", memory_free_pct=5)
    assert reg.pick("Qwen3-8B-4bit").node_id == "mini"


def test_session_sticks_unless_much_busier(reg):
    reg.nodes["m5"].inflight = 1
    assert reg.pick("Qwen3-8B-4bit", session="s1").node_id == "mini"
    reg.nodes["m5"].inflight = 0
    reg.nodes["mini"].inflight = 2  # within slack of m5's 0
    assert reg.pick("Qwen3-8B-4bit", session="s1").node_id == "mini"
    reg.nodes["mini"].inflight = 3
    assert reg.pick("Qwen3-8B-4bit", session="s1").node_id == "m5"


def test_exclude_for_retries(reg):
    assert reg.pick("Qwen3-8B-4bit", exclude={"m5"}).node_id == "mini"
    with pytest.raises(NoNodeAvailable):
        reg.pick("Qwen3-8B-4bit", exclude={"m5", "mini"})


def test_reregister_with_new_url_resets_counters(reg):
    reg.nodes["m5"].served = 5
    reg.register("m5", "http://a:9000", "Qwen3-8B-4bit", speed=122)
    assert reg.nodes["m5"].served == 0


def test_metrics_spread_tokens_over_generation_time():
    from swarm.metrics import BUCKET_S, Metrics

    now = [1000.0]
    m = Metrics(clock=lambda: now[0])
    m.record("m5", "q", "chat", prompt_tokens=10, completion_tokens=200, seconds=10.0, ttft_s=0.5)
    s = m.summary()
    row = s["series"]["m5"]
    assert abs(sum(row) * BUCKET_S - 200) < 1e-6  # all tokens accounted for
    assert row[-1] == row[-2] == 20.0  # 20 tok/s in the last two buckets
    assert s["nodes"]["m5"]["tok_s"] == round(200 / 60, 2)
    assert s["nodes"]["m5"]["ttft_p50_s"] == 0.5
    now[0] += 700  # beyond the window
    assert m.summary()["series"] == {} and m.summary()["totals"]["completion_tokens"] == 200
