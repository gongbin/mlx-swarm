import json
import struct

import pytest

from swarm.planner import (
    GB, InfeasibleError, ModelProfile, NodeSpec, evaluate, nodes_from_hostfile, plan, profile_model,
)


def uniform(n_layers=10, layer_gb=1.0, shared_gb=0.5, moe=False):
    b = int(layer_gb * GB)
    return ModelProfile([b] * n_layers, [b] * n_layers, int(shared_gb * GB), int(shared_gb * GB), moe)


def test_latency_mode_fills_the_fast_node_first():
    nodes = [NodeSpec("fast", 8.0, 200), NodeSpec("slow", 8.0, 100)]
    p = plan(uniform(), nodes, kv_reserve_gb=1)
    # fast: 6 layers + 0.5 shared + 1 KV = 7.5 <= 8; 7 layers would be 8.5.
    assert p.counts == [6, 4] and p.ranges == [(0, 6), (6, 10)]
    assert all(m <= n.budget_gb for m, n in zip(p.mem_gb, nodes))


def test_throughput_mode_balances_stage_times():
    nodes = [NodeSpec("fast", 20.0, 200), NodeSpec("slow", 20.0, 100)]
    p = plan(uniform(12), nodes, mode="throughput")
    assert p.counts == [8, 4]  # 8/200 == 4/100


def test_infeasible_reports_sizes():
    with pytest.raises(InfeasibleError, match="no split fits"):
        plan(uniform(), [NodeSpec("a", 4, 100), NodeSpec("b", 4, 100)])
    with pytest.raises(InfeasibleError):
        plan(uniform(2), [NodeSpec(str(i), 99, 100) for i in range(3)])


def test_estimate_and_efficiency():
    nodes = [NodeSpec("a", 99, 100), NodeSpec("b", 99, 100)]
    dense = evaluate(uniform(), nodes, [5, 5], hop_ms=0)
    # 10 GiB of layers + 0.5 GiB head at 100 GB/s.
    assert dense.token_ms == pytest.approx(10.5 * GB / 100e9 * 1000)
    assert dense.expected_tok_s == pytest.approx(dense.tok_s * 0.95)
    assert evaluate(uniform(moe=True), nodes, [5, 5]).efficiency == 0.82


def write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, size in tensors.items():
        header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def test_profile_counts_only_routed_experts_as_read(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"num_hidden_layers": 2, "num_experts": 8, "num_experts_per_tok": 2}))
    write_safetensors(tmp_path / "model.safetensors", {
        "model.embed_tokens.weight": 100,
        "model.layers.0.self_attn.q_proj.weight": 40,
        "model.layers.0.mlp.switch_mlp.up_proj.weight": 800,
        "model.layers.1.self_attn.q_proj.weight": 40,
        "model.norm.weight": 4,
        "lm_head.weight": 100,
    })
    p = profile_model(tmp_path)
    assert p.layer_mem == [840, 40] and p.layer_read == [240, 40]
    assert p.shared_mem == 204 and p.shared_read == 104  # embedding lookup not counted
    assert p.moe


def test_hostfile_needs_budget_and_bandwidth(tmp_path):
    f = tmp_path / "hosts.json"
    f.write_text(json.dumps([{"name": "m5", "ssh": "127.0.0.1", "mem_budget_gb": 20, "bandwidth_gbs": 122},
                             {"ssh": "mini"}]))
    with pytest.raises(ValueError, match="mini"):
        nodes_from_hostfile(f)
