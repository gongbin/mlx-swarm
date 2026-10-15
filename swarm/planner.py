"""Pipeline planner: split a model's layers across heterogeneous Macs.

Decoding is memory-bandwidth bound, so the time a node spends on a token is
roughly the bytes its layers read divided by its memory bandwidth. For MoE
layers only the routed experts are read (``num_experts_per_tok / num_experts``
of the expert weights), while all of them must fit in memory.

Each rank also keeps the embedding, final norm and LM head (every rank computes
the logits, see ``swarm.pipeline``), so those count against every node's memory.

Two objectives:

- ``latency`` (default): one stream visits the stages one after another, so
  minimise the sum of stage times -- put as many layers as fit on fast nodes;
- ``throughput``: with many requests interleaved the slowest stage limits the
  rate, so minimise the maximum stage time.

Splits are enumerated exhaustively; for a handful of nodes and <200 layers
that takes well under a second.
"""

from __future__ import annotations

import itertools
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

GB = 1024**3
DEFAULT_HOP_MS = 0.3  # one activation hop over Thunderbolt/10GbE (measured 0.26 ms p50 for 16 KiB)
DEFAULT_KV_RESERVE_GB = 2.0  # KV cache and activations for a few concurrent requests
# Measured decode speed / bandwidth bound on M5 + M4 mini over Thunderbolt (P0):
# dense Qwen3-8B 95%; MoE Qwen3-30B-A3B 81% (4bit) and 83% (6bit), whose expert
# gather kernels do not reach full memory bandwidth.
EFFICIENCY = {"dense": 0.95, "moe": 0.82}
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.(switch_mlp|experts)\.")


@dataclass
class ModelProfile:
    layer_mem: list[int]  # bytes held per layer
    layer_read: list[int]  # bytes read per decoded token, per layer
    shared_mem: int  # embedding, final norm, LM head: on every rank
    shared_read: int  # bytes read per token for the logits
    moe: bool = False

    @property
    def total_mem(self) -> int:
        return sum(self.layer_mem) + self.shared_mem


@dataclass
class NodeSpec:
    name: str
    budget_gb: float
    bandwidth_gbs: float


@dataclass
class Plan:
    counts: list[int]
    ranges: list[tuple[int, int]]
    mem_gb: list[float]
    stage_ms: list[float]
    token_ms: float  # single-stream time per token (stages + hops + logits)
    mode: str
    efficiency: float = 1.0

    @property
    def tok_s(self) -> float:
        """Bandwidth-bound upper limit."""
        return 1000.0 / self.token_ms

    @property
    def expected_tok_s(self) -> float:
        return self.tok_s * self.efficiency

    @property
    def split(self) -> str:
        return ",".join(map(str, self.counts))


def _tensor_sizes(path: Path) -> dict[str, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {k: v["data_offsets"][1] - v["data_offsets"][0] for k, v in header.items() if k != "__metadata__"}


def profile_model(model_dir: str | Path) -> ModelProfile:
    """Per-layer sizes from the safetensors headers (no weights are read)."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    config = {**config.get("text_config", {}), **config}
    n_layers = config["num_hidden_layers"]
    experts = config.get("num_experts") or config.get("n_routed_experts")
    active = config.get("num_experts_per_tok")
    expert_frac = active / experts if experts and active else 1.0

    layer_mem, layer_read = [0] * n_layers, [0.0] * n_layers
    shared_mem = shared_read = 0
    files = sorted(model_dir.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {model_dir}")
    for f in files:
        for name, size in _tensor_sizes(f).items():
            m = LAYER_RE.search(name)
            if m and int(m.group(1)) < n_layers:
                i = int(m.group(1))
                layer_mem[i] += size
                layer_read[i] += size * (expert_frac if EXPERT_RE.search(name) else 1.0)
            else:
                shared_mem += size
                # The embedding is a lookup (a few rows); the LM head (or the tied
                # embedding used as one) is a full matmul.
                if "embed_tokens" not in name or config.get("tie_word_embeddings"):
                    shared_read += size
    return ModelProfile(layer_mem, [int(x) for x in layer_read], shared_mem, shared_read, expert_frac < 1)


def evaluate(profile: ModelProfile, nodes: list[NodeSpec], counts: list[int],
             kv_reserve_gb: float = DEFAULT_KV_RESERVE_GB, hop_ms: float = DEFAULT_HOP_MS,
             mode: str = "latency") -> Plan:
    ranges, start = [], 0
    for c in counts:
        ranges.append((start, start + c))
        start += c
    mem, stage = [], []
    for (a, b), node in zip(ranges, nodes):
        mem.append((sum(profile.layer_mem[a:b]) + profile.shared_mem) / GB + kv_reserve_gb)
        stage.append(sum(profile.layer_read[a:b]) / (node.bandwidth_gbs * 1e9) * 1000)
    logits_ms = max(profile.shared_read / (n.bandwidth_gbs * 1e9) * 1000 for n in nodes)
    token_ms = sum(stage) + logits_ms + hop_ms * len(nodes)  # n-1 sends + the final gather
    return Plan(list(counts), ranges, mem, stage, token_ms, mode,
                EFFICIENCY["moe" if profile.moe else "dense"])


class InfeasibleError(ValueError):
    pass


def plan(profile: ModelProfile, nodes: list[NodeSpec], mode: str = "latency",
         kv_reserve_gb: float = DEFAULT_KV_RESERVE_GB, hop_ms: float = DEFAULT_HOP_MS) -> Plan:
    """Best contiguous split of the layers over ``nodes`` (in pipeline order)."""
    n_layers, n = len(profile.layer_mem), len(nodes)
    if not 1 <= n <= n_layers:
        raise InfeasibleError(f"cannot split {n_layers} layers over {n} nodes")
    if mode not in ("latency", "throughput"):
        raise ValueError(f"unknown mode {mode!r}")

    best, best_key = None, None
    for cuts in itertools.combinations(range(1, n_layers), n - 1):
        bounds = (0, *cuts, n_layers)
        counts = [bounds[i + 1] - bounds[i] for i in range(n)]
        p = evaluate(profile, nodes, counts, kv_reserve_gb, hop_ms, mode)
        if any(m > node.budget_gb for m, node in zip(p.mem_gb, nodes)):
            continue
        # Primary objective, then the other one as a tie-break; then prefer
        # more headroom on the tightest node.
        headroom = min(node.budget_gb - m for m, node in zip(p.mem_gb, nodes))
        primary, secondary = (sum(p.stage_ms), max(p.stage_ms))[:: 1 if mode == "latency" else -1]
        key = (round(primary, 3), round(secondary, 3), -headroom)
        if best_key is None or key < best_key:
            best, best_key = p, key
    if best is None:
        need = profile.total_mem / GB + kv_reserve_gb * n
        have = sum(node.budget_gb for node in nodes)
        raise InfeasibleError(
            f"no split fits: the model needs about {need:.1f} GB in total "
            f"(including {profile.shared_mem / GB:.1f} GB of embedding/head on every node "
            f"and {kv_reserve_gb:g} GB KV reserve each); budgets add up to {have:.1f} GB"
        )
    return best


def nodes_from_hostfile(path: str | Path) -> list[NodeSpec]:
    """Hostfile entries may carry ``mem_budget_gb`` and ``bandwidth_gbs`` next to mlx.launch's fields."""
    hosts = json.loads(Path(path).read_text())
    nodes = []
    for i, h in enumerate(hosts):
        missing = [k for k in ("mem_budget_gb", "bandwidth_gbs") if k not in h]
        if missing:
            raise ValueError(f"{path}: host {i} ({h.get('ssh')}) lacks {', '.join(missing)}")
        nodes.append(NodeSpec(h.get("name") or h["ssh"], float(h["mem_budget_gb"]), float(h["bandwidth_gbs"])))
    return nodes


def report(profile: ModelProfile, nodes: list[NodeSpec], p: Plan) -> str:
    lines = [
        f"model: {profile.total_mem / GB:.1f} GB in {len(profile.layer_mem)} layers "
        f"(+{profile.shared_mem / GB:.2f} GB embedding/head per node); "
        f"reads {(sum(profile.layer_read) + profile.shared_read) / GB:.2f} GB per token",
        f"split ({p.mode}): {p.split}",
        f"{'node':<24}{'layers':<10}{'memory':>16}{'stage':>10}",
    ]
    for node, (a, b), mem, ms in zip(nodes, p.ranges, p.mem_gb, p.stage_ms):
        lines.append(f"{node.name:<24}{f'{a}-{b - 1}':<10}{f'{mem:.1f}/{node.budget_gb:g} GB':>16}{ms:>8.1f}ms")
    lines.append(f"expected single-stream decode: ~{p.expected_tok_s:.1f} tok/s "
                 f"(bandwidth bound {p.tok_s:.1f} tok/s x {p.efficiency:.0%} measured efficiency)")
    alone = [n.name for n in nodes if profile.total_mem / GB + DEFAULT_KV_RESERVE_GB <= n.budget_gb]
    if alone:
        lines.append(f"note: the model fits on {', '.join(alone)} alone; replicas (one copy per Mac) "
                     f"give more throughput than a pipeline for models that fit")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="swarm plan", description="Dry run: how a model splits over a group")
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument("--hostfile", required=True,
                        help="mlx.launch hostfile whose entries also have mem_budget_gb and bandwidth_gbs")
    parser.add_argument("--mode", choices=("latency", "throughput"), default="latency")
    parser.add_argument("--kv-reserve-gb", type=float, default=DEFAULT_KV_RESERVE_GB)
    parser.add_argument("--hop-ms", type=float, default=DEFAULT_HOP_MS)
    parser.add_argument("--split-only", action="store_true", help="print just the split, e.g. 34,14")
    args = parser.parse_args(argv)
    profile = profile_model(args.model)
    nodes = nodes_from_hostfile(args.hostfile)
    try:
        p = plan(profile, nodes, args.mode, args.kv_reserve_gb, args.hop_ms)
    except InfeasibleError as e:
        sys.exit(str(e))
    print(p.split if args.split_only else report(profile, nodes, p))
