"""LLM inference benchmark: single node, or pipeline-parallel across Macs.

Single node (also writes the greedy reference used to check pipeline output):
    python bench/llm_bench.py --model <path|repo> --save-ref

Pipeline across processes/Macs (launched by mlx.launch, ring backend):
    mlx.launch --backend ring -n 2 bench/llm_bench.py --model <path|repo>
    mlx.launch --backend ring --hostfile hosts.json bench/llm_bench.py \
        --model <path|repo> --split w:32,16

Reports prefill tok/s, decode tok/s and per-rank peak memory for each
(prompt length, batch size) pair, and checks that pipeline greedy output is
token-identical to the single-node reference.
"""

import argparse
import json
import re
import socket
import sys
import time
import zlib
from importlib import metadata
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate, load, stream_generate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm.pipeline import apply_pipeline, barrier, parse_split  # noqa: E402
from swarm.resources import ResourceError, default_budget_gb, guard_resources, num_layers  # noqa: E402

REF_PROMPT = (
    "Explain, step by step, why the sky appears blue during the day "
    "and red at sunset."
)
REF_TOKENS = 64


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(model).name or model)


def load_model(model_path: str, group, split: str | None):
    if group.size() == 1:
        model, tokenizer, config = load(model_path, return_config=True)
        return model, tokenizer, config, [(0, len(model.layers))]

    model, tokenizer, config = load(model_path, lazy=True, return_config=True)
    ranges = parse_split(split, len(model.model.layers), group.size())
    apply_pipeline(model, group, ranges)
    mx.eval(model.parameters())
    barrier(group)
    return model, tokenizer, config, ranges


def greedy_tokens(model, tokenizer) -> list[int]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": REF_PROMPT}], add_generation_prompt=True
    )
    return [
        r.token for r in stream_generate(model, tokenizer, prompt, max_tokens=REF_TOKENS)
    ]


def run_once(model, tokenizer, prompts, gen_tokens, prefill_step):
    if len(prompts) == 1:
        for r in stream_generate(
            model,
            tokenizer,
            prompts[0],
            max_tokens=gen_tokens,
            prefill_step_size=prefill_step,
        ):
            pass
        return r.prompt_tps, r.generation_tps
    stats = batch_generate(
        model, tokenizer, prompts, max_tokens=gen_tokens, prefill_step_size=prefill_step
    ).stats
    return stats.prompt_tps, stats.generation_tps


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default=None, help="even | 10,18 | w:1,2.5")
    parser.add_argument("--prompt-tokens", default="128,1024,4096")
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prefill-step", type=int, default=2048)
    parser.add_argument("--save-ref", action="store_true")
    parser.add_argument("--label", default="", help="free-form tag, e.g. 2xM5-tb4")
    parser.add_argument("--out", default="results")
    parser.add_argument(
        "--mem-budget-gb",
        default=None,
        help="max MLX memory per rank in GB, one value or one per rank (e.g. 20,8.5); "
        "default: 60%% of each node's GPU recommended working set",
    )
    parser.add_argument("--min-free-disk-gb", type=float, default=20)
    args = parser.parse_args()

    group = mx.distributed.init()
    rank, size = group.rank(), group.size()
    mode = "single" if size == 1 else "pipeline"

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    if args.mem_budget_gb:
        budgets = [float(x) for x in args.mem_budget_gb.split(",")]
        budget = budgets[rank] if len(budgets) > 1 else budgets[0]
    else:
        budget = default_budget_gb()
    if size > 1 and len(set(mx.distributed.all_gather(
        mx.array([zlib.crc32(socket.gethostname().encode()) % 2**31]), group=group, stream=mx.cpu
    ).tolist())) == 1:
        # Ranks on one Mac share one GPU and RAM; split the budget between them.
        budget /= size
    # This rank's share of the weights, from the planned layer split.
    fraction = 1 / size
    if size > 1 and (n := num_layers(args.model)):
        start, end = parse_split(args.split, n, size)[rank]
        fraction = (end - start) / n
    try:
        shard_gb = guard_resources(args.model, fraction, budget, args.min_free_disk_gb)
    except ResourceError as e:
        sys.exit(f"refusing to run: {e}")
    print(f"[rank {rank} {socket.gethostname()}] weights ~{shard_gb:.1f} GB, "
          f"MLX memory limit {budget:.1f} GB", flush=True)

    tic = time.perf_counter()
    model, tokenizer, config, ranges = load_model(args.model, group, args.split)
    load_s = time.perf_counter() - tic
    log(f"[{mode} x{size}] loaded {args.model} in {load_s:.1f}s, layers per rank: {ranges}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_path = out / f"ref-{model_slug(args.model)}.json"

    # Correctness: greedy output must match the single-node reference.
    tokens = greedy_tokens(model, tokenizer)
    correctness = None
    if mode == "single" and args.save_ref:
        ref_path.write_text(json.dumps({"model": args.model, "tokens": tokens}))
        log(f"saved greedy reference to {ref_path}")
    elif mode == "pipeline" and ref_path.exists():
        ref = json.loads(ref_path.read_text())["tokens"]
        match = next((i for i, (a, b) in enumerate(zip(ref, tokens)) if a != b), None)
        correctness = {
            "identical": tokens == ref,
            "first_mismatch": match,
            "checked_tokens": len(ref),
        }
        log(f"greedy check vs single node: {correctness}")
    elif mode == "pipeline":
        # No single-node reference (e.g. the model does not fit on one Mac):
        # at least show the greedy answer so a broken split is obvious.
        log(f"no single-node reference; greedy output: {tokenizer.decode(tokens)!r}")

    # Random prompts and no EOS, so every trial generates exactly gen_tokens.
    tokenizer._eos_token_ids = set()
    vocab = config.get("vocab_size") or config["text_config"]["vocab_size"]
    mx.random.seed(0)

    results = []
    for p in [int(x) for x in args.prompt_tokens.split(",")]:
        for b in [int(x) for x in args.batch_sizes.split(",")]:
            prompts = mx.random.randint(0, vocab, (b, p)).tolist()
            mx.reset_peak_memory()
            run_once(model, tokenizer, prompts, args.gen_tokens, args.prefill_step)
            trials = [
                run_once(model, tokenizer, prompts, args.gen_tokens, args.prefill_step)
                for _ in range(args.trials)
            ]
            peak = mx.distributed.all_gather(
                mx.array([mx.get_peak_memory() / 2**30]), group=group, stream=mx.cpu
            )
            row = {
                "prompt_tokens": p,
                "batch_size": b,
                "gen_tokens": args.gen_tokens,
                "prompt_tps": round(sum(t[0] for t in trials) / len(trials), 1),
                "generation_tps": round(sum(t[1] for t in trials) / len(trials), 2),
                "peak_memory_gb_per_rank": [round(x, 2) for x in peak.tolist()],
            }
            results.append(row)
            log(
                f"prompt={p:>5} batch={b:>2}  prefill {row['prompt_tps']:>9.1f} tok/s  "
                f"decode {row['generation_tps']:>7.2f} tok/s  "
                f"peak GB/rank {row['peak_memory_gb_per_rank']}"
            )

    hosts = mx.distributed.all_gather(
        mx.array([zlib.crc32(socket.gethostname().encode()) % 2**31]), group=group, stream=mx.cpu
    )
    report = {
        "model": args.model,
        "mode": mode,
        "world_size": size,
        "distinct_hosts": len(set(hosts.tolist())),
        "layer_ranges": ranges,
        "label": args.label,
        "host": socket.gethostname(),
        "mlx": mx.__version__,
        "mlx_lm": metadata.version("mlx-lm"),
        "load_seconds": round(load_s, 1),
        "correctness": correctness,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if rank == 0:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = f"-{args.label}" if args.label else ""
        path = out / f"llm-{model_slug(args.model)}-{mode}{size}{tag}-{stamp}.json"
        path.write_text(json.dumps(report, indent=2))
        log(f"Saved to {path}")


if __name__ == "__main__":
    main()
