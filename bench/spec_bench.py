"""Speculative decoding benchmark: target model alone vs. with a small draft model.

For each prompt and each draft length k, measures decode tok/s, the share of
tokens that came from accepted drafts, and whether the greedy output is
identical to the target alone (it must be: verification is exact).

    uv run python bench/spec_bench.py --model models/Qwen3-8B-4bit \\
        --draft models/Qwen3-0.6B-4bit --num-draft 2,3,4,6
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402

from swarm.resources import ResourceError, default_budget_gb, guard_resources  # noqa: E402

PROMPTS = {
    "explain-zh": "/no_think 用大约200字解释一下什么是统一内存架构，以及它对大模型推理的好处。",
    "code": "/no_think Write a Python function that parses an ISO 8601 date string and returns a datetime, "
            "with docstring and error handling.",
    "list-en": "/no_think List ten practical tips for writing clear technical documentation, one sentence each.",
    "think": "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Think step by step.",
}


def run(model, tokenizer, prompt, max_tokens, draft=None, k=0):
    kwargs = {"draft_model": draft, "num_draft_tokens": k} if draft is not None else {}
    tokens, drafted, last = [], 0, None
    for r in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, **kwargs):
        tokens.append(r.token)
        drafted += r.from_draft
        last = r
    return tokens, drafted, last


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--num-draft", default="2,3,4,6")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompts", default=",".join(PROMPTS))
    parser.add_argument("--mem-budget-gb", type=float, default=None)
    parser.add_argument("--min-free-disk-gb", type=float, default=20)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    budget = args.mem_budget_gb or default_budget_gb()
    try:
        target_gb = guard_resources(args.model, 1.0, budget, args.min_free_disk_gb)
        draft_gb = guard_resources(args.draft, 1.0, budget - target_gb, args.min_free_disk_gb)
    except ResourceError as e:
        sys.exit(f"refusing to run: {e}")
    mx.set_memory_limit(int(budget * 2**30))
    print(f"target {target_gb:.1f} GB + draft {draft_gb:.1f} GB, memory limit {budget:.1f} GB", flush=True)

    model, tokenizer = load(args.model)
    draft, _ = load(args.draft)
    ks = [int(k) for k in args.num_draft.split(",")]
    run(model, tokenizer, "Hi", 8)  # warm up both kernels sets
    run(model, tokenizer, "Hi", 8, draft, 2)

    rows = []
    for name in args.prompts.split(","):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPTS[name]}], add_generation_prompt=True
        )
        base, _, r = run(model, tokenizer, prompt, args.max_tokens)
        rows.append({"prompt": name, "k": 0, "tokens": len(base), "decode_tps": round(r.generation_tps, 2),
                     "accepted_share": 0.0, "identical": True, "peak_gb": round(r.peak_memory, 2)})
        print(f"{name:<11} k=0  {r.generation_tps:6.1f} tok/s  ({len(base)} tokens)", flush=True)
        for k in ks:
            toks, drafted, r = run(model, tokenizer, prompt, args.max_tokens, draft, k)
            row = {"prompt": name, "k": k, "tokens": len(toks), "decode_tps": round(r.generation_tps, 2),
                   "accepted_share": round(drafted / max(len(toks), 1), 3), "identical": toks == base,
                   "peak_gb": round(r.peak_memory, 2)}
            rows.append(row)
            print(f"{name:<11} k={k}  {r.generation_tps:6.1f} tok/s  x{r.generation_tps / rows[-1 - ks.index(k) - 1]['decode_tps']:.2f}"
                  f"  accepted {row['accepted_share']:.0%}  identical={row['identical']}", flush=True)

    summary = {}
    for k in [0, *ks]:
        sel = [r for r in rows if r["k"] == k]
        summary[k] = round(sum(r["decode_tps"] for r in sel) / len(sel), 2)
    print("mean decode tok/s by k:", summary)
    out = Path("results") / (
        f"spec-{Path(args.model).name}-{Path(args.draft).name}{'-' + args.label if args.label else ''}"
        f"-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.write_text(json.dumps({"model": args.model, "draft": args.draft, "host": socket.gethostname(),
                               "max_tokens": args.max_tokens, "summary": summary, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
