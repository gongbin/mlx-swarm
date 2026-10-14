"""Benchmark a Prism Bonsai MLX pack with its bundled runtime.

Bonsai packs use custom weight transforms that mlx_lm.load() cannot apply, and
pin their own mlx version, so run this from a separate venv built from the
pack's runtime/requirements.txt:

    ~/bonsai-venv/bin/python bench/bonsai_bench.py models/Ternary-Bonsai-2-27B-mlx-2bit \
        --mem-budget-gb 10 --label M4-16G

Reports load time, prefill and greedy decode tok/s and peak memory, and writes
results/bonsai-<pack>-<label>-<stamp>.json.
"""

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pack")
    parser.add_argument("--prompt-tokens", default="128,1024")
    parser.add_argument("--gen-tokens", type=int, default=128)
    parser.add_argument("--prefill-step", type=int, default=512)
    parser.add_argument("--mem-budget-gb", type=float, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    pack = Path(args.pack).resolve()
    weights_gb = sum(f.stat().st_size for f in pack.glob("*.safetensors")) / 2**30
    if weights_gb > args.mem_budget_gb:
        sys.exit(f"refusing to run: weights {weights_gb:.1f} GB > budget {args.mem_budget_gb} GB")
    mx.set_memory_limit(int(args.mem_budget_gb * 2**30))
    mx.set_cache_limit(2**30)

    sys.path.insert(0, str(pack / "runtime"))
    from tokenizers import Tokenizer

    tic = time.perf_counter()
    config = json.loads((pack / "config.json").read_text())
    if config.get("components", {}).get("vision"):
        # Schema-2 packs with a vision tower load through mlx-vlm; we drive only
        # its language model with text tokens.
        from vision_artifact import load_vl_model

        lm = load_vl_model(pack, load_processor=False)[0].language_model

        def logits(x, cache):
            return lm(x, cache=cache).logits[:, -1, :]

        def prefill(x, cache):
            lm(x, cache=cache, skip_logits=True)

        make_cache = lm.make_cache
    else:
        from artifact import load_model

        model, _ = load_model(pack)

        def logits(x, cache):
            return model.lm_head(model.model(x, cache=cache)[:, -1:, :])[:, -1, :]

        def prefill(x, cache):
            model.model(x, cache=cache)

        make_cache = model.make_cache
    mx.eval(mx.zeros(1))
    load_s = time.perf_counter() - tic
    print(f"loaded {pack.name} ({weights_gb:.1f} GB) in {load_s:.1f}s, "
          f"active {mx.get_active_memory() / 2**30:.2f} GB", flush=True)

    # Sanity check that the transforms are right: a coherent greedy answer.
    tokenizer = Tokenizer.from_file(str(pack / "tokenizer.json"))
    prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one word.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    x = mx.array([tokenizer.encode(prompt, add_special_tokens=False).ids])
    cache = make_cache()
    out = []
    for _ in range(8):
        x = mx.argmax(logits(x, cache), axis=-1)[:, None]
        out.append(int(x.item()))
    sample = tokenizer.decode(out)
    print(f"sanity: {sample!r}", flush=True)

    vocab = tokenizer.get_vocab_size()
    mx.random.seed(0)
    results = []
    for p in [int(v) for v in args.prompt_tokens.split(",")]:
        tokens = mx.random.randint(0, vocab, (1, p))
        mx.reset_peak_memory()
        cache = make_cache()

        tic = time.perf_counter()
        for i in range(0, p - 1, args.prefill_step):
            prefill(tokens[:, i : min(i + args.prefill_step, p - 1)], cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        x = tokens[:, -1:]
        x = mx.argmax(logits(x, cache), axis=-1)[:, None]
        mx.eval(x)
        prefill_s = time.perf_counter() - tic

        tic = time.perf_counter()
        for _ in range(args.gen_tokens):
            x = mx.argmax(logits(x, cache), axis=-1)[:, None]
            mx.async_eval(x)
        mx.eval(x)
        decode_s = time.perf_counter() - tic

        row = {
            "prompt_tokens": p,
            "gen_tokens": args.gen_tokens,
            "prompt_tps": round(p / prefill_s, 1),
            "generation_tps": round(args.gen_tokens / decode_s, 2),
            "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 2),
        }
        results.append(row)
        print(f"prompt={p:>5}  prefill {row['prompt_tps']:>8.1f} tok/s  "
              f"decode {row['generation_tps']:>6.2f} tok/s  peak {row['peak_memory_gb']} GB", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.label}" if args.label else ""
    path = out_dir / f"bonsai-{pack.name}{tag}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({
        "pack": pack.name,
        "host": socket.gethostname(),
        "mlx": mx.__version__,
        "weights_gb": round(weights_gb, 2),
        "load_seconds": round(load_s, 1),
        "sanity_output": sample,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
