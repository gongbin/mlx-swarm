"""Inference worker: mlx-lm's OpenAI-compatible server, made safe for a shared Mac.

mlx_lm.server already provides /v1/chat/completions (streaming), continuous
batching and speculative decoding. This wrapper adds what a swarm node needs:

- the memory/disk guard runs before the model is loaded, and MLX's memory and
  wired limits are capped to the node budget (stock mlx_lm.server wires the full
  recommended working set, which leaves the OS too little on small Macs);
- the server only ever serves the model it was started with. Stock mlx_lm.server
  loads whatever the request's "model" field names, which would bypass the guard
  and can pull a model from the Hugging Face Hub;
- with ``--draft-model``, speculative decoding is used while a request runs
  alone and batching as soon as others arrive (``swarm.adaptive``);
- it exits when the process that started it (the agent) is gone, so a killed
  agent does not leave an orphan holding the port and the model's memory.

Under ``mlx.launch`` (a pipeline group) every rank runs this worker: each one
loads only its layer range (``--pipeline-split``, see ``swarm.planner``), checks
memory against its own budget (``--mem-budget-gb 20,8.5``: one per rank), and
all collective communication runs on the CPU stream. Rank 0 serves HTTP; the
other ranks follow its batches in lockstep (mlx_lm.server's distributed mode).

Any argument not listed here is passed through to mlx_lm.server:

    swarm worker --model models/Qwen3-8B-4bit --port 8081 --decode-concurrency 8
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time

import mlx.core as mx

from swarm.resources import GB, ResourceError, default_budget_gb, guard_resources

COLLECTIVES = ("all_sum", "all_gather", "all_max", "all_min", "sum_scatter", "send", "recv", "recv_like")


def pin_model(provider) -> None:
    """Make the provider serve only its default (command-line) model."""
    load = provider.load

    def pinned_load(model_path=None, adapter_path=None, draft_model_path=None):
        if model_path not in (None, "default_model"):
            logging.debug("ignoring requested model %r; serving the pinned model", model_path)
        return load("default_model", None, "default_model")

    provider.load = pinned_load


def exit_with_parent(interval: float = 2.0) -> None:
    parent = os.getppid()

    def watch():
        while os.getppid() == parent:
            time.sleep(interval)
        logging.warning("parent process %d is gone; worker exiting", parent)
        os._exit(1)

    threading.Thread(target=watch, daemon=True, name="parent-watch").start()


def collectives_on_cpu() -> None:
    """Default every collective to the CPU stream.

    mlx_lm.server shares requests between ranks with ``all_sum`` on the default
    (GPU) stream; an idle rank > 0 then blocks inside a GPU command buffer
    waiting for rank 0, which trips the Metal watchdog -- the freeze seen in P0.
    """
    for name in COLLECTIVES:
        op = getattr(mx.distributed, name)

        def on_cpu(*args, _op=op, stream=None, **kwargs):
            return _op(*args, stream=mx.cpu if stream is None else stream, **kwargs)

        setattr(mx.distributed, name, on_cpu)


def pipeline_loader(ranges):
    """Replacement for mlx_lm's sharded_load: our split, any llama-like model."""
    from mlx_lm.utils import load

    from swarm.pipeline import apply_pipeline, barrier

    def sharded_load(repo, pipeline_group=None, tensor_group=None, return_config=False, *,
                     tokenizer_config=None):
        group = pipeline_group or mx.distributed.init()
        model, tokenizer, config = load(repo, lazy=True, return_config=True,
                                        tokenizer_config=tokenizer_config)
        start, end = apply_pipeline(model, group, ranges)
        mx.eval(model.parameters())
        barrier(group)
        logging.info("rank %d/%d: layers %d-%d loaded", group.rank(), group.size(), start, end - 1)
        return (model, tokenizer, config) if return_config else (model, tokenizer)

    return sharded_load


def rank_fraction(model: str, ranges, rank: int) -> float:
    """Share of the model's bytes this rank holds (its layers plus embedding/head)."""
    from swarm.planner import profile_model

    try:
        p = profile_model(model)
    except (FileNotFoundError, KeyError, OSError):
        a, b = ranges[rank]
        return (b - a) / ranges[-1][1]
    a, b = ranges[rank]
    return (sum(p.layer_mem[a:b]) + p.shared_mem) / p.total_mem


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="swarm worker", description=__doc__.splitlines()[0], add_help=False
    )
    parser.add_argument("--mem-budget-gb", default=None, help="GB; in a pipeline group one per rank: 20,8.5")
    parser.add_argument("--min-free-disk-gb", type=float, default=20)
    parser.add_argument("--pipeline-split", default=None,
                        help="layers per rank, e.g. 37,11 (see `swarm plan`); default: even")
    args, rest = parser.parse_known_args(argv)
    exit_with_parent()

    group = mx.distributed.init()
    rank, size = group.rank(), group.size()
    budgets = [float(x) for x in args.mem_budget_gb.split(",")] if args.mem_budget_gb else []
    budget = (budgets[rank] if len(budgets) > 1 else budgets[0]) if budgets else default_budget_gb()

    from mlx_lm import server

    serve = server.run

    def guarded_run(host, port, provider, *a, **kw):
        model = provider.cli_args.model
        if not model:
            sys.exit("swarm worker: --model is required")
        fraction = 1.0
        if size > 1:
            from swarm.pipeline import parse_split
            from swarm.resources import num_layers

            ranges = parse_split(args.pipeline_split, num_layers(model), size)
            fraction = rank_fraction(model, ranges, rank)
            server.sharded_load = pipeline_loader(ranges)
        try:
            shard = guard_resources(model, fraction, budget, args.min_free_disk_gb)
        except ResourceError as e:
            sys.exit(f"swarm worker (rank {rank}): refusing to load {model}: {e}")
        # mlx_lm.server.main() already wired the whole recommended working set,
        # and every BatchGenerator would do it again; adaptive.install undoes that.
        wired = int(min(budget, shard + 2) * GB)
        mx.set_wired_limit(wired)
        from swarm import adaptive

        adaptive.install(server, wired)
        pin_model(provider)
        logging.info("serving %s (%.1f GB) with memory limit %.1f GB", model, shard, budget)
        serve(host, port, provider, *a, **kw)

    server.run = guarded_run
    if size > 1:
        collectives_on_cpu()
        # A sleeping rank drops the Thunderbolt link and stalls the whole group.
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
        except OSError:
            pass
        if "--pipeline" not in rest:
            rest.append("--pipeline")
        logging.basicConfig(format=f"%(asctime)s [rank {rank}] %(message)s", level=logging.INFO)
    sys.argv = ["swarm worker", *rest]
    server.main()


if __name__ == "__main__":
    main()
