"""Inference worker: mlx-lm's OpenAI-compatible server, made safe for a shared Mac.

mlx_lm.server already provides /v1/chat/completions (streaming), continuous
batching and speculative decoding. This wrapper adds what a swarm node needs:

- the memory/disk guard runs before the model is loaded, and MLX's memory and
  wired limits are capped to the node budget (stock mlx_lm.server wires the full
  recommended working set, which leaves the OS too little on small Macs);
- the server only ever serves the model it was started with. Stock mlx_lm.server
  loads whatever the request's "model" field names, which would bypass the guard
  and can pull a model from the Hugging Face Hub;
- it exits when the process that started it (the agent) is gone, so a killed
  agent does not leave an orphan holding the port and the model's memory.

Any argument not listed here is passed through to mlx_lm.server:

    swarm worker --model models/Qwen3-8B-4bit --port 8081 --decode-concurrency 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

import mlx.core as mx

from swarm.resources import GB, ResourceError, default_budget_gb, guard_resources


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="swarm worker", description=__doc__.splitlines()[0], add_help=False
    )
    parser.add_argument("--mem-budget-gb", type=float, default=None)
    parser.add_argument("--min-free-disk-gb", type=float, default=20)
    args, rest = parser.parse_known_args(argv)
    exit_with_parent()
    budget = args.mem_budget_gb or default_budget_gb()

    from mlx_lm import server

    serve = server.run

    def guarded_run(host, port, provider, *a, **kw):
        model = provider.cli_args.model
        if not model:
            sys.exit("swarm worker: --model is required")
        try:
            shard = guard_resources(model, 1.0, budget, args.min_free_disk_gb)
        except ResourceError as e:
            sys.exit(f"swarm worker: refusing to load {model}: {e}")
        # mlx_lm.server.main() already wired the whole recommended working set.
        mx.set_wired_limit(int(min(budget, shard + 2) * GB))
        pin_model(provider)
        logging.info("serving %s (%.1f GB) with memory limit %.1f GB", model, shard, budget)
        serve(host, port, provider, *a, **kw)

    server.run = guarded_run
    sys.argv = ["swarm worker", *rest]
    server.main()


if __name__ == "__main__":
    main()
