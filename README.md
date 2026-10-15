# mlx-swarm

English | [简体中文](README.zh-CN.md)

Turn the Apple Silicon Macs you already have into one LLM inference cluster. mlx-swarm runs
[MLX](https://github.com/ml-explore/mlx) models across several Macs and serves them behind a single
OpenAI-compatible API.

## Features

- **One API for many Macs**: any OpenAI SDK works against the cluster. Requests go to the least-loaded Mac, conversations stay on the same machine to reuse its prompt cache, and requests move to another Mac when one fails.
- **Batch jobs that finish exactly once**: submit thousands of prompts as a JSONL file. Every Mac pulls work at its own pace. Leases, fencing tokens and retries make sure each task's result is saved exactly once, even when a Mac crashes or sleeps mid-task.
- **Models bigger than one Mac**: pipeline parallelism splits a model by layers across Macs over Thunderbolt or Ethernet, with output identical to a single Mac. A planner picks the split for Macs of different sizes and speeds and predicts the speed before anything is loaded. The group is served through the same API.
- **Fast single streams, full throughput under load**: with a small draft model a worker uses speculative decoding while a request runs alone (about 2x faster), and switches to batching the moment other requests arrive. A running request is handed over to the batch mid-stream without a pause.
- **Web dashboard**: open the coordinator's address in a browser to see every Mac's health, memory and live tok/s, chat with any served model, submit and download batch jobs, and create or revoke API keys with per-key usage.
- **Zero configuration**: the coordinator announces itself over Bonjour (mDNS). Nodes find it on their own, pick the fastest link, and run as login services that restart after a crash.
- **Safe on shared Macs**: memory and disk checks run before every model load, and memory use is capped below what macOS needs. Nodes stay awake only while they are serving.

## Current functionality

| Component | What it does |
|---|---|
| `swarm coordinator` | OpenAI-compatible gateway (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming), node registry, routing and failover, batch job queue (`/v1/jobs`, OpenAI Batch format), optional API keys |
| Dashboard (`http://<coordinator>:8000/`, or `swarm ui`) | Node status and throughput, drain or remove nodes, chat playground, batch jobs, API keys and examples, and a built-in guide to setting up and running the cluster. Managing the cluster needs the coordinator's Mac or its `--admin-token` |
| `swarm agent` | Runs on every Mac: starts and supervises an MLX worker, finds the coordinator, reports health, runs batch tasks. With `--hostfile` it runs a pipeline group across several Macs as one node |
| `swarm plan` | Dry run: how a model splits over a group of Macs, memory per Mac, and expected tok/s |
| `swarm jobs` | Submit, track, cancel and download batch jobs |
| `swarm service` | Installs the coordinator or an agent as a launchd login service |
| `swarm nodes` / `chat` / `discover` | Cluster status, quick chat, mDNS lookup |
| `bench/` | Node probe, network micro-benchmark, single-Mac and pipeline benchmarks |

```bash
uv sync
uv run swarm service install coordinator                              # on one Mac
uv run swarm service install agent --model models/Qwen3-8B-4bit      # on every Mac
uv run swarm chat "Hello" --model Qwen3-8B-4bit
uv run swarm jobs submit tasks.jsonl --model Qwen3-8B-4bit --wait -o results.jsonl
```

Roadmap: two-Mac run of the 30B MoE, request interleaving in the pipeline and group failover (P2); more single-Mac speedups such as prefix caching and KV quantization (P3).

## Benchmarks

Test setup: an M5 MacBook Pro (32 GB, 122 GB/s) and an M4 Mac mini (16 GB, 101 GB/s), connected by a Thunderbolt 3 cable (6.5 Gbit/s, 0.26 ms round trip).

**Pipeline parallelism (decode speed, one stream)**

| Model | M5 alone | M5 + mini | Notes |
|---|---|---|---|
| Qwen3-8B-4bit | 27.9 tok/s | 25.0 tok/s | 93.5% of the theoretical limit, output identical to one Mac |
| Qwen3-32B-4bit | 6.7 tok/s | 6.1 tok/s | 91% of the theoretical limit, output identical to one Mac |
| Qwen3-30B-A3B-4bit (MoE) | 59.1 tok/s | 51.4 tok/s | output identical to one Mac |
| Qwen3-30B-A3B-6bit (MoE, 23 GiB) | does not fit | 35–37 tok/s | runs only when split across both Macs |

**Replica pool (both Macs serving Qwen3-8B-4bit)**

| Test | Result |
|---|---|
| 8 concurrent chat requests | spread 4 / 4 across the two Macs |
| Kill one worker during requests | all requests answered by the other Mac; the worker restarted in about 2 s |
| Batch of 40 prompts | 16 s |
| `kill -9` one Mac and restart the coordinator during a batch | 40/40 succeeded, no task ran twice |
| `kill -9` a Mac holding 4 tasks | its tasks were picked up again after about 30 s and completed on the other Mac |
| `kill -9` an agent running as a service | launchd restarted it after 7 s; it re-registered by 11 s |

**Adaptive speculative decoding (M5, Qwen3-8B-4bit with Qwen3-0.6B-4bit as the draft)**

| Load | Batching only | Speculative only | Adaptive |
|---|---|---|---|
| One request at a time | 23.4 tok/s | 40.9 tok/s | 48–52 tok/s |
| 4 concurrent requests (total) | 80.2 tok/s | 40.2 tok/s | 75–81 tok/s |
| 3 short requests arriving during a long one | — | wait for it to finish | answered in about 1 s |

## License

[Apache License 2.0](LICENSE)
