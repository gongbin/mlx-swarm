# mlx-swarm

Distributed LLM inference across Apple Silicon Macs on MLX: a pool of Macs behind one
OpenAI-compatible API, plus pipeline parallelism for models that do not fit on one Mac.

- Design: `doc2-技术评估与开发方案.md`

```bash
uv sync          # uses the Tsinghua PyPI mirror configured in pyproject.toml
uv run swarm     # list commands
```

- `swarm coordinator` / `swarm agent`: OpenAI-compatible pool with routing and failover
- `swarm jobs submit tasks.jsonl --wait`: batch queue with leases, fencing tokens and retries

## License

Apache License 2.0, see [LICENSE](LICENSE).
