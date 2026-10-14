# mlx-swarm

Distributed LLM inference across Apple Silicon Macs on MLX: a pool of Macs behind one
OpenAI-compatible API, plus pipeline parallelism for models that do not fit on one Mac.

- Design: `doc2-技术评估与开发方案.md`
- P0 benchmarks and runbook: `docs/P0-runbook.md`
- Running the pool (P1): `docs/P1-quickstart.md`

```bash
uv sync          # uses the Tsinghua PyPI mirror configured in pyproject.toml
uv run swarm     # list commands
```
