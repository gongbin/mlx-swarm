"""Memory and disk guards shared by benchmarks and workers.

The M5 test machine froze twice under memory pressure with a nearly full disk
(macOS could not grow swap). Everything that loads a model goes through
``guard_resources`` first, which refuses to start instead of overcommitting.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlx.core as mx

GB = 2**30


class ResourceError(RuntimeError):
    """The model would not fit comfortably on this node."""


def default_budget_gb() -> float:
    """60% of the GPU's recommended working set, leaving room for the OS and apps."""
    return 0.6 * mx.device_info()["max_recommended_working_set_size"] / GB


def _local_path(model: str) -> Path | None:
    path = Path(model)
    if path.exists():
        return path
    try:
        from mlx_lm.utils import hf_repo_to_path

        return hf_repo_to_path(model)
    except Exception:
        return None


def weight_bytes(model: str) -> tuple[int, bool]:
    """Total safetensors size of a local dir or HF repo, and whether it is on disk."""
    path = _local_path(model)
    if path is not None:
        return sum(f.stat().st_size for f in path.glob("model*.safetensors")), True
    from huggingface_hub import HfApi

    info = HfApi().model_info(model, files_metadata=True)
    return sum(
        s.size or 0
        for s in info.siblings
        if s.rfilename.startswith("model") and s.rfilename.endswith(".safetensors")
    ), False


def num_layers(model: str) -> int | None:
    path = _local_path(model)
    if path is None:
        return None
    config = json.loads((path / "config.json").read_text())
    return config.get("num_hidden_layers") or config.get("text_config", {}).get(
        "num_hidden_layers"
    )


def guard_resources(
    model: str, fraction: float, budget_gb: float, min_free_disk_gb: float
) -> float:
    """Check that this node's share of the weights fits, then cap MLX memory.

    Raises ``ResourceError`` when the shard exceeds ``budget_gb`` or the disk
    would have less than ``min_free_disk_gb`` free. Returns the shard size in GB.
    """
    weights, on_disk = weight_bytes(model)
    shard = weights * fraction / GB
    free_disk = shutil.disk_usage(Path.home()).free / GB
    if not on_disk:
        free_disk -= weights / GB
    problems = []
    if shard > budget_gb:
        problems.append(f"weights per rank {shard:.1f} GB > memory budget {budget_gb:.1f} GB")
    if free_disk < min_free_disk_gb:
        problems.append(
            f"free disk {'after download ' if not on_disk else ''}{free_disk:.1f} GB "
            f"< {min_free_disk_gb:.0f} GB (needed for swap headroom)"
        )
    if problems:
        raise ResourceError("; ".join(problems))
    # Hard cap for MLX allocations; leave the rest for the OS and other apps.
    mx.set_memory_limit(int(budget_gb * GB))
    mx.set_cache_limit(int(2 * GB))
    return shard
