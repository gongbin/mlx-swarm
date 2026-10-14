"""Heterogeneous pipeline parallelism for mlx-lm models.

mlx-lm 0.31.x ships ``PipelineMixin`` only for a few model families
(deepseek_v2/v3, glm4_moe, ministral3) and always splits layers evenly.
This module adds pipeline support to "llama-like" decoder models
(``model.model.{embed_tokens, layers, norm}`` with ``layer(h, mask, cache)``)
and lets each rank own an arbitrary contiguous range of layers, so slower or
smaller Macs can take fewer layers.

Rank 0 owns the first layers. Activations flow rank r -> r+1, and the final
hidden state is all-gathered so every rank computes the same logits and
samples the same token (same scheme as mlx-lm's own pipeline models).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask

# Model types whose inner model matches the generic forward below.
SUPPORTED_MODEL_TYPES = {"llama", "mistral", "qwen2", "qwen3", "qwen3_moe"}

LayerRange = Tuple[int, int]


def split_layers(num_layers: int, weights: Sequence[float]) -> List[LayerRange]:
    """Split ``num_layers`` into contiguous ranges proportional to ``weights``.

    Every rank gets at least one layer, and the counts always sum to
    ``num_layers``.
    """
    n = len(weights)
    if n == 0 or n > num_layers:
        raise ValueError(f"cannot split {num_layers} layers across {n} ranks")
    if any(w <= 0 for w in weights):
        raise ValueError("weights must be positive")

    total = sum(weights)
    raw = [w / total * num_layers for w in weights]
    counts = [max(1, int(r)) for r in raw]
    # Fix rounding: add to ranks furthest below their share, take from ranks
    # furthest above it (never below one layer).
    while sum(counts) < num_layers:
        i = max(range(n), key=lambda i: raw[i] - counts[i])
        counts[i] += 1
    while sum(counts) > num_layers:
        i = max((i for i in range(n) if counts[i] > 1), key=lambda i: counts[i] - raw[i])
        counts[i] -= 1

    ranges, start = [], 0
    for c in counts:
        ranges.append((start, start + c))
        start += c
    return ranges


def parse_split(spec: Optional[str], num_layers: int, size: int) -> List[LayerRange]:
    """Parse a split spec.

    - ``None`` / ``"even"``: equal split
    - ``"10,18"``: explicit layer counts per rank (must sum to ``num_layers``)
    - ``"w:1,2.5"``: proportional weights per rank (e.g. memory budgets)
    """
    if spec in (None, "", "even"):
        return split_layers(num_layers, [1.0] * size)
    if spec.startswith("w:"):
        weights = [float(x) for x in spec[2:].split(",")]
        if len(weights) != size:
            raise ValueError(f"split has {len(weights)} weights for {size} ranks")
        return split_layers(num_layers, weights)

    counts = [int(x) for x in spec.split(",")]
    if len(counts) != size or sum(counts) != num_layers or min(counts) < 1:
        raise ValueError(
            f"split {counts} must have {size} positive entries summing to {num_layers}"
        )
    ranges, start = [], 0
    for c in counts:
        ranges.append((start, start + c))
        start += c
    return ranges


def _pipelined_forward(self, inputs, cache=None, input_embeddings=None):
    if input_embeddings is not None:
        h = input_embeddings
    else:
        h = self.embed_tokens(inputs)

    if cache is None:
        cache = [None] * len(self.layers)
    mask = create_attention_mask(h, cache[0])

    rank, size, group = self.pipeline_rank, self.pipeline_size, self.pipeline_group

    # Communication runs on the CPU stream and is evaluated eagerly before any
    # GPU work that depends on it is submitted. Otherwise the GPU command buffer
    # carries a wait on the network (via a cross-stream event), which on slow
    # links trips the Metal watchdog (kIOGPUCommandBufferCallbackErrorTimeout)
    # and can freeze the whole Mac. Unified memory makes the CPU hop free.
    if rank > 0:
        h = mx.distributed.recv_like(h, rank - 1, group=group, stream=mx.cpu)
        mx.eval(h)

    for layer, c in zip(self.layers, cache):
        h = layer(h, mask, c)

    if rank < size - 1:
        h = mx.distributed.send(h, rank + 1, group=group, stream=mx.cpu)
        # Keep the send in the graph: the cache is always evaluated.
        if cache[-1] is not None:
            cache[-1].keys = mx.depends(cache[-1].keys, h)

    if size > 1:
        # Every rank receives the last stage's output.
        # The slice must stay on the CPU stream too, or the GPU waits on the gather.
        with mx.stream(mx.cpu):
            h = mx.distributed.all_gather(h, group=group)[-h.shape[0] :]
        mx.eval(h)

    return self.norm(h)


_CLASS_CACHE: dict = {}


def _pipelined_class(cls):
    if cls not in _CLASS_CACHE:
        _CLASS_CACHE[cls] = type(
            f"Pipelined{cls.__name__}", (cls,), {"__call__": _pipelined_forward}
        )
    return _CLASS_CACHE[cls]


def apply_pipeline(
    model, group: mx.distributed.Group, ranges: Sequence[LayerRange]
) -> LayerRange:
    """Keep only this rank's layers and switch the inner model to pipelined forward.

    Call on a lazily loaded model (``mlx_lm.load(..., lazy=True)``) *before*
    evaluating parameters, so weights of dropped layers are never materialized.
    Returns this rank's layer range.
    """
    model_type = getattr(model, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"model_type={model_type!r} not supported; "
            f"supported: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    inner = model.model
    if len(ranges) != group.size():
        raise ValueError(f"{len(ranges)} ranges for group of size {group.size()}")
    if ranges[-1][1] != len(inner.layers):
        raise ValueError(f"ranges cover {ranges[-1][1]} of {len(inner.layers)} layers")

    rank = group.rank()
    start, end = ranges[rank]
    inner.layers = inner.layers[start:end]
    inner.pipeline_rank = rank
    inner.pipeline_size = group.size()
    inner.pipeline_group = group
    inner.__class__ = _pipelined_class(type(inner))
    return start, end


def barrier(group: Optional[mx.distributed.Group] = None):
    mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=mx.cpu))
