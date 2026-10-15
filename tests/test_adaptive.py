from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import pytest

from swarm.adaptive import _HandoffQueue

TOKENIZER = Path("models/Qwen3-0.6B-4bit")
TEXT = '    date_str : str\n        A string 表示 ISO 8601 的日期，例如 "2025-04-05T14:30:00Z"。\n'


@dataclass
class Response:  # same shape as mlx_lm.server.Response
    text: str
    token: int


@dataclass
class GenerationContext:
    _should_stop: bool = False


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER.exists():
        pytest.skip("needs the local Qwen3-0.6B-4bit tokenizer")
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    return load_tokenizer(TOKENIZER)


def streamed(tokenizer, tokens):
    detok, text = tokenizer.detokenizer, ""
    for t in tokens:
        detok.add_token(t)
        text += detok.last_segment
    return text


def test_text_is_continuous_across_a_handoff_at_any_token(tokenizer):
    tokens = tokenizer.encode(TEXT, add_special_tokens=False)
    expected = streamed(tokenizer, tokens)  # one uninterrupted stream
    for cut in range(1, len(tokens)):
        # Before the hand-off: what the speculative path streamed.
        detok = tokenizer.detokenizer
        sent = ""
        for t in tokens[:cut]:
            detok.add_token(t)
            sent += detok.last_segment
        # After: the batch path's text (from a fresh detokenizer) is replaced.
        out = Queue()
        q = _HandoffQueue(out, GenerationContext(), tokenizer, tokens[:cut])
        q.put(GenerationContext())  # the batch's own context is swallowed
        fresh = tokenizer.detokenizer
        for t in tokens[cut:]:
            fresh.add_token(t)
            q.put(Response(fresh.last_segment, t))
        while not out.empty():
            sent += out.get().text
        assert sent == expected, f"hand-off after token {cut}: {sent!r}"


def test_stop_request_reaches_the_batch_context(tokenizer):
    client, batch = GenerationContext(), GenerationContext()
    q = _HandoffQueue(Queue(), client, tokenizer, [])
    q.put(batch)
    client._should_stop = True
    q.put(Response("x", tokenizer.encode("x", add_special_tokens=False)[0]))
    assert batch._should_stop
