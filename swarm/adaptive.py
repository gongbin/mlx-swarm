"""Adaptive speculative decoding for mlx_lm.server, and a wired-memory fix for its batches.

mlx_lm.server either speculates with a draft model (one request at a time) or
batches concurrent requests -- never both. Measured on an M5 with Qwen3-8B-4bit
and Qwen3-0.6B-4bit as the draft: speculation gives one stream 41 tok/s instead
of 23, but 4 concurrent requests get 40 tok/s in total instead of 80.

With ``install(server, wired_limit)`` the worker gets both:

- a request that arrives while the server is idle is decoded speculatively;
- requests that arrive while others are queued or a batch is running are batched;
- when a request shows up during a speculative generation, that generation
  stops after the current token, saves its KV cache into the prompt cache and
  re-enters the queue as a continuation, which joins the batch (prefill is one
  token, thanks to the cache) and keeps streaming into the same response.

The prompt cache only ever holds the target model's KV; the speculative path
catches the draft model up on a reused prefix itself (cheap: the draft is small).

Independently of speculation, ``BatchGenerator`` wires the GPU's whole
recommended working set when it starts; the subclass installed here puts the
worker's own wired limit back.
"""

from __future__ import annotations

import dataclasses
import logging
import weakref

import mlx.core as mx

log = logging.getLogger("swarm.adaptive")

PREFILL_CHUNK = 2048


class Continuation:
    """Stands in for a CompletionRequest: the rest of a speculative generation."""

    request_type = "continuation"

    def __init__(self, tokens: list[int], state: str):
        self.tokens = tokens
        self.state = state


class _HandoffQueue:
    """The response queue of a continued request.

    The batch path announces a new GenerationContext; the client already has
    one, so it is swallowed (and stop requests are forwarded to it). The batch
    also detokenizes with a fresh detokenizer, which drops the leading space of
    its first token; the text is recomputed here with a detokenizer that has
    seen everything generated before the hand-off.
    """

    def __init__(self, rqueue, client_ctx, tokenizer, generated: list[int]):
        self.rqueue = rqueue
        self.client_ctx = client_ctx
        self.batch_ctx = None
        self.detokenizer = tokenizer.detokenizer
        for t in generated:
            self.detokenizer.add_token(t)
        self.detokenizer.last_segment  # already sent to the client

    def put(self, item):
        if self.batch_ctx is None and type(item).__name__ == "GenerationContext":
            self.batch_ctx = item
            return
        if self.batch_ctx is not None and self.client_ctx._should_stop:
            self.batch_ctx._should_stop = True
        if dataclasses.is_dataclass(item) and hasattr(item, "token") and hasattr(item, "text"):
            self.detokenizer.add_token(item.token)
            item = dataclasses.replace(item, text=self.detokenizer.last_segment)
        self.rqueue.put(item)


def _active(bg) -> int:
    n = 0
    for attr in ("_generation_batch", "_prompt_batch", "_unprocessed_sequences", "_currently_processing"):
        try:
            n += len(getattr(bg, attr))
        except (AttributeError, TypeError):
            pass
    return n


def install(server, wired_limit: int | None = None) -> None:
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    generators: weakref.WeakSet = weakref.WeakSet()

    class BatchGenerator(server.BatchGenerator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if wired_limit is not None and mx.metal.is_available():
                mx.set_wired_limit(wired_limit)
            generators.add(self)

    server.BatchGenerator = BatchGenerator

    def batch_busy() -> bool:
        return any(_active(bg) for bg in list(generators))

    # -- models: keep batching possible when a draft model is loaded ----------
    provider_load = server.ModelProvider._load

    def _load(self, *args, **kwargs):
        provider_load(self, *args, **kwargs)
        if self.draft_model is not None and not self.is_batchable:
            self.is_batchable = all(hasattr(c, "merge") for c in make_prompt_cache(self.model))

    server.ModelProvider._load = _load

    RG = server.ResponseGenerator
    tokenize = RG._tokenize
    serve_single = RG._serve_single

    def _is_batchable(self, args):
        if args.seed is not None or not self.model_provider.is_batchable:
            return False
        if self.model_provider.draft_model is None or self._is_distributed:
            return True
        if getattr(args, "continuation", False):
            return True
        # Speculate only when this request would otherwise run alone.
        return batch_busy() or self.requests.qsize() > 0

    def _tokenize(self, tokenizer, request, args):
        if isinstance(request, Continuation):
            return list(request.tokens), [list(request.tokens)], ["assistant"], request.state
        return tokenize(self, tokenizer, request, args)

    def _serve_speculative(self, item):
        rqueue, request, args = item
        provider = self.model_provider
        if provider.draft_model is None or isinstance(request, Continuation):
            return serve_single(self, item)
        try:
            _speculate(self, provider, rqueue, request, args)
        except Exception as e:  # same contract as mlx_lm: errors go to the client
            log.exception("speculative generation failed")
            rqueue.put(e)

    def _speculate(self, provider, rqueue, request, args):
        from mlx_lm.generate import stream_generate

        model, draft, tokenizer = provider.model, provider.draft_model, provider.tokenizer
        prompt, _, _, initial_state = self._tokenize(tokenizer, request, args)
        sm, sequences = self._make_state_machine(provider.model_key, tokenizer, args.stop_words, initial_state)
        sm_state = sm.make_state()
        ctx = server.GenerationContext(
            has_thinking=tokenizer.has_thinking, has_tool_calling=tokenizer.has_tool_calling,
            tool_parser=tokenizer.tool_parser, sequences=sequences, prompt=prompt,
        )
        rqueue.put(ctx)

        target_cache, rest = self.prompt_cache.fetch_nearest_cache(provider.model_key, prompt)
        if target_cache is None:
            target_cache, rest = make_prompt_cache(model), prompt
        if not rest:  # exact hit: re-feed the last prompt token
            trim_prompt_cache(target_cache, 1)
            rest = prompt[-1:]
        cached = len(prompt) - len(rest)
        ctx.prompt_cache_count = cached
        # The draft model starts from scratch: catch it up on the cached prefix.
        draft_cache = make_prompt_cache(draft)
        for i in range(0, cached, PREFILL_CHUNK):
            draft(mx.array([prompt[i:min(i + PREFILL_CHUNK, cached)]]), cache=draft_cache)
            mx.eval([c.state for c in draft_cache])

        def progress(done, total):
            rqueue.put((done, total))

        generated, state = [], initial_state
        handoff = False
        for gen in stream_generate(
            model=model, tokenizer=tokenizer, prompt=rest, max_tokens=args.max_tokens,
            sampler=server._make_sampler(args, tokenizer),
            logits_processors=server._make_logits_processors(args),
            prompt_cache=target_cache + draft_cache, draft_model=draft,
            num_draft_tokens=args.num_draft_tokens, prompt_progress_callback=progress,
            prefill_step_size=self.cli_args.prefill_step_size,
        ):
            finish_reason = gen.finish_reason
            sm_state, match, state = sm.match(sm_state, gen.token)
            if match is not None and state is None:
                finish_reason = "stop"
            rqueue.put(server.Response(
                gen.text, gen.token, state, match, gen.logprobs[gen.token].item(), finish_reason,
                server._format_top_logprobs(gen.logprobs, args.top_logprobs, tokenizer),
            ))
            generated.append(gen.token)
            if ctx._should_stop or finish_reason is not None:
                break
            if self.requests.qsize() > 0 and state is not None and len(generated) < args.max_tokens:
                handoff = True
                break

        tokens = prompt + generated
        if not handoff:
            rqueue.put(None)
            # Keep the target's cache for the next turn (it may hold a few
            # unaccepted draft positions beyond the last token: trim them).
            excess = target_cache[0].offset - (len(tokens) - 1)
            if excess > 0:
                trim_prompt_cache(target_cache, excess)
            self.prompt_cache.insert_cache(provider.model_key, tokens[:-1], target_cache)
            return

        # Hand over to the batch: cache everything but the last token, which the
        # batch feeds to produce the next one.
        excess = target_cache[0].offset - (len(tokens) - 1)
        if excess > 0:
            trim_prompt_cache(target_cache, excess)
        self.prompt_cache.insert_cache(provider.model_key, tokens[:-1], target_cache)
        rest_args = dataclasses.replace(args, max_tokens=args.max_tokens - len(generated))
        rest_args.continuation = True
        log.info("speculative request handed to the batch after %d tokens", len(generated))
        self.requests.put((_HandoffQueue(rqueue, ctx, tokenizer, generated), Continuation(tokens, state), rest_args))

    RG._is_batchable = _is_batchable
    RG._tokenize = _tokenize
    RG._serve_single = _serve_speculative
