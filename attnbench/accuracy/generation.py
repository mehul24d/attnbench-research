"""The per-cell execution path: prompt in, `Generated` out.

Extracted from scripts/run_accuracy.py rather than left as a closure inside
it, for the same reason `runner.build_cells` and `runner.plan` are pure
functions: this is where a Stage 3 row is actually produced, and a body that
only exists inside a script needs a GPU and a downloaded model to exercise
even once. Here it runs against a toy model and a fake tokenizer on CPU, so
the substitutions it makes -- real head geometry onto a placeholder cell
config, the real tokenized length onto the grid's nominal one, the per-task
cap, the decode-backend choice -- are all covered before a rented hour is
spent discovering one of them was wrong.

The script keeps what genuinely belongs to it: loading the model, wrapping
it, and deriving the stop-token sets once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Optional

from . import stopping
from .grid_configs import decode_backend_for, gate_source_of
from .schema import Generated
from ..backends.base import AttentionBackend
from ..config import AttnConfig


@dataclass(frozen=True)
class StopTokens:
    """The three vocabulary-derived id sets the stopping rule needs.

    Derived once and passed in, not recomputed per example: a byte-level BPE
    vocabulary is ~150K entries, and scanning it per cell would cost more
    across the grid than the decode steps it governs.
    """

    eos: frozenset[int]
    newline: frozenset[int]
    whitespace: frozenset[int]

    @classmethod
    def from_tokenizer(cls, tokenizer, generation_config=None) -> "StopTokens":
        return cls(eos=stopping.eos_token_ids(tokenizer, generation_config),
                   newline=stopping.newline_token_ids(tokenizer),
                   whitespace=stopping.whitespace_token_ids(tokenizer))


@dataclass(frozen=True)
class ModelGeometry:
    """The real model's head shape and dtype.

    A cell's `AttnConfig` carries placeholder geometry (`n_heads_q=1`,
    `head_dim=128`) -- correct as a cell identity, wrong as a description of
    what runs. The same placeholder-vs-real confusion already made one FLOPs
    estimate wrong (see timing_probe._with_real_heads); this is the runtime
    half of the same fix.
    """

    n_heads_q: int
    n_heads_kv: int
    head_dim: int
    dtype: str

    @classmethod
    def from_config(cls, config, dtype: str) -> "ModelGeometry":
        n_heads_q = config.num_attention_heads
        return cls(n_heads_q=n_heads_q,
                   n_heads_kv=config.num_key_value_heads,
                   head_dim=getattr(config, "head_dim",
                                    config.hidden_size // n_heads_q),
                   dtype=dtype)

    def onto(self, cfg: AttnConfig, *, seq_len: int) -> AttnConfig:
        """`cfg`'s identity (mask, sparsity, block_size, mask_source) with
        this model's geometry and the example's real tokenized length."""
        return replace(cfg, n_heads_q=self.n_heads_q, n_heads_kv=self.n_heads_kv,
                       head_dim=self.head_dim, dtype=self.dtype, batch=1,
                       seq_len=seq_len)


def generate_one(wrapped, tokenizer, *, cfg: AttnConfig, backend: AttentionBackend,
                 example, geometry: ModelGeometry, stop_tokens: StopTokens,
                 score_cache_dir: str, device: str = "cuda",
                 synchronize=None) -> Generated:
    """One Stage 3 row's text and the wall time it took.

    `synchronize` is called on both sides of the timer where the device is
    asynchronous. Defaulted to a real `torch.cuda.synchronize` when the
    device is cuda, because forgetting it does not raise -- it just reports a
    kernel launch instead of a kernel, in the direction that flatters
    whichever backend queues most work.
    """
    import torch

    if synchronize is None:
        synchronize = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    input_ids = tokenizer(example.context, return_tensors="pt").input_ids.to(device)
    run_cfg = geometry.onto(cfg, seq_len=int(input_ids.shape[-1]))

    layer_scores: Optional[dict] = None
    if run_cfg.mask == "block_sparse":
        # Cache-checked inside; the dense scoring pass runs once per
        # (model, task, example, seq_len) and is shared across every
        # sparsity level at that cell.
        layer_scores = wrapped.compute_importance_scores(
            input_ids, task=example.task, example_id=example.example_id,
            cache_dir=score_cache_dir)

    decode_backend = decode_backend_for(backend)

    synchronize()
    t0 = time.perf_counter()
    result = wrapped.generate(
        input_ids, backend, cfg=run_cfg,
        max_new_tokens=stopping.token_cap(example.task),
        eos_token_ids=stop_tokens.eos,
        newline_token_ids=stop_tokens.newline,
        whitespace_token_ids=stop_tokens.whitespace,
        layer_scores=layer_scores, decode_backend=decode_backend)
    synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return Generated(text=tokenizer.decode(result.token_ids, skip_special_tokens=True),
                     latency_ms=latency_ms, stop_reason=result.stop_reason,
                     n_generated=result.n_generated,
                     decode_backend=result.decode_backend,
                     gate_source=gate_source_of(backend))
