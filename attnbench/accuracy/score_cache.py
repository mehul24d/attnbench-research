"""Disk cache for importance scores.

Stage 4's bootstrap over sparsity levels reuses the same dense-softmax pass
(sparsity only changes the top-k threshold at mask-build time, not the
underlying ranking -- see masks.importance_block_mask), so without a cache
the most expensive pass in the study reruns on every sparsity point. Cached
per (model, task, example, seq_len); layer and KV-head are array dimensions
inside the stored tensor, not separate entries.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import torch


def cache_key(model_id: str, task: str, example_id: str, seq_len: int) -> str:
    """Stable hash over all four fields (same style as AttnConfig.key() /
    masks._mask_identity_key: json.dumps, sha1, truncate).

    `model_id` must actually be part of the hashed blob, not merely a
    parameter this function accepts -- the cached tensor's shape
    (n_layers, n_heads_kv, ...) depends on which model produced it, and two
    models in this study genuinely differ (Qwen2.5-1.5B: 28 layers / 2 KV
    heads; Llama-3.2-1B: 16 layers / 8 KV heads). Dropping model_id from the
    hash would let a run against model B silently load model A's
    wrong-shaped tensor. Deliberately excludes block_size: entries are
    always stored at the grid's finest configured block_size (see
    AccuracyGrid.finest_block_size), and a coarser block_size is served by
    pooling the cached finer tensor further at use time, never a separate
    cache entry.
    """
    blob = json.dumps(
        {"model_id": model_id, "task": task, "example_id": example_id,
         "seq_len": seq_len},
        sort_keys=True,
    ).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def _path_for(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.pt"


def save(cache_dir: str | Path, key: str, scores: torch.Tensor) -> None:
    """Write `scores` (shape (n_layers, n_heads_kv, n_blocks, n_blocks),
    fp16) to the cache, atomically.

    fp16 rather than bf16: these are softmax probabilities in [0, 1], where
    fp16's larger mantissa gives better precision than bf16's wider-but-
    shallower range, and nothing here ever feeds a gradient. Written to a
    temp file and renamed into place so a crash mid-write can't leave a
    corrupt entry a later run would silently load as a cache hit.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = _path_for(cache_dir, key)
    tmp = final.with_suffix(f".tmp.{os.getpid()}")
    torch.save(scores.to(torch.float16), tmp)
    os.replace(tmp, final)


def load(cache_dir: str | Path, key: str, *, expected_n_layers: int,
         expected_n_heads_kv: int) -> Optional[torch.Tensor]:
    """Return the cached tensor, or None on a miss.

    Raises on a shape mismatch against `expected_n_layers`/
    `expected_n_heads_kv` rather than silently returning a wrong-shaped
    tensor -- the key is what *prevents* a cross-model collision; this is
    what *catches* one if it happens anyway (a hash truncation collision, a
    hand-edited cache dir, a bug in a future caller that computes the key
    wrong).
    """
    p = _path_for(cache_dir, key)
    if not p.exists():
        return None
    scores = torch.load(p, map_location="cpu")
    n_layers, n_heads_kv = scores.shape[0], scores.shape[1]
    if (n_layers, n_heads_kv) != (expected_n_layers, expected_n_heads_kv):
        raise ValueError(
            f"cache entry {p} has shape (n_layers={n_layers}, "
            f"n_heads_kv={n_heads_kv}), but this run expects "
            f"(n_layers={expected_n_layers}, "
            f"n_heads_kv={expected_n_heads_kv}) -- refusing to hand back a "
            f"wrong-shaped tensor. Delete the entry if it's genuinely stale."
        )
    return scores


# Bytes per stored element. fp16, per `save` above.
_STORED_DTYPE_BYTES = 2


def cache_bytes_for_grid(grid, *, n_layers: int, n_heads_kv: int) -> dict[int, int]:
    """Bytes the score cache will occupy, per seq_len, for a whole grid.

    A function rather than a table in a planning doc. The grid has moved
    twice since the 4.09 GiB figure was first written down (the two-sided
    rule restored 16384 to n=300 and sparsity 0.75), and a doc table cannot
    notice that. This recomputes from whatever the grid says today, and a
    test asserts the total against the pinned grid so a future edit that
    outgrows the disk fails on CPU rather than at hour four of a rented
    session.

    Independent of sparsity: one entry per (model, task, example, seq_len)
    holds every layer, and every sparsity level at that cell reads it.
    """
    out: dict[int, int] = {}
    for seq_len, n_per_length in grid.seq_lens.items():
        n_blocks = seq_len // grid.finest_block_size
        per_example = (n_layers * n_heads_kv * n_blocks * n_blocks
                       * _STORED_DTYPE_BYTES)
        out[seq_len] = per_example * n_per_length * len(grid.tasks)
    return out
