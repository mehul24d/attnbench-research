"""The recall pass of the estimator-frontier study (pre-registration sec. 2.3,
4.7): exact attention mass per block, and each arm's recall against it.

Everything here runs on the device the layer's q and k are on, and nothing
is copied to the host inside a forward.

    M_b(i, j) = (1/|B_i|) * sum_{q in B_i} sum_{k in B_j, k <= q} softmax_q(q.k/sqrt(d))[k]

is computed once at b = 16 as unnormalised sums, in chunks of query tokens,
and summed into 32, 64 and 128. Each arm's mask is then scored:

    R  = sum_{j in S} M(i, j)
    R~ = R / R*,   R* = the free blocks' mass plus the largest masses of as
                        many candidate blocks as the mask keeps

`RecallPass` is used like `ThresholdProfiler`: the model wrapper hands it
each layer's q, k and v, and it returns dense causal attention, so later
layers see the hidden states the real model produces (teacher-forced).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import torch

from .. import masks
from ..analysis import frontier_prereg as fp
from ..analysis.frontier_recall import D_NOMS, RecallRow, split_of

FINEST = 16
BLOCK_SIZES = (16, 32, 64, 128)
MIN_ROW = 2                       # rows i < 2 are filled by the free blocks
Scorer = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


def exact_block_sums(q: torch.Tensor, k: torch.Tensor, *, block: int = FINEST,
                     chunk_tokens: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """(sums, counts). `sums` is (H, n, n) fp32: for query block i and key
    block j, the attention mass summed over the block's query tokens.
    `counts` is (n,): the tokens in each query block. M = sums / counts.

    q and k are (H, S, D), k with the KV heads already repeated. The softmax
    is causal and in fp32, whatever dtype q and k arrive in."""
    if q.dim() != 3 or q.shape != k.shape:
        raise ValueError(f"q and k must both be (H, S, D); got {tuple(q.shape)}, {tuple(k.shape)}")
    if chunk_tokens % block:
        raise ValueError("chunk_tokens must be a multiple of the block size")
    h, s, d = q.shape
    n = -(-s // block)
    pad = n * block - s
    k32 = k.float()
    key_pos = torch.arange(s, device=q.device)
    sums = torch.zeros(h, n, n, dtype=torch.float32, device=q.device)
    for start in range(0, s, chunk_tokens):
        stop = min(start + chunk_tokens, s)
        logits = torch.matmul(q[:, start:stop].float(), k32.transpose(-2, -1)) / math.sqrt(d)
        future = key_pos.unsqueeze(0) > torch.arange(start, stop, device=q.device).unsqueeze(1)
        p = torch.softmax(logits.masked_fill(future, float("-inf")), dim=-1)
        if pad:
            p = torch.nn.functional.pad(p, (0, pad))
        per_token = p.view(h, stop - start, n, block).sum(dim=-1)        # (H, tokens, n)
        rows = torch.arange(start, stop, device=q.device) // block
        sums.index_add_(1, rows, per_token)
    counts = torch.full((n,), block, dtype=torch.float32, device=q.device)
    if pad:
        counts[-1] = block - pad
    return sums, counts


def coarsen(sums: torch.Tensor, counts: torch.Tensor, factor: int) -> tuple:
    """Sums and counts at `factor` times the block size: rows and columns
    are added in groups of `factor`."""
    if factor == 1:
        return sums, counts
    h, n, _ = sums.shape
    m = -(-n // factor)
    pad = m * factor - n
    if pad:
        sums = torch.nn.functional.pad(sums, (0, pad, 0, pad))
        counts = torch.nn.functional.pad(counts, (0, pad))
    sums = sums.view(h, m, factor, m, factor).sum(dim=(2, 4))
    return sums, counts.view(m, factor).sum(dim=1)


def block_mass(sums: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """M: each row sums to 1."""
    return sums / counts.view(1, -1, 1)


def recall(mass: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(R, R~), each (H, n), for a per-head mask `active` (H, n, n).

    R* is the mass of the free blocks (the sink and the diagonal) plus the
    largest masses of as many candidate blocks 0 < j < i as the mask keeps
    in that row. A mask that drops a free block is refused: the definition
    needs F inside S."""
    if mass.shape != active.shape or active.dtype != torch.bool:
        raise ValueError(f"mass {tuple(mass.shape)} and a bool mask {tuple(active.shape)} "
                         f"{active.dtype} must agree")
    h, n, _ = mass.shape
    device = mass.device
    cand = masks._candidate_matrix(n, True, device)
    free = torch.eye(n, dtype=torch.bool, device=device)
    free[:, 0] = True
    if not bool((active | ~free).all()):
        raise ValueError("the mask drops a free block (the sink or the diagonal)")
    if bool((active & ~free & ~cand).any()):
        raise ValueError("the mask keeps a block above the diagonal")
    raw = (mass * active).sum(dim=-1)
    kept = (active & cand).sum(dim=-1)                                   # (H, n)
    ranked = mass.masked_fill(~cand, 0.0).sort(dim=-1, descending=True).values
    best = torch.cat([torch.zeros(h, n, 1, device=device, dtype=mass.dtype),
                      ranked.cumsum(dim=-1)], dim=-1)                   # best[..., c] = top-c mass
    star = (mass * free).sum(dim=-1) + best.gather(-1, kept.unsqueeze(-1)).squeeze(-1)
    return raw, raw / star


def row_budgets(n: int, block: int, d_nom: float, device) -> Optional[torch.Tensor]:
    """None at b = 128 (the nominal budget); the matched budgets below it."""
    if block == 128:
        return None
    return torch.tensor(fp.matched_row_budgets(n, block, d_nom), dtype=torch.long, device=device)


@dataclass
class _Cell:
    raw: float = 0.0
    norm: float = 0.0
    kept: float = 0.0
    valid: float = 0.0
    rows: int = 0


@dataclass
class RecallPass:
    """One example's recall, layer by layer.

    `scorers` maps an arm's name to a function (q, k_rep, block) -> per-head
    scores (H_q, n, n) on the device. The oracle arm "O" needs no scorer:
    its scores are the exact mass. Every arm is selected by the era-4 rule
    at each d_nom and each block size, from one seeded jitter."""
    scorers: Mapping[str, Scorer]
    identity_seed: str
    block_sizes: Sequence[int] = BLOCK_SIZES
    d_noms: Sequence[float] = D_NOMS
    chunk_tokens: int = 2048
    layer_idx: Optional[int] = None
    cells: dict = field(default_factory=dict)
    layers_seen: set = field(default_factory=set)
    _dense: object = None

    def __post_init__(self):
        if "O" in self.scorers:
            raise ValueError('"O" is the oracle; it takes no scorer')
        if any(b not in BLOCK_SIZES for b in self.block_sizes):
            raise ValueError(f"block sizes are {BLOCK_SIZES}")

    def set_layer(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def score_layer(self, q: torch.Tensor, k_rep: torch.Tensor) -> None:
        """q and k_rep are (H_q, S, D)."""
        if self.layer_idx is None:
            raise RuntimeError("set_layer() comes before a layer is scored")
        if self.layer_idx in self.layers_seen:
            raise RuntimeError(f"layer {self.layer_idx} was scored twice for one example")
        self.layers_seen.add(self.layer_idx)
        s = q.shape[1]
        sums16, counts16 = exact_block_sums(q, k_rep, chunk_tokens=self.chunk_tokens)
        for b in self.block_sizes:
            mass = block_mass(*coarsen(sums16, counts16, b // FINEST))
            n = mass.shape[-1]
            valid = torch.arange(1, n + 1, device=mass.device, dtype=torch.float32)
            for arm in ("O", *self.scorers):
                scores = mass if arm == "O" else self.scorers[arm](q, k_rep, b)
                for d in self.d_noms:
                    m = masks.importance_block_mask_per_head(
                        s, b, 1.0 - d, scores, causal=True, identity_seed=self.identity_seed,
                        row_budgets=row_budgets(n, b, d, mass.device))
                    raw, norm = recall(mass, m.active)
                    cell = self.cells.setdefault((arm, b, d), _Cell())
                    cell.raw += float(raw[:, MIN_ROW:].sum())
                    cell.norm += float(norm[:, MIN_ROW:].sum())
                    cell.rows += raw[:, MIN_ROW:].numel()
                    cell.kept += float(m.active.sum())
                    cell.valid += float(valid.sum()) * m.active.shape[0]

    def forward(self, q, k, v, cfg, mask=None):
        """The model wrapper's hook: score this layer, return dense causal
        attention."""
        from ..backends.block_sparse import _expand_kv
        k_rep, _ = _expand_kv(k, v, cfg)
        self.score_layer(q[0], k_rep[0])
        if self._dense is None:
            from ..backends.impls import SDPABackend
            self._dense = SDPABackend("flash" if q.is_cuda else "math")
        return self._dense.forward(q, k, v, cfg)

    def rows(self, *, model: str, task: str, band: int, example_id: str, n_layers: int,
             git_commit: str, git_dirty: bool) -> list[RecallRow]:
        """One row per (arm, block size, d_nom): the example's unweighted mean
        over (layer, head, row). Refused unless every layer was scored."""
        if self.layers_seen != set(range(n_layers)):
            raise RuntimeError(f"scored layers {sorted(self.layers_seen)} of {n_layers}")
        split = split_of(example_id)
        return [RecallRow(model=model, task=task, band=band, example_id=example_id, split=split,
                          arm=arm, mask_selector="per_head", block_size=b, d_nom=d,
                          recall_raw=c.raw / c.rows, recall_norm=c.norm / c.rows,
                          realised_density=c.kept / c.valid,
                          git_commit=git_commit, git_dirty=git_dirty)
                for (arm, b, d), c in sorted(self.cells.items())]


def meanpool_scorer() -> Scorer:
    """The MP arm: per-head mean-pool scores. k arrives with the KV heads
    already repeated, so each query head pools its own key copy."""
    from .model import meanpool_scores_per_head_on_device

    def score(q, k_rep, block):
        return meanpool_scores_per_head_on_device(
            q.unsqueeze(0), k_rep.unsqueeze(0), n_heads_kv=q.shape[0], block_size=block)
    return score
