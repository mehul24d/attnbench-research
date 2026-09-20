"""Block-sparse mask generation: random (timing-only) and importance-derived
(accuracy-only), behind one typed interface, deterministic under a seed.

Random and importance masks answer different questions and must never be
compared -- see AttnConfig.mask_source. Every mask this module produces
records which kind it is.

Masks at different sparsity levels for an otherwise-identical config must
nest: the 0.9 pattern is a strict subset of the 0.75 pattern's active
blocks. Deriving the shuffle/ranking from an identity that excludes
`sparsity` and `mask_source`, and taking a prefix/top-k of one fixed
ordering, makes that automatic rather than something callers arrange.
Without it, accuracy-vs-sparsity would carry resampling noise on top of the
sparsity effect itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal, Optional

import torch

from .config import AttnConfig


@dataclass(frozen=True)
class BlockSparseMask:
    """The one source of truth for a block-sparsity pattern: which
    (q_block, kv_block) pairs are active, at a fixed block_size, for a fixed
    seq_len. Every backend-specific representation is a view derived from
    `active`, generated on demand by the to_*() methods below -- never
    stored redundantly, so two backends asked for "the same" mask can never
    silently diverge.
    """

    seq_len: int
    block_size: int
    active: torch.Tensor          # bool, shape (n_blocks, n_blocks), CPU. True = attend.
    seed: int
    source: Literal["random", "importance"]
    causal: bool                  # every backend consuming this mask reads causal-ness
                                    # from here rather than re-hardcoding the same
                                    # assumption mask_for() makes independently

    def __post_init__(self) -> None:
        n = _n_blocks(self.seq_len, self.block_size)
        if tuple(self.active.shape) != (n, n):
            raise ValueError(
                f"active mask shape {tuple(self.active.shape)} does not "
                f"match (n_blocks, n_blocks)=({n}, {n}) for "
                f"seq_len={self.seq_len}, block_size={self.block_size}"
            )
        if self.active.dtype != torch.bool:
            raise ValueError("active must be a bool tensor")
        if self.active.device.type != "cpu":
            raise ValueError(
                "BlockSparseMask stores its pattern on CPU; backends move it "
                "to device in their own to_*() conversion"
            )

    # ---- backend-specific views --------------------------------------

    def to_dense_bool(self, device: str = "cuda") -> torch.Tensor:
        """Expand to a full (seq_len, seq_len) boolean mask. Used by
        NaiveAttention (the correctness oracle) and any backend with no
        native sparse representation.

        **Sub-block causality is applied here, and must be.** `active` is a
        grid over BLOCKS, so it can say "the diagonal block is active" but it
        cannot express the triangle inside that block. `_candidate_rows`
        excludes whole blocks above the diagonal and states the intent that
        "the generated pattern must already exclude [the future] entirely" --
        but a block grid structurally cannot carry that intent onto the
        diagonal, where every q_block attends to its own kv_block.

        Without the `tril` below, a query could attend to up to
        `block_size - 1` strictly future keys inside its own diagonal block:
        measured at seq_len=256/block_size=64, 8064 such (query, key) pairs,
        63 future keys for the worst-case query. That is future leakage in
        the correctness oracle itself.

        It also silently disagreed with the real kernel:
        `backends/block_sparse.py` passes `is_causal=mask.causal` to
        `block_sparse_attn_func`, so BSA masks those positions. The oracle did
        not. A correctness gate comparing the two would have reported a
        mismatch attributable to BSA's kernel, when the defect was in this
        expansion -- and Stage 3 accuracy numbers taken through the oracle
        path would have been optimistic by whatever a 63-token lookahead is
        worth on a retrieval task.

        `to_block_sparse_attn_mask` deliberately does NOT do this: BSA takes
        causality as a separate argument and applies it in-kernel, so encoding
        it in the block grid too would mask the diagonal twice.
        """
        s, bs = self.seq_len, self.block_size
        dense = self.active.repeat_interleave(bs, dim=0).repeat_interleave(bs, dim=1)
        dense = dense[:s, :s].to(device)
        if self.causal:
            dense = dense.tril()
        return dense

    def to_flex_block_mask(self, device: str = "cuda"):
        """FlexAttention `BlockMask` over the same active-block grid.

        Two details here are load-bearing, and getting either wrong produces
        a result that looks fine and measures the wrong thing:

        **BLOCK_SIZE must be this mask's block_size.** FlexAttention decides
        which blocks to skip at *its* granularity, which defaults to 128. Left
        at the default, a `block_size=64` pattern would still be computed
        correctly -- `mask_mod` is exact -- but the kernel would skip work in
        128-wide chunks, so the sparsity actually exploited would not be the
        sparsity configured, and its latency would not be comparable to the
        other backends running the same nominal pattern. Passing BLOCK_SIZE
        explicitly makes flex's skipping grid identical to `active`.

        **Causality is applied inside `mask_mod`, elementwise.** Same reason
        `to_dense_bool` needs a `tril`: `active` is a grid over blocks and
        cannot express the triangle within the diagonal block. FlexAttention
        classifies each block as fully-masked, fully-unmasked, or partial, and
        evaluates `mask_mod` elementwise only in the partial ones -- so the
        diagonal blocks land in the partial set and get exact per-element
        causality, at no cost to the fully-unmasked interior blocks.

        The result agrees elementwise with `to_dense_bool()`; a test asserts
        it, because "two representations of the same mask" is precisely the
        kind of claim that silently stops being true.
        """
        from torch.nn.attention.flex_attention import create_block_mask

        active = self.active.to(device)
        block_size = self.block_size
        causal = self.causal

        def mask_mod(b, h, q_idx, kv_idx):
            allowed = active[q_idx // block_size, kv_idx // block_size]
            if causal:
                allowed = allowed & (q_idx >= kv_idx)
            return allowed

        return create_block_mask(
            mask_mod, B=None, H=None,
            Q_LEN=self.seq_len, KV_LEN=self.seq_len,
            device=device, BLOCK_SIZE=block_size,
        )

    def to_block_sparse_attn_mask(self, batch: int, n_heads: int,
                                   device: str = "cuda") -> torch.Tensor:
        """(batch, n_heads, n_q_blocks, n_kv_blocks) bool tensor, broadcasting
        `active` across batch and heads -- the exact layout
        mit-han-lab/Block-Sparse-Attention's `block_sparse_attn_func` expects
        for `base_blockmask`, confirmed against that repo's source
        (block_sparse_attn_interface.py) and its own fwd_bwd performance
        test, not assumed. This study never varies the sparsity pattern per
        batch element or per head, so broadcasting is exactly right, not a
        simplification of something the library could do differently.
        """
        return (self.active.to(device=device, dtype=torch.bool)
                .unsqueeze(0).unsqueeze(0).expand(batch, n_heads, -1, -1))

    def to_xformers_bias(self):
        """xFormers-native sparse attention bias. Deferred until
        backends/xformers.py lands."""
        raise NotImplementedError("wire up alongside backends/xformers.py")


# ---------------------------------------------------------------------------
# Identity, pool construction, and seeding
# ---------------------------------------------------------------------------

def _n_blocks(seq_len: int, block_size: int) -> int:
    return -(-seq_len // block_size)   # ceil div


def _int_seed(s: str) -> int:
    """Deterministic across processes and runs -- unlike builtin hash(),
    which is randomized per-process (PEP 456) unless PYTHONHASHSEED is
    pinned. A mask's reproducibility must not depend on that env var."""
    return int(hashlib.sha1(s.encode()).hexdigest()[:16], 16)


def _mask_identity_key(cfg: AttnConfig) -> str:
    """Hash over every field that determines the *pool* of maskable blocks,
    excluding `sparsity` and `mask_source`. This is what makes nesting
    across sparsity automatic: every sparsity level for an otherwise-
    identical config derives the same shuffle/ranking and only differs in
    how much of it is kept.
    """
    d = asdict(cfg)
    d.pop("sparsity", None)
    d.pop("mask_source", None)
    blob = json.dumps(d, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _free_blocks(n: int, *, kind: str, seed_int: int) -> list[int]:
    """The kv block granted free to each query block, outside the budget.

    `kind="sink"` gives kv block 0 to every row: the study's rule since
    37675a0, matching Sparse Frontier's Appendix A.1.1.

    `kind="random"` is the CONTROL for that rule (audit 2026-09-20). It grants
    one arbitrary off-diagonal block per row instead, drawn deterministically
    from the row's own candidates. Forcing the sink both grants the sink and
    adds a block per row -- at 2048/0.9 that is +53.8% of active blocks -- and
    the accuracy gain could come from either. This arm holds the block COUNT
    identical and changes only WHICH block is free, so the difference between
    them is the sink's own contribution and nothing else.

    Row 0 has no off-diagonal candidate and gets none; row 1's only candidate
    is block 0, so the control coincides with the sink there by necessity.
    """
    if kind == "sink":
        return [0] * n
    g = torch.Generator().manual_seed(seed_int ^ 0x5F3E_C0DE)
    out = []
    for qb in range(n):
        if qb == 0:
            out.append(-1)                      # nothing free: no candidates
        else:
            out.append(int(torch.randint(0, qb, (1,), generator=g).item()))
    return out


def _candidate_rows(n: int, causal: bool) -> list[list[int]]:
    """kv_block indices each q_block may legally SPEND BUDGET ON.

    Two blocks are granted free and excluded from every row here, so neither
    is spent from the sparsity budget:

      - the **diagonal** block (local context), and
      - **kv_block 0**, the attention sink.

    The sink was previously an ordinary candidate, surviving only if it won a
    top-k. Measured across 1,107 cached oracle score tensors it did not: at
    0.9 sparsity the oracle kept it for 65.0% of query blocks and a random
    mask for 7.5%, against the reference implementation's 100%. 212 of 217
    drops on one example were genuine rank-outs with budget remaining, not
    budget exhaustion.

    The cause is pooling dilution. A sink's attention mass sits on token 0,
    and these are block MEANS over `block_size` tokens, so a coarse block
    spreads that mass and underranks the block holding it. Forcing the sink
    is the standard correction (Sparse Frontier, Appendix A.1.1) and it is
    what this study now does.

    Note the consequence for realised sparsity: two blocks per row are now
    free rather than one, so the achieved density is very slightly above the
    nominal `1 - sparsity`. The reference implementation binary-searches k to
    hit a target exactly; this study does not, and `BlockSparseMask` records
    the realised density so the difference is visible rather than assumed.

    Under causal block-sparsity, kv_block > q_block is structurally invalid
    (not just low-priority) and is never a candidate: NaiveAttention's
    block_sparse branch applies no separate causal mask, so the generated
    pattern must already exclude it entirely.
    """
    if causal:
        return [[kv for kv in range(qb) if kv != 0] for qb in range(n)]
    return [[kv for kv in range(n) if kv != qb and kv != 0] for qb in range(n)]


def _candidate_rows_excluding(n: int, causal: bool, free: list[int]) -> list[list[int]]:
    """`_candidate_rows` with a per-row free block excluded instead of kv 0.

    Same shape and the same budget arithmetic, so a mask built this way has
    the same active-block count as the forced-sink mask it is the control for.
    """
    if causal:
        return [[kv for kv in range(qb) if kv != free[qb]] for qb in range(n)]
    return [[kv for kv in range(n) if kv != qb and kv != free[qb]] for qb in range(n)]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def random_block_mask(seq_len: int, block_size: int, sparsity: float,
                       *, causal: bool, identity_seed: str) -> BlockSparseMask:
    """Timing-only. Samples which off-diagonal blocks are active to hit the
    target sparsity. Deterministic in `identity_seed` alone: every sparsity
    level derived from the same identity_seed takes a prefix of one fixed
    shuffle of the maskable-block pool, so higher sparsity is always a
    strict subset of lower sparsity's active set, never an independent
    resample.
    """
    n = _n_blocks(seq_len, block_size)
    active = torch.zeros(n, n, dtype=torch.bool)
    active.fill_diagonal_(True)
    active[:, 0] = True          # attention sink, granted free -- see _candidate_rows

    rows = _candidate_rows(n, causal)
    pool = [(qb, kv) for qb, kvs in enumerate(rows) for kv in kvs]

    seed_int = _int_seed(identity_seed)
    if pool:
        g = torch.Generator().manual_seed(seed_int)
        order = torch.randperm(len(pool), generator=g).tolist()
        n_active = round((1.0 - sparsity) * len(pool))
        for idx in order[:n_active]:
            qb, kv = pool[idx]
            active[qb, kv] = True

    return BlockSparseMask(seq_len=seq_len, block_size=block_size,
                            active=active, seed=seed_int, source="random",
                            causal=causal)


def importance_block_mask(seq_len: int, block_size: int, sparsity: float,
                           importance_scores: torch.Tensor, *, causal: bool,
                           identity_seed: str,
                           free_block: str = "sink") -> BlockSparseMask:
    """Accuracy-only. `importance_scores` (n_blocks, n_blocks) comes from
    Stage 3's model wrapper -- a real attention-score-derived ranking, not
    computed here, keeping this module hardware/model-agnostic. Keeps the
    top-(1 - sparsity) fraction of off-diagonal candidates per query block
    by score. Nests across sparsity automatically: growing the per-row
    budget is a top-k over a fixed ranking, so it always keeps the smaller
    budget's picks as a subset, never a resample. `identity_seed` only
    breaks exact score ties.

    `free_block` selects which block is granted outside the budget: `"sink"`
    (kv block 0, the study's rule since 37675a0) or `"random"` (the control
    for that rule -- see `_free_blocks`). The block COUNT is identical either
    way, so a difference between the two arms is the sink's own contribution
    and not the extra block the sink fix also grants.
    """
    n = _n_blocks(seq_len, block_size)
    if tuple(importance_scores.shape) != (n, n):
        raise ValueError(
            f"importance_scores shape {tuple(importance_scores.shape)} != "
            f"({n}, {n}) for seq_len={seq_len}, block_size={block_size}"
        )
    if free_block not in ("sink", "random"):
        raise ValueError(f"free_block must be 'sink' or 'random', got {free_block!r}")
    active = torch.zeros(n, n, dtype=torch.bool)
    active.fill_diagonal_(True)
    seed_int = _int_seed(identity_seed)
    if free_block == "sink":
        active[:, 0] = True      # attention sink, granted free -- see _candidate_rows
        rows = _candidate_rows(n, causal)
    else:
        free = _free_blocks(n, kind="random", seed_int=seed_int)
        for qb, kv in enumerate(free):
            if kv >= 0:
                active[qb, kv] = True
        rows = _candidate_rows_excluding(n, causal, free)
    g = torch.Generator().manual_seed(seed_int)

    for qb, kvs in enumerate(rows):
        if not kvs:
            continue
        # DRAW BEFORE THE BUDGET CHECK, and never conditionally. `g` is one
        # stream shared by every row, so a row that skips its draw shifts
        # every later row's jitter. `budget` depends on `sparsity`; the draw
        # must not, or the three arms of the ladder stop sharing a tie-break
        # and nesting fails wherever scores tie.
        #
        # This is instance 45 (docs/silent_failure_patterns.md). The draw sat
        # after `if budget <= 0: continue` until 2026-09-20. Short rows cross
        # that threshold at different sparsities -- at 2048/block128 the 0.5
        # stream has consumed 14 draws by qb=7, 0.75 has consumed 12 and 0.9
        # has consumed 0 -- so from the first short row onward no two arms
        # drew the same jitter. Where scores tie, the tie-break decides the
        # mask, and three different tie-breaks are three unrelated masks.
        #
        # Measured on banked oracle score tensors: at n_blocks=65 nesting
        # broke in 70 of 112 layer-masks, and in 126 of the 128 seq_lens that
        # produce that block count, so it was never a seed artifact. The
        # breaks are concentrated in the final query block, which
        # `_scoring_forward_chunked` zero-pads -- 52.8% of that row's
        # candidates are exactly 0.0 in fp16 there, so its top-k is decided
        # entirely by jitter. Drawing unconditionally takes that count to 0.
        #
        # Note what this does NOT change: with no ties, adding a different
        # 1e-9 perturbation cannot reorder distinct scores, so every
        # untied row builds the identical mask it did before.
        jitter = torch.rand(len(kvs), generator=g) * 1e-9
        budget = round((1.0 - sparsity) * len(kvs))
        if budget <= 0:
            continue
        scores = importance_scores[qb, kvs]
        top = (scores + jitter).topk(k=min(budget, len(kvs))).indices
        for j in top.tolist():
            active[qb, kvs[j]] = True

    return BlockSparseMask(seq_len=seq_len, block_size=block_size,
                            active=active, seed=seed_int,
                            source=("importance" if free_block == "sink"
                                    else "importance_randfree"),
                            causal=causal)


def mask_for(cfg: AttnConfig, *,
             importance_scores: Optional[torch.Tensor] = None) -> BlockSparseMask:
    """The one entry point sweep.py / accuracy/ actually call.

    Dispatches on cfg.mask_source. Seed for nesting is derived from an
    identity that excludes sparsity and mask_source (see
    `_mask_identity_key`), never from a caller-supplied seed -- two callers
    building "the same" cfg always get the identical pattern.

    Every block_sparse config in this study is causal (autoregressive LM
    attention) -- hardcoded here rather than inferred, since AttnConfig.mask
    doesn't carry a separate causal flag for block_sparse today. There is no
    non-causal block-sparse use case in the current grid; revisit this if
    one is added.
    """
    if cfg.mask != "block_sparse":
        raise ValueError(f"mask_for requires cfg.mask == 'block_sparse', got {cfg.mask!r}")
    if cfg.sparsity is None or cfg.block_size is None:
        raise ValueError("block_sparse config requires sparsity and block_size")

    identity_seed = _mask_identity_key(cfg)
    causal = True

    if cfg.mask_source == "random":
        return random_block_mask(cfg.seq_len, cfg.block_size, cfg.sparsity,
                                  causal=causal, identity_seed=identity_seed)
    if cfg.mask_source in ("importance", "importance_randfree"):
        if importance_scores is None:
            raise ValueError(
                f"mask_source == {cfg.mask_source!r} requires importance_scores")
        return importance_block_mask(
            cfg.seq_len, cfg.block_size, cfg.sparsity, importance_scores,
            causal=causal, identity_seed=identity_seed,
            free_block=("sink" if cfg.mask_source == "importance" else "random"))
    raise ValueError(f"unknown mask_source: {cfg.mask_source!r}")
