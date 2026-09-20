"""masks.py: determinism and nesting. Everything downstream (Stage 2 timing
comparisons, Stage 4's accuracy-vs-sparsity curve) depends on mask identity
being correct, so this validates it in isolation, on CPU, before any backend
consumes it.
"""

from __future__ import annotations

import torch

from attnbench.config import AttnConfig
from attnbench.masks import mask_for


def _sparse_cfg(sparsity: float, **overrides) -> AttnConfig:
    fields = dict(seq_len=1024, batch=2, n_heads_q=8, n_heads_kv=8, head_dim=64,
                  dtype="bfloat16", mask="block_sparse", pass_kind="fwd",
                  sparsity=sparsity, block_size=64, mask_source="random")
    fields.update(overrides)
    return AttnConfig(**fields)


def test_random_mask_is_deterministic():
    cfg = _sparse_cfg(0.9)
    m1 = mask_for(cfg)
    m2 = mask_for(cfg)
    assert torch.equal(m1.active, m2.active)
    assert m1.seed == m2.seed


def test_random_mask_nests_across_sparsity():
    """The 0.9 mask's active set must be a strict subset of the 0.75 mask's
    -- higher sparsity is a prefix of the same shuffle, not an independent
    resample, or accuracy-vs-sparsity would carry resampling noise on top
    of the sparsity effect itself."""
    sparse = mask_for(_sparse_cfg(0.9))
    looser = mask_for(_sparse_cfg(0.75))
    assert bool((sparse.active <= looser.active).all())
    assert sparse.active.sum() < looser.active.sum()


def test_importance_mask_nests_across_sparsity():
    cfg_09 = _sparse_cfg(0.9, mask_source="importance")
    cfg_075 = _sparse_cfg(0.75, mask_source="importance")
    n = -(-cfg_09.seq_len // cfg_09.block_size)
    g = torch.Generator().manual_seed(0)
    scores = torch.rand(n, n, generator=g)

    sparse = mask_for(cfg_09, importance_scores=scores)
    looser = mask_for(cfg_075, importance_scores=scores)
    assert bool((sparse.active <= looser.active).all())
    assert sparse.active.sum() < looser.active.sum()


def test_unrelated_field_change_does_not_collapse_identity():
    """Two configs differing only in an unrelated field (dtype) must not be
    treated as 'the same' mask identity -- each still generates
    deterministically, but independently."""
    a = mask_for(_sparse_cfg(0.9, dtype="bfloat16"))
    b = mask_for(_sparse_cfg(0.9, dtype="float16"))
    assert a.seed != b.seed


def test_diagonal_always_active():
    m = mask_for(_sparse_cfg(0.99))
    assert bool(m.active.diagonal().all())


def test_no_active_block_above_diagonal():
    """Causal is structural for block_sparse configs in this study: no
    separate causal mask is applied on top (see NaiveAttention.forward),
    so the generated pattern must never mark a strictly-upper-triangular
    block active."""
    m = mask_for(_sparse_cfg(0.0))   # sparsity=0 -> every legal block active
    n = m.active.shape[0]
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    assert not bool((m.active & upper).any())


def test_to_dense_bool_shape_and_dtype():
    cfg = _sparse_cfg(0.9)
    m = mask_for(cfg)
    dense = m.to_dense_bool(device="cpu")
    assert dense.shape == (cfg.seq_len, cfg.seq_len)
    assert dense.dtype == torch.bool


def test_causal_dense_mask_never_permits_attending_to_the_future():
    """A block grid cannot express the triangle inside its own diagonal
    block, so `to_dense_bool` must apply sub-block causality itself.

    Before this was fixed, a causal block-sparse mask at seq_len=256 /
    block_size=64 permitted 8064 (query, key) pairs where the key was
    strictly in the future -- up to 63 per query, all inside diagonal
    blocks. `backends/block_sparse.py` passes `is_causal` to the real kernel
    and so masked them, meaning the correctness oracle and the kernel under
    test computed different things by construction.
    """
    import torch
    from attnbench.masks import random_block_mask

    for block_size in (64, 128):
        m = random_block_mask(seq_len=512, block_size=block_size, sparsity=0.5,
                              causal=True, identity_seed="causal-check")
        dense = m.to_dense_bool(device="cpu")
        qi = torch.arange(512).unsqueeze(1)
        ki = torch.arange(512).unsqueeze(0)
        future = dense & (ki > qi)
        assert int(future.sum()) == 0, (
            f"block_size={block_size}: {int(future.sum())} positions let a "
            f"query attend to a strictly future key")


def test_non_causal_dense_mask_is_not_triangularised():
    """The tril applies only when the mask is causal -- a non-causal pattern
    must keep its full block structure, or bidirectional configs silently
    lose half their mask."""
    import torch
    from attnbench.masks import random_block_mask

    m = random_block_mask(seq_len=256, block_size=64, sparsity=0.5,
                          causal=False, identity_seed="noncausal-check")
    dense = m.to_dense_bool(device="cpu")
    qi = torch.arange(256).unsqueeze(1)
    ki = torch.arange(256).unsqueeze(0)
    assert int((dense & (ki > qi)).sum()) > 0, (
        "a non-causal mask should still permit attending to later positions")


def test_diagonal_block_is_still_active_after_triangularisation():
    """The fix must not remove the diagonal block -- every query needs to
    attend to at least itself, and the diagonal is granted for free rather
    than spent from the sparsity budget."""
    import torch
    from attnbench.masks import random_block_mask

    m = random_block_mask(seq_len=256, block_size=64, sparsity=0.9,
                          causal=True, identity_seed="diag-check")
    dense = m.to_dense_bool(device="cpu")
    assert bool(dense.diagonal().all()), "every query must attend to itself"
    assert int(dense.sum(1).min()) >= 1


# --------------------------------------------------------------------------
# The attention sink is granted free, in every arm, at every sparsity.
#
# Until 2026-09-16 kv_block 0 was an ordinary candidate that had to win a
# top-k. It usually did not: measured across 1,107 cached oracle score
# tensors, at 0.9 sparsity the oracle kept it for 65.0% of query blocks and a
# random mask for 7.5%. That is a known-destructive divergence from the
# reference implementation, and it put an 8.7x confound inside the
# random-versus-importance comparison, since the two arms dropped the sink at
# very different rates.
#
# Nothing in this suite tested the sink either way, which is why the defect
# survived every green run. These tests exist so it cannot come back silently.
# --------------------------------------------------------------------------

import pytest
import torch

from attnbench import masks as M


_SPARSITIES = (0.5, 0.75, 0.9)


def _scores(n: int) -> torch.Tensor:
    """Scores that deliberately rank the sink LAST, so a passing test proves
    the sink is granted outside the budget rather than merely winning it."""
    g = torch.Generator().manual_seed(0)
    s = torch.rand(n, n, generator=g) + 1.0
    s[:, 0] = 0.0
    return s


@pytest.mark.parametrize("sparsity", _SPARSITIES)
def test_importance_masks_keep_the_sink_even_when_it_scores_worst(sparsity):
    n = 32
    a = M.importance_block_mask(n * 128, 128, sparsity, _scores(n),
                                causal=True, identity_seed="sink").active
    assert bool(a[:, 0].all()), (
        f"sink dropped at sparsity={sparsity} despite being granted free; "
        f"kept for {100 * a[:, 0].float().mean():.1f}% of query blocks"
    )


@pytest.mark.parametrize("sparsity", _SPARSITIES)
def test_random_masks_keep_the_sink(sparsity):
    n = 32
    a = M.random_block_mask(n * 128, 128, sparsity,
                            causal=True, identity_seed="sink").active
    assert bool(a[:, 0].all())


@pytest.mark.parametrize("sparsity", _SPARSITIES)
def test_both_arms_retain_the_sink_at_the_same_rate(sparsity):
    """The confound this fix removes was a DIFFERENCE between the arms, so the
    property to assert is equality, not merely that each is high."""
    n = 32
    imp = M.importance_block_mask(n * 128, 128, sparsity, _scores(n),
                                  causal=True, identity_seed="s").active
    rnd = M.random_block_mask(n * 128, 128, sparsity,
                              causal=True, identity_seed="s").active
    assert imp[:, 0].float().mean() == rnd[:, 0].float().mean() == 1.0


def test_the_sink_is_not_spent_from_the_budget():
    """Granting the sink free must ADD a block, not displace a scored pick.

    If the sink were taken out of budget instead, density would be unchanged
    and the highest-scoring real block would silently vanish.
    """
    n = 64
    sp = 0.9
    s = _scores(n)
    a = M.importance_block_mask(n * 128, 128, sp, s, causal=True,
                                identity_seed="b").active
    # every query block keeps: diagonal + sink + its budgeted picks
    for qb in range(2, n):
        cand = [kv for kv in range(qb) if kv != 0]
        budget = round((1.0 - sp) * len(cand))
        expected = 1 + 1 + budget            # diagonal, sink, budget
        assert int(a[qb].sum()) == expected, (
            f"query block {qb}: got {int(a[qb].sum())} active, expected {expected}"
        )


@pytest.mark.parametrize("sparsity", _SPARSITIES)
def test_forcing_the_sink_does_not_break_nesting(sparsity):
    """Nesting is the property that makes Stage 4's sparsity bootstrap valid:
    a larger budget must be a superset of a smaller one, never a resample."""
    n = 32
    s = _scores(n)
    tight = M.importance_block_mask(n * 128, 128, 0.9, s, causal=True,
                                    identity_seed="n").active
    loose = M.importance_block_mask(n * 128, 128, sparsity, s, causal=True,
                                    identity_seed="n").active
    assert int((tight & ~loose).sum()) == 0


# ---------------------------------------------------------------------------
# Nesting when scores TIE -- instance 45
#
# Every nesting test above seeds with `torch.rand`, which produces essentially
# no ties, so the jitter that breaks ties never decides anything and a bug in
# how it is drawn cannot show. `limitations.md` says so in as many words:
# "torch.rand produces essentially no ties". That made the whole nesting suite
# blind to the defect below for as long as it existed.
#
# The real scores tie constantly. `score_cache` stores fp16, and the final
# query block is zero-padded by `_scoring_forward_chunked` whenever the prompt
# does not land on a block boundary -- which is 95-100% of banked rows. In that
# row, measured on banked oracle tensors, 52.8% of candidates are exactly 0.0,
# so the top-k is decided entirely by the tie-break.
#
# The defect: `jitter` was drawn AFTER `if budget <= 0: continue`, and `budget`
# depends on `sparsity`, so the three arms of the ladder ran off different
# positions in one shared generator and broke ties differently. These fixtures
# are built to make that visible rather than to be realistic.
# ---------------------------------------------------------------------------

def _tied_scores(n: int, *, zeros_in_last_row: bool = True) -> torch.Tensor:
    """Scores shaped like a real fp16-cached tensor with a padded last block.

    Distinct and well separated everywhere except the final query block, where
    most candidates collapse to exactly 0.0 -- the pattern that makes the
    tie-break, not the score, decide which blocks are kept.
    """
    g = torch.Generator().manual_seed(7)
    s = (torch.rand(n, n, generator=g) + 1.0)
    if zeros_in_last_row:
        s[n - 1, :] = 0.0
        s[n - 1, 1] = 5.0          # one genuine winner, the rest tied at zero
    return s.half().float()         # round-trip through the cache's dtype


@pytest.mark.parametrize("tight,loose", ((0.9, 0.75), (0.9, 0.5), (0.75, 0.5)))
def test_nesting_holds_when_most_candidates_tie(tight, loose):
    """The fixture the suite was missing: ties, not `torch.rand`.

    Before the fix this failed at n=65 with 8-17 blocks in the tighter mask
    absent from the looser one. Parametrised over ordered PAIRS rather than
    over a single sparsity: comparing 0.9 against a loose value drawn from a
    list that contains 0.9 makes one case compare a mask with itself, which
    passes whatever the code does.
    """
    n = 65
    s = _tied_scores(n)
    tight_mask = M.importance_block_mask(n * 128, 128, tight, s, causal=True,
                                         identity_seed="tied").active
    loose_mask = M.importance_block_mask(n * 128, 128, loose, s, causal=True,
                                         identity_seed="tied").active
    violations = int((tight_mask & ~loose_mask).sum())
    assert violations == 0, (
        f"nesting broken under ties: {violations} block(s) in the {tight} mask "
        f"are absent from the {loose} mask. The two arms broke the same tie "
        f"differently, which means they did not share a jitter draw."
    )


@pytest.mark.parametrize("n", (16, 17, 32, 65, 128))
def test_nesting_holds_under_ties_at_every_block_count(n):
    """n=17/65 are the shapes that actually occur: a prompt one token over a
    block boundary gives a final query block that is almost entirely padding.
    n=16/32/128 are the exact-multiple cases, kept so a fix that only helps
    the ragged shapes is not mistaken for a general one."""
    s = _tied_scores(n)
    masks_by_sparsity = {
        sp: M.importance_block_mask(n * 128, 128, sp, s, causal=True,
                                    identity_seed=f"tied{n}").active
        for sp in _SPARSITIES
    }
    for tight, loose in ((0.9, 0.75), (0.9, 0.5), (0.75, 0.5)):
        bad = int((masks_by_sparsity[tight] & ~masks_by_sparsity[loose]).sum())
        assert bad == 0, (
            f"n_blocks={n}: {bad} block(s) of the {tight} mask are missing "
            f"from the {loose} mask")


def test_every_sparsity_draws_the_same_jitter_for_the_same_row():
    """The structural form of the property, which holds with or without ties.

    Asserting on masks can only catch the bug when the fixture happens to tie.
    This asserts the mechanism directly: for one identity, the jitter handed to
    query block `qb` must not depend on `sparsity`. Recording the draws in
    order is what makes a shifted stream visible even where it changes nothing.
    """
    n = 65
    s = _tied_scores(n, zeros_in_last_row=False)
    seen: dict[float, list[tuple[int, float]]] = {}
    real_rand = torch.rand

    for sp in _SPARSITIES:
        draws: list[tuple[int, float]] = []

        def recording_rand(*args, **kwargs):
            out = real_rand(*args, **kwargs)
            if kwargs.get("generator") is not None and out.ndim == 1:
                draws.append((int(out.numel()), float(out[0])))
            return out

        torch.rand = recording_rand
        try:
            M.importance_block_mask(n * 128, 128, sp, s, causal=True,
                                    identity_seed="stream")
        finally:
            torch.rand = real_rand
        seen[sp] = draws

    reference = seen[_SPARSITIES[0]]
    for sp in _SPARSITIES[1:]:
        assert seen[sp] == reference, (
            f"the jitter stream differs between sparsity {_SPARSITIES[0]} and "
            f"{sp}: {len(reference)} vs {len(seen[sp])} draws, first divergence "
            f"at index "
            f"{next((i for i, (a, b) in enumerate(zip(reference, seen[sp])) if a != b), 'length')}."
            f"\n\nA draw that is skipped on one arm and taken on another shifts "
            f"every later row's tie-break, so the arms stop agreeing about "
            f"equal-scoring blocks. Draw unconditionally, before any "
            f"sparsity-dependent branch.")
