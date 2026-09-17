"""The equivalence guard in scripts/run_vectorised_endtoend.py, driven directly.

Both 2026-09-07 failures were in a script body no test could reach, and item 4
repeated the shape: its first guard was written against synthetic continuous
scores, passed 36 configs, and fired on the first real band. These tests drive
the SAME function the script calls, on data shaped like what the score cache
actually stores (fp16, tie-dense), and -- most importantly -- assert the
relaxed guard still REJECTS a builder that is genuinely wrong.

A guard relaxed after it fires is worthless unless something proves it still
bites. That is what test_rejects_* are for.
"""

import sys
from pathlib import Path

import pytest
import torch

from attnbench import masks

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_vectorised_endtoend as R  # noqa: E402


def _tie_dense_scores(n_layers=3, n_kv=2, n=48, seed=0):
    """fp16-rounded pooled softmax probabilities: the dtype score_cache.save
    writes, and therefore the tie structure the real builders see."""
    torch.manual_seed(seed)
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    probs = torch.softmax(
        (torch.randn(n, n) * 6).masked_fill(upper, float("-inf")), dim=1)
    p16 = probs.half().float()
    return {i: p16.unsqueeze(0).repeat(n_kv, 1, 1) for i in range(n_layers)}


def _call(scores, n=48, sparsities=(0.5, 0.75, 0.9)):
    return R._assert_equivalent(
        seq_len=n * 64, block_size=64, finest_block_size=64,
        sparsities=list(sparsities), scores=scores, seed="t")


def test_tie_dense_fp16_scores_are_accepted():
    """The case that fired the original bitwise guard must now pass, and must
    report a NON-ZERO tie-break count -- a guard that passes because it found
    nothing to compare is the vacuous case."""
    diff, zero_frac, tie_frac = _call(_tie_dense_scores())
    assert diff > 0, ("expected some tie-break disagreement on fp16 scores; "
                      "zero would mean this fixture is not exercising ties")
    assert 0.0 <= zero_frac <= 1.0
    assert tie_frac > 0.0


def test_continuous_fp32_scores_show_no_disagreement():
    """The regime the original laptop check used. It should pass with zero
    differences -- which is exactly why it proved nothing about fp16."""
    torch.manual_seed(1)
    scores = {i: torch.rand(2, 48, 48) for i in range(3)}
    diff, _, tie_frac = _call(scores)
    assert diff == 0
    assert tie_frac == pytest.approx(0.0, abs=1e-6)


def test_rejects_a_builder_that_ranks_differently():
    """Negative control. Inverting the score inverts the ranking while keeping
    per-row budgets identical, so ONLY the kept-score-multiset check can catch
    it. If this ever passes, the relaxation has become a deletion."""
    orig = R._vectorised_builder

    def inverted(seq_len, block_size, sparsity, importance_scores, *,
                 causal, identity_seed):
        return orig(seq_len, block_size, sparsity, -importance_scores,
                    causal=causal, identity_seed=identity_seed)

    R._vectorised_builder = inverted
    try:
        with pytest.raises(SystemExit, match="RANKING"):
            _call(_tie_dense_scores())
    finally:
        R._vectorised_builder = orig


def test_rejects_a_builder_that_changes_how_much_work_the_kernel_does():
    """The other half. A builder keeping a different NUMBER of blocks would
    make the two timings incomparable, which is the whole point of item 4."""
    orig = R._vectorised_builder

    def greedier(seq_len, block_size, sparsity, importance_scores, *,
                 causal, identity_seed):
        # same ranking, larger budget
        return orig(seq_len, block_size, max(0.0, sparsity - 0.1),
                    importance_scores, causal=causal,
                    identity_seed=identity_seed)

    R._vectorised_builder = greedier
    try:
        with pytest.raises(SystemExit, match="active count differs"):
            _call(_tie_dense_scores())
    finally:
        R._vectorised_builder = orig


def test_reference_builder_is_restored_after_a_rejection():
    """A guard that leaves masks.importance_block_mask monkeypatched on its
    way out would corrupt every measurement after it."""
    from attnbench import masks
    before = masks.importance_block_mask
    orig = R._vectorised_builder

    def inverted(seq_len, block_size, sparsity, importance_scores, *,
                 causal, identity_seed):
        return orig(seq_len, block_size, sparsity, -importance_scores,
                    causal=causal, identity_seed=identity_seed)

    R._vectorised_builder = inverted
    try:
        with pytest.raises(SystemExit):
            _call(_tie_dense_scores())
    finally:
        R._vectorised_builder = orig
    assert masks.importance_block_mask is before


def test_jitter_tolerance_has_a_ceiling():
    """The 1e-9 tolerance is bounded by the reference's own jitter, so a
    disagreement larger than that must still be rejected. Without this,
    "differs only by a tolerance" could grow to cover anything.

    The perturbation swaps one genuinely-kept block for a genuinely-unkept
    one of clearly different score, holding the per-row count fixed -- so
    ONLY the multiset branch can catch it, which is the branch under test.
    """
    orig = R._vectorised_builder

    def swapped(seq_len, block_size, sparsity, importance_scores, *,
                causal, identity_seed):
        m = orig(seq_len, block_size, sparsity, importance_scores,
                 causal=causal, identity_seed=identity_seed)
        act = m.active.clone()
        n = act.shape[0]
        for qb in range(n - 1, 1, -1):
            cand = torch.zeros(n, dtype=torch.bool)
            cand[1:qb] = True
            kept = cand & act[qb]
            unkept = cand & ~act[qb]
            if not kept.any() or not unkept.any():
                continue
            ki = torch.nonzero(kept).flatten()
            ui = torch.nonzero(unkept).flatten()
            lo = ki[importance_scores[qb][ki].argmin()]
            hi = ui[importance_scores[qb][ui].argmax()]
            if importance_scores[qb, lo] == importance_scores[qb, hi]:
                continue          # a tie swap is exactly what IS allowed
            act[qb, lo] = False
            act[qb, hi] = True
            break
        return masks.BlockSparseMask(
            seq_len=seq_len, block_size=block_size, active=act, seed=m.seed,
            source=m.source, causal=m.causal)

    R._vectorised_builder = swapped
    try:
        with pytest.raises(SystemExit, match="RANKING"):
            _call(_tie_dense_scores())
    finally:
        R._vectorised_builder = orig


def test_jitter_reach_matches_the_reference_implementation():
    """If masks.py ever changes its jitter magnitude, this tolerance is no
    longer justified by anything and must be revisited rather than silently
    becoming either too tight or too permissive."""
    import inspect
    from attnbench import masks
    src = inspect.getsource(masks.importance_block_mask)
    assert "1e-9" in src, (
        "importance_block_mask no longer uses a 1e-9 jitter; "
        "run_vectorised_endtoend.JITTER_REACH was derived from it and must "
        "be re-derived.")
    assert R.JITTER_REACH == 1e-9
