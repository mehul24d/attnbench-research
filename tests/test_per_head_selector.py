"""The era-4 per-head selector (estimator-frontier pre-registration sec. 2.1),
on CPU. Gate G3: with one score matrix broadcast to every head it is era 3
bitwise."""
from __future__ import annotations

import pytest
import torch

from attnbench import masks
from attnbench.analysis import frontier_prereg as fp

SEED = "per-head-selector-test"


def _scores(h, n, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(h, n, n, generator=g).to(dtype)


@pytest.mark.parametrize("sparsity", [0.5, 0.75, 0.9, 0.95])
@pytest.mark.parametrize("n", [3, 17, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_g3_one_score_matrix_on_every_head_is_era_3_bitwise(sparsity, n, dtype):
    """fp16 scores tie densely, so the jitter decides and must decide alike."""
    one = _scores(1, n, seed=n, dtype=dtype)[0]
    era3 = masks.importance_block_mask_device(n * 128, 128, sparsity, one, causal=True,
                                              identity_seed=SEED)
    era4 = masks.importance_block_mask_per_head(n * 128, 128, sparsity,
                                                one.expand(5, n, n), causal=True,
                                                identity_seed=SEED)
    assert era4.active.shape == (5, n, n) and era4.to_bsa().shape == (1, 5, n, n)
    for h in range(5):
        assert torch.equal(era4.active[h], era3.active)
    assert era4.seed == era3.seed and era4.mask_selector == "per_head"


@pytest.mark.parametrize("sparsity", [0.5, 0.9])
def test_g3_holds_where_the_jitter_alone_decides(sparsity):
    """Four score levels, so nearly every budget boundary falls inside a tie.
    Without the era-3 jitter the stable sort would break ties by position.

    The levels are 1e-6 apart on purpose. The jitter is at most 1e-9, and
    fp32 absorbs it into any score above about 0.02, where ties then break
    by position in era 3 and era 4 alike."""
    n = 64
    g = torch.Generator().manual_seed(7)
    one = torch.randint(0, 4, (n, n), generator=g).float() * 1e-6
    era3 = masks.importance_block_mask_device(n * 128, 128, sparsity, one, causal=True,
                                              identity_seed=SEED)
    era4 = masks.importance_block_mask_per_head(n * 128, 128, sparsity, one.expand(3, n, n),
                                                causal=True, identity_seed=SEED)
    assert all(torch.equal(era4.active[h], era3.active) for h in range(3))
    by_position = masks.importance_block_mask_per_head(
        n * 128, 128, sparsity, one.expand(3, n, n), causal=True, identity_seed="another seed")
    assert not torch.equal(by_position.active, era4.active)   # the seed is what breaks the ties


def test_each_head_selects_on_its_own_scores():
    n, H = 64, 4
    scores = _scores(H, n, seed=3)
    m = masks.importance_block_mask_per_head(n * 128, 128, 0.75, scores, causal=True,
                                             identity_seed=SEED)
    for h in range(H):
        alone = masks.importance_block_mask_device(n * 128, 128, 0.75, scores[h], causal=True,
                                                   identity_seed=SEED)
        assert torch.equal(m.active[h], alone.active)
    assert not torch.equal(m.active[0], m.active[1])
    # Not the head mean: that is era 3.
    mean = masks.importance_block_mask_device(n * 128, 128, 0.75, scores.mean(0), causal=True,
                                              identity_seed=SEED)
    assert any(not torch.equal(m.active[h], mean.active) for h in range(H))


@pytest.mark.parametrize("d_nom", [0.50, 0.25, 0.10, 0.05])
def test_every_row_keeps_the_two_free_blocks_plus_its_budget(d_nom):
    n = 128
    m = masks.importance_block_mask_per_head(n * 128, 128, 1 - d_nom, _scores(3, n), causal=True,
                                             identity_seed=SEED)
    kept = m.active.sum(dim=2)
    for i in range(n):
        assert kept[:, i].tolist() == [fp.kept_128(i, d_nom)] * 3, i
        assert bool(m.active[:, i, 0].all()) and bool(m.active[:, i, i].all())
    assert not bool(m.active.triu(1).any())               # nothing above the diagonal
    want = sum(fp.kept_128(i, d_nom) for i in range(n)) / (n * (n + 1) / 2)
    assert m.realised_density() == pytest.approx(want)


def test_the_budget_is_the_era_3_expression_not_d_nom_times_candidates():
    """They differ by one block on a few rows at 0.10 and 0.05."""
    assert masks.era_budget(0.10, 15) == 1 and round(0.10 * 15) == 2
    assert masks.era_budget(0.05, 10) == 1 and round(0.05 * 10) == 0
    for d in (0.50, 0.25):
        assert all(masks.era_budget(d, c) == round(d * c) for c in range(300))
    differ = {d: sum(masks.era_budget(d, i - 1) != round(d * (i - 1)) for i in range(2, 128))
              for d in (0.10, 0.05)}
    assert differ == {0.10: 6, 0.05: 3}
    for s in (0.5, 0.75, 0.9, 0.95):
        for c in range(300):
            assert masks.era_budget(1 - s, c) == round((1.0 - s) * c)


@pytest.mark.parametrize("b", [16, 32, 64])
@pytest.mark.parametrize("d_nom", [0.25, 0.10])
def test_matched_budgets_below_128_keep_the_parent_rows_density(b, d_nom):
    tokens = 16384
    n = tokens // b
    budgets = torch.tensor(fp.matched_row_budgets(n, b, d_nom))
    m = masks.importance_block_mask_per_head(tokens, b, 1 - d_nom, _scores(2, n, seed=b),
                                             causal=True, identity_seed=SEED,
                                             row_budgets=budgets)
    kept = m.active.sum(dim=2)
    for i in fp.h7_rows(tokens, b):
        assert kept[:, i].tolist() == [2 + fp.matched_budget(i, b, d_nom)] * 2
        p = i * b // 128
        err = abs(int(kept[0, i]) / (i + 1) - fp.kept_128(p, d_nom) / (p + 1))
        # The nearest count a row of i + 1 blocks can hold. H7 is scored at
        # b = 16, where that is within the 0.01 tolerance on every scored row.
        # At b = 32 and 64 the early rows are too short for 0.01 (15 blocks at
        # b = 64 move the density in steps of 0.067); those sizes are
        # descriptive.
        assert err <= 0.5 / (i + 1) + 1e-12
        if b == 16:
            assert err <= fp.H7_DENSITY_TOL and fp.row_density_matches(i, b, d_nom)
    # Without them the plain rule keeps a different count, and a lower density.
    plain = masks.importance_block_mask_per_head(tokens, b, 1 - d_nom, _scores(2, n, seed=b),
                                                 causal=True, identity_seed=SEED)
    assert plain.realised_density() < m.realised_density()


def test_bad_inputs_are_refused():
    n = 8
    kw = dict(causal=True, identity_seed=SEED)
    with pytest.raises(ValueError):
        masks.importance_block_mask_per_head(n * 128, 128, 0.5, torch.rand(n, n), **kw)
    with pytest.raises(ValueError):
        masks.importance_block_mask_per_head(n * 128, 128, 0.5, torch.rand(2, n, n + 1), **kw)
    for bad in (torch.ones(n), torch.ones(n + 1, dtype=torch.long), -torch.ones(n, dtype=torch.long)):
        with pytest.raises(ValueError):
            masks.importance_block_mask_per_head(n * 128, 128, 0.5, torch.rand(2, n, n),
                                                 row_budgets=bad, **kw)
    # A budget above a row's candidates is clamped to them.
    full = masks.importance_block_mask_per_head(
        n * 128, 128, 0.5, torch.rand(2, n, n), row_budgets=torch.full((n,), 99), **kw)
    assert torch.equal(full.active[0], torch.ones(n, n, dtype=torch.bool).tril())


def test_the_selector_reports_per_head_for_the_row_column():
    assert masks.mask_selector_for("block_sparse", sparse=True, per_head=True) == "per_head"
    assert "per_head" in masks.MASK_SELECTORS
