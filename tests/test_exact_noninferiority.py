"""The exact paired non-inferiority bound (audit T4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attnbench.analysis import exact_noninferiority as ni


def test_clopper_pearson_matches_the_closed_forms():
    # k = 0 and k = n have closed forms: 1 - a**(1/n) and a**(1/n)
    for n in (5, 50, 100):
        assert ni.cp_upper(0, n, 0.0125) == pytest.approx(1 - 0.0125 ** (1 / n))
        assert ni.cp_lower(n, n, 0.0125) == pytest.approx(0.0125 ** (1 / n))
    assert ni.cp_lower(0, 50, 0.0125) == 0.0
    assert ni.cp_upper(50, 50, 0.0125) == 1.0


def test_the_dense_pilot_intervals_are_reproduced():
    """docs/t4_dense_pilot.md quotes 95% Clopper-Pearson intervals; the
    module's two-sided interval must give the same numbers from the counts."""
    quoted = {16: (19.5, 46.7), 17: (21.2, 48.8), 13: (14.6, 40.3),
              3: (1.3, 16.5), 30: (45.2, 73.6), 20: (26.4, 54.8)}
    for k, (lo, hi) in quoted.items():
        got = ni._two_sided_cp(k, 50, 0.05)
        assert (round(got[0], 1), round(got[1], 1)) == (lo, hi), k


def test_attainable_bounds_quoted_in_the_plan():
    assert round(100 * ni.attainable_lower_bound(50), 1) == -8.4
    assert round(100 * ni.attainable_lower_bound(100), 1) == -4.3
    assert round(100 * ni.paired_lower_bound(0, 1, 50), 1) == -12.1
    assert round(100 * ni.paired_lower_bound(3, 3, 100), 1) == -8.9


def test_coverage_is_at_least_nominal():
    """The claim the module rests on, checked by simulation: across true
    (p_b, p_c), including the floor and ceiling the bootstrap fails at, the
    bound lies above the true difference at most alpha of the time."""
    rng = np.random.default_rng(0)
    n, alpha, sims = 30, 0.025, 4000
    table = np.array([[ni.paired_lower_bound(b, c, n, alpha) if b + c <= n else np.nan
                       for c in range(n + 1)] for b in range(n + 1)])
    for p_b, p_c in [(0.0, 0.0), (0.0, 0.05), (0.02, 0.02), (0.1, 0.3),
                     (0.3, 0.1), (0.45, 0.45), (0.0, 0.9)]:
        draws = rng.multinomial(n, [p_b, p_c, 1 - p_b - p_c], size=sims)
        bounds = table[draws[:, 0], draws[:, 1]]
        miss = np.mean(bounds > p_b - p_c)
        assert miss <= alpha, (p_b, p_c, miss)


def test_counts_and_decision():
    dense = np.array([1, 1, 1, 1, 0, 0, 1, 0, 1, 1] * 10, dtype=bool)
    sparse = dense.copy()
    sparse[:2] = False            # two dense-only
    sparse[4] = True              # one sparse-only
    r = ni.paired_exact_ni(dense, sparse, margin_pts=10)
    assert (r.n, r.b_sparse_only, r.c_dense_only) == (100, 1, 2)
    assert r.diff_pts == pytest.approx(-1.0)
    assert r.lower_pts == pytest.approx(100 * ni.paired_lower_bound(1, 2, 100))
    assert r.non_inferior == (r.lower_pts > -10)
    assert ni.paired_exact_ni(dense, dense, margin_pts=1).non_inferior is False


def _rows(task, band, ids, correct, *, sparse=None):
    d = dict(task=task, example_id=ids, context_length=band - 3, correct=correct)
    if sparse is None:
        return pd.DataFrame({**d, "backend_role": "dense_reference",
                             "score_source": None, "sparsity": None})
    src, s = sparse
    return pd.DataFrame({**d, "backend_role": "block_sparse",
                         "score_source": src, "sparsity": s})


def test_run_pairs_on_example_and_separates_arms_by_scorer():
    ids = [f"qa_1_16384_{i}" for i in range(20)]
    dense = _rows("qa_1", 16384, ids, [True] * 20)
    oracle = _rows("qa_1", 16384, ids, [True] * 20, sparse=("dense_softmax_fp32", 0.5))
    inline = _rows("qa_1", 16384, ids[::-1], [False] + [True] * 19,
                   sparse=("minference_meanpool_inline", 0.5))
    out = ni.run_exact_ni(dense, pd.concat([oracle, inline]), bands=(16384, 32768),
                          margin_pts=10).set_index("arm")
    assert set(out.index) == {"dense_softmax_fp32@0.5", "minference_meanpool_inline@0.5"}
    assert out.loc["dense_softmax_fp32@0.5", "c_dense_only"] == 0
    assert out.loc["minference_meanpool_inline@0.5", "c_dense_only"] == 1


def test_unpaired_or_mixed_input_refuses():
    ids = [f"e{i}" for i in range(5)]
    dense = _rows("qa_1", 16384, ids, [True] * 5)
    short = _rows("qa_1", 16384, ids[:4], [True] * 4, sparse=("dense_softmax_fp32", 0.5))
    with pytest.raises(ValueError, match="unpaired"):
        ni.run_exact_ni(dense, short, bands=(16384,), margin_pts=10)
    with pytest.raises(ValueError, match="repeat"):
        ni.run_exact_ni(pd.concat([dense, dense]), short, bands=(16384,), margin_pts=10)
    unknown = short.assign(score_source=None)
    with pytest.raises(ValueError, match="arm is unknown"):
        ni.run_exact_ni(dense, unknown, bands=(16384,), margin_pts=10)
