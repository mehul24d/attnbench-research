"""The estimator-frontier pre-registration's plan test (lock gate L1,
2026-10-03).

`attnbench/analysis/frontier_prereg.py` is the pre-registration as code. This
file holds it to the document three ways:

1. Every pre-registered number in the module appears, worded as the module
   uses it, in `docs/estimator_frontier_preregistration.md`.
2. Every scorer (H1-H8, R1-R4, P-T4) has a pass case, a fail case and an
   indeterminate case (sec. 11.3).
3. The other sec. 11.3 break-tests: the split leak, the reservation STOP,
   rebracket's leak guard, the torch-fallback refusal and the era-4 stripped
   column (in `tests/test_eras.py`, checked here to exist). The bracket's
   numbers in sec. 8.2-8.3 equal what the code computes.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attnbench.analysis import frontier_prereg as fp

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "estimator_frontier_preregistration.md"
TEXT = " ".join(DOC.read_text().split())


def _has(phrase: str) -> bool:
    return " ".join(phrase.split()) in TEXT


# --- 1. the numbers ---------------------------------------------------------

@pytest.mark.parametrize("phrase,value", [
    ("τ_k = 0.05", fp.TAU_K), ("τ_e = 0.07", fp.TAU_E),
    ("f = max( 0.02,", fp.F_MIN),
    ("Margin 10 points, one-sided alpha 0.025", (fp.MARGIN_PTS, fp.ALPHA)),
    ("B = 10,000, seed 20261003", (fp.BOOT_B, fp.BOOT_SEED)),
    ("∈ [0.75, 1.33]", fp.H1_BAND),
    ("at least 5 of every 6 determinate cells (83%) in band", fp.H1_MIN_FRACTION),
    ("by a further factor of 1.10", fp.H1_GUARD_MARGIN),
    ("fewer than 6 determinate cells", fp.H1_MIN_DETERMINATE),
    ("σ_Q = 0.035", fp.H1_SIGMA_Q),
    ("at least 6 of the twelve 7B cells", fp.H3_MIN_7B_CELLS),
    ("parent row p ≥ 7", fp.H7_MIN_PARENT_ROW),
    ("evaluation indices start at 3000", fp.EVAL_INDEX_OFFSET),
    ("matches the sign of 1 − m in at least 90%", fp.H2A_MIN_MATCH),
    ("no wrong sign where |1 − m| > 0.15", fp.H2A_WRONG_SIGN_GUARD),
    ("max(0.20·|S_pred|, f·T_dense^e2e) in at least 80%", (fp.H2B_REL, fp.H2B_MIN_WITHIN)),
    ("point speedups in [1.25, 1.80]", fp.H2D_RANGE),
    ("R̃_XA8 − R̃_MP ≥ +0.05", fp.H3_MIN_DIFF),
    ("**Pass:** at least 16 of 24.", fp.H3_MIN_CELLS),
    ("1 ± 0.01, enforced by gate G7", fp.H3_DENSITY_TOL),
    ("is below 0.95 in at least 2/3", fp.H3B_LIMIT),
    ("is ≥ 2.0× that in the highest", fp.H4_RATIO),
    ("trend test, p < 0.025", fp.H4_P),
    ("within ±0.005 of a tercile boundary go to the lower", fp.H4_BOUNDARY),
    ("at least 30 dense-correct examples per tercile", fp.H4_MIN_PER_TERCILE),
    ("mean raw R ≥ 0.85", fp.H6A_PASS), ("STOP at R < 0.70", fp.H6A_STOP),
    ("≥ 0.80 over its 996 non-zero entries", (fp.H6B_MIN_RHO, fp.H6B_N_ENTRIES)),
    ("≤ 0.75 × (1 − R̃_MP) at b = 128", fp.H7A_FACTOR),
    ("over the scored rows, within 1 ± 0.01", fp.H7_DENSITY_TOL),
    ("within ±5% (relative) of the L4 run on at least 90% of the 400 ids",
     (fp.R1_REL, fp.R1_MIN_FRAC, fp.R_N_IDS)),
    ("T4's L4 dense predictions on at least 90%", fp.R2_MIN_FRAC),
    ("on at least 85% of ids", fp.R3_MIN_AGREE),
    ("|Δ accuracy| ≤ 5 points", fp.R3_MAX_DELTA_PTS),
    ("**₹12,000** for this study", fp.CAP_INR),
])
def test_the_module_states_the_documents_numbers(phrase, value):
    assert _has(phrase), f"the pre-registration no longer says {phrase!r}"
    for v in np.ravel([value]):
        forms = {re.sub(r"\.0$", "", f"{v:,}"), f"{v:g}", f"{v:.2f}", f"{int(v * 100)}%"}
        if float(v) == int(v):
            forms.add(str(int(v)))
        # As whole numbers: "0.7" must not match inside "0.75".
        assert any(re.search(rf"(?<![\d.,]){re.escape(f)}(?![\d])", phrase.replace("+", ""))
                   for f in forms), (phrase, v)


# --- 2. every scorer: pass, fail, indeterminate -------------------------------

def _h1(n_in, n_out, n_ind, pair=("L4", "A100")):
    good = [fp.H1Cell(c1=1.0, c2=0.5, D=3.3, W=6.6, pair=pair)] * n_in    # Q = 1.0
    bad = [fp.H1Cell(c1=1.0, c2=2.0, D=3.3, W=6.6, pair=pair)] * n_out    # Q = 4.0
    # D/W = 1.34: the null's Q is 0.75, inside the band, so no discrimination.
    ind = [fp.H1Cell(c1=1.0, c2=1.34, D=1.34, W=1.0, pair=("A100", "H100"))] * n_ind
    return good + bad + ind


def test_h1_band_edges_after_the_tolerance_widening():
    """[0.75, 1.33] widened x/÷ 1.05^2 is [0.680, 1.466]."""
    lo, hi = 0.75 / 1.05 ** 2, 1.33 * 1.05 ** 2
    assert fp.h1_band() == pytest.approx((lo, hi))

    def cell(q):  # D/W = 0.5, so Q = 2 * c2 / c1
        return fp.H1Cell(c1=1.0, c2=q / 2, D=3.3, W=6.6)
    assert fp.score_h1([cell(lo + 0.002)] * 6).verdict == fp.PASS
    assert fp.score_h1([cell(lo - 0.002)] * 6).verdict == fp.FAIL
    assert fp.score_h1([cell(hi - 0.002)] * 6).verdict == fp.PASS
    assert fp.score_h1([cell(hi + 0.002)] * 6).verdict == fp.FAIL


def test_h1():
    assert fp.score_h1(_h1(5, 1, 6)).verdict == fp.PASS            # 5 of 6
    assert fp.score_h1(_h1(4, 2, 6)).verdict == fp.FAIL
    assert fp.score_h1(_h1(10, 2, 0)).verdict == fp.PASS           # 10 of 12
    assert fp.score_h1(_h1(9, 3, 0)).verdict == fp.FAIL
    assert fp.score_h1(_h1(5, 0, 6)).verdict == fp.UNRESOLVED      # 5 determinate
    assert fp.score_h1(_h1(0, 0, 12)).verdict == fp.UNRESOLVED


def test_h1_a_pair_that_passes_under_the_null_is_not_counted():
    """The referee's case (2026-10-03): for A100-H100 the expected D/W is
    1.34, so c the same on both cards gives Q = 0.75, inside the widened
    band. Such cells used to count as passes."""
    null_cells = [fp.H1Cell(c1=1.0, c2=1.0, D=1.34, W=1.0, pair=("A100", "H100"))] * 6
    assert not fp.h1_determinate(1.34, 1.0)
    v = fp.score_h1(null_cells)
    assert v.verdict == fp.UNRESOLVED and v.counts["in_band"] == 0
    # A null cell on a discriminating pair is out of band.
    assert fp.h1_determinate(3.3, 6.8)
    assert fp.score_h1([fp.H1Cell(c1=1.0, c2=1.0, D=3.3, W=6.8)] * 6).verdict == fp.FAIL
    # The guard needs the margin too: the null just outside the band is not enough.
    lo, hi = fp.h1_band()
    assert not fp.h1_determinate(1.0, hi * 1.05)
    assert fp.h1_determinate(1.0, hi * 1.11)


def test_h1_l4_h100_is_descriptive():
    cells = _h1(6, 0, 0) + _h1(0, 6, 0, pair=("L4", "H100"))
    v = fp.score_h1(cells)
    assert v.verdict == fp.PASS and v.counts["descriptive"] == 6


def test_h1_false_pass_rate_as_the_document_states_it():
    p = fp.h1_null_cell_pass_probability()
    assert p == pytest.approx(0.0032, abs=5e-5)
    independent, correlated = fp.h1_false_pass(p)
    assert correlated == p and independent < 1e-11
    assert _has("0.32%") and _has("up to 50%")
    assert fp.h1_false_pass(0.5)[0] == pytest.approx(7 / 64)


def _h2(m, lo, hi, s_meas=10.0, s_pred=10.0):
    return fp.H2Cell(m=m, s_lower=lo, s_upper=hi, f=0.02, s_meas=s_meas,
                     s_pred=s_pred, t_dense_e2e=100.0)


def test_h2ab():
    good = [_h2(0.5, 1.5, 1.7)] * 9 + [_h2(1.5, 0.5, 0.7)]
    a, b = fp.score_h2ab(good)
    assert (a.verdict, b.verdict) == (fp.PASS, fp.PASS)
    wrong_far = good[:-1] + [_h2(1.5, 1.5, 1.7, s_meas=0.0)]       # profitable at m = 1.5
    a, b = fp.score_h2ab(wrong_far)
    assert a.verdict == fp.FAIL and a.counts["wrong_far"] == 1
    assert b.verdict == fp.PASS                                     # 9/10 within
    a, b = fp.score_h2ab([_h2(1.03, 0.9, 1.1)])                    # near parity only
    assert (a.verdict, b.verdict) == (fp.UNRESOLVED, fp.UNRESOLVED)


def test_h2c():
    p = dict(band=8192, card="A100", deployable=True, f=0.02)
    assert fp.score_h2c([dict(p, s_lower=0.9, s_upper=0.97)]).verdict == fp.PASS
    # A noisy session must not pass it: mostly unresolved points are unresolved.
    assert fp.score_h2c([dict(p, s_lower=0.9, s_upper=1.01)]).verdict == fp.UNRESOLVED
    assert fp.score_h2c([dict(p, s_lower=0.9, s_upper=0.97)] * 2
                        + [dict(p, s_lower=0.9, s_upper=1.01)] * 2).verdict == fp.PASS
    assert fp.score_h2c([dict(p, s_lower=1.1, s_upper=1.2)]).verdict == fp.FAIL
    assert fp.score_h2c([dict(p, band=16384, s_lower=1.1, s_upper=1.2)]).verdict == fp.UNSCORABLE


def test_h2d():
    p = dict(card="A100", band=32768, arm="MP", f=0.02)
    good = [dict(p, d_nom=0.25, s=1.4, s_lower=1.3, s_upper=1.5),
            dict(p, d_nom=0.10, s=1.6, s_lower=1.5, s_upper=1.7)]
    assert fp.score_h2d(good).verdict == fp.PASS
    assert fp.score_h2d([good[0], dict(good[1], s=1.9)]).verdict == fp.FAIL
    assert fp.score_h2d([good[0], dict(good[1], s_lower=0.99)]).verdict == fp.UNRESOLVED
    assert fp.score_h2d(good[:1]).verdict == fp.UNSCORABLE


def test_h3_and_h3b():
    def cells(kind, n15, n7):
        return ([fp.H3Cell(model="1.5B", **kind)] * n15 + [fp.H3Cell(model="7B", **kind)] * n7)
    win = dict(diffs=[0.08, 0.1, 0.06, 0.07] * 5, density_ratio=1.0)
    lose = dict(diffs=[0.0, 0.01, -0.01, 0.02] * 5, density_ratio=1.0)
    off = dict(diffs=[0.08] * 20, density_ratio=1.02)
    assert fp.score_h3(cells(win, 10, 6) + cells(lose, 2, 6)).verdict == fp.PASS
    assert fp.score_h3(cells(win, 5, 5) + cells(lose, 7, 7)).verdict == fp.FAIL
    # 16 of 24, but only 4 of the 7B cells: the T4-informed half cannot carry it.
    assert fp.score_h3(cells(win, 12, 4) + cells(lose, 0, 8)).verdict == fp.FAIL
    assert fp.score_h3(cells(win, 12, 4) + cells(off, 0, 4) + cells(lose, 0, 4)).verdict == fp.UNRESOLVED
    assert fp.score_h3b([0.9, 0.9, 0.99]).verdict == fp.PASS
    assert fp.score_h3b([0.9, 0.99, 0.99]).verdict == fp.FAIL
    assert fp.score_h3b([]).verdict == fp.UNSCORABLE


def _h4_stratum(rng, loss_by_tercile, n=150):
    r = rng.uniform(0, 1, n)
    t = fp.terciles(r)
    dense = np.ones(n, bool)
    sparse = rng.uniform(0, 1, n) >= np.array([loss_by_tercile[k - 1] for k in t])
    return {"r_tilde": r, "dense_correct": dense, "sparse_correct": sparse}


def test_h4():
    rng = np.random.default_rng(0)
    strong = {f"s{i}": _h4_stratum(rng, (0.5, 0.25, 0.1)) for i in range(3)}
    assert fp.score_h4(strong).verdict == fp.PASS
    flat = {f"s{i}": _h4_stratum(rng, (0.2, 0.2, 0.2)) for i in range(3)}
    assert fp.score_h4(flat).verdict == fp.FAIL
    small = {"s": _h4_stratum(rng, (0.5, 0.25, 0.1), n=40)}
    assert fp.score_h4(small).verdict == fp.UNDERPOWERED


def test_h4_boundary_rule():
    """Within +/-0.005 of a boundary goes to the lower tercile."""
    r = np.linspace(0, 1, 301)                  # step 1/300; q1 = 1/3
    t = fp.terciles(r)
    assert r[101] - 1 / 3 < 0.005 and t[101] == 1
    assert r[102] - 1 / 3 > 0.005 and t[102] == 2
    assert t.min() == 1 and t.max() == 3


def test_h5():
    pt = dict(arm="MP", is_xa=False, profit="profitable", certified=True)
    a, b = fp.score_h5([dict(pt, band=16384, profit="unresolved"),
                        dict(pt, band=32768, is_xa=True)])
    assert (a.verdict, b.verdict) == (fp.PASS, fp.PASS)
    a, b = fp.score_h5([dict(pt, band=16384), dict(pt, band=32768)])
    assert (a.verdict, b.verdict) == (fp.FAIL, fp.FAIL)
    a, b = fp.score_h5([dict(pt, band=32768, is_xa=True)])
    assert a.verdict == fp.UNSCORABLE
    # Both were written with T4's cells in view (sec. 3), and say so.
    assert b.labels == ("prior anchored on visible data",)


def test_h6():
    assert fp.score_h6a(0.9).verdict == fp.PASS
    v = fp.score_h6a(0.8)
    assert v.verdict == fp.FAIL and "triggers I2c-B (double-BOS arm)" in v.labels
    assert "G6 STOP" in fp.score_h6a(0.6).labels
    assert fp.score_h6a(None).verdict == fp.UNSCORABLE
    assert fp.i2cb_readout(0.9).startswith("passes on the authors'")
    assert fp.score_h6b(0.85, 996).verdict == fp.PASS
    assert fp.score_h6b(0.5, 996).verdict == fp.FAIL
    assert fp.score_h6b(0.85, 1024).verdict == fp.UNSCORABLE


def test_h7():
    good = fp.H7Cell(r_mp_16=0.95, r_mp_128=0.9, gap_16=0.01, gap_128=0.05, density_ratio=1.0)
    bad = fp.H7Cell(r_mp_16=0.9, r_mp_128=0.9, gap_16=0.05, gap_128=0.01, density_ratio=1.0)
    off = fp.H7Cell(r_mp_16=0.95, r_mp_128=0.9, gap_16=0.01, gap_128=0.05, density_ratio=1.02)
    rows = fp.H7Cell(r_mp_16=0.95, r_mp_128=0.9, gap_16=0.01, gap_128=0.05,
                     density_ratio=1.0, rows_match=False)
    a, b = fp.score_h7([good] * 16 + [bad] * 8)
    assert (a.verdict, b.verdict) == (fp.PASS, fp.PASS)
    a, b = fp.score_h7([good] * 10 + [bad] * 14)
    assert (a.verdict, b.verdict) == (fp.FAIL, fp.FAIL)
    a, b = fp.score_h7([good] * 19 + [off] * 5)
    assert (a.verdict, b.verdict) == (fp.UNRESOLVED, fp.UNRESOLVED)
    a, b = fp.score_h7([good] * 19 + [rows] * 5)                 # the per-row check
    assert (a.verdict, b.verdict) == (fp.UNRESOLVED, fp.UNRESOLVED)
    # Descriptive cells are not counted: 15 good + 4 descriptive is short of 16.
    a, b = fp.score_h7([good] * 15 + [off] * 4 + [bad] * 5)
    assert a.verdict == fp.FAIL and a.counts["descriptive"] == 4


def test_h7_under_cut_4_can_still_pass_and_fail():
    """Cut 4 leaves the 12 1.5B cells. Against the old fixed 16 it could not
    pass at all (the referee's finding)."""
    good = fp.H7Cell(r_mp_16=0.95, r_mp_128=0.9, gap_16=0.01, gap_128=0.05, density_ratio=1.0)
    bad = fp.H7Cell(r_mp_16=0.9, r_mp_128=0.9, gap_16=0.05, gap_128=0.01, density_ratio=1.0)
    a, _ = fp.score_h7([good] * 8 + [bad] * 4)
    assert a.verdict == fp.PASS and a.labels == ("1.5B only (cut 4)",)
    assert fp.score_h7([good] * 7 + [bad] * 5)[0].verdict == fp.FAIL
    assert fp.score_h7([good] * 20)[0].verdict == fp.UNSCORABLE


def test_h7_the_matched_budget_equalises_realised_density():
    """Under the plain rule the b=16 / b=128 density ratio is fixed by
    arithmetic at about 0.69-0.96 (the referee's table), so the old 1 +/- 0.05
    check could not be met. The matched budget puts every scored row within
    0.01 of its parent row, and every cell's ratio within 1 +/- 0.01."""
    for n in (8192, 16384, 32768):
        for d in (0.25, 0.10):
            r128 = fp.h7_rows(n, 128)
            d128 = sum(fp.kept_128(p, d) for p in r128) / sum(p + 1 for p in r128)
            rows = fp.h7_rows(n, 16)
            assert all(fp.row_density_matches(i, 16, d) for i in rows)
            d16 = sum(2 + fp.matched_budget(i, 16, d) for i in rows) / sum(i + 1 for i in rows)
            assert abs(d16 / d128 - 1) <= fp.H7_DENSITY_TOL
            # The plain rule, for the record:
            plain = sum(2 + round(d * (i - 1)) for i in rows) / sum(i + 1 for i in rows)
            assert abs(plain / d128 - 1) > 0.02
    # Below the scored rows a b=16 row cannot always match: that is why p >= 7.
    assert not all(fp.row_density_matches(i, 16, 0.25) for i in range(2, 56))
    assert fp.matched_budget(0, 16, 0.25) == 0 and fp.kept_128(1, 0.25) == 2


def _pair(n, b, c):
    dense = np.array([True] * n)
    sparse = dense.copy()
    sparse[:c] = False
    dense2 = dense.copy()
    dense2[c:c + b] = False
    return dense2, sparse


def test_h8_and_p_t4():
    ok = dict(zip(("dense", "sparse"), _pair(300, 0, 0)))
    bad = dict(zip(("dense", "sparse"), _pair(300, 0, 60)))
    assert fp.score_h8([ok], gates_pass=True).verdict == fp.PASS
    assert fp.score_h8([ok, bad], gates_pass=True).verdict == fp.FAIL
    assert fp.score_h8([ok], gates_pass=False).verdict == fp.UNSCORABLE
    assert fp.score_h8([], gates_pass=True).verdict == fp.UNSCORABLE
    n100bad = dict(zip(("dense", "sparse"), _pair(100, 0, 20)))
    n100ok = dict(zip(("dense", "sparse"), _pair(100, 0, 0)))
    assert fp.score_p_t4({16384: n100bad, 32768: n100bad}).verdict == fp.PASS
    assert fp.score_p_t4({16384: n100ok, 32768: n100bad}).verdict == fp.FAIL
    assert fp.score_p_t4({16384: n100bad}).verdict == fp.UNSCORABLE


def _ids(v):
    return {f"e{i}": v(i) for i in range(400)}


def test_r1_r2():
    assert fp.score_r1(_ids(lambda i: 0.3), _ids(lambda i: 0.31)).verdict == fp.PASS
    assert fp.score_r1(_ids(lambda i: 0.3), _ids(lambda i: 0.4)).verdict == fp.FAIL
    assert fp.score_r1(_ids(lambda i: 0.3), {"x": 0.3}).verdict == fp.UNSCORABLE
    assert fp.score_r2(_ids(str), _ids(str)).verdict == fp.PASS
    assert fp.score_r2(_ids(str), _ids(lambda i: str(i) if i % 5 else "no")).verdict == fp.FAIL
    assert fp.score_r2(_ids(str), {}).verdict == fp.UNSCORABLE


def test_r3_r4_and_their_labels():
    ok1, ok2 = fp.Verdict("R1", fp.PASS), fp.Verdict("R2", fp.PASS)
    same = {"a100": [True] * 90 + [False] * 10, "l4": [True] * 90 + [False] * 10}
    diff = {"a100": [True] * 100, "l4": [True] * 80 + [False] * 20}
    assert fp.score_r3({"qa_1/16384": same}, primary=["qa_1/16384"], r1=ok1, r2=ok2).verdict == fp.PASS
    v = fp.score_r3({"qa_1/16384": diff}, primary=["qa_1/16384"],
                    r1=fp.Verdict("R1", fp.FAIL), r2=ok2)
    assert v.verdict == fp.FAIL and v.labels == ("path-divergent",)
    assert fp.score_r3({}, primary=[], r1=ok1, r2=ok2).verdict == fp.UNSCORABLE
    assert fp.score_r4({16384: (False, False), 32768: (True, True)}, r1=ok1, r2=ok2).verdict == fp.PASS
    v = fp.score_r4({16384: (True, False), 32768: (False, False)}, r1=ok1,
                    r2=fp.Verdict("R2", fp.FAIL))
    assert v.verdict == fp.FAIL and v.labels == ("card-confounded",)
    assert fp.score_r4({16384: (False, False)}, r1=ok1, r2=ok2).verdict == fp.UNSCORABLE


def test_profit_classes_and_floor():
    assert fp.classify_profit(1.05, 1.1, 0.02) == "profitable"
    assert fp.classify_profit(0.9, 0.97, 0.02) == "unprofitable"
    assert fp.classify_profit(1.01, 1.05, 0.02) == "unresolved"
    assert fp.resolution_floor(0.01) == 0.02
    assert fp.resolution_floor(0.03) == 0.03
    assert fp.resolution_floor(0.0, sigma_s=0.02, k=4) == pytest.approx(3.182 * 0.01, rel=1e-3)
    assert fp.near_parity(1.06) and not fp.near_parity(1.08)
    with pytest.raises(TypeError):
        fp.near_parity(1.5, s=1.01)      # the measured speedup never triggers a replicate
    assert fp.marginal(0.83, 0.88, 0.85) and not fp.marginal(0.86, 0.9, 0.85)


# --- 3. the other sec. 11.3 break-tests --------------------------------------

def test_the_split_leak_is_refused():
    """sec. 5: a split sharing one qa_1 question index with evaluation, under
    another seed and context, is refused."""
    evaluation = [{"example_id": "qa_1_32768_5", "context_sha256": "aaa", "qa_question": (5, ("d1",))}]
    calibration = [{"example_id": "qa_1_32768_2000", "context_sha256": "bbb",
                    "qa_question": (5, ("d9",))}]
    with pytest.raises(fp.SplitLeak, match="qa_question"):
        fp.check_splits_disjoint({"evaluation": evaluation, "calibration": calibration})
    calibration[0]["qa_question"] = (2000, ("d9",))
    fp.check_splits_disjoint({"evaluation": evaluation, "calibration": calibration})
    with pytest.raises(fp.SplitLeak, match="niah_needles"):
        fp.check_splits_disjoint({"a": [{"niah_needles": [("k", "1")]}],
                                  "b": [{"niah_needles": [("k", "1")]}]})


def test_evaluation_indices_are_fresh():
    """sec. 5: disjoint from every banked index of the task, at any band, and
    from the other splits."""
    idx = list(fp.evaluation_indices(300))
    assert idx[0] == 3000 and idx[-1] == 3299
    tasks = {t for t, _ in fp.BANKED_INDEX_RANGES}
    fp.check_evaluation_unbanked({t: idx for t in tasks})
    with pytest.raises(fp.SplitLeak, match="banked"):
        fp.check_evaluation_unbanked({"qa_1": range(0, 300)})      # the old split
    with pytest.raises(fp.SplitLeak, match="banked"):
        fp.check_evaluation_unbanked({"vt": [299]})                 # banked at 2048 only
    spans = {k: set(range(a, a + n)) for k, (a, n) in fp.SPLIT_INDEX.items()}
    names = sorted(spans)
    assert all(spans[a].isdisjoint(spans[b]) for i, a in enumerate(names) for b in names[i + 1:])
    with pytest.raises(ValueError):
        fp.evaluation_indices(301)


def test_calibration_texts_are_held_to_the_firewall():
    q = "Which document states the year the bridge across the northern river opened?"
    fp.check_calibration_texts_disjoint(["an unrelated calibration text " * 5], [q])
    with pytest.raises(fp.SplitLeak, match="calibration text 1"):
        fp.check_calibration_texts_disjoint(["x", f"Docs ... {q}\n Answer:"], [q])
    fp.check_calibration_texts_disjoint(["short probe here"], ["short"])   # under min_len


def test_a_cut_hypothesis_is_not_run_whatever_the_scorer_says():
    v = {"H8": fp.Verdict("H8", fp.FAIL), "H6b": fp.Verdict("H6b", fp.PASS),
         "H1": fp.Verdict("H1", fp.PASS)}
    out = fp.apply_cuts(v, ["9", "11"])
    assert out["H8"].verdict == out["H6b"].verdict == fp.NOT_RUN
    assert out["H1"].verdict == fp.PASS
    # I4 is never cut, so no cut can leave H1 short of cells.
    assert next(l for l in fp.bracket() if l.name == "I4").star


def test_the_reservation_check_stops_over_the_cap():
    assert fp.reservation_check(spent=9000, next_u=1000, remaining_star_u=[2000],
                                s_res_inr=600, r_b=262) == "STOP"
    assert fp.reservation_check(spent=9000, next_u=1000, remaining_star_u=[1138],
                                s_res_inr=600, r_b=262) == "PROCEED"   # exactly 12,000
    assert fp.reservation_check(spent=1000, next_u=1000, remaining_star_u=[2000],
                                s_res_inr=600, r_b=262) == "PROCEED"
    assert fp.s_res(2, 33.3) == pytest.approx(424 + 2.4 * 33.3)


@pytest.mark.parametrize("col", ["correct", "predicted", "R", "R_tilde",
                                 "realised_density", "recall_norm"])
def test_rebracket_refuses_any_result_column(col):
    frame = pd.DataFrame({"latency_ms": [1.0], col: [1]})
    with pytest.raises(fp.LeakRefusal):
        fp.rebracket([frame], measured={})


def test_rebracket_reads_timing_only():
    frame = pd.DataFrame({"latency_ms": [1.0], "phase": ["prefill"]})
    out = fp.rebracket([frame], measured={"mu": (7.0, 8.0)})
    assert out["worst_case"]["total"] < fp.worst_case(fp.bracket())["total"]


def test_torch_fallback_rows_are_refused_cross_card():
    a = pd.DataFrame({"t": [1.0], "xattn_path": ["triton"]})
    b = pd.DataFrame({"t": [2.0], "xattn_path": ["torch_fallback"]})
    with pytest.raises(fp.TorchFallbackRefusal):
        fp.cross_card_ratio(a, b, "t")
    assert fp.cross_card_ratio(a, b, "t", allow_fallback=True) == 2.0
    with pytest.raises(fp.TorchFallbackRefusal):
        fp.score_h1([fp.H1Cell(1, 1, 3.3, 6.6, xattn_path="torch_fallback")])


def test_the_era_4_stripped_column_break_test_exists():
    src = (REPO / "tests" / "test_eras.py").read_text()
    assert "def test_a_stripped_per_head_row_is_refused_not_read_as_era_3" in src


def test_every_row_writer_carries_mask_selector():
    """sec. 4.5: accuracy, component, end-to-end, phase and recall rows."""
    for f in ("attnbench/accuracy/runner.py", "scripts/measure_estimator_cost.py",
              "scripts/run_vectorised_endtoend.py", "attnbench/accuracy/phase_timing.py",
              "attnbench/analysis/frontier_recall.py"):
        assert "mask_selector" in (REPO / f).read_text(), f


# --- the bracket against the document --------------------------------------

def _inr(x):
    return f"{round(x):,}"


def test_the_bracket_in_the_document_is_what_the_code_computes():
    lines = fp.bracket()
    t = fp.totals(lines)
    core, full = t["core"], t["all"]
    assert _has(f"**₹{_inr(core[0])}–{_inr(core[1])}**")
    assert _has(f"**₹{_inr(full[0])}–{_inr(full[1])}**")
    assert _has(f"**{core[2]:.1f}–{core[3]:.1f} h**")
    assert _has(f"**{full[2]:.1f}–{full[3]:.1f} h**")
    w = fp.worst_case(lines)
    assert _has(f"which is ₹{_inr(w['total'])}, leaving ₹{_inr(w['spare'])}")
    assert w["total"] <= fp.CAP_INR
    for name, row_label in (("C", "★ C: XA calibration"), ("I1", "★ I1:"),
                            ("I2a", "★ I2a:"), ("I2c", "★ I2c:"), ("I2d", "I2d:"),
                            ("I2e", "I2e:"), ("I4", "★ I4:"), ("X primary", "★ X: 1.5B primary"),
                            ("X secondary", "★ X: 1.5B secondary"), ("X-rep", "★ X-rep:"),
                            ("X0", "★ X0:")):
        line = next(l for l in lines if l.name == name)
        cell = f"| {_inr(line.inr[0])}–{_inr(line.inr[1])} |"
        row = next(r for r in DOC.read_text().splitlines() if r.startswith(f"| {row_label}"))
        assert cell in row, (name, cell, row)


# --- the 2026-10-04 re-bracket ---------------------------------------------

def test_the_rerun_term_is_the_larger_band_of_x_primary_not_half_the_line():
    lines = fp.bracket()
    xp = next(l for l in lines if l.name == "X primary")
    unit = xp.rerun_unit_min / 60 * fp.RATE["A100"]
    assert unit > xp.inr[1] / 2                       # the 32768 band is more than half
    w = fp.worst_case(lines)
    assert w["rerun"] == pytest.approx(unit)
    assert w["rerun"] >= max(l.inr[1] for l in lines if l.star and l.name != "X primary")
    assert _has(f"| {_inr(w['rerun'])} (eighth draft 1,698, which should have been 2,251) |")
    assert _has(f"| {_inr(w['core'])} (7,975 before the second calibration table; eighth draft 9,231")
    assert _has(f"| {_inr(w['s_res'])} (₹424 storage")
    assert _has(f"with ₹{_inr(w['spare'])} to spare")


def test_a_costlier_single_phase_line_becomes_the_rerun_term():
    lines = [l if l.name != "I1" else fp.BracketLine("I1", "A100", (0, 600), True, "intrinsic")
             for l in fp.bracket()]
    assert fp.worst_case(lines)["rerun"] == pytest.approx(600 / 60 * 284)


def test_the_row_upper_bound_is_prefill_plus_capped_tokens():
    from attnbench.accuracy import stopping
    inp = fp.BracketInputs()
    caps = stopping.TASK_TOKEN_CAPS
    assert inp.cap_tokens["primary"] == caps["qa_1"]
    assert inp.cap_tokens["secondary"] == max(caps["niah_multivalue"], caps["niah_multiquery"])
    assert inp.cap_tokens["rep"] == (caps["qa_1"] + inp.cap_tokens["secondary"]) / 2
    rows = [inp.pre_a100[b] * inp.ex[1] + 42 * inp.decode_s[1] for b in (16384, 32768)]
    assert _has(f"{rows[0]:.2f} s and {rows[1]:.2f} s for `qa_1`")
    xp = next(l for l in fp.bracket(inp) if l.name == "X primary")
    want = (300 * 9 * sum(rows) * inp.wall[1] / 60
            + 300 * sum(inp.pre_a100[b] for b in (16384, 32768)) * inp.mu[1] / 60
            + 2 * inp.overhead_min[1])
    assert xp.minutes[1] == pytest.approx(want)
    # A longer cap or a slower decode must raise the worst case.
    base = fp.worst_case(fp.bracket(inp))["total"]
    import dataclasses
    for change in ({"cap_tokens": {"primary": 62, "secondary": 62, "rep": 62}},
                   {"decode_s": (0.020, 0.050)}, {"ex": (1.0, 2.0)}):
        assert fp.worst_case(fp.bracket(dataclasses.replace(inp, **change)))["total"] > base


def test_the_eighth_drafts_worst_case_was_over_the_cap_once_the_rerun_is_priced():
    """Sec. 8.2 (d). The old row proxies (2.726 s and 5.398 s), the old core
    and the larger band give 12,248."""
    band = (300 * 9 * 5.398 * 1.69 / 60 + 300 * 1.10181 * 9.08 / 60 + 15) / 60 * 284
    assert round(band) == 2251
    assert round(9231.26 + 503.94 + band + 262.14) == 12248
    assert _has("₹12,248, which is ₹248 over the cap")


# --- the reservation rule, the cut order and DP1's trigger (2026-10-04) -----

def test_h8s_funding_cannot_be_consumed_by_a_lower_priority_item_running_first():
    """At worst-case inputs I2b, I2d and I2e (cut items 7, 3 and 9) come
    before the Llama runs (item 11) in sec. 8.5's order. They must not
    launch, and H8's runs must. The case is the one where H8 can be funded
    at all: H6a passes, so I2c-B does not run."""
    lines = fp.bracket()
    new = fp.fund(lines, fp.RUN_ORDER, skip=("I2c-B",))
    assert {"XL0", "XL primary"} <= set(new["ran"])
    assert not {"I2b", "I2d", "I2e"} & set(new["ran"])
    assert new["stopped_at"] is None and new["spent"] <= fp.CAP_INR
    # The rule before 2026-10-04 let them run, and H8 then went unfunded.
    old = fp.fund(lines, fp.RUN_ORDER, skip=("I2c-B",), reserve_higher=False)
    assert {"I2b", "I2d", "I2e"} <= set(old["ran"]) and "XL primary" in old["cut"]
    # If H6a misses and I2c-B runs, every input at its upper end leaves H8
    # unfunded under either rule; the lower items still do not run.
    worst = fp.fund(lines, fp.RUN_ORDER)
    assert {"XL0", "XL primary"} <= set(worst["cut"]) and "I2c-B" in worst["ran"]
    assert not {"I2b", "I2d", "I2e"} & set(worst["ran"])
    assert _has("If H6a misses and I2c-B runs, H8 is unfunded at worst-case inputs")


def test_no_walk_leaves_a_never_cut_session_unrun_or_goes_over_the_cap():
    for kw in ({}, {"xa_withheld": True}, {"reserve_higher": False}):
        out = fp.fund(fp.bracket(), fp.RUN_ORDER, **kw)
        assert out["stopped_at"] is None
        assert out["spent"] + fp.s_res(2, 0) <= fp.CAP_INR      # with two months of storage
        assert {l.name for l in fp.bracket() if l.star} <= set(out["ran"])


def test_launch_check_reserves_higher_priority_items_only():
    kw = dict(spent=0, next_u=500, remaining_star_u=[8000], s_res_inr=500, r_b=0)
    # Item 3 must leave room for item 11; item 11 need not leave room for item 3.
    assert fp.launch_check(next_item="3", remaining_cuttable_u={"11": 3081}, **kw) == "STOP"
    assert fp.launch_check(next_item="11", remaining_cuttable_u={"3": 3081}, **kw) == "PROCEED"
    # A never-cut session reserves no cuttable item.
    assert fp.launch_check(next_item=None, remaining_cuttable_u={"11": 3081}, **kw) == "PROCEED"
    # The rest of a session's own item is reserved: XL0 needs XL primary covered.
    assert fp.launch_check(next_item="11", remaining_cuttable_u={"11": 3081}, **kw) == "STOP"
    with pytest.raises(ValueError):
        fp.launch_check(next_item="8", remaining_cuttable_u={}, **kw)
    with pytest.raises(ValueError):
        fp.launch_check(next_item="3", remaining_cuttable_u={"8": 1}, **kw)


def test_a_cut_takes_the_whole_item_and_the_llama_runs_that_need_the_probe():
    out = fp.fund(fp.bracket(), fp.RUN_ORDER, xa_withheld=True)
    assert {"XL0", "XL primary", "XL secondary", "XL 65536"} <= set(out["cut"])
    assert "SL" in out["ran"]


def test_withheld_publication_moves_sl_and_vs_to_the_end_of_the_cut_order():
    assert fp.cut_order() == fp.CUT_ORDER and fp.CUT_ORDER.index("0c") == 2
    w = fp.cut_order(xa_withheld=True)
    assert w[-2:] == ("0c", "10") and sorted(w) == sorted(fp.CUT_ORDER)
    assert "8" not in fp.CUT_ORDER
    assert _has("if publication of XAttention results is withheld at any point, cuts 0c and 10 move to the end")
    kw = dict(spent=0, next_u=500, remaining_star_u=[8000], s_res_inr=500,
              next_item="11", remaining_cuttable_u={"0c": 3081})
    assert fp.launch_check(**kw) == "PROCEED"
    assert fp.launch_check(xa_withheld=True, **kw) == "STOP"


def test_every_cut_item_has_a_row_in_the_cut_effects_table():
    rows = [r for r in DOC.read_text().splitlines() if r.startswith("| ")]
    for c in fp.CUT_ORDER:
        assert any(r.startswith(f"| {c} |") for r in rows), c
    for c, pairs in fp.CUT_LABEL.items():
        for _, label in pairs:
            assert any(r.startswith(f"| {c} |") and label in r for r in rows), (c, label)
    # And the other way: every label the table prints is in the code.
    import re
    for c in fp.CUT_ORDER:
        row = next(r for r in rows if r.startswith(f"| {c} |"))
        for label in re.findall(r'abelled "([^"]+)"', row):
            assert label in [l for _, l in fp.CUT_LABEL.get(c, ())], (c, label)
    for names in fp.CUT_LINES.values():
        assert all(any(l.name == n for l in fp.bracket()) for n in names)


def test_sl_is_priced_as_cuttable_and_as_never_cut():
    lines = fp.bracket()
    sl = next(l for l in lines if l.name == "SL")
    assert not sl.star and fp.cut_item_of("SL") == "0c"
    assert _has(f"₹{_inr(sl.inr[0])}–{_inr(sl.inr[1])}, one more arm")
    star = fp.bracket(fp.BracketInputs(sl_star=True))
    w0, w1 = fp.worst_case(lines), fp.worst_case(star)
    assert w1["core"] - w0["core"] == pytest.approx(sl.inr[1])
    assert w1["rerun"] > w0["rerun"]
    assert _has(f"₹{_inr(w1['total'])}, leaving ₹{_inr(w1['spare'])}")
    # As never-cut it takes H8's funding at worst-case inputs, even when
    # I2c-B does not run.
    assert "XL primary" in fp.fund(star, fp.RUN_ORDER, skip=("I2c-B",))["cut"]
    assert _has("SL as never-cut leaves H8 unfunded at worst-case inputs")


def test_dp1s_trigger_on_the_arm_factor_is_a_number():
    assert fp.ARM_FACTOR_UPPER == fp.BracketInputs().ex[1] == 1.6
    be = fp.arm_factor_break_even()
    assert 2.4 < be < 2.6 and _has(f"at or above {be:.2f}")
    assert fp.dp1_arm_factor(1.6) == "within"
    assert fp.dp1_arm_factor(1.61) == "rebracket"
    assert fp.dp1_arm_factor(be + 0.01) == "cut_or_stop"
    assert _has("measured arm factor above 1.6")


def test_c_makes_two_tables_and_the_claim_table_is_the_smaller():
    inp = fp.BracketInputs()
    assert inp.c_text_factor < inp.c_text_factor_full
    c = next(l for l in fp.bracket(inp) if l.name == "C")
    want = (inp.a7_row_16384 * (inp.c_text_factor + inp.c_text_factor_full) * inp.mu[1] / 60
            + inp.overhead_min[1] + 10)
    assert c.minutes[1] == pytest.approx(want) and c.star
    assert _has("the claim table on 104 texts and the descriptive one on 121")
