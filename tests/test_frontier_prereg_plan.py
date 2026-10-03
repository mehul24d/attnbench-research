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
    ("at least 15 of the determinate cells in band, and no more than 3 indeterminate",
     (fp.H1_MIN_IN_BAND, fp.H1_MAX_INDETERMINATE)),
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
    ("within 1 ± 0.05", fp.H7_DENSITY_TOL),
    ("more than 4 indeterminate makes H7 unresolved", fp.H7_MAX_INDETERMINATE),
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

def _h1(n_in, n_out, n_ind):
    good = [fp.H1Cell(c1=1.0, c2=0.5, D=3.3, W=6.6)] * n_in       # Q = 1.0
    bad = [fp.H1Cell(c1=1.0, c2=2.0, D=3.3, W=6.6)] * n_out       # Q = 4.0
    ind = [fp.H1Cell(c1=1.0, c2=1.0, D=1.0, W=1.0)] * n_ind       # ln(D/W) = 0
    return good + bad + ind


def test_h1_band_edges_after_the_tolerance_widening():
    """[0.75, 1.33] widened x/÷ 1.05^2 is [0.680, 1.466]."""
    lo, hi = 0.75 / 1.05 ** 2, 1.33 * 1.05 ** 2

    def cell(q):  # D/W = 0.5, so Q = 2 * c2 / c1
        return fp.H1Cell(c1=1.0, c2=q / 2, D=3.3, W=6.6)
    assert fp.score_h1([cell(lo + 0.002)] * 15).verdict == fp.PASS
    assert fp.score_h1([cell(lo - 0.002)] * 15).verdict == fp.FAIL
    assert fp.score_h1([cell(hi - 0.002)] * 15).verdict == fp.PASS
    assert fp.score_h1([cell(hi + 0.002)] * 15).verdict == fp.FAIL


def test_h1():
    assert fp.score_h1(_h1(15, 0, 3)).verdict == fp.PASS
    assert fp.score_h1(_h1(14, 4, 0)).verdict == fp.FAIL
    assert fp.score_h1(_h1(14, 0, 4)).verdict == fp.UNRESOLVED


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
    assert fp.score_h2c([dict(p, s_lower=0.9, s_upper=1.01)]).verdict == fp.PASS
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
    win = fp.H3Cell(diffs=[0.08, 0.1, 0.06, 0.07] * 5, density_ratio=1.0)
    lose = fp.H3Cell(diffs=[0.0, 0.01, -0.01, 0.02] * 5, density_ratio=1.0)
    off = fp.H3Cell(diffs=[0.08] * 20, density_ratio=1.02)
    assert fp.score_h3([win] * 16 + [lose] * 8).verdict == fp.PASS
    assert fp.score_h3([win] * 10 + [lose] * 14).verdict == fp.FAIL
    assert fp.score_h3([win] * 12 + [off] * 4 + [lose] * 8).verdict == fp.UNRESOLVED
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
    off = fp.H7Cell(r_mp_16=0.95, r_mp_128=0.9, gap_16=0.01, gap_128=0.05, density_ratio=1.1)
    a, b = fp.score_h7([good] * 16 + [bad] * 8)
    assert (a.verdict, b.verdict) == (fp.PASS, fp.PASS)
    a, b = fp.score_h7([good] * 10 + [bad] * 14)
    assert (a.verdict, b.verdict) == (fp.FAIL, fp.FAIL)
    a, b = fp.score_h7([good] * 19 + [off] * 5)
    assert (a.verdict, b.verdict) == (fp.UNRESOLVED, fp.UNRESOLVED)


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
    assert fp.near_parity(m=1.06) and not fp.near_parity(m=1.08)


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


def test_the_reservation_check_stops_over_the_cap():
    assert fp.reservation_check(spent=9000, next_u=1000, remaining_star_u=[2000],
                                s_res_inr=600, r_b=262) == "STOP"
    assert fp.reservation_check(spent=9000, next_u=1000, remaining_star_u=[1138],
                                s_res_inr=600, r_b=262) == "PROCEED"   # exactly 12,000
    assert fp.reservation_check(spent=1000, next_u=1000, remaining_star_u=[2000],
                                s_res_inr=600, r_b=262) == "PROCEED"
    assert fp.s_res(2, 32.4) == pytest.approx(424 + 2.4 * 32.4)


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
    """sec. 4.5: accuracy, component, end-to-end and phase rows. The recall
    writer does not exist yet (sec. 11.2); when it does, add it here."""
    for f in ("attnbench/accuracy/runner.py", "scripts/measure_estimator_cost.py",
              "scripts/run_vectorised_endtoend.py", "attnbench/accuracy/phase_timing.py"):
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
                            ("I2e", "I2e:"), ("X primary", "★ X: 1.5B primary"),
                            ("X secondary", "★ X: 1.5B secondary"), ("X-rep", "★ X-rep:")):
        line = next(l for l in lines if l.name == name)
        cell = f"| {_inr(line.inr[0])}–{_inr(line.inr[1])} |"
        row = next(r for r in DOC.read_text().splitlines() if r.startswith(f"| {row_label}"))
        assert cell in row, (name, cell, row)
