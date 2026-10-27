"""T2.28: the density sensitivity, recomputed from a committed file's own
components at plus and minus the cross-path density difference.

Hand-computed. The family used throughout is K(h1, dD) S(item, 4 B)
K(h1, d1), then the closing synchronise, all in layer 0, with COSTS from
test_predict_recursion (item: s_fixed 1, bandwidth 4, so s_0 = 1 + 4/4 = 2;
synchronise: s_fixed 1):

    H1 = 1, D1 = 1 + D;  sync: H2 = D2 = (1 + D) + 2 = D + 3;
    K: H3 = D + 4, D3 = D + 5;  close: max(D + 4, D + 5) + 1 = D + 6.

So the recursion is D + 6, the Sum is h 2 + d (D + 1) = D + 3, the Sum + s_0
adds the two synchronisations' 2 + 1 = D + 6, and the step form (one layer)
is max(2, D + 1) + 3 = D + 4 for D >= 1. The device cost D is made linear in
density, D = 40 * density, so the arm committed at 0.25, 0.5 and 0.75 has
D = 10, 20, 30 and recursion 16, 26, 36.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from attnbench.predict import commit as C
from attnbench.predict import learned as L
from attnbench.predict.baselines import fit_fitted_sum
from attnbench.predict.commit import (CodeStamp, PredictionCommitError,
                                      build_record, verify_bytes, write_prediction)
from attnbench.predict.density_sensitivity import (
    NOT_RECOMPUTABLE, density_sensitivity, decision_flips)
from attnbench.predict.recursion import close_prefill, kernel_op, price, sync_op
from attnbench.predict.training import (Cell, GroundTruthRow, HardwareFeatures,
                                        committed_training_set)

from test_predict_commit import (CODE_PATHS, CONFIG, CP, MODEL, MeanModel, TOY,
                                 fit_learned)
from test_predict_recursion import COSTS

T0 = datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)
CLEAN = CodeStamp("a" * 40, False)
HW = HardwareFeatures(pinned_bandwidth=1024, pageable_bandwidth=512)
L_ = 4096


def seq(D):
    return close_prefill([kernel_op(1, D, layer=0),
                          sync_op("item", payload_bytes=4, layer=0),
                          kernel_op(1, 1, layer=0)])


def dense_seq(d):
    return close_prefill([kernel_op(1, d, layer=0)])


def cell(ops, arm="x", density=0.5, L=L_, pairing="P1", batch=1,
         model_name=MODEL, code_path=CP):
    return Cell(pairing=pairing, model_name=model_name, code_path=code_path,
                arm=arm, context_length=L, density=density, batch=batch,
                priced=price(ops, COSTS), hardware=HW, model=TOY)


def family(densities=(0.25, 0.5, 0.75), **kw):
    return [cell(seq(40 * d), density=d, **kw) for d in densities]


def committed(tmp_path, cells, Q=8, **kw):
    kw.setdefault("code_paths", CODE_PATHS)
    rec = build_record("P1", cells, Q, banked=[], fitted_sum=None, learned=None,
                       code=CLEAN, written_at=T0, **kw)
    f = write_prediction(rec, tmp_path)
    return verify_bytes(f.path.read_bytes(), f.sidecar.read_text(), f.path.name)


def key(arm="x", density=0.5, L=L_):
    return C.cell_key(MODEL, CP, arm, density, L, 1)


def by(res, k, predictor="recursion"):
    (c,) = [c for c in res.cells if c.key == k and c.predictor == predictor]
    return c


def test_the_family_is_the_hand_computed_one(tmp_path):
    cp = committed(tmp_path, family())
    assert [c.predictions["recursion"] for c in cp.cells] == [16, 26, 36]
    assert [c.predictions["sum"] for c in cp.cells] == [13, 23, 33]
    assert [c.predictions["step_form"] for c in cp.cells] == [14, 24, 34]


def test_recomputed_at_plus_and_minus_the_difference(tmp_path):
    cp = committed(tmp_path, family())
    res = density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"})
    r = by(res, key("x", 0.5))
    assert r.committed == 26 and r.delta == 0.25
    assert [(p.sign, p.density, p.predicted, p.extrapolated) for p in r.points] == \
        [(-1, 0.25, 16, False), (0, 0.5, 26, False), (1, 0.75, 36, False)]
    assert r.at(-1).relative_change == pytest.approx(-10 / 26)
    assert r.at(1).relative_change == pytest.approx(10 / 26)
    assert r.at(0).relative_change == 0
    assert res.abscissa == "key density"
    # Only the 0.25 cell cannot take -0.25 (density 0).
    assert [n.key for n in res.not_recomputed] == [key("x", 0.25)]


def test_every_predictor_that_reads_only_the_components_is_recomputed(tmp_path):
    cp = committed(tmp_path, family(), constant_host_cost=3.0)
    res = density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"})
    got = {p: by(res, key("x", 0.5), p) for p in
           ("recursion", "sum", "sum_plus_s0", "step_form", "constant_host_cost")}
    assert [p.predicted for p in got["recursion"].points] == [16, 26, 36]
    assert [p.predicted for p in got["sum"].points] == [13, 23, 33]
    assert [p.predicted for p in got["sum_plus_s0"].points] == [16, 26, 36]
    assert [p.predicted for p in got["step_form"].points] == [14, 24, 34]
    # Constant host cost 3: both kernels' h become 3; D3 = D+4+... worked
    # independently: H1 = 3, D1 = 3 + D; sync H2 = D + 5; K: H3 = D + 8,
    # D3 = D + 9; close max(D + 8, D + 9) + 1 = D + 10.
    assert [p.predicted for p in got["constant_host_cost"].points] == [20, 30, 40]


def test_the_committed_point_is_the_committed_prediction_exactly(tmp_path):
    cp = committed(tmp_path, family(), constant_host_cost=3.0)
    res = density_sensitivity(cp, deltas={"x": 0.125}, dense_arms={"dense"})
    for c in res.cells:
        assert c.at(0).predicted == c.committed
        assert c.committed == cp.cell(*c.key).predictions[c.predictor]


def test_between_committed_densities_it_interpolates(tmp_path):
    # 0.5 -/+ 0.125 is 0.375 and 0.625: D = 15 and 25.
    cp = committed(tmp_path, family())
    r = by(density_sensitivity(cp, deltas={"x": 0.125}, dense_arms={"dense"}),
           key("x", 0.5))
    assert [p.predicted for p in r.points] == [21, 26, 31]
    assert not any(p.extrapolated for p in r.points)


def test_outside_the_committed_range_it_extrapolates_and_says_so(tmp_path):
    # 0.25 - 0.125 = 0.125: D = 5 (from 0.25 and 0.5); 0.75 + 0.125 = 0.875:
    # D = 35; 0.75 + 0.25 = 1.0 is allowed: D = 40.
    cp = committed(tmp_path, family())
    res = density_sensitivity(cp, deltas={"x": 0.125}, dense_arms={"dense"})
    lo = by(res, key("x", 0.25))
    assert [(p.predicted, p.extrapolated) for p in lo.points] == \
        [(11, True), (16, False), (21, False)]
    hi = by(res, key("x", 0.75))
    assert [(p.predicted, p.extrapolated) for p in hi.points] == \
        [(31, False), (36, False), (41, True)]
    res = density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"})
    assert by(res, key("x", 0.75)).at(1).predicted == 46


def test_a_density_outside_zero_to_one_is_not_recomputed_and_listed(tmp_path):
    cp = committed(tmp_path, family())
    res = density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"})
    assert key("x", 0.25) not in {c.key for c in res.cells}
    (nr,) = res.not_recomputed
    assert nr.key == key("x", 0.25) and "outside (0, 1]" in nr.reason


def test_a_negative_extrapolated_component_is_not_recomputed(tmp_path):
    # Device 1 at 0.5 and 10 at 0.75: at 0.25 the line is 1 - 9 = -8.
    cp = committed(tmp_path, [cell(seq(1), density=0.5), cell(seq(10), density=0.75)])
    res = density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"})
    assert key("x", 0.5) not in {c.key for c in res.cells}
    assert any(n.key == key("x", 0.5) and "negative" in n.reason
               for n in res.not_recomputed)


def test_one_committed_density_has_no_response_to_interpolate(tmp_path):
    cp = committed(tmp_path, [cell(seq(20))])
    res = density_sensitivity(cp, deltas={"x": 0.125}, dense_arms={"dense"})
    assert not res.cells
    (nr,) = res.not_recomputed
    assert "only one density" in nr.reason


def test_different_operation_sequences_across_densities_are_not_compared(tmp_path):
    other = close_prefill([kernel_op(1, 20, layer=0), kernel_op(1, 1, layer=0)])
    cp = committed(tmp_path, [cell(seq(10), density=0.25), cell(other, density=0.5)])
    res = density_sensitivity(cp, deltas={"x": 0.125}, dense_arms={"dense"})
    assert not res.cells
    assert all("different operation sequences" in n.reason for n in res.not_recomputed)


def test_an_arm_with_no_measured_difference_is_listed_not_assumed_zero(tmp_path):
    cp = committed(tmp_path, family() + [cell(seq(5), arm="y", density=0.5),
                                         cell(seq(9), arm="y", density=0.75)])
    res = density_sensitivity(cp, deltas={"x": 0.125}, dense_arms={"dense"})
    assert {c.key.arm for c in res.cells} == {"x"}
    assert {n.key.arm for n in res.not_recomputed} == {"y"}
    assert all("no cross-path density difference" in n.reason
               for n in res.not_recomputed)


def test_a_difference_for_one_length_overrides_the_arm_default(tmp_path):
    cp = committed(tmp_path, family() + family(L=8192))
    res = density_sensitivity(
        cp, deltas={"x": 0.125, ("x", 8192): 0.25}, dense_arms={"dense"})
    assert by(res, key("x", 0.5, 4096)).delta == 0.125
    assert by(res, key("x", 0.5, 8192)).delta == 0.25


def test_a_zero_difference_gives_the_committed_value_three_times(tmp_path):
    cp = committed(tmp_path, family())
    r = by(density_sensitivity(cp, deltas={"x": 0}, dense_arms={"dense"}),
           key("x", 0.5))
    assert [p.predicted for p in r.points] == [26, 26, 26]


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_a_bad_difference_is_refused(tmp_path, bad):
    cp = committed(tmp_path, family())
    with pytest.raises(PredictionCommitError, match="absolute change"):
        density_sensitivity(cp, deltas={"x": bad}, dense_arms={"dense"})


def test_the_dense_arm_has_no_density_and_is_reported_unchanged(tmp_path):
    cp = committed(tmp_path, family() + [cell(dense_seq(14), arm="dense", density=1.0)])
    res = density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"})
    d = by(res, key("dense", 1.0))
    assert d.committed == 16 and [p.predicted for p in d.points] == [16, 16, 16]
    assert d.delta == 0.0


def test_the_whole_file_uses_one_abscissa_and_it_can_be_the_measured_mask_density(
        tmp_path):
    cp = committed(tmp_path, family())
    md = {key("x", 0.25): 0.3, key("x", 0.5): 0.5, key("x", 0.75): 0.7}
    res = density_sensitivity(cp, deltas={"x": 0.1}, dense_arms={"dense"},
                              mask_density=md)
    r = by(res, key("x", 0.5))
    assert res.abscissa == "mask density"
    assert [p.predicted for p in r.points] == pytest.approx([21, 26, 31])
    assert [p.density for p in r.points] == pytest.approx([0.4, 0.5, 0.6])
    with pytest.raises(PredictionCommitError, match="one abscissa"):
        density_sensitivity(cp, deltas={"x": 0.1}, dense_arms={"dense"},
                            mask_density={key("x", 0.5): 0.5})
    with pytest.raises(PredictionCommitError, match="not in the file"):
        density_sensitivity(cp, deltas={"x": 0.1}, dense_arms={"dense"},
                            mask_density={**md, key("x", 0.9): 0.9})


def test_two_cells_at_one_mask_density_are_not_recomputed(tmp_path):
    cp = committed(tmp_path, family(densities=(0.25, 0.5)))
    md = {key("x", 0.25): 0.4, key("x", 0.5): 0.4}
    res = density_sensitivity(cp, deltas={"x": 0.05}, dense_arms={"dense"},
                              mask_density=md)
    assert not res.cells and all("share a density" in n.reason
                                 for n in res.not_recomputed)


def test_families_are_per_model_code_path_length_and_batch(tmp_path):
    # The same arm at the same densities under another model has its own
    # family and its own response (here twice as steep).
    steep = [cell(seq(80 * d), density=d, model_name="m-large") for d in (0.25, 0.5, 0.75)]
    cp = committed(tmp_path, family() + steep)
    res = density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"})
    small = by(res, key("x", 0.5))
    large = by(res, C.cell_key("m-large", CP, "x", 0.5, L_, 1))
    assert [p.predicted for p in small.points] == [16, 26, 36]
    assert [p.predicted for p in large.points] == [26, 46, 66]


def test_nonlinearity_reports_how_far_a_middle_density_is_from_the_chord(tmp_path):
    # Recursion 16, 26, 36 is a line: 0. Moving the middle device cost from
    # 20 to 24 puts the middle 4 above the chord (chord at 0.5 is 26), which
    # is 4 / 36 of the larger end.
    (tmp_path / "a").mkdir()
    line = committed(tmp_path / "a", family())
    r = by(density_sensitivity(line, deltas={"x": 0.25}, dense_arms={"dense"}),
           key("x", 0.5))
    assert r.nonlinearity == 0
    bent = [cell(seq(10), density=0.25), cell(seq(24), density=0.5),
            cell(seq(30), density=0.75)]
    (tmp_path / "b").mkdir()
    cp = committed(tmp_path / "b", bent)
    r = by(density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"}),
           key("x", 0.5))
    assert r.nonlinearity == pytest.approx(4 / 36)
    # Two densities cannot say.
    (tmp_path / "c").mkdir()
    two = committed(tmp_path / "c", family(densities=(0.25, 0.5)))
    r = by(density_sensitivity(two, deltas={"x": 0.125}, dense_arms={"dense"}),
           key("x", 0.25))
    assert r.nonlinearity is None


def test_what_cannot_be_recomputed_is_named(tmp_path):
    cp = committed(tmp_path, family())
    c0 = cp.cells[1]
    withl = dataclasses.replace(c0, predictions={**c0.predictions, "learned": 1.0,
                                                 "nearest_pairing": 2.0})
    cp = dataclasses.replace(cp, cells=(cp.cells[0], withl, cp.cells[2]))
    res = density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"})
    assert {n.predictor for n in res.not_recomputed if n.key == c0.key} == \
        set(NOT_RECOMPUTABLE)
    assert not [c for c in res.cells if c.predictor in NOT_RECOMPUTABLE]
    with pytest.raises(PredictionCommitError, match="learned predictor's fitted model"):
        density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"},
                            predictors=["learned"])
    with pytest.raises(PredictionCommitError, match="has no committed"):
        density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"},
                            predictors=["fitted_sum"])
    with pytest.raises(PredictionCommitError, match="name the dense"):
        density_sensitivity(cp, deltas={"x": 0.25}, dense_arms=set())


def test_the_fitted_sum_is_recomputed_from_its_committed_coefficients(
        tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LOCKED_CONFIG", CONFIG)
    p1 = family()
    rows = [GroundTruthRow(cell=c, measured=2.0 * c.priced.device[0] + 5.0) for c in p1]
    ts = committed_training_set(rows, "P2", banked=["P1"])
    fs, lm = fit_fitted_sum(ts), fit_learned(ts, CONFIG, model_factory=MeanModel)
    p2 = family(pairing="P2")
    rec = build_record("P2", p2, 8, banked=["P1"], fitted_sum=fs, learned=lm,
                       code=CLEAN, written_at=T0, code_paths=CODE_PATHS)
    f = write_prediction(rec, tmp_path)
    cp = verify_bytes(f.path.read_bytes(), f.sidecar.read_text(), f.path.name)
    res = density_sensitivity(cp, deltas={"x": 0.125}, dense_arms={"dense"})
    r = by(res, key("x", 0.5), "fitted_sum")
    # Independently: price the interpolated sequences directly (D = 15, 25).
    want = [fs.predict(price(seq(D), COSTS)) for D in (15, 20, 25)]
    assert [p.predicted for p in r.points] == pytest.approx(want)
    assert r.committed == cp.cell(*key("x", 0.5)).predictions["fitted_sum"]
    assert {n.predictor for n in res.not_recomputed} == {"learned"}


# --------------------------------------------------------------------------
# Does the decision flip?
# --------------------------------------------------------------------------

def test_decision_flips_names_the_offsets_that_change_the_sign(tmp_path):
    # Dense K(h1, d19), close: 21. Sparse 16 / 26 / 36 at 0.25 / 0.5 / 0.75.
    # At 0.5 the committed sparse (26) is slower; at -0.25 it is 16, faster:
    # flips at -1; at +0.25 it is 36, slower still: no flip.
    cp = committed(tmp_path, family() + [cell(dense_seq(19), arm="dense", density=1.0)])
    res = density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"})
    out = {d.sparse_key.density: d for d in decision_flips(res, dense_arm="dense")}
    assert out[0.5].dense_key == key("dense", 1.0)
    assert not out[0.5].committed_sparse_faster and out[0.5].flips_at == (-1,)
    # At 0.75 (36): committed slower; its minus point is D = 20 -> 26: still
    # slower, no flip. Its plus point is extrapolated (46): no flip.
    # At 0.25 the plus-minus points are not both recomputable (0), so it is
    # absent: nothing is compared on a half-answer.
    assert out[0.75].flips_at == ()
    assert 0.25 not in out


def test_a_tie_is_dense_in_the_flip_check_too(tmp_path):
    # Dense 26 equals the committed sparse at 0.5: a tie, so "not faster".
    # At -0.25 sparse is 16 (faster): flips; at +0.25 it is 36: no flip.
    cp = committed(tmp_path, family() + [cell(dense_seq(24), arm="dense", density=1.0)])
    res = density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"})
    out = {d.sparse_key.density: d for d in decision_flips(res, dense_arm="dense")}
    assert not out[0.5].committed_sparse_faster and out[0.5].flips_at == (-1,)


def test_flips_with_no_dense_cell_is_refused(tmp_path):
    cp = committed(tmp_path, family())
    res = density_sensitivity(cp, deltas={"x": 0.25}, dense_arms={"dense"})
    with pytest.raises(PredictionCommitError, match="no 'dense' cells"):
        decision_flips(res, dense_arm="dense")


# --------------------------------------------------------------------------
# Which committed densities are interpolated between, and at which Q
# --------------------------------------------------------------------------

def test_it_interpolates_between_the_two_nearest_densities_not_the_first_two(
        tmp_path):
    # A kink: D = 10, 24, 30 at 0.25, 0.5, 0.75 (recursion D + 6). Between
    # 0.5 and 0.75 the line is 24 + 24 * (x - 0.5): at 0.625 D = 27. Between
    # 0.25 and 0.5 it is 10 + 56 * (x - 0.25): at 0.375 D = 17. Taking the
    # first pair for both, or the last, gives other numbers.
    bent = [cell(seq(10), density=0.25), cell(seq(24), density=0.5),
            cell(seq(30), density=0.75)]
    cp = committed(tmp_path, bent)
    res = density_sensitivity(cp, deltas={"x": 0.125}, dense_arms={"dense"})
    assert [p.predicted for p in by(res, key("x", 0.5)).points] == [23, 30, 33]
    # And from the 0.75 cell: 0.625 is inside (D = 27); 0.875 extrapolates
    # from the last pair, slope 24 per unit: D = 30 + 24 * 0.125 = 33.
    assert [p.predicted for p in by(res, key("x", 0.75)).points] == [33, 36, 39]
    assert by(res, key("x", 0.75)).at(1).extrapolated


def qseq(D):
    # Three kernels of device cost D, then a long host operation, then close.
    return close_prefill([kernel_op(1, D, layer=0)] * 3 + [kernel_op(10, 1, layer=0)])


def test_the_recomputation_uses_the_committed_queue_depth(tmp_path):
    # D = 2, 4, 6 at 0.25, 0.5, 0.75. With no queue the totals are 15, 15, 21;
    # at Q = 1 (a host launch waits for the previous one to finish) they are
    # 17, 21, 25 (worked: D = 4 -> H 1, 5, 9, 19, D 5, 9, 13, 20, close 21;
    # D = 2 -> H 1, 3, 5, 15, close 17; D = 6 -> H 1, 7, 13, 23, close 25).
    fam = [cell(qseq(8 * d), density=d) for d in (0.25, 0.5, 0.75)]
    (tmp_path / "q1").mkdir()
    cp1 = committed(tmp_path / "q1", fam, Q=1)
    assert [c.predictions["recursion"] for c in cp1.cells] == [17, 21, 25]
    r = by(density_sensitivity(cp1, deltas={"x": 0.25}, dense_arms={"dense"}),
           key("x", 0.5))
    assert [p.predicted for p in r.points] == [17, 21, 25]
    (tmp_path / "q8").mkdir()
    cp8 = committed(tmp_path / "q8", fam, Q=8)
    r = by(density_sensitivity(cp8, deltas={"x": 0.25}, dense_arms={"dense"}),
           key("x", 0.5))
    assert [p.predicted for p in r.points] == [15, 15, 21]
