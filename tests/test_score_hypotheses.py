"""The scoring driver for H0 to H3 on P2 and P4 (`scripts/score_hypotheses.py`;
the adviser's rulings dated 16 Nov 2026, scoring wave, items 1 and 6).

Fixtures with known verdicts: `test_predict_commit`'s hand-worked records
(xattn recursion 16, dense 8; measured 20 and 10, so the recursion's median
absolute relative error is 0.20), P1's and P2's cells built from the same
operations (H0's differences are all 0), and a two-length grid whose
predicted and measured crossovers are placed by hand for H3.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import score_hypotheses as SH  # noqa: E402
from attnbench.analysis import hypotheses as HY  # noqa: E402
from attnbench.lock import checklist as CL  # noqa: E402
from attnbench.lock.floors import FloorTable  # noqa: E402
from attnbench.predict import intervals as IV  # noqa: E402
from attnbench.predict.commit import load_committed  # noqa: E402
from attnbench.predict.training import CellKey  # noqa: E402

import lock_fixture as LF  # noqa: E402
from test_predict_commit import (MODEL, NO_SKEW, T0, Clock, MemoryStore, committed,  # noqa: E402,F401
                                 locked, p1_record, p2_record, trained)

ROW = {"arm": "dense", "arm_kind": "dense", "deployable": False, "arm_label": "dense",
       "compile_setting": None, "traced": False, "reps": 3, "rounds_run": 10,
       "clean_rounds": 10, "prefill_ms_samples": [], "prefill_ms_median": 0.0,
       "flagged_ms_samples": [], "throttle_reasons_seen": [], "context_length": 4,
       "batch": 1, "builder_kind": None, "mask_density": None, "executed_density": 1.0,
       "model": MODEL, "cell_arm": "dense", "nominal_density": None, "budget": 4,
       "group": "eager"}


def row(arm, samples_ms, density=1.0):
    r = copy.deepcopy(ROW)
    r.update(arm=arm, arm_label=arm, cell_arm=arm, prefill_ms_samples=list(samples_ms),
             prefill_ms_median=sorted(samples_ms)[len(samples_ms) // 2])
    if arm != "dense":
        r.update(arm_kind="precomputed", mask_density=density, executed_density=density)
    return r


def result(*rows, clean=10):
    return {"traced": False, "complete": True, "clean_rounds": clean, "rounds_run": 10,
            "rows": list(rows)}


def lk(**over):
    floors = FloorTable({(4, 1): 0.0, (16384, 1): 0.01377, (16384, 4): 0.02759,
                         (32768, 1): 0.10116})
    base = dict(tolerances={"H0": 0.1, "H1": 0.25, "H2": 0.25}, floors=floors,
                bootstrap=IV.Settings(0.95, 200, 0), h4_bound_pts=float("nan"), h4_tests=(),
                h4_alpha=float("nan"), h4_n=0, scored={}, sensitivity_s={(16384, 1): 0.11105},
                analysis_commit="c" * 40)
    base.update(over)
    return HY.Locked(**base)


NOT = {"reason": "never ran", "appendix_a": "Appendix A, 12 Nov 2026"}
SCORED = {"H0": ["P2"], "H1": ["P2", "P4"], "H2": ["P2"], "H3": ["P4"],
          "H4": {"not_scored": NOT}, "H5": {"not_scored": NOT}, "H6": {"not_scored": NOT}}
LOCK = {"scored_pairings": SCORED, "values": {"floors_quantile": {"value": 0.95}}}


@pytest.fixture
def cps(tmp_path, locked):
    store = MemoryStore(Clock(T0))
    for d in ("p1", "p2"):
        (tmp_path / d).mkdir()
    f1, r1 = committed(tmp_path / "p1", p1_record(), store, T0 + timedelta(seconds=5))
    p1 = load_committed(store, r1.remote, ground_truth_started_at=T0 + timedelta(seconds=6),
                        clock_skew=NO_SKEW, sync_gap=NO_SKEW, banked_at={}, local=f1.path)
    f2, r2 = committed(tmp_path / "p2", p2_record(), store, T0 + timedelta(seconds=5))
    p2 = load_committed(store, r2.remote, ground_truth_started_at=T0 + timedelta(seconds=6),
                        clock_skew=NO_SKEW, sync_gap=NO_SKEW,
                        banked_at={"P1": T0 - timedelta(days=1)}, local=f2.path, **trained())
    return {"P1": p1, "P2": p2, "P4": p2}


# --- item 1: the lock is the tag's ----------------------------------------------------

def test_the_lock_is_the_tags_and_main_may_differ_only_in_the_plan_hash():
    tag = {"plan_sha256": "a" * 64, "values": {"x": {"value": 1}}, "blocks": {"P4": [1]}}
    assert SH.check_lock_source(tag, {**tag, "plan_sha256": "b" * 64}) == ["/plan_sha256"]
    assert SH.check_lock_source(tag, copy.deepcopy(tag)) == []
    for main in ({**tag, "values": {"x": {"value": 2}}}, {**tag, "blocks": {}},
                 {**tag, "extra": 1}, {k: v for k, v in tag.items() if k != "blocks"}):
        with pytest.raises(SH.ScoreError, match="differs from the tag's"):
            SH.check_lock_source(tag, main)


def test_mains_lock_differs_from_the_tags_only_in_the_plan_hash():
    try:
        tagged = SH.lock_at(SH.TAG)
    except SH.ScoreError:
        pytest.skip(f"no tag {SH.TAG} in this clone")
    main = json.loads((ROOT / SH.LOCK_FILE).read_text())
    assert SH.check_lock_source(tagged, main) in ([], ["/plan_sha256"])


def test_the_lock_reader_reads_h0_to_h3_as_from_lock_does_and_reads_the_real_lock():
    lock = LF.complete_lock(CL.PLAN.read_text(encoding="utf-8"))
    lock["scored_pairings"] = LF.SCORED
    lock["values"]["h4_confirmatory_family"]["value"] = {"m": 2, "tests": ["a", "b"]}
    tol = lock["values"]["h0_to_h3_tolerances"]["value"]
    tol.update(H0=0.11, H1=0.12, H2=0.13)            # distinct, so a swap shows
    ref = HY.Locked.from_lock(lock)
    assert ref.tolerances == {"H0": 0.11, "H1": 0.12, "H2": 0.13}
    assert SH.locked_from(lock) == ref               # H4 scored: from_lock itself
    lock["scored_pairings"] = {**LF.SCORED, "H4": {"not_scored": NOT}}
    lock["values"]["power_target_and_n"]["value"] = None
    with pytest.raises(HY.HypothesisError, match="power_target_and_n"):
        HY.Locked.from_lock(lock)                    # the analysis commit's reader refuses
    got = SH.locked_from(lock)
    for f in ("tolerances", "floors", "bootstrap", "sensitivity_s", "analysis_commit"):
        assert getattr(got, f) == getattr(ref, f), f
    assert "H4" not in got.scored and got.scored["H1"] == ref.scored["H1"]
    lock["values"]["h0_to_h3_tolerances"]["value"] = None
    with pytest.raises(HY.HypothesisError, match="h0_to_h3_tolerances"):
        SH.locked_from(lock)


# --- the analysis commit --------------------------------------------------------------

def test_an_imported_file_that_differs_from_the_analysis_commit_is_refused(tmp_path):
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True,
                                    capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "b.py").write_text("y = 1\n")
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "c")
    commit = git("rev-parse", "HEAD")
    mods = {"a": SimpleNamespace(__file__=str(tmp_path / "pkg" / "a.py")),
            "b": SimpleNamespace(__file__=str(tmp_path / "pkg" / "b.py")),
            "elsewhere": SimpleNamespace(__file__="/usr/lib/x.py")}
    files = SH.imported_files(tmp_path.resolve(), mods)
    assert files == ["pkg/a.py", "pkg/b.py"]
    assert SH.not_at_commit(files, commit, tmp_path.resolve()) == []
    (tmp_path / "pkg" / "b.py").write_text("y = 2\n")
    assert SH.not_at_commit(files, commit, tmp_path.resolve()) == ["pkg/b.py"]
    L = lk(analysis_commit=commit)
    with pytest.raises(SH.ScoreError, match="pkg/b.py"):
        SH.at_analysis_commit(L, tmp_path.resolve(), mods)


# --- the measurements -----------------------------------------------------------------

def test_a_cell_in_two_blocks_pools_its_samples_and_the_spread_is_per_block(cps):
    cp = cps["P2"]
    blocks = [("b3", [("g", result(row("xattn", [19.0, 20.0, 21.0], 0.5),
                                   row("dense", [9.0, 10.0, 11.0])))]),
              ("b4", [("g", result(row("dense", [12.0, 13.0, 14.0, 15.0])))])]
    m = SH.measure(cp, blocks, min_clean_rounds=10)
    key = {c.arm: c.key for c in cp.cells}
    assert m["seconds"][key["xattn"]] == pytest.approx(0.020)
    # pooled dense: 9, 10, 11, 12, 13, 14, 15 ms -> median 12 ms, not either block's
    assert m["seconds"][key["dense"]] == pytest.approx(0.012)
    assert m["pooled_cells"] == [list(key["dense"])]
    s = SH.spread(m, quantile=1.0, budget_of=lk().floors.budget_of)
    # about each block's own median: at most |9 - 10| and |12 - 13.5| ms, never |9 - 12|
    assert s == {(4, 1): pytest.approx(0.0015)}


def test_a_group_short_of_clean_rounds_leaves_its_cells_unmeasured(cps):
    cp = cps["P2"]
    blocks = [("s", [("g1", result(row("xattn", [20.0, 20.0], 0.5))),
                     ("g2", result(row("dense", [10.0, 10.0]), clean=9))])]
    m = SH.measure(cp, blocks, min_clean_rounds=10)
    assert len(m["excluded"]) == 1 and [k.arm for k in m["unmeasured"]] == ["dense"]


def test_a_cell_timed_twice_within_one_block_is_refused(cps):
    blocks = [("b3", [("g1", result(row("dense", [10.0, 10.0]))),
                      ("g2", result(row("dense", [11.0, 11.0])))])]
    with pytest.raises(SH.G.GateAError, match="measured twice"):
        SH.measure(cps["P2"], blocks, min_clean_rounds=10)


# --- the verdicts ---------------------------------------------------------------------

def measured(cp, xattn_ms, dense_ms):
    return SH.measure(cp, [("s", [("g", result(row("xattn", xattn_ms, 0.5),
                                               row("dense", dense_ms)))])], min_clean_rounds=10)


def test_the_hand_worked_records_score_as_worked_out(cps):
    ms = {p: measured(cps[p], [20.0, 20.0, 20.0], [10.0, 10.0, 10.0]) for p in ("P2", "P4")}
    # the file has no step-9 times in seconds: the records are in hand-worked units
    for m in ms.values():
        m["seconds"] = {k: v * 1000 for k, v in m["seconds"].items()}
        m["samples"] = {k: [x * 1000 for x in v] for k, v in m["samples"].items()}
        m["per_block"] = {k: [x * 1000 for x in v] for k, v in m["per_block"].items()}
    out = SH.score(LOCK, lk(), cps, ms, dense_arm="dense")
    h1 = out["H1"]["P2"]
    assert h1["recursion_median_abs_relative"] == pytest.approx(0.20)
    assert h1["parts"]["tolerance"] == HY.HOLDS and h1["sign"]["agree"] == 1
    # P1 and P2 share every operation's prices: H0 reads 0 and holds
    h0 = out["H0"]["P2"]
    assert h0["median_abs_relative_difference"] == 0.0 and h0["verdict"] == HY.HOLDS
    assert h0["operations_scored"] > 0
    # H2 re-runs P2's recursion on P1's (equal) host times: H1's 0.20 again, within 0.25
    h2 = out["H2"]["P2"]
    assert h2["median_abs_relative"] == pytest.approx(0.20) and h2["parts"]["tolerance"] == HY.HOLDS
    # one length per estimator: no crossover bracket, so H3 is not scored
    assert out["H3"]["P4"]["verdict"] == HY.NOT_SCORED
    assert {out[h]["verdict"] for h in ("H4", "H5", "H6")} == {HY.NOT_SCORED}
    tight = SH.score(LOCK, lk(tolerances={"H0": 0.1, "H1": 0.1, "H2": 0.1}), cps, ms,
                     dense_arm="dense")
    assert tight["H1"]["P2"]["verdict"] == HY.FAILS and tight["H2"]["P2"]["verdict"] == HY.FAILS


def test_h2_is_not_tested_when_h0_fails(cps):
    ms = {p: measured(cps[p], [20.0] * 3, [10.0] * 3) for p in ("P2", "P4")}
    for m in ms.values():
        m["seconds"] = {k: v * 1000 for k, v in m["seconds"].items()}
    out = SH.score(LOCK, lk(tolerances={"H0": 1e-9, "H1": 0.25, "H2": 0.25}), cps, ms,
                   dense_arm="dense")
    assert out["H0"]["P2"]["verdict"] == HY.HOLDS        # 0 <= any tolerance
    # P1's host prices doubled: every relative difference is -0.5
    p1 = dataclasses.replace(cps["P1"], cells=tuple(
        dataclasses.replace(c, priced=dataclasses.replace(
            c.priced, host=tuple(2 * h for h in c.priced.host)))
        for c in cps["P1"].cells))
    out = SH.score(LOCK, lk(), {**cps, "P1": p1}, ms, dense_arm="dense")
    assert out["H0"]["P2"]["median_abs_relative_difference"] == pytest.approx(0.5)
    assert out["H0"]["P2"]["verdict"] == HY.FAILS
    assert out["H2"]["P2"]["verdict"] == HY.NOT_TESTED


def test_a_different_sequence_leaves_h0_and_h2_unscored_and_the_report_says_why(cps):
    ms = {p: measured(cps[p], [20.0] * 3, [10.0] * 3) for p in ("P2", "P4")}
    for m in ms.values():
        m["seconds"] = {k: v * 1000 for k, v in m["seconds"].items()}
    # P1's sequences one operation longer, as the L4's are than the A100's
    p1 = dataclasses.replace(cps["P1"], cells=tuple(
        dataclasses.replace(c, priced=dataclasses.replace(
            c.priced, **{f: tuple(getattr(c.priced, f)) + (getattr(c.priced, f)[-1],)
                         for f in ("kinds", "mechanisms", "layers", "payload_bytes",
                                   "host", "device")}))
        for c in cps["P1"].cells))
    out = SH.score(LOCK, lk(), {**cps, "P1": p1}, ms, dense_arm="dense")
    h0, h2 = out["H0"]["P2"], out["H2"]["P2"]
    assert h0["verdict"] == HY.NOT_SCORED and h0["cells_compared"] == 0
    assert len(h0["cells_with_a_different_sequence"]) == len(cps["P2"].cells)
    assert h2["verdict"] == HY.NOT_SCORED and len(h2["left_out"]) == len(cps["P2"].cells)
    rep = {"lock": {"tag": SH.TAG, "main_differs_at": ["/plan_sha256"]},
           "analysis": {"analysis_commit": "8" * 40, "files_checked": 1},
           "files": {}, "measured": {}, "scores": out}
    md = SH.to_markdown(rep)
    row = lambda h: next(x for x in md.splitlines() if x.startswith(f"| {h} "))
    assert "no operation is above" in row("H0") and "0 cells with the same sequence" in row("H0")
    assert "None" not in row("H0") and "None" not in row("H2")
    assert "no transferred cell is measured" in row("H2") and "**not scored**" in row("H2")


def test_the_driver_refuses_a_lock_that_scores_otherwise(cps):
    ms = {p: measured(cps[p], [20.0] * 3, [10.0] * 3) for p in ("P2", "P4")}
    for bad in ({**SCORED, "H3": ["P2", "P4"]}, {**SCORED, "H6": ["P4"]}):
        with pytest.raises(SH.ScoreError):
            SH.score({**LOCK, "scored_pairings": bad}, lk(), cps, ms, dense_arm="dense")


def grid_cp(pred_ms):
    """Cells of meanpool@0.5 and dense at 2048 and 4096, batch 1, with the
    recursion's predictions `pred_ms` {(arm, L): ms}."""
    cells = [SimpleNamespace(key=CellKey("m", "cp", a, d, L, 1),
                             predictions={"recursion": pred_ms[(a, L)] / 1000})
             for a, d in (("meanpool", 0.5), ("dense", 1.0)) for L in (2048, 4096)]
    return SimpleNamespace(cells=cells)


def grid_measured(cp, meas_ms):
    samples = {c.key: [meas_ms[(c.key.arm, c.key.context_length)] / 1000 + j * 1e-5
                       for j in (-1, 0, 1)] for c in cp.cells}
    return {"seconds": {k: v[1] for k, v in samples.items()}, "samples": samples,
            "nominal": {c.key: (0.5 if c.key.arm == "meanpool" else None) for c in cp.cells}}


def test_h3_holds_when_the_measured_crossover_is_in_the_predicted_bracket():
    floors = FloorTable({(2048, 1): 0.001, (4096, 1): 0.001})
    # predicted: sparse slower at 2048 (+5 ms), faster at 4096 (-10 ms)
    cp = grid_cp({("meanpool", 2048): 25, ("dense", 2048): 20,
                  ("meanpool", 4096): 30, ("dense", 4096): 40})
    st = IV.Settings(0.95, 200, 0)
    same = grid_measured(cp, {("meanpool", 2048): 26, ("dense", 2048): 20,
                              ("meanpool", 4096): 31, ("dense", 4096): 41})
    pred, meas = SH.h3_inputs(cp, same, dense_arm="dense", settings=st)
    assert list(pred) == ["meanpool@0.5"]
    assert [round(g.point, 6) for g in pred["meanpool@0.5"]] == [0.005, -0.01]
    assert HY.h3(pred, meas, floors=floors, pairing="P4")["verdict"] == HY.HOLDS
    # measured: sparse slower at both lengths, so no crossover in range is measured
    none = grid_measured(cp, {("meanpool", 2048): 26, ("dense", 2048): 20,
                              ("meanpool", 4096): 45, ("dense", 4096): 41})
    pred, meas = SH.h3_inputs(cp, none, dense_arm="dense", settings=st)
    assert HY.h3(pred, meas, floors=floors, pairing="P4")["verdict"] == HY.FAILS


# --- the mirror -----------------------------------------------------------------------

def test_the_mirror_returns_what_was_read_and_refuses_what_was_not(tmp_path):
    live = MemoryStore(Clock(T0))
    (tmp_path / "f").write_bytes(b"abc")
    live.put(tmp_path / "f", "gs://b/x")
    live.exists = lambda r: r in live.objs
    rec = SH.RecordingStore(live, tmp_path / "m")
    assert rec.get("gs://b/x") == b"abc" and rec.created("gs://b/x") == T0
    assert rec.exists("gs://b/none") is False
    mir = SH.MirrorStore(tmp_path / "m")
    assert mir.get("gs://b/x") == b"abc" and mir.created("gs://b/x") == T0
    assert mir.exists("gs://b/none") is False
    with pytest.raises(FileNotFoundError):
        mir.get("gs://b/none")
    with pytest.raises(SH.ScoreError, match="did not record"):
        mir.get("gs://b/other")
    (tmp_path / "m" / "objects" / "b" / "x").write_bytes(b"abd")
    with pytest.raises(SH.ScoreError, match="not what was read"):
        mir.get("gs://b/x")
