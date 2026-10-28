"""C10: `scripts/paper_artifacts.py` and `attnbench/figures/paper_*.py`.

Every expected number is worked by hand from `_paper_fixtures` (its
docstring gives the committed and measured times). No GPU and no torch.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
from pathlib import Path

import pytest

from attnbench.figures import paper_artifacts as PA
from attnbench.figures import paper_inputs as PI
from attnbench.figures import paper_numbers as PN
from attnbench.figures import paper_tables as PT
from attnbench.figures import results as RS
from attnbench.figures.results import sha256_file
from attnbench.predict import scoring as SC
from attnbench.predict.commit import CodeStamp, verify_bytes

import _paper_fixtures as F
from _figure_fixtures import STARTED_AT, SYNC_AT

ROOT = Path(__file__).resolve().parents[1]
CLEAN_RUN = CodeStamp(git_commit="b" * 40, git_dirty=False)


def _script():
    spec = importlib.util.spec_from_file_location("paper_artifacts_script",
                                                  ROOT / "scripts" / "paper_artifacts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "pred").mkdir()
    cp, local, store, remote = F.loaded(tmp_path / "pred")
    return cp, local, store, remote, tmp_path


def _inputs(setup, **over):
    cp, _, _, _, tmp = setup
    spec = F.write_spec(tmp / "in", cp, **over)
    return PI.build_inputs(PI.read_spec(spec), spec.parent, cp)


def _rel(p, m):
    return (p - m) / m


# --- the command, end to end ---------------------------------------------------------

def test_command_writes_every_artifact(setup, monkeypatch):
    cp, local, store, remote, tmp = setup
    spec = F.write_spec(tmp / "in", cp)
    mod = _script()
    monkeypatch.setattr(mod, "code_stamp", lambda: CLEAN_RUN)
    out = tmp / "out"
    rc = mod.main(["--remote", remote, "--local", str(local), "--spec", str(spec),
                   "--clock-skew-s", "0", "--out-dir", str(out)], store=store)
    assert rc == 0
    want = {f"tables/{n}.tex" for n in ("cells", "predictors", "ablations", "queue",
                                          "residuals_sync", "decision", "crossover")}
    want |= {f"figures/{n}.pdf" for n in ("error_vs_length", "decision_error", "crossover")}
    want |= {"numbers.json", "numbers.tex"}
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["outputs"]) == want
    for name, sha in manifest["outputs"].items():
        assert sha256_file(out / name) == sha
    assert manifest["prediction"]["sha256"] == cp.sha256
    assert manifest["generated_by"] == {"git_commit": "b" * 40, "git_dirty": False}
    roles = sorted(f["role"] for f in manifest["inputs"])
    assert roles == ["crossover_intervals"] + ["ground_truth"] * 3 + ["prediction"]


def test_command_refuses_a_prediction_synced_after_the_workload(setup, monkeypatch):
    cp, local, store, remote, tmp = setup
    early = SYNC_AT - (STARTED_AT - SYNC_AT)
    spec = F.write_spec(tmp / "in", cp, results={4: F.result(4, started=early)})
    mod = _script()
    monkeypatch.setattr(mod, "code_stamp", lambda: CLEAN_RUN)
    rc = mod.main(["--remote", remote, "--local", str(local), "--spec", str(spec),
                   "--clock-skew-s", "0", "--out-dir", str(tmp / "out")], store=store)
    assert rc == 2
    assert not (tmp / "out").exists()


def test_command_refuses_a_dirty_checkout(setup, monkeypatch):
    cp, local, store, remote, tmp = setup
    spec = F.write_spec(tmp / "in", cp)
    mod = _script()
    monkeypatch.setattr(mod, "code_stamp", lambda: CodeStamp("b" * 40, True))
    rc = mod.main(["--remote", remote, "--local", str(local), "--spec", str(spec),
                   "--clock-skew-s", "0", "--out-dir", str(tmp / "out")], store=store)
    assert rc == 2
    assert not (tmp / "out").exists()


def test_command_refuses_a_local_copy_that_is_not_the_store_s(setup, monkeypatch):
    cp, local, store, remote, tmp = setup
    spec = F.write_spec(tmp / "in", cp)
    other = tmp / "other.json"
    other.write_bytes(local.read_bytes() + b" ")
    mod = _script()
    monkeypatch.setattr(mod, "code_stamp", lambda: CLEAN_RUN)
    assert mod.main(["--remote", remote, "--local", str(other), "--spec", str(spec),
                     "--clock-skew-s", "0", "--out-dir", str(tmp / "out")], store=store) == 2


# --- what is refused ------------------------------------------------------------------

def test_a_prediction_read_by_hash_alone_is_refused(setup):
    cp, local, _, _, tmp = setup
    side = Path(str(local) + ".sha256").read_text()
    bare = verify_bytes(local.read_bytes(), side, local.name)
    assert bare.synced_at is None
    spec = F.write_spec(tmp / "in", cp)
    with pytest.raises(RS.ResultsError, match="sync time"):
        PI.build_inputs(PI.read_spec(spec), spec.parent, bare)


def test_a_spec_for_another_prediction_is_stale(setup):
    with pytest.raises(PI.PaperInputError, match="stale"):
        _inputs(setup, prediction_sha256="0" * 64)


def test_a_changed_ground_truth_file_is_stale(setup):
    cp, _, _, _, tmp = setup
    spec = F.write_spec(tmp / "in", cp)
    p = tmp / "in" / "gt_L8.json"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(PI.PaperInputError, match="stale"):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp)


def test_an_unhashed_ground_truth_file_is_refused(setup):
    cp, _, _, _, tmp = setup
    spec = F.write_spec(tmp / "in", cp)
    doc = json.loads(spec.read_text())
    del doc["results"][0]["sha256"]
    spec.write_text(json.dumps(doc))
    with pytest.raises(PI.PaperInputError, match="unhashed"):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp)


def test_intervals_made_for_another_prediction_are_stale(setup):
    cp, _, _, _, tmp = setup
    spec = F.write_spec(tmp / "in", cp)
    iv = F.write_json(tmp / "in" / "intervals.json", F.intervals_doc(cp, sha="1" * 64))
    doc = json.loads(spec.read_text())
    doc["crossover"]["predicted_intervals"]["sha256"] = sha256_file(iv)
    spec.write_text(json.dumps(doc))
    with pytest.raises(PI.PaperInputError, match="another prediction"):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp)


def test_an_interval_not_centred_on_the_committed_difference_is_refused(setup):
    cp, _, _, _, tmp = setup
    spec = F.write_spec(tmp / "in", cp)
    d = F.intervals_doc(cp)
    d["entries"][0]["point"] += 0.25
    iv = F.write_json(tmp / "in" / "intervals.json", d)
    doc = json.loads(spec.read_text())
    doc["crossover"]["predicted_intervals"]["sha256"] = sha256_file(iv)
    spec.write_text(json.dumps(doc))
    with pytest.raises(PI.PaperInputError, match="not the committed difference"):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp)


@pytest.mark.parametrize("field", ["relative_floor", "resolution_floor_s",
                                   "floor_provisional"])
def test_lock_values_have_no_default(setup, field):
    cp, _, _, _, tmp = setup
    spec = F.write_spec(tmp / "in", cp)
    doc = json.loads(spec.read_text())
    del doc[field]
    spec.write_text(json.dumps(doc))
    with pytest.raises(PI.PaperInputError):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp)


def test_decision_error_needs_admissibility_stated(setup):
    inp = _inputs(setup)
    k = PI.cell_key(*F.key("xattn", 8))
    inp = dataclasses.replace(inp, admissible={a: v for a, v in inp.admissible.items()
                                               if a != k})
    with pytest.raises(PI.PaperInputError, match="admissibility"):
        PT.decision(inp)


def test_the_output_directory_must_be_empty(setup):
    inp = _inputs(setup)
    out = setup[4] / "out"
    out.mkdir()
    (out / "numbers.json").write_text("{}")
    with pytest.raises(PI.PaperInputError, match="not empty"):
        PA.build(inp, out, CLEAN_RUN)
    assert (out / "numbers.json").read_text() == "{}"


@pytest.mark.parametrize("code", [CodeStamp("b" * 40, True), CodeStamp("b" * 7, False)])
def test_artifacts_are_made_at_a_clean_known_commit(setup, code):
    with pytest.raises(PI.PaperInputError, match="clean"):
        PA.build(_inputs(setup), setup[4] / "out", code)


# --- classes: oracle, deployable, CPU and GPU builders kept apart ------------------------

def test_every_class_has_its_own_label():
    labels = list(PI.CLASSES.values())
    assert len(set(labels)) == len(labels)
    for c in (PI.ORACLE, PI.DEPLOYABLE, PI.CPU_BUILDER, PI.GPU_BUILDER):
        assert c in PI.CLASSES


@pytest.mark.parametrize("arm,cls,kinds,match", [
    ("oracle", "deployable_estimator", None, "is a 'oracle' arm, stated as"),
    ("oracle", "cpu_builder", None, "is a 'oracle' arm, stated as"),
    ("xattn", "oracle", None, "is a 'in_backend' arm, stated as"),
    ("xattn", "dense", None, "dense"),
    ("xattn", "deployable_estimator",
     {**F.KINDS, "xattn": ("in_backend", False)}, "its row says not"),
])
def test_a_stated_class_must_agree_with_the_row(setup, arm, cls, kinds, match):
    cp, _, _, _, tmp = setup
    results = {L: F.result(L, kinds=kinds) for L in F.LENGTHS} if kinds else None
    spec = F.write_spec(tmp / "in", cp, results=results,
                        result_classes={**F.CLASSES, arm: cls},
                        builder_kinds={arm: "cpu"} if cls == "cpu_builder" else {})
    with pytest.raises(PI.PaperInputError, match=match):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp)


def test_a_row_without_its_kind_is_refused(setup):
    cp, _, _, _, tmp = setup
    r = F.result(4)
    del r["rows"][0]["arm_kind"]
    spec = F.write_spec(tmp / "in", cp, results={4: r})
    with pytest.raises(PI.PaperInputError, match="arm_kind"):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp)


def test_every_committed_arm_needs_a_class(setup):
    with pytest.raises(PI.PaperInputError, match="no result class"):
        _inputs(setup, result_classes={"xattn": "deployable_estimator", "dense": "dense"})


def test_tables_and_figures_name_the_class_and_never_share_a_panel(setup):
    inp = _inputs(setup)
    cells = PT.tex_cells(inp)
    body = cells.split("\\midrule")[1].split("\\bottomrule")[0].strip().splitlines()
    want = {"xattn": "deployable estimator", "oracle": "oracle (non-deployable reference)",
            "dense": "dense"}
    assert len(body) == 9
    for line in body:
        c = line.split(" & ")
        assert c[6] == want[c[2]]
    # One decision group per (predictor, model, code path, class).
    gs = PT.decision(inp)
    keys = [(g.predictor, g.model, g.code_path, g.result_class) for g in gs]
    assert len(keys) == len(set(keys)) == 2 * len(inp.predictors)


# --- numbers, worked by hand ------------------------------------------------------------

def test_per_predictor_rows_by_hand(setup):
    inp = _inputs(setup)
    rows = {(r["predictor"], r["code_path"], r["result_class"]): r
            for r in PT.predictor_rows(inp)}
    # xattn recursion 16 s; measured 16.8, 15.2, 16.8.
    x = rows[("recursion", F.CP, "deployable_estimator")]
    errs = [_rel(16, 16.8), _rel(16, 15.2), _rel(16, 16.8)]
    assert x["n"] == 3
    assert x["median_abs_relative"] == pytest.approx(0.8 / 16.8)
    assert x["max_abs_relative"] == pytest.approx(0.8 / 15.2)
    assert x["mean_relative"] == pytest.approx(sum(errs) / 3)
    assert x["n_inside_floor"] == 2              # 4.76% twice, 5.26% once; floor 5%
    # dense recursion d + 2: 6, 14, 32 against 6.2, 15.5, 31.
    d = rows[("recursion", F.CP, "dense")]
    assert d["max_abs_relative"] == pytest.approx(1.5 / 15.5)
    assert d["n_inside_floor"] == 2
    # Two code paths, never pooled: oracle is on CP2 alone.
    o = rows[("recursion", F.CP2, "oracle")]
    assert o["median_abs_relative"] == pytest.approx(0.4 / 8.4)
    assert o["max_abs_relative"] == pytest.approx(3.0 / 5.0)
    assert not any(cp == F.CP2 and cls != "oracle" for _, cp, cls in rows)


def test_residuals_by_synchronisation_count(setup):
    rows = PT.sync_rows(_inputs(setup))
    rec = {(r["code_path"], r["result_class"]): r["n_synchronisations"]
           for r in rows if r["predictor"] == "recursion"}
    # OPS has one .item(); OPS_HOST and the dense ops have none.
    assert rec == {(F.CP, "deployable_estimator"): 1, (F.CP, "dense"): 0,
                   (F.CP2, "oracle"): 0}


def test_ablations_by_hand(setup):
    rows = {(r["key"].arm, r["key"].context_length): r for r in PT.ablation_rows(_inputs(setup))}
    d = rows[("dense", 8)]
    # No synchronisation of its own: both ablations equal the recursion, 14 s.
    assert d["no_sync_cost_s"] == d["no_sync_wait_s"] == d["recursion_s"] == 14
    assert d["host_only_bound_s"] == 1 + 0          # h of the kernel; closing sync h = 0
    x = rows[("xattn", 4)]
    # OPS: removing the .item() wait (s_0 kept) lets the host run ahead, so the
    # no-wait time is below the full recursion; dropping its cost is too.
    assert x["no_sync_wait_s"] < x["recursion_s"]
    assert x["no_sync_cost_s"] < x["recursion_s"]
    assert x["measured_s"] == 16.8
    assert x["recursion_relative"] == pytest.approx(_rel(16, 16.8))
    # oracle at L4 measured 5.0 s: its device bound (l + d = 1 + 1 + ...) is checked.
    o = rows[("oracle", 4)]
    assert o["bound_above_measured"] == [b for b in PT.BOUNDS
                                         if o[f"{b}_s"] > 5.0]


def test_decision_error_by_hand(setup):
    inp = _inputs(setup)
    g = {(g.predictor, g.result_class): g for g in PT.decision(inp)}
    x = g[("recursion", "deployable_estimator")]
    # L4: 16 vs 6 predicted, 16.8 vs 6.2 measured: dense both, correct.
    # L8: measured 15.2 vs 15.5, inside the 0.5 s floor: not scored.
    # L16: 16 vs 32 and 16.8 vs 31: sparse both, correct.
    assert x.scored.decision == {SC.CORRECT: 2, SC.FALSE_PROFITABLE: 0,
                                 SC.MISSED_PROFITABLE: 0}
    assert x.scored.inside_floor == (PI.cell_key(*F.key("xattn", 8)),)
    o = g[("recursion", "oracle")]
    # L4: 8 vs 6 predicted dense; 5.0 vs 6.2 measured sparse, admissible: missed.
    # L8: 8 vs 14 and 8.4 vs 15.5, admissible: correct. L16: inadmissible: correct.
    assert o.scored.decision == {SC.CORRECT: 2, SC.FALSE_PROFITABLE: 0,
                                 SC.MISSED_PROFITABLE: 1}
    assert (o.scored.sign_agree, o.scored.sign_disagree) == (2, 1)


def test_crossover_by_hand(setup):
    inp = _inputs(setup)
    s = {x.series[2]: x for x in PT.crossover(inp)}
    x = s["xattn"]
    # Predicted 16 - (6, 14, 32) = +10, +2, -16: bracket [8, 16].
    assert [(b.lower, b.upper) for b in x.predicted.brackets] == [(8, 16)]
    # Measured +10.6, -0.3 (inside the floor), -14.2: bracket widened to [4, 16].
    assert [(b.lower, b.upper) for b in x.measured.brackets] == [(4, 16)]
    c, = x.error.comparisons
    assert (c.contains, c.overlaps, c.direction_agrees) == (False, True, True)
    assert c.shift_grid_steps == -0.5
    o = s["oracle"]
    # Predicted 8 - (6, 14, 32) = +2, -6, -24: [4, 8]. Measured all negative.
    assert [(b.lower, b.upper) for b in o.predicted.brackets] == [(4, 8)]
    assert o.measured.status == "none in range"
    assert o.error.status == "spurious"


def test_no_crossover_without_intervals_and_the_outputs_say_so(setup):
    cp, _, _, _, tmp = setup
    spec = F.write_spec(tmp / "in", cp, crossover=False)
    inp = PI.build_inputs(PI.read_spec(spec), spec.parent, cp)
    assert PT.crossover(inp) is None
    m = PA.build(inp, tmp / "out", CLEAN_RUN)
    assert m["crossover"].startswith("not computed")
    assert "figures/crossover.pdf" not in m["outputs"]
    assert "tables/crossover.tex" not in m["outputs"]


def test_queue_sensitivity_covers_every_committed_cell(setup):
    inp = _inputs(setup)
    rows = PT.queue_rows(inp)
    assert len(rows) == len(inp.cp.cells) == 9
    for r in rows:
        assert [m for m, _, _ in r["points"]] == [0.25, 0.5, 2, 4, None]
        assert r["measured_q"] == 8
        if r["key"].arm == "dense":
            assert r["flips_at"] is None
        else:
            assert isinstance(r["flips_at"], list)


def test_measured_samples_must_give_the_row_s_statistic(setup):
    cp, _, _, _, tmp = setup
    r = F.result(4)
    r["rows"][0]["prefill_ms_samples"] = [1.0, 2.0, 3.0]
    spec = F.write_spec(tmp / "in", cp, results={4: r})
    with pytest.raises(PI.PaperInputError, match="samples"):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp)


# --- numbers.json and numbers.tex ----------------------------------------------------------

def test_numbers_tex_matches_numbers_json(setup):
    inp = _inputs(setup)
    m = PA.build(inp, setup[4] / "out", CLEAN_RUN)
    out = setup[4] / "out"
    nums = json.loads((out / "numbers.json").read_text())
    tex = (out / "numbers.tex").read_text()
    macros = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", tex))
    assert len(macros) == len(nums["numbers"]) == tex.count("\\newcommand")
    for name, e in nums["numbers"].items():
        assert macros[name] == e["tex"]
        assert re.fullmatch(r"ab[A-Za-z]+", name)
    assert nums["code_paths"] == {"CodePathA": F.CP, "CodePathB": F.CP2} or \
        nums["code_paths"] == {"CodePathA": F.CP2, "CodePathB": F.CP}
    letter = {v: k for k, v in nums["code_paths"].items()}
    n = f"abMedianAbsErrorRecursionMMinusSmall{letter[F.CP]}DeployableEstimator"
    assert nums["numbers"][n]["value"] == pytest.approx(0.8 / 16.8)
    assert macros[n] == "4.76"
    n = f"abDecisionMissedProfitableRecursionMMinusSmall{letter[F.CP2]}Oracle"
    assert macros[n] == "1"
    assert macros["abRelativeNoiseFloor"] == "5.00"
    assert macros["abResolutionFloor"] == "0.5000"
    assert m["floors"] == {"relative": 0.05, "resolution_s": 0.5, "provisional": True}


def test_two_numbers_with_one_name_are_refused(setup):
    N = PN.Numbers(_inputs(setup))
    N.add("cells", 1, "count")
    with pytest.raises(PI.PaperInputError, match="two numbers"):
        N.add("cells", 2, "count")


@pytest.mark.parametrize("part,want", [("sum_plus_s0", "SumPlusSZero"),
                                       ("m-small", "MMinusSmall"), ("0.25x", "ZeroPointTwoFiveX"),
                                       ("no queue", "NoQueue")])
def test_macro_names_are_letters_only(part, want):
    assert PN.camel(part) == want


def test_tex_values():
    assert PN.tex_value(0.047619, "fraction") == "4.76"
    assert PN.tex_value(-0.1, "fraction") == "-10.00"
    assert PN.tex_value(0.5, "seconds") == "0.5000"
    assert PN.tex_value(3, "count") == "3"
    with pytest.raises(PI.PaperInputError):
        PN.tex_value(2.5, "count")


# --- LaTeX and PDF form ---------------------------------------------------------------------

def test_tables_are_booktabs_and_escaped(setup):
    inp = _inputs(setup)
    for t in (PT.tex_cells(inp), PT.tex_predictors(inp), PT.tex_ablations(inp),
              PT.tex_queue(inp), PT.tex_sync(inp), PT.tex_decision(inp, PT.decision(inp)),
              PT.tex_crossover(inp, PT.crossover(inp))):
        assert t.count("\\toprule") == t.count("\\midrule") == t.count("\\bottomrule") == 1
        assert "\\hline" not in t
        body = t.split("\\midrule")[1].split("\\bottomrule")[0]
        assert "|" not in body                   # code path labels are escaped
        ncols = len(re.search(r"\\begin\{tabular\}\{([lrc]+)\}", t).group(1))
        for line in body.strip().splitlines():
            assert line.count(" & ") == ncols - 1
    assert "sum\\_plus\\_s0" in PT.tex_predictors(inp)
    assert PT.tex_escape("a_b|c%") == "a\\_b\\textbar{}c\\%"


def test_figures_embed_truetype_fonts_and_are_reproducible(setup):
    inp = _inputs(setup)
    tmp = setup[4]
    PA.build(inp, tmp / "a", CLEAN_RUN)
    PA.build(inp, tmp / "b", CLEAN_RUN)
    for n in ("error_vs_length", "decision_error", "crossover"):
        a = (tmp / "a" / "figures" / f"{n}.pdf").read_bytes()
        assert a.startswith(b"%PDF")
        assert b"/FontFile2" in a and b"/Type3" not in a
        assert a == (tmp / "b" / "figures" / f"{n}.pdf").read_bytes()


# --- added for mutants that survived the first break test --------------------------------

def test_two_code_paths_of_one_class_are_never_pooled(setup):
    # Bypasses the row check on purpose: both sparse arms in one class, on
    # two code paths, must still give two groups everywhere.
    inp = dataclasses.replace(_inputs(setup), classes={**F.CLASSES,
                                                       "oracle": "deployable_estimator"})
    for rows in (PT.predictor_rows(inp), PT.sync_rows(inp)):
        dep = [r for r in rows if r["predictor"] == "recursion"
               and r["result_class"] == "deployable_estimator"]
        assert sorted(r["code_path"] for r in dep) == sorted([F.CP, F.CP2])
    gs = [g for g in PT.decision(inp) if g.predictor == "recursion"]
    assert sorted(g.code_path for g in gs) == sorted([F.CP, F.CP2])
    assert {g.code_path: g.scored.n_scored for g in gs} == {F.CP: 2, F.CP2: 3}


def test_an_interval_against_another_dense_cell_is_refused(setup):
    cp, _, _, _, tmp = setup
    spec = F.write_spec(tmp / "in", cp)
    d = F.intervals_doc(cp)
    e = d["entries"][0]                               # xattn at L4
    e["dense_cell"] = F.key("dense", 8)
    e["point"] = 16 - 14
    e["lo"], e["hi"] = e["point"] - 0.5, e["point"] + 0.5
    iv = F.write_json(tmp / "in" / "intervals.json", d)
    doc = json.loads(spec.read_text())
    doc["crossover"]["predicted_intervals"]["sha256"] = sha256_file(iv)
    spec.write_text(json.dumps(doc))
    inp = PI.build_inputs(PI.read_spec(spec), spec.parent, cp)
    with pytest.raises(PI.PaperInputError, match="not the measured dense cell"):
        PT.crossover(inp)


def test_errors_inside_the_noise_floor_carry_a_dagger(setup):
    t = PT.tex_cells(_inputs(setup))
    body = t.split("\\midrule")[1].split("\\bottomrule")[0].strip().splitlines()
    rec = {}
    for line in body:
        c = line.split(" & ")
        rec[(c[2], int(c[4]))] = c[11]                # the recursion's error column
    assert rec[("xattn", 4)] == "-4.76\\dag"          # inside 5%
    assert rec[("xattn", 8)] == "+5.26"               # outside
    assert rec[("dense", 8)] == "-9.68"


def test_decision_counts_in_numbers(setup):
    nums = PN.collect(_inputs(setup), PT.decision(_inputs(setup)), None).entries
    N = PN.Numbers(_inputs(setup))
    a, b = N.code_paths[F.CP], N.code_paths[F.CP2]
    assert nums[f"abDecisionCorrectRecursionMMinusSmall{a}DeployableEstimator"]["tex"] == "2"
    assert nums[f"abDecisionInsideFloorRecursionMMinusSmall{a}DeployableEstimator"]["tex"] == "1"
    assert nums[f"abDecisionCorrectRecursionMMinusSmall{b}Oracle"]["tex"] == "2"
    assert nums[f"abSignDisagreesRecursionMMinusSmall{b}Oracle"]["tex"] == "1"
    assert nums[f"abDecisionFalseProfitableRecursionMMinusSmall{b}Oracle"]["tex"] == "0"


# --- builder kind (researcher's decision, 28 Oct 2026, relayed) ---------------------------

def _with_builder_rows(kinds_by_length):
    """Ground truth whose xattn rows carry `builder_kind`, per length."""
    out = {}
    for L in F.LENGTHS:
        r = F.result(L)
        if kinds_by_length.get(L) is not None:
            r["rows"][0]["builder_kind"] = kinds_by_length[L]
        out[L] = r
    return out


def test_a_builder_class_needs_its_kind_stated_while_rows_lack_it(setup):
    cp, _, _, _, tmp = setup
    classes = {**F.CLASSES, "xattn": "gpu_builder"}
    with pytest.raises(PI.PaperInputError, match="builder kind"):
        _inputs(setup, result_classes=classes)
    inp = _inputs(setup, result_classes=classes, builder_kinds={"xattn": "gpu"})
    assert inp.builder("xattn") == "gpu" and inp.builder_sources["xattn"] == ("stated",)
    with pytest.raises(PI.PaperInputError, match="builder kind"):
        _inputs(setup, result_classes=classes, builder_kinds={"xattn": "cpu"})


def test_a_row_s_own_builder_kind_is_read_and_must_agree(setup):
    cp, _, _, _, tmp = setup
    results = _with_builder_rows({4: "cpu", 8: "cpu", 16: "cpu"})
    inp = _inputs(setup, results=results,
                  result_classes={**F.CLASSES, "xattn": "cpu_builder"})
    assert inp.builder("xattn") == "cpu" and inp.builder_sources["xattn"] == ("row",)
    with pytest.raises(PI.PaperInputError, match="spec states"):
        _inputs(setup, results=results, builder_kinds={"xattn": "gpu"})


def test_rows_of_one_arm_with_different_builder_kinds_are_never_pooled(setup):
    with pytest.raises(PI.PaperInputError, match="never pooled"):
        _inputs(setup, results=_with_builder_rows({4: "cpu", 8: "gpu", 16: "cpu"}))
    # A row with a kind and a row with none, nothing stated: refused too.
    with pytest.raises(PI.PaperInputError, match="never pooled"):
        _inputs(setup, results=_with_builder_rows({4: "gpu"}))


@pytest.mark.parametrize("bad", [{"nosuch": "cpu"}, {"xattn": "tpu"}])
def test_a_stated_builder_kind_must_be_a_kind_of_a_committed_arm(setup, bad):
    with pytest.raises(PI.PaperInputError, match="builder kind"):
        _inputs(setup, builder_kinds=bad)


def test_two_builder_kinds_in_one_class_and_code_path_are_kept_apart(tmp_path, monkeypatch):
    # The oracle arm moved onto xattn's code path and made a deployable
    # estimator: the two arms then differ only in where they build masks.
    monkeypatch.setattr(F, "CP2", F.CP)
    (tmp_path / "pred").mkdir()
    cp, *_ = F.loaded(tmp_path / "pred")
    kinds = {**F.KINDS, "oracle": ("inline", True)}
    spec = F.write_spec(tmp_path / "in", cp, crossover=False,
                        results={L: F.result(L, kinds=kinds) for L in F.LENGTHS},
                        result_classes={**F.CLASSES, "oracle": "deployable_estimator"},
                        builder_kinds={"xattn": "gpu", "oracle": "cpu"})
    inp = PI.build_inputs(PI.read_spec(spec), spec.parent, cp)
    for rows in (PT.predictor_rows(inp), PT.sync_rows(inp)):
        dep = [r for r in rows if r["predictor"] == "recursion"
               and r["result_class"] == "deployable_estimator"]
        assert sorted(r["builder_kind"] for r in dep) == ["cpu", "gpu"]
        assert {r["code_path"] for r in dep} == {F.CP}
    gs = [g for g in PT.decision(inp) if g.predictor == "recursion"
          and g.result_class == "deployable_estimator"]
    assert sorted(g.builder_kind for g in gs) == ["cpu", "gpu"]
    nums = PN.collect(inp, PT.decision(inp), None).entries
    a = PN.Numbers(inp).code_paths[F.CP]
    assert nums[f"abMedianAbsErrorRecursionMMinusSmall{a}DeployableEstimatorGpuBuilder"][
        "value"] == pytest.approx(0.8 / 16.8)
    assert nums[f"abMedianAbsErrorRecursionMMinusSmall{a}DeployableEstimatorCpuBuilder"][
        "value"] == pytest.approx(0.4 / 8.4)
    m = PA.build(inp, tmp_path / "out", CLEAN_RUN)
    assert m["builder_kinds"]["oracle"] == {"kind": "cpu", "from": ["stated"]}
    t = (tmp_path / "out" / "tables" / "predictors.tex").read_text()
    assert " & CPU & " in t and " & GPU & " in t


# --- crossover intervals from the committed file -----------------------------------------

@pytest.fixture
def with_committed_intervals(tmp_path):
    (tmp_path / "plain").mkdir()
    plain, *_ = F.loaded(tmp_path / "plain")
    entries = F.intervals_doc(plain)["entries"]
    (tmp_path / "pred").mkdir()
    cp, local, store, remote = F.loaded(tmp_path / "pred", crossover_intervals=entries)
    return cp, local, store, remote, tmp_path


def test_intervals_are_taken_from_the_committed_file_when_it_has_them(with_committed_intervals):
    cp, local, _, _, tmp = with_committed_intervals
    spec = F.write_spec(tmp / "in", cp)
    doc = json.loads(spec.read_text())
    del doc["crossover"]["predicted_intervals"]
    spec.write_text(json.dumps(doc))
    inp = PI.build_inputs(PI.read_spec(spec), spec.parent, cp,
                          committed_bytes=local.read_bytes())
    assert inp.crossover["source"] == "committed prediction"
    assert "crossover_intervals" not in [f["role"] for f in inp.files]
    s = {x.series[2]: x for x in PT.crossover(inp)}
    assert [(b.lower, b.upper) for b in s["xattn"].predicted.brackets] == [(8, 16)]


def test_a_spec_file_beside_committed_intervals_is_refused(with_committed_intervals):
    cp, local, _, _, tmp = with_committed_intervals
    spec = F.write_spec(tmp / "in", cp)
    with pytest.raises(PI.PaperInputError, match="names another file"):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp,
                        committed_bytes=local.read_bytes())


def test_committed_intervals_need_the_measured_protocol(with_committed_intervals):
    cp, local, _, _, tmp = with_committed_intervals
    spec = F.write_spec(tmp / "in", cp, crossover=False)
    with pytest.raises(PI.PaperInputError, match="measured side"):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp,
                        committed_bytes=local.read_bytes())


def test_committed_intervals_are_checked_like_the_spec_s(tmp_path):
    (tmp_path / "plain").mkdir()
    plain, *_ = F.loaded(tmp_path / "plain")
    entries = F.intervals_doc(plain)["entries"]
    entries[1]["point"] += 1.0
    (tmp_path / "pred").mkdir()
    cp, local, *_ = F.loaded(tmp_path / "pred", crossover_intervals=entries)
    spec = F.write_spec(tmp_path / "in", cp)
    doc = json.loads(spec.read_text())
    del doc["crossover"]["predicted_intervals"]
    spec.write_text(json.dumps(doc))
    with pytest.raises(PI.PaperInputError, match="not the committed difference"):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp,
                        committed_bytes=local.read_bytes())


def test_bytes_that_are_not_the_committed_file_are_refused(setup):
    cp, local, _, _, tmp = setup
    spec = F.write_spec(tmp / "in", cp)
    with pytest.raises(PI.PaperInputError, match="are not it"):
        PI.build_inputs(PI.read_spec(spec), spec.parent, cp,
                        committed_bytes=local.read_bytes() + b" ")


def test_the_command_reads_committed_intervals(with_committed_intervals, monkeypatch):
    cp, local, store, remote, tmp = with_committed_intervals
    spec = F.write_spec(tmp / "in", cp)
    doc = json.loads(spec.read_text())
    del doc["crossover"]["predicted_intervals"]
    spec.write_text(json.dumps(doc))
    mod = _script()
    monkeypatch.setattr(mod, "code_stamp", lambda: CLEAN_RUN)
    out = tmp / "out"
    assert mod.main(["--remote", remote, "--local", str(local), "--spec", str(spec),
                     "--clock-skew-s", "0", "--out-dir", str(out)], store=store) == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["crossover_intervals_from"] == "committed prediction"
    assert "figures/crossover.pdf" in m["outputs"]
