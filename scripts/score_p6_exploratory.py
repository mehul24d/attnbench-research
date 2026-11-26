#!/usr/bin/env python3
"""P6's prediction against P6's ground truth: EXPLORATORY (the researcher's task
of 26 Nov 2026 after P3's push, item 2b). It scores no hypothesis. Built as
`scripts/score_p3_exploratory.py` (26 Nov 2026), for P6 against P4:

    python3 scripts/score_p6_exploratory.py --inputs <16 Nov inputs.json> \\
        --mirror <16 Nov mirror> --session <P6 session> --out <dir>

`scripts/score_hypotheses.py` is not changed; its own functions are called here,
unchanged, on P6's data: the lock at the scorer's tag (`lock_at`, `locked_from`),
the analysis-commit check (`at_analysis_commit`) on every module this run imports
but this file; P4's committed T6.4 prediction through `load` on the 16 Nov 2026
mirror (`MirrorStore`, offline); P6's ground truth (`09_ground_truth`,
`gate_a.ground_truth`) through `measure`, with the lock's minimum clean rounds;
the residuals of every predictor in the file (`residuals.residuals`, relative
floor 0, the dense arm `dense`) and their median absolute relative error
(`hypotheses._median_abs`), overall and by arm class.

The committed P6 file (`docs/p6_prediction/`, the same-GPU mirror of P4) is
checked against its sha256 and its sidecar, and cell for cell against P4's
committed values, before anything is measured; every step of the session must
have passed. (This file does not import `score_p3_exploratory.py`, whose module
is not at the analysis commit: its two session readers are repeated below.)

The backend split (item 2b): the prediction prices `sdpa_flash` for every dense
cell (`shakedown_arms.ARMS["dense"]`); P6's dense selection (`03_dense_select`)
chose per cell. Each dense cell is *matched* when the kernel chosen is the one
priced and *mismatched* otherwise; every predictor's error is reported for each
group. P4 itself chose `fa3` at all six dense cells (16 Nov 2026), so on P4
every dense cell was mismatched. The image's own backends are block 1's record
(`b1_backends.json`, `scripts/p6_backend_check.py`), listed when given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import score_hypotheses as SH  # noqa: E402  (unchanged)
import gate_a as GA  # noqa: E402
from attnbench.analysis import hypotheses as HY  # noqa: E402
from attnbench.predict import residuals as R  # noqa: E402
from attnbench.predict.ledger import Ledger, budget_of  # noqa: E402

LABEL = "EXPLORATORY"
PREDICTION = ROOT / "docs/p6_prediction/prediction_P6_exploratory_from_P4_20261126T153102766985Z.json"
PREDICTION_SHA256 = "11f2361fcf508cb75c84c7f9a6d12a914b9130ff7ff1a0a319f20e5ecd2da843"
PRICED_DENSE = "sdpa_flash"


class P6ScoreError(ValueError):
    pass


def check_prediction(path: Path) -> dict:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    side = Path(str(path) + ".sha256").read_text().split()[0]
    if sha != side or sha != PREDICTION_SHA256:
        raise P6ScoreError(f"{path.name}: sha256 {sha}, sidecar {side}, committed {PREDICTION_SHA256}")
    rec = json.loads(raw)
    if rec.get("label") != LABEL or rec.get("pairing") != "P6" or (rec.get("source") or {}).get("pairing") != "P4":
        raise P6ScoreError(f"{path.name} is not P6's {LABEL} file from P4")
    return rec


def same_as_p4(rec: dict, cp4) -> int:
    """Every P6 cell's predictions equal P4's committed ones for that cell."""
    p4 = {(c.key.arm, float(c.key.density), int(c.key.context_length), int(c.key.batch)): c.predictions
          for c in cp4.cells}
    n = 0
    for c in rec["cells"]:
        k = (c["arm"], float(c["density"]), int(c["context_length"]), int(c["batch"]))
        if k not in p4:
            raise P6ScoreError(f"P6 cell {k} is not one of P4's")
        if c["predictions_s"] != {p: p4[k].get(p) for p in c["predictions_s"]} or set(c["predictions_s"]) != set(p4[k]):
            raise P6ScoreError(f"P6 cell {k} is not P4's committed values")
        n += 1
    if n != len(p4):
        raise P6ScoreError(f"P6 has {n} cells, P4 {len(p4)}")
    return n


def session_passed(session: Path) -> None:      # as score_p3_exploratory.session_passed
    m = json.loads((session / "manifest.json").read_text())
    if m.get("dry_run") or m.get("rehearsal"):
        raise P6ScoreError(f"{session} is a dry run's or a rehearsal's")
    last = {}
    for a in m.get("attempts") or ():
        last[a["step"]] = a.get("status")
    bad = [s for s, v in last.items() if v != "passed"] + (["ground_truth"] if "ground_truth" not in last else [])
    if bad:
        raise P6ScoreError(f"{session}: steps not passed: {bad}")


def dense_choices(session: Path) -> dict:       # as score_p3_exploratory.dense_choices
    out = {}
    for f in sorted((session / "03_dense_select").glob("dense_choice_*.json")):
        d = json.loads(f.read_text())
        _, _, length, batch = f.stem.rsplit("_", 3)
        out[(int(length), int(batch))] = d["chosen"]
    return out


def backend_split(res, chosen: dict, predictors) -> dict:
    """Dense residuals by whether the chosen kernel is the priced one: for each
    group, its cells and every predictor's median absolute relative error."""
    groups = {"matched": [], "mismatched": []}
    rows = []
    for r in sorted((r for r in res if r.arm_class == R.DENSE), key=lambda r: (r.context_length, r.batch, r.predictor)):
        budget = budget_of(r.context_length)
        k = chosen.get((int(budget), int(r.batch)))
        g = "matched" if k == PRICED_DENSE else "mismatched"
        groups[g].append(r)
        if r.predictor == "recursion":
            rows.append({"context_length": r.context_length, "batch": r.batch, "budget": budget,
                         "chosen_kernel": k, "priced_kernel": PRICED_DENSE, "group": g,
                         "predicted_s": r.predicted, "measured_s": r.measured, "relative": r.relative})
    split = {g: {"cells": sum(r.predictor == "recursion" for r in xs),
                 "median_abs_relative": {p: (HY._median_abs(xs, p) if any(r.predictor == p for r in xs) else None)
                                         for p in predictors}}
             for g, xs in groups.items()}
    return {"rows": rows, "split": split}


def load_p4(store, prefix: str, spec: dict, tagged: dict, banked: dict) -> tuple:
    """P4's committed prediction as the scorer loads it (`score_hypotheses.main`):
    its training rows and files (`training_inputs`), its ground-truth starts from
    its passed sessions (`passed_sessions`), then `load`."""
    rows, files = SH.training_inputs(spec, tagged, banked)
    _blocks, starts, _left = SH.passed_sessions(spec["P4"]["sessions"], spec["P4"]["ground_truth_dir"])
    return SH.load(store, prefix, "P4", spec["P4"], starts, banked=banked, rows=rows, files=files)


def run(args) -> dict:
    spec = json.loads(Path(args.inputs).read_text())
    session = Path(args.session)
    rec = check_prediction(Path(args.prediction))
    session_passed(session)
    tagged = SH.lock_at(SH.TAG)
    locked = SH.locked_from(tagged)
    store = SH.MirrorStore(Path(args.mirror))
    prefix = spec["bucket"].rstrip("/") + "/"
    banked = {}
    for p in ("P1", "P2", "P4"):
        b = Ledger(store, prefix, p).banked()
        if b is not None:
            banked[p] = b
    cp4, read4 = load_p4(store, prefix, spec, tagged, banked)
    n_same = same_as_p4(rec, cp4)
    predictors = sorted({p for c in rec["cells"] for p in c["predictions_s"]})
    mcr = int(tagged["values"]["rerun_budget"]["value"]["min_clean_rounds"])
    results, _starts = GA.ground_truth(session / "09_ground_truth")
    ms = SH.measure(cp4, [(session.name, results)], min_clean_rounds=mcr)
    rs = R.residuals(cp4, ms["seconds"], statistic=cp4.statistic, dense_arms=["dense"],
                     relative_floor=0.0, predictors=predictors)
    res = rs.residuals
    overall = {p: HY._median_abs(res, p) for p in predictors}
    by_class = {}
    for cls in (R.DENSE, R.SYNC_FREE, R.SYNCHRONISING):
        xs = [r for r in res if r.arm_class == cls]
        if xs:
            by_class[cls] = {"n": sum(r.predictor == "recursion" for r in xs),
                             **{p: HY._median_abs(xs, p) for p in predictors}}
    chosen = dense_choices(session)
    dense = backend_split(res, chosen, predictors)
    backends = None
    if args.backends:
        b = json.loads(Path(args.backends).read_text())
        backends = {k: b.get(k) for k in ("gpu", "distributions", "archs", "supplied", "required_correct", "passed")}
    cells = [{"arm": r.arm, "density": r.density, "context_length": r.context_length, "batch": r.batch,
              "arm_class": r.arm_class, "predicted_s": r.predicted, "measured_s": r.measured,
              "relative": r.relative, "inside_floor": r.inside_floor}
             for r in sorted(res, key=lambda r: (r.arm, r.density, r.context_length, r.batch))
             if r.predictor == "recursion"]
    return {"label": LABEL, "scores_no_hypothesis": True, "pairing": "P6",
            "prediction": {"file": Path(args.prediction).name, "sha256": PREDICTION_SHA256,
                           "cells_equal_to_p4": n_same, "predictors": predictors},
            "source": read4, "lock_tag": SH.TAG,
            # every imported module of this repo at the analysis commit, this file left out
            "analysis": SH.at_analysis_commit(locked, modules={k: v for k, v in sys.modules.items()
                                                               if k != "__main__"}),
            "measured": {"cells": len(ms["seconds"]), "excluded": ms["excluded"],
                         "unmeasured": [list(k) for k in ms["unmeasured"]], "min_clean_rounds": mcr},
            "median_abs_relative": overall, "by_arm_class": by_class,
            "dense_backend": {**dense, "priced_kernel": PRICED_DENSE,
                              "choices": {f"{a}x{b}": k for (a, b), k in sorted(chosen.items())},
                              "image": backends},
            "cells": cells}


def _f(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def to_markdown(rep: dict) -> str:
    preds = rep["prediction"]["predictors"]
    L = [f"# P6 prediction against P6's ground truth — **{rep['label']}**", "",
         "Scores no hypothesis. The committed P6 file (the same-GPU mirror of P4, sha256 "
         f"`{rep['prediction']['sha256'][:12]}…`) equals P4's committed values in all "
         f"{rep['prediction']['cells_equal_to_p4']} cells. Measured cells: {rep['measured']['cells']}; "
         f"excluded {len(rep['measured']['excluded'])}, unmeasured {len(rep['measured']['unmeasured'])}.", "",
         "| predictor | median abs relative error |", "|---|---|"]
    L += [f"| {p} | {_f(v)} |" for p, v in rep["median_abs_relative"].items()]
    d = rep["dense_backend"]
    L += ["", f"Dense backend: priced `{d['priced_kernel']}`; P6's dense selection chose "
          + ", ".join(f"{k} at {c}" for c, k in d["choices"].items()) + ".", "",
          "| group | dense cells | " + " | ".join(preds) + " |", "|---|---|" + "---|" * len(preds)]
    for g, v in d["split"].items():
        L.append(f"| {g} | {v['cells']} | " + " | ".join(_f(v["median_abs_relative"][p]) for p in preds) + " |")
    L += ["", "| length | batch | chosen | group | predicted s | measured s | relative |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['context_length']} | {r['batch']} | {r['chosen_kernel']} | {r['group']} | {_f(r['predicted_s'])} | "
          f"{_f(r['measured_s'])} | {_f(r['relative'], 3)} |" for r in d["rows"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--prediction", default=str(PREDICTION))
    ap.add_argument("--backends", default=None, help="block 1's b1_backends.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        rep = run(args)
    except (P6ScoreError, SH.ScoreError, HY.HypothesisError, OSError, KeyError, ValueError) as e:
        print(f"score_p6_exploratory: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p6_exploratory_scored.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    (out / "p6_exploratory_scored.md").write_text(to_markdown(rep))
    print(f"wrote {out}/p6_exploratory_scored.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
