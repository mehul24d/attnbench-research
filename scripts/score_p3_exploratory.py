#!/usr/bin/env python3
"""P3's 22 Nov 2026 prediction against P3's ground truth: EXPLORATORY (the
researcher's post-run task of 26 Nov 2026, item 6). It scores no hypothesis.

    python3 scripts/score_p3_exploratory.py --inputs <16 Nov inputs.json> \\
        --mirror <16 Nov mirror> --session <P3 session> --out <dir>

`scripts/score_hypotheses.py` scores H0 to H3 on P2 and P4 only and is not
changed; its own functions are called here, unchanged, on P3's data:

- the lock at the scorer's tag (`lock_at`, `locked_from`), the analysis-commit
  check (`at_analysis_commit`) on every module this run imports but this file
  (as the scorer leaves itself out);
- P1's committed T3.2d prediction through `load` (`commit.load_committed`) on
  the 16 Nov 2026 mirror (`MirrorStore`, offline: nothing reads the bucket).
  The 22 Nov 2026 P3 file is the full transfer from P1, so its cells must equal
  P1's committed values one for one; this is checked, with its sha256 against
  its sidecar, before anything is measured;
- P3's ground truth (`09_ground_truth`, `gate_a.ground_truth`) through
  `measure`, with the lock's minimum clean rounds;
- the residuals of every predictor in the file (`residuals.residuals`, as H1
  calls it: relative floor 0, the dense arm `dense`), their median absolute
  relative error (`hypotheses._median_abs`), overall and by arm class.

Not H1: H1 refuses a P1 file other than as a dry run, and needs rivals the
transfer has not got. `passed_sessions` is not used: P3's session has no
late-check step; the manifest's last attempt of every step is checked here.

The dense-backend check: the prediction prices `sdpa_flash` for every dense
cell; P3's dense selection (`03_dense_select`) chose per cell. Each dense cell
is listed with the kernel chosen, and the dense residuals are split by whether
it is `sdpa_flash` (disclosed, Appendix A draft of 26 Nov 2026; not rescored).
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
PREDICTION = ROOT / "docs/p3_prediction/prediction_P3_exploratory_from_P1_20261122T211249520458Z.json"
PREDICTION_SHA256 = "72b907cf4605336bba91ef3f04f8503945e49dabef207d4e3c4fc7b26183e1b2"
PREDICTORS = ("recursion", "step_form", "sum", "sum_plus_s0")
PRICED_DENSE = "sdpa_flash"


class P3ScoreError(ValueError):
    pass


def check_prediction(path: Path) -> dict:
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    side = Path(str(path) + ".sha256").read_text().split()[0]
    if sha != side or sha != PREDICTION_SHA256:
        raise P3ScoreError(f"{path.name}: sha256 {sha}, sidecar {side}, committed {PREDICTION_SHA256}")
    rec = json.loads(raw)
    if rec.get("label") != LABEL or rec.get("pairing") != "P3":
        raise P3ScoreError(f"{path.name} is not P3's {LABEL} file")
    return rec


def same_as_p1(rec: dict, cp1) -> int:
    """Every P3 cell's predictions equal P1's committed ones for that cell."""
    p1 = {(c.key.arm, float(c.key.density), int(c.key.context_length), int(c.key.batch)): c.predictions
          for c in cp1.cells}
    n = 0
    for c in rec["cells"]:
        k = (c["arm"], float(c["density"]), int(c["context_length"]), int(c["batch"]))
        if k not in p1:
            raise P3ScoreError(f"P3 cell {k} is not one of P1's")
        for p, v in c["predictions_s"].items():
            if p1[k].get(p) != v:
                raise P3ScoreError(f"P3 cell {k} {p}: {v} is not P1's {p1[k].get(p)}")
        n += 1
    if n != len(p1):
        raise P3ScoreError(f"P3 has {n} cells, P1 {len(p1)}")
    return n


def session_passed(session: Path) -> None:
    m = json.loads((session / "manifest.json").read_text())
    if m.get("dry_run") or m.get("rehearsal"):
        raise P3ScoreError(f"{session} is a dry run's or a rehearsal's")
    last = {}
    for a in m.get("attempts") or ():
        last[a["step"]] = a.get("status")
    bad = [s for s, v in last.items() if v != "passed"] + (["ground_truth"] if "ground_truth" not in last else [])
    if bad:
        raise P3ScoreError(f"{session}: steps not passed: {bad}")


def dense_choices(session: Path) -> dict:
    out = {}
    for f in sorted((session / "03_dense_select").glob("dense_choice_*.json")):
        d = json.loads(f.read_text())
        _, _, length, batch = f.stem.rsplit("_", 3)
        out[(int(length), int(batch))] = d["chosen"]
    return out


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
    cp1, read1 = SH.load(store, prefix, "P1", spec["P1"], [], banked=banked, rows=[], files={})
    n_same = same_as_p1(rec, cp1)
    mcr = int(tagged["values"]["rerun_budget"]["value"]["min_clean_rounds"])
    results, _starts = GA.ground_truth(session / "09_ground_truth")
    ms = SH.measure(cp1, [(session.name, results)], min_clean_rounds=mcr)
    rs = R.residuals(cp1, ms["seconds"], statistic=cp1.statistic, dense_arms=["dense"],
                     relative_floor=0.0, predictors=list(PREDICTORS))
    res = rs.residuals
    overall = {p: HY._median_abs(res, p) for p in PREDICTORS}
    by_class = {}
    for cls in (R.DENSE, R.SYNC_FREE, R.SYNCHRONISING):
        xs = [r for r in res if r.arm_class == cls]
        if xs:
            by_class[cls] = {"n": sum(r.predictor == "recursion" for r in xs),
                             **{p: HY._median_abs(xs, p) for p in PREDICTORS}}
    chosen = dense_choices(session)
    dense_rows, split = [], {"priced_kernel": [], "other_kernel": []}
    for r in sorted((r for r in res if r.predictor == "recursion" and r.arm_class == R.DENSE),
                    key=lambda r: (r.context_length, r.batch)):
        budget = budget_of(r.context_length)
        k = chosen.get((int(budget), int(r.batch)))
        dense_rows.append({"context_length": r.context_length, "batch": r.batch, "budget": budget,
                           "chosen_kernel": k, "priced_kernel": PRICED_DENSE,
                           "predicted_s": r.predicted, "measured_s": r.measured,
                           "relative": r.relative})
        split["priced_kernel" if k == PRICED_DENSE else "other_kernel"].append(r)
    dense_split = {g: {"n": len(xs), "recursion_median_abs_relative": HY._median_abs(xs, "recursion")}
                   for g, xs in split.items()}
    cells = [{"arm": r.arm, "density": r.density, "context_length": r.context_length, "batch": r.batch,
              "arm_class": r.arm_class, "predicted_s": r.predicted, "measured_s": r.measured,
              "relative": r.relative, "inside_floor": r.inside_floor}
             for r in sorted(res, key=lambda r: (r.arm, r.density, r.context_length, r.batch))
             if r.predictor == "recursion"]
    return {"label": LABEL, "scores_no_hypothesis": True, "pairing": "P3",
            "prediction": {"file": str(Path(args.prediction).relative_to(ROOT)) if Path(args.prediction).is_relative_to(ROOT) else args.prediction,
                           "sha256": PREDICTION_SHA256, "cells_equal_to_p1": n_same},
            "source": read1, "lock_tag": SH.TAG,
            # every imported module of this repo at the analysis commit, this file left out
            # as the scorer leaves itself out
            "analysis": SH.at_analysis_commit(locked, modules={k: v for k, v in sys.modules.items()
                                                               if k != "__main__"}),
            "measured": {"cells": len(ms["seconds"]), "excluded": ms["excluded"],
                         "unmeasured": [list(k) for k in ms["unmeasured"]], "min_clean_rounds": mcr},
            "median_abs_relative": overall, "by_arm_class": by_class,
            "dense_backend": {"rows": dense_rows, "split": dense_split,
                              "choices": {f"{a}x{b}": k for (a, b), k in sorted(chosen.items())}},
            "cells": cells}


def _f(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def to_markdown(rep: dict) -> str:
    L = [f"# P3 prediction against P3's ground truth — **{rep['label']}**", "",
         "Scores no hypothesis. The 22 Nov 2026 file (full transfer from P1, sha256 "
         f"`{rep['prediction']['sha256'][:12]}…`) equals P1's committed values in all "
         f"{rep['prediction']['cells_equal_to_p1']} cells. Measured cells: {rep['measured']['cells']}; "
         f"excluded {len(rep['measured']['excluded'])}, unmeasured {len(rep['measured']['unmeasured'])}.", "",
         "| predictor | median abs relative error |", "|---|---|"]
    for p, v in rep["median_abs_relative"].items():
        L.append(f"| {p} | {_f(v)} |")
    L += ["", "| arm class | n | recursion | step_form | sum | sum_plus_s0 |", "|---|---|---|---|---|---|"]
    for c, v in rep["by_arm_class"].items():
        L.append(f"| {c} | {v['n']} | " + " | ".join(_f(v[p]) for p in PREDICTORS) + " |")
    L += ["", f"Dense backend: priced `{PRICED_DENSE}`; P3's dense selection chose "
          + ", ".join(f"{k} at {c}" for c, k in rep["dense_backend"]["choices"].items()) + ".", "",
          "| length | batch | chosen | predicted s | measured s | relative |", "|---|---|---|---|---|---|"]
    for r in rep["dense_backend"]["rows"]:
        L.append(f"| {r['context_length']} | {r['batch']} | {r['chosen_kernel']} | {_f(r['predicted_s'])} | "
                 f"{_f(r['measured_s'])} | {_f(r['relative'], 3)} |")
    s = rep["dense_backend"]["split"]
    L += ["", f"Dense recursion, median abs relative: kernel as priced {_f(s['priced_kernel']['recursion_median_abs_relative'])} "
          f"(n {s['priced_kernel']['n']}), another kernel {_f(s['other_kernel']['recursion_median_abs_relative'])} "
          f"(n {s['other_kernel']['n']}).", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--prediction", default=str(PREDICTION))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        rep = run(args)
    except (P3ScoreError, SH.ScoreError, HY.HypothesisError, OSError, KeyError, ValueError) as e:
        print(f"score_p3_exploratory: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p3_exploratory_scored.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    (out / "p3_exploratory_scored.md").write_text(to_markdown(rep))
    print(f"wrote {out}/p3_exploratory_scored.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
