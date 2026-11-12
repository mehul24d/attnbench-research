"""The confirmatory analysis's dry run on P1's banked data (T4.3 wave 2a,
adviser's ruling 2a of 11 Nov 2026; `attnbench/analysis/hypotheses.py`).

P1 is the development pairing: nothing here is a test, and every output says
so. The run reads one P1 repeat's committed prediction (the local copy, served
at the sync time its own `predict` step recorded, which its `verify` step
checked against the bucket), its ground truth (step 9's group files) and a
file of working values in the lock's shapes (`docs/t43/dry_run_working_values.json`,
the draft's proposals, never the lock), and writes:

  - H1 read on P1 (`development=True`): the recursion against the rivals the
    file has (P1 is trained on nothing, so no fitted sum or learned
    predictor), the sign and the XAttention pair, by arm class;
  - P1's within-session spread and the marginal labels it and the
    sensitivity floor give;
  - H3's two halves on P1: the measured crossover brackets per estimator,
    each length's difference bootstrapped at the working settings, and the
    predicted ones from the committed points;
  - H0 and H2 of P1 against itself, which must read zero difference and the
    file's own recursion: a check of the plumbing, nothing more.

    python3 scripts/hypotheses.py --local <prediction.json> \\
        --predict-record <session>/07_predict/result.json \\
        --ground-truth <step 9 dir> --working docs/t43/dry_run_working_values.json --out <dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

from attnbench.analysis import gate_a as G  # noqa: E402
from attnbench.analysis import hypotheses as HY  # noqa: E402
from attnbench.lock.floors import FloorTable  # noqa: E402
from attnbench.predict import crossover as X  # noqa: E402
from attnbench.predict import intervals as IV  # noqa: E402

LABEL = "dry run on P1, the development pairing: not a test (T4.3 wave 2a)"


class RecordedStore:
    """The local copy and its sidecar at their bucket paths, created when
    `predict` recorded."""

    def __init__(self, remote: str, local: Path, synced_at: datetime):
        local = Path(local)
        self.objects = {remote: local.read_bytes(),
                        remote + ".sha256": Path(str(local) + ".sha256").read_bytes()}
        self.synced_at = synced_at

    def get(self, remote):
        if remote not in self.objects:
            raise FileNotFoundError(remote)
        return self.objects[remote]

    def exists(self, remote):
        return remote in self.objects

    def created(self, remote):
        self.get(remote)
        return self.synced_at


def working_locked(path: Path) -> HY.Locked:
    w = json.loads(Path(path).read_text())
    b = w["bootstrap_settings"]
    return HY.Locked(tolerances=w["tolerances"],
                     floors=FloorTable.from_ms(w["resolution_floor_ms"], pairing="P1"),
                     bootstrap=IV.Settings(b["level"], b["n_boot"], b["seed"]),
                     h4_bound_pts=w["h4_bound_pts"], h4_tests=tuple(w["h4_tests"]),
                     h4_alpha=w["h4_alpha"], h4_n=w["h4_n"], scored={},
                     sensitivity_s={HY._cell(k): v / 1000.0
                                    for k, v in w["sensitivity_floor_ms"].items()},
                     analysis_commit="working")


def grids(cp, seconds, nominal, *, dense_arm):
    """{estimator: ([sparse keys], [dense keys])} at batch 1, over the lengths
    where both are measured, an estimator being an arm at a nominal density."""
    dense = {(k.model, k.context_length): k for k in seconds if k.arm == dense_arm and k.batch == 1}
    out: dict = {}
    for k in sorted(seconds, key=lambda k: k.context_length):
        if k.arm == dense_arm or k.batch != 1 or (k.model, k.context_length) not in dense:
            continue
        e = f"{k.arm}@{nominal.get(k, 'calibrated')}"
        s, d = out.setdefault(e, ([], []))
        s.append(k)
        d.append(dense[(k.model, k.context_length)])
    return {e: v for e, v in out.items() if len(v[0]) >= 2}


def run(args) -> dict:
    import gate_a as GS                  # scripts/gate_a.py: step 9's group files
    import _load
    rec = json.loads(Path(args.predict_record).read_text())["findings"]
    remote = "gs://dry-run/" + rec["remote_in_bucket"]
    store = RecordedStore(remote, Path(args.local), datetime.fromisoformat(rec["synced_at"]))
    results, starts = GS.ground_truth(Path(args.ground_truth))
    cp = _load.load(remote, args.local, started_at=starts, clock_skew_s=args.clock_skew_s,
                    banked_at=None, store=store)
    L = working_locked(Path(args.working))
    m = G.measured(cp, results)
    seconds, samples, nominal = m["seconds"], {}, {}
    for name, result in results:
        if G.short_of_rounds(result):
            continue
        for row in result["rows"]:
            k = G.place_row(cp, row, name).key
            samples[k] = [x / 1000.0 for x in row["prefill_ms_samples"]]
            # an arm with no nominal density (XAttention: its density is the mask's) is
            # one estimator across lengths
            nominal[k] = row.get("nominal_density") or "calibrated"
    spread = HY.within_session_spread(samples, quantile=args.spread_quantile,
                                      budget_of=L.floors.budget_of)
    h1 = HY.h1(cp, seconds, locked=L, dense_arm=args.dense_arm, spread_s=spread,
               development=True)
    measured, predicted = {}, {}
    pred = {c.key: c.predictions["recursion"] for c in cp.cells}
    for e, (s, d) in grids(cp, seconds, nominal, dense_arm=args.dense_arm).items():
        measured[e] = HY.measured_differences(samples, s, d, statistic=statistics.median,
                                              settings=L.bootstrap)
        predicted[e] = [X.GridDifference(int(a.context_length), pred[a] - pred[b],
                                         pred[a] - pred[b], pred[a] - pred[b])
                        for a, b in zip(s, d)]
    h3 = HY.h3(predicted, measured, floors=L.floors, pairing=cp.pairing)
    h0 = HY.h0(cp, cp, side="host", floors=L.floors, tolerance=L.tolerances["H0"])
    tr, left = HY.transferred(cp, cp)
    plumbing = {"h0_self": {k: h0[k] for k in ("operations_scored", "operations_unscorable",
                                               "verdict")},
                "h0_self_median": h0.get("median_abs_relative_difference"),
                "h2_self_equals_recursion": all(abs(tr[k] - pred[k]) <= 1e-12 * max(1.0, pred[k])
                                                for k in tr),
                "h2_self_left_out": left}
    return {"label": LABEL, "prediction_sha256": cp.sha256, "pairing": cp.pairing,
            "working_values": str(args.working), "excluded_groups": m["excluded"],
            "unmeasured": [list(k) for k in m["unmeasured"]],
            "within_session_spread_ms": {f"{a}x{b}": v * 1000 for (a, b), v in spread.items()},
            "h1": h1, "h3": h3, "plumbing": plumbing}


def to_markdown(rep: dict) -> str:
    h1, h3 = rep["h1"], rep["h3"]
    L = [f"# The confirmatory analysis on P1: {rep['label']}", "",
         f"Prediction `{rep['prediction_sha256'][:12]}`; working values `{rep['working_values']}`.",
         f"Groups left out (short of clean rounds): {len(rep['excluded_groups'])}; "
         f"unmeasured cells: {len(rep['unmeasured'])}.", "",
         "## H1, read on P1 (not a test)", "",
         f"- Recursion median |relative error| {h1['recursion_median_abs_relative']:.4f} over "
         f"{h1['n_cells']} cells (tolerance {h1['tolerance']}): {h1['parts']['tolerance']}.",
         "- Rivals: " + ", ".join(f"{p} {v:.4f} (recursion below: {h1['recursion_below'][p]})"
                                  for p, v in h1["rivals_median_abs_relative"].items())
         + f"; not in the file: {h1['rivals_not_in_file']}.",
         f"- Sign: {h1['sign']['agree']} agree, {len(h1['sign']['disagree'])} disagree, "
         f"{h1['sign']['inside_floor']} inside the floor, {len(h1['sign']['marginal'])} marginal.",
         f"- XAttention pair: {h1['xattention_pair']['agree']} agree, "
         f"{h1['xattention_pair']['disagree']} disagree ({h1['xattention_pair']['verdict']}).",
         f"- As H1's verdict would read: **{h1['verdict']}** (development pairing).", "",
         "## H3's two halves on P1 (not a test)", "",
         "| Estimator | Status | Verdict |", "|---|---|---|"]
    for e, r in h3["estimators"].items():
        L.append(f"| {e} | {r['status']} | {r['verdict']} |")
    L += ["", "## Within-session spread (descriptive)", "",
          ", ".join(f"{k}: {v:.2f} ms" for k, v in rep["within_session_spread_ms"].items()), "",
          "## Plumbing", "", f"H0 of P1 against itself: {rep['plumbing']['h0_self']}; "
          f"H2 of P1 on its own host times equals its recursion: "
          f"{rep['plumbing']['h2_self_equals_recursion']}."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--local", required=True)
    ap.add_argument("--predict-record", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--working", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dense-arm", default="dense")
    ap.add_argument("--clock-skew-s", type=float, default=60.0)
    ap.add_argument("--spread-quantile", type=float, default=0.95)
    args = ap.parse_args(argv)
    try:
        rep = run(args)
    except (HY.HypothesisError, G.GateAError, OSError, KeyError, ValueError) as e:
        print(f"hypotheses: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hypotheses_p1_dry_run.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    (out / "hypotheses_p1_dry_run.md").write_text(to_markdown(rep))
    print(f"wrote {out}/hypotheses_p1_dry_run.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
