"""Projected T3.1 step durations from Shakedown 6, against the session's
budgets (researcher's request, 1 Nov 2026): the expected 5 h and the cost
cap of Rs 760 (the halt, 7 h at the L4's Rs 108.64/h planning rate).

A projection, not a measurement. Shakedown 6 (L4, 1 Nov 2026) timed whole
steps 1 to 4 of the rehearsal and two cells of step 5; steps 6 to 11 have
never run on a GPU. Every number below is one of:

  - an anchor read from Shakedown 6's banked results (`ANCHORS`, each with
    its file; the values are also written here so the projection runs where
    `results/` is absent, and checked against the files where present);
  - a scaling rule from those anchors to T3.1's grid (`configs/t31/
    p1_session.json` and the working values), stated per step;
  - a constant that no session has measured, with a low and a high value
    (`UNMEASURED`), so the projection is a range.

The dense prefill's wall time is fitted as a*L + c*L^2 per sequence from
the session floor's series at 2048 and 16384 tokens, and a batch of b is b
times one sequence; each sparse arm's wall is taken as dense's (central),
or 1.5 times it (high). Step 5 now records each phase's seconds per cell
(`cells_index.json`), so the first T3.1 cells replace these constants.

    python3 scripts/t31_projection.py            # the table
    python3 scripts/t31_projection.py --json     # the same, as JSON
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

S6 = ROOT / "results" / "shakedown6_l4_20261101" / "results"
RATE_INR_H = 108.64
EXPECTED_H = 5.0
CAP_INR = 760.0
LAYERS = 28

# Shakedown 6's measurements (file, value). Seconds unless named.
ANCHORS = {
    "preflight_s": ("t31_rehearsal_20261101T1400Z/manifest.json: preflight", 54.2),
    "link_s": ("t31_rehearsal_20261101T1400Z/manifest.json: link", 159.8),
    "dense_select_s": ("t31_rehearsal_20261101T1400Z/manifest.json: dense_select", 62.2),
    "probe_s": ("t31_rehearsal_20261101T1400Z/manifest.json: probe", 401.5),
    "dense_2048_ms": ("t31_rehearsal_20261101T1400Z/03_dense_select/session_floor.json: "
                      "median of dense L2048 b1", 131.9),
    "dense_16384_ms": ("t31_rehearsal_20261101T1400Z/03_dense_select/session_floor.json: "
                       "median of dense L16384 b1", 1632.0),
    # host operators per prefill (the recording's cpu_op events)
    "ops_dense": ("05_host_components/dense_d1_L2048_b1/recording/trace.json", 4231),
    "ops_xattention_2048": ("s6_diag/s6_diag_fixed/xattention_None_2048/recording/trace.json",
                            19267),
    "ops_oracle_2048": ("05_host_components/oracle_d0.25_L2048_b1/recording/trace.json", 21507),
    "ops_oracle_8192": ("t228_shakedown_20261101T1417Z/09_pricing_zoo/oracle/recording/"
                        "trace.json", 125751),
    # kernels per prefill
    "kernels_dense": ("05_host_components/dense_d1_L2048_b1/recording/sequence.json", 1224),
    "kernels_block_sparse": ("05_host_components/oracle_d0.25_L2048_b1/recording/sequence.json",
                             2204),
    "kernels_xattention": ("s6_diag/s6_diag_fixed/xattention_None_2048/recording/sequence.json",
                           4948),
    "bare_signatures_per_cell": ("s6_diag/s6_diag_fixed/oracle_0.25_2048/bare_prices.json", 71),
}

# Constants no session has measured: (low, central, high).
UNMEASURED = {
    "null_pass_s_per_host_op": (20e-6, 40e-6, 100e-6),   # the dispatch mode, per host operator
    "isolated_s_per_kernel": (0.3e-3, 0.5e-3, 1.0e-3),   # a synchronise and a flush per call
    "trace_export_s_per_op": (10e-6, 20e-6, 50e-6),      # writing and reading a profiler trace
    "bare_s_per_signature_rep": (25e-6, 35e-6, 60e-6),   # synchronise plus the call's host time
    "sparse_wall_over_dense": (0.8, 1.0, 1.5),
    "compile_s_per_arm_cell": (30.0, 60.0, 120.0),        # Dynamo + inductor + CUDA graphs
    "xattention_chunk_tokens": (16384, 16384, 8192),     # kernels grow per chunk
}
SETUP_MIN, TEARDOWN_MIN = 6.0, 8.0                        # Shakedown 6: launch to first step; pull


def check_anchors(root: Path = S6) -> list[str]:
    """Where the banked results are present, the anchors that differ from
    them (by more than 1%); an empty list when they agree or are absent."""
    if not root.exists():
        return []
    bad = []
    man = root / "t31_rehearsal_20261101T1400Z" / "manifest.json"
    if man.exists():
        secs = {a["step"]: a["seconds"] for a in json.loads(man.read_text())["attempts"]}
        for k, step in (("preflight_s", "preflight"), ("link_s", "link"),
                        ("dense_select_s", "dense_select"), ("probe_s", "probe")):
            if abs(secs[step] - ANCHORS[k][1]) > 0.01 * ANCHORS[k][1]:
                bad.append(f"{k}: {ANCHORS[k][1]} here, {secs[step]} banked")
    fl = root / "t31_rehearsal_20261101T1400Z" / "03_dense_select" / "session_floor.json"
    if fl.exists():
        series = json.loads(fl.read_text())["series"]
        for k, s in (("dense_2048_ms", "dense L2048 b1"), ("dense_16384_ms", "dense L16384 b1")):
            got = statistics.median(series[s])
            if abs(got - ANCHORS[k][1]) > 0.01 * ANCHORS[k][1]:
                bad.append(f"{k}: {ANCHORS[k][1]} here, {got:.1f} banked")
    return bad


def a(k):
    return ANCHORS[k][1]


def dense_fit() -> tuple[float, float]:
    """(a, c) of the dense prefill wall a*L + c*L^2, ms, from the two anchors."""
    l1, l2 = 2048, 16384
    w1, w2 = a("dense_2048_ms"), a("dense_16384_ms")
    c = (w2 - w1 * l2 / l1) / (l2 * l2 - l1 * l2)
    return (w1 - c * l1 * l1) / l1, c


def wall_s(L: int, b: int) -> float:
    lin, quad = dense_fit()
    return b * (lin * L + quad * L * L) / 1e3


def oracle_ops(L: int) -> float:
    """The oracle's host operators per prefill, by the power law between
    its 2048 and 8192 recordings (its mask built on the host from scores)."""
    o1, o2 = a("ops_oracle_2048"), a("ops_oracle_8192")
    k = math.log(o2 / o1) / math.log(4)
    return o1 * (L / 2048) ** k


def grid():
    from t31_steps import BATCH1_ONLY, cells, fill_working_values
    cfg = json.loads((ROOT / "configs" / "t31" / "p1_session.json").read_text())
    cfg = fill_working_values(cfg)
    return cfg, cells(cfg), BATCH1_ONLY


def project(which: int) -> dict:
    """Minutes per step for scenario `which` (0 low, 1 central, 2 high)."""
    u = {k: v[which] for k, v in UNMEASURED.items()}
    cfg, cells, batch1 = grid()
    lengths = cfg["lengths"]
    rehearsal_b1 = [2048, 16384]
    by_lb: dict = {}
    for arm, d, L, b in cells:
        by_lb.setdefault((L, b), []).append(arm)

    def arm_wall(arm, L, b):
        return wall_s(L, b) * (1.0 if arm == "dense" else u["sparse_wall_over_dense"])

    def host_ops(arm, L, b):
        if arm == "oracle":
            return oracle_ops(L) * b
        if arm.startswith("xattention"):
            return a("ops_xattention_2048") * math.ceil(L / u["xattention_chunk_tokens"])
        return a("ops_dense") * (1.4 if arm == "meanpool" else 1.0)

    def kernels(arm, L):
        if arm.startswith("xattention"):
            return a("kernels_xattention") * math.ceil(L / u["xattention_chunk_tokens"])
        return a("kernels_dense") if arm == "dense" else a("kernels_block_sparse")

    sum_b1 = lambda Ls: sum(wall_s(L, 1) for L in Ls)                    # noqa: E731
    m = {}
    m["1 preflight"] = a("preflight_s") / 60
    # the mechanism table is per length; the rest of the step is fixed
    m["2 link"] = a("link_s") / 60 * (0.5 + 0.5 * len(lengths) / len(rehearsal_b1))
    sel = [wall_s(L, b) for (L, b) in by_lb]
    m["3 dense_select"] = a("dense_select_s") / 60 * sum(sel) / (
        wall_s(2048, 1) + wall_s(2048, 4) + wall_s(16384, 1))
    m["4 probe"] = a("probe_s") / 60 * (0.4 + 0.6 * sum_b1(lengths) / sum_b1(rehearsal_b1))
    phase = {"record": 0.0, "isolated replay": 0.0, "null passes": 0.0, "bare pass": 0.0,
             "residual": 0.0}
    shared = set()
    for arm, d, L, b in cells:
        w, ops = arm_wall(arm, L, b), host_ops(arm, L, b)
        phase["record"] += 3 * w + ops * u["trace_export_s_per_op"]
        phase["isolated replay"] += (kernels(arm, L) * b * u["isolated_s_per_kernel"] + w
                                     + 2 * ops * u["trace_export_s_per_op"])
        phase["null passes"] += 6 * ops * u["null_pass_s_per_host_op"]
        # each signature timed once per process: one layer's device work, once
        # per (length, batch) for what the arms share, once per cell for the rest
        layer = w / LAYERS
        phase["bare pass"] += 1001 * (layer * (0.5 if (L, b) in shared else 1.0)
                                      + a("bare_signatures_per_cell")
                                      * u["bare_s_per_signature_rep"])
        shared.add((L, b))
        phase["residual"] += 7 * w
    m["5 host_components"] = sum(phase.values()) / 60
    dev = 0.0
    for arm, d, L, b in cells:
        w = arm_wall(arm, L, b)
        dev += 4 * w + 10 + (0 if which < 2 else
                             kernels(arm, L) * b * u["isolated_s_per_kernel"] + w)
    m["6 device_components"] = dev / 60
    m["7 predict"] = 1.0
    m["8 hold"] = 1.5
    gt = 0.0
    rounds = cfg["reps"] + cfg.get("warmup", 3) + 1        # clean rounds, warm-up, the twin
    for group, arms in cfg["gt_groups"].items():
        for (L, b), cell_arms in by_lb.items():
            present = [x for x in cell_arms if x in arms or x == "dense"]
            n = sum(len(cfg["densities"][x]) for x in set(present))
            gt += rounds * sum(arm_wall(x, L, b) for x in present for _ in cfg["densities"][x])
            gt += cfg["dense_rounds"] * wall_s(L, b)
            if group != "eager":
                gt += (n - 1) * u["compile_s_per_arm_cell"]
    m["9 ground_truth"] = gt / 60
    m["10 verify"] = 1.0
    m["11 postflight"] = 2.0
    return {"steps": m, "step5_phases": {k: v / 60 for k, v in phase.items()},
            "cells": len(cells)}


def table() -> dict:
    runs = [project(i) for i in range(3)]
    steps = list(runs[0]["steps"])
    rows = [(s, *(r["steps"][s] for r in runs)) for s in steps]
    tot = [SETUP_MIN + TEARDOWN_MIN + sum(r["steps"].values()) for r in runs]
    return {"rows": rows, "phases": [(p, *(r["step5_phases"][p] for r in runs))
                                     for p in runs[0]["step5_phases"]],
            "total_min": tot, "total_inr": [t / 60 * RATE_INR_H for t in tot],
            "cells": runs[0]["cells"], "setup_min": SETUP_MIN, "teardown_min": TEARDOWN_MIN,
            "expected_h": EXPECTED_H, "cap_inr": CAP_INR, "rate_inr_h": RATE_INR_H,
            "halt_h": CAP_INR / RATE_INR_H}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    bad = check_anchors()
    if bad:
        print("REFUSING: the anchors differ from the banked results: " + "; ".join(bad),
              file=sys.stderr)
        return 2
    t = table()
    if args.json:
        print(json.dumps(t, indent=1))
        return 0
    print(f"T3.1 projected from Shakedown 6 ({t['cells']} cells; minutes, low / central / high)")
    print(f"{'setup':24s} {t['setup_min']:7.1f}")
    for s, lo, mid, hi in t["rows"]:
        print(f"{s:24s} {lo:7.1f} {mid:7.1f} {hi:7.1f}")
        if s.startswith("5 "):
            for p, plo, pmid, phi in t["phases"]:
                print(f"    {p:20s} {plo:7.1f} {pmid:7.1f} {phi:7.1f}")
    print(f"{'teardown':24s} {t['teardown_min']:7.1f}")
    lo, mid, hi = t["total_min"]
    print(f"{'total':24s} {lo:7.1f} {mid:7.1f} {hi:7.1f}   "
          f"({lo / 60:.1f} / {mid / 60:.1f} / {hi / 60:.1f} h)")
    rl, rm, rh = t["total_inr"]
    print(f"cost at Rs {RATE_INR_H}/h: Rs {rl:.0f} / {rm:.0f} / {rh:.0f}")
    print(f"against the expected {EXPECTED_H:.0f} h: "
          + " / ".join("within" if m <= EXPECTED_H * 60 else "over" for m in t["total_min"]))
    print(f"against the cap Rs {CAP_INR:.0f} (halt at {t['halt_h']:.1f} h): "
          + " / ".join("within" if r <= CAP_INR else "over" for r in t["total_inr"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
