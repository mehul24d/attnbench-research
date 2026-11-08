"""Projected T3.1 step durations from Shakedown 6, against the session's
budgets (researcher's request, 1 Nov 2026): the expected 5 h and the cost
cap of Rs 760 (the halt, 7 h at the L4's Rs 108.64/h planning rate).

A projection, not a measurement. Shakedown 6 (L4, 1 Nov 2026) timed whole
steps 1 to 4 of the rehearsal and two cells of step 5. T3.1's third session
(L4, 2 Nov 2026, `results/t31_p1_s3`) ran steps 1 to 4 on the whole grid and
step 5 for 93 minutes, 31 of its 36 cells and part of the 32nd, before the
out-of-memory kill: steps 1 to 5 are projected from it (`S3_ANCHORS`), the
rest from Shakedown 6. Steps 6 to 11 have never run on a GPU. Every number
below is one of:

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
S3 = ROOT / "results" / "t31_p1_s3" / "results" / "t31_p1_20261102T0643Z"
RATE_INR_H = 108.64
EXPECTED_H = 5.0
CAP_INR = 760.0
LAYERS = 28

# Shakedown 6's measurements (file, value). Seconds unless named.
ANCHORS = {
    "dense_2048_ms": ("t31_rehearsal_20261101T1400Z/03_dense_select/session_floor.json: "
                      "median of dense L2048 b1", 131.9),
    "dense_16384_ms": ("t31_rehearsal_20261101T1400Z/03_dense_select/session_floor.json: "
                       "median of dense L16384 b1", 1632.0),
    # kernels per prefill
    "kernels_dense": ("05_host_components/dense_d1_L2048_b1/recording/sequence.json", 1224),
    "kernels_block_sparse": ("05_host_components/oracle_d0.25_L2048_b1/recording/sequence.json",
                             2204),
    "kernels_xattention": ("s6_diag/s6_diag_fixed/xattention_None_2048/recording/sequence.json",
                           4948),
}

# T3.1's third session (manifest.json, seconds per step): steps 1 to 4 whole;
# step 5 until the kill, in its 32nd cell (`oracle_d0.5_L32768`), after 31.
S3_ANCHORS = {
    "preflight_s": ("manifest.json: preflight", 94.6),
    "link_s": ("manifest.json: link", 162.7),
    "dense_select_s": ("manifest.json: dense_select", 214.0),
    "probe_s": ("manifest.json: probe", 834.8),
    "host_components_s": ("manifest.json: host_components, killed in its 32nd cell", 5596.4),
}

# Constants no session has measured: (low, central, high).
UNMEASURED = {
    "isolated_s_per_kernel": (0.3e-3, 0.5e-3, 1.0e-3),   # a synchronise and a flush per call
    # step 5 after the third session's kill point (ruling, 2 Nov 2026): the rest
    # of `oracle_d0.5_L32768` (its isolated trace's export and compaction, the
    # null and bare passes, the residual; it had run 24 minutes), each of the
    # four mean-pool and XAttention cells at 32768, and a fresh process per cell
    # (imports, CUDA, the model load)
    "step5_rest_of_oracle_32768_min": (10.0, 20.0, 35.0),
    "step5_small_32768_cell_min": (1.0, 2.0, 3.0),
    "cell_process_start_s": (20.0, 30.0, 50.0),
    "sparse_wall_over_dense": (0.8, 1.0, 1.5),
    "compile_s_per_arm_cell": (30.0, 60.0, 120.0),        # Dynamo + inductor + CUDA graphs
    "xattention_chunk_tokens": (16384, 16384, 8192),     # kernels grow per chunk
    # step 12, Gate A's leg 3 (8 Nov 2026): the dispatch observer's wall over the
    # bare prefill (the null pass's 2.1 to 2.5 on S4's leaf calls, rounded out), the
    # largest injection's own time per pass (idle 4000 at up to 13 us; busy 64 at
    # about 0.28 ms each, predicted), and the process, model load, calibration and plan
    "observer_wall_factor": (1.5, 2.5, 3.5),
    "leg3_injection_s_per_pass": (0.01, 0.02, 0.05),
    "leg3_setup_s": (60.0, 120.0, 240.0),
}
SETUP_MIN, TEARDOWN_MIN = 6.0, 8.0                        # Shakedown 6: launch to first step; pull


def check_anchors(root: Path = S6) -> list[str]:
    """Where the banked results are present, the anchors that differ from
    them (by more than 1%); an empty list when they agree or are absent."""
    if not root.exists():
        return []
    bad = []
    fl = root / "t31_rehearsal_20261101T1400Z" / "03_dense_select" / "session_floor.json"
    if fl.exists():
        series = json.loads(fl.read_text())["series"]
        for k, s in (("dense_2048_ms", "dense L2048 b1"), ("dense_16384_ms", "dense L16384 b1")):
            got = statistics.median(series[s])
            if abs(got - ANCHORS[k][1]) > 0.01 * ANCHORS[k][1]:
                bad.append(f"{k}: {ANCHORS[k][1]} here, {got:.1f} banked")
    s3 = S3 / "manifest.json"
    if s3.exists():
        secs = {x["step"]: x["seconds"] for x in json.loads(s3.read_text())["attempts"]}
        for k, (_, v) in S3_ANCHORS.items():
            step = k.removesuffix("_s")
            if abs(secs[step] - v) > 0.01 * v:
                bad.append(f"S3 {k}: {v} here, {secs[step]} banked")
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


def grid(cfg=None):
    """The session's config (a debug session's, if given), its cells, and the
    batch-1-only arms."""
    from t31_steps import BATCH1_ONLY, cells, fill_working_values
    if cfg is None:
        cfg = json.loads((ROOT / "configs" / "t31" / "p1_session.json").read_text())
        cfg = fill_working_values(cfg)
    return cfg, cells(cfg), BATCH1_ONLY


def project(which: int, cfg=None) -> dict:
    """Minutes per step for scenario `which` (0 low, 1 central, 2 high), for
    the config `cfg` if given."""
    u = {k: v[which] for k, v in UNMEASURED.items()}
    cfg, cells, batch1 = grid(cfg)
    by_lb: dict = {}
    for arm, d, L, b in cells:
        by_lb.setdefault((L, b), []).append(arm)

    def arm_wall(arm, L, b):
        return wall_s(L, b) * (1.0 if arm == "dense" else u["sparse_wall_over_dense"])

    def kernels(arm, L):
        if arm.startswith("xattention"):
            return a("kernels_xattention") * math.ceil(L / u["xattention_chunk_tokens"])
        return a("kernels_dense") if arm == "dense" else a("kernels_block_sparse")

    m = {}
    # steps 1 to 4 as the third session ran them, on this grid
    for n, step in ((1, "preflight"), (2, "link"), (3, "dense_select"), (4, "probe")):
        m[f"{n} {step}"] = S3_ANCHORS[f"{step}_s"][1] / 60
    phase = {"measured (31 cells, part of the 32nd)": S3_ANCHORS["host_components_s"][1] / 60,
             "rest of oracle_d0.5_L32768": u["step5_rest_of_oracle_32768_min"],
             "four small cells at 32768": 4 * u["step5_small_32768_cell_min"],
             f"a process per cell ({len(cells)})": len(cells) * u["cell_process_start_s"] / 60}
    m["5 host_components"] = sum(phase.values())
    dev = len(cells) * u["cell_process_start_s"]           # a process per cell, as step 5's
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
        if not arms:                                       # a group the session does not run
            continue
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
    d = cfg.get("delay_injection")
    if d:
        variants = len(d["mechanisms"]) * sum(len(c) for c in d["counts"].values())
        passes = (variants + 1) * (d["reps"] + d["warmup"]) + 1      # + unobserved, calibration
        per = wall_s(d["length"], d["batch"]) * u["observer_wall_factor"] \
            + u["leg3_injection_s_per_pass"]
        m["12 delay_injection"] = (passes * per + u["leg3_setup_s"]) / 60
    return {"steps": m, "step5_phases": dict(phase),
            "cells": len(cells)}


def debug_cap_min(start: str, not_measured=(), run_here=()) -> float:
    """The cap of a debug session (ruling, 2 Nov 2026): 1.5 times the central
    projection of the steps from `start` to the last, on the grid without the
    `not_measured` cells. Step 5, if the session starts there, is the cells it
    runs (`run_here`, the ones the donor never ran): each a small 32768 cell and
    a process, since the staged cells resume."""
    from t31_debug import CAP_FACTOR
    from t31_steps import STEPS
    cfg, _, _ = grid()
    cfg = dict(cfg, not_measured=list(not_measured))
    steps = project(1, cfg)["steps"]
    u = {k: v[1] for k, v in UNMEASURED.items()}
    steps = {s: m for s, m in steps.items() if STEPS.index(s.split(" ", 1)[1]) >= STEPS.index(start)}
    if "5 host_components" in steps:
        steps["5 host_components"] = len(run_here) * (
            u["step5_small_32768_cell_min"] + u["cell_process_start_s"] / 60)
    return CAP_FACTOR * sum(steps.values())


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
    print(f"T3.1 projected from its third session and Shakedown 6 ({t['cells']} cells; "
          f"minutes, low / central / high)")
    print(f"{'setup':24s} {t['setup_min']:7.1f}")
    for s, lo, mid, hi in t["rows"]:
        print(f"{s:24s} {lo:7.1f} {mid:7.1f} {hi:7.1f}")
        if s.startswith("5 "):
            for p, plo, pmid, phi in t["phases"]:
                print(f"    {p:36s} {plo:7.1f} {pmid:7.1f} {phi:7.1f}")
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
