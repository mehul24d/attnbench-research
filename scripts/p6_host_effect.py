#!/usr/bin/env python3
"""The host-effect rule that gates P6 (Appendix A, 21 Nov 2026; EXPLORATORY).

P3 (AMD EPYC 7R13 + L4) against P1 (Intel Xeon 8273CL + L4): the same GPU,
another host. For each grid cell the two sessions share, take the operators
both host-component price lists name (`<step>/<cell>/bare_prices.json`, the
bare timer), price each per call (`bare_s / calls`) and compute
|a / b - 1|. A cell's value is the median over its operators; the statistic
d is the median over cells. The comparison is per operator name, not per
strict operation sequence.

The rule, fixed before P3's host components are read:

    d > 0.10           run      a host effect on the L4: P6 runs
    d <= 0.034         skip     no host effect: P6 does not run
    0.034 < d <= 0.10  skip     unless the researcher overrules in writing
                                (dated, in docs/permission_log.md) before
                                P3's result is read

    python3 scripts/p6_host_effect.py <P3 host dir> <P1 host dir> [--override-run]

prints the cell count, d, the 95th percentile of the cells, the median ratio
of host totals minus 1, and the verdict.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

TAU_RUN = 0.10      # above: run P6 (the largest P2-against-P1 cell p95, and H0's 0.1)
TAU_SKIP = 0.034    # at or below: skip (the largest P2-against-P1 d)


def load(host_dir: Path) -> dict:
    """cell -> {"total": host_total_s, "ops": {operator: bare seconds per call}}."""
    out = {}
    for f in sorted(Path(host_dir).glob("*/bare_prices.json")):
        j = json.loads(f.read_text())
        ops = {k: v["bare_s"] / v["calls"] for k, v in j["operators"].items() if v["calls"]}
        out[f.parent.name] = {"total": j["host_total_s"], "ops": ops}
    return out


def compare(a: dict, b: dict) -> dict:
    per_cell, totals = [], []
    for cell in sorted(set(a) & set(b)):
        oa, ob = a[cell]["ops"], b[cell]["ops"]
        common = sorted(set(oa) & set(ob))
        if not common:
            continue
        per_cell.append(statistics.median(abs(oa[o] / ob[o] - 1) for o in common))
        totals.append(a[cell]["total"] / b[cell]["total"] - 1)
    if not per_cell:
        raise ValueError("no shared cell with a shared operator")
    q = sorted(per_cell)
    p95 = q[min(len(q) - 1, int(0.95 * len(q)))]
    return {"cells": len(q), "d": statistics.median(q), "p95": p95,
            "total_ratio_minus_1": statistics.median(totals)}


def verdict(d: float, override_run: bool = False) -> str:
    if d > TAU_RUN:
        return "run"
    if d <= TAU_SKIP:
        return "skip"
    return "run (researcher's written override)" if override_run else "skip (inconclusive band)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("p3_dir", type=Path)
    ap.add_argument("p1_dir", type=Path)
    ap.add_argument("--override-run", action="store_true",
                    help="only with the researcher's dated written override, logged before P3's result is read")
    args = ap.parse_args(argv)
    r = compare(load(args.p3_dir), load(args.p1_dir))
    r["verdict"] = verdict(r["d"], args.override_run)
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
