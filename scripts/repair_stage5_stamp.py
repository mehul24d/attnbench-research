#!/usr/bin/env python3
"""Repair the `clocks_locked` column that Stage 5's provenance merge destroyed.

    python scripts/repair_stage5_stamp.py \
        --parquet results/stage5/phases.parquet \
        --log results/gpu_session_20260907_stage5/stage5.log

This is NOT hand-correcting a measurement. It applies a deterministic rule
under a precondition, refuses if the precondition does not hold, and writes an
audit record beside the parquet naming every assertion it checked and the
value each one read. See docs/silent_failure_patterns.md #23.

Background
----------
The 2026-09-07 Stage 5 run locked clocks successfully. `lock_clocks()`
returned True, `measure_band` threaded `clocks_locked=True` onto all 27
PhaseMeasurement rows, and the final write did:

    prov = provenance.capture().to_dict()   # clocks_locked DEFAULTS to False
    for k, v in prov.items():
        df[k] = v                           # measured True -> stamped False

`clocks_locked` is a GATED field -- `cross_arch.Speedup` and
`canary.CanaryDrift` read it -- so leaving a known-false value in it is a
landmine, and a caveat in a markdown file is not where a gate looks.

The three corroborations
------------------------
Each is independent of the others and independent of the flag being repaired.
All three must hold or this script writes nothing.

1. `persistence_mode == "Enabled"` on every row. Persistence mode is set by
   `_smi(["-pm", "1"])` *inside* `lock_clocks()` and by nothing else in this
   codebase. A fresh GCP L4 boots with it Disabled. So Enabled proves
   `lock_clocks()` ran AND that its `sudo -n` escalation worked -- which is
   the only thing that could have made the subsequent `-lgc` fail.

2. `sm_clock_mhz` sits at the requested pin. `lock_clocks()` requests
   `int(max_sm * 0.85)`; the L4's max SM clock is 2040, giving 1734. NVIDIA
   clocks quantise to 15 MHz steps, so a successful pin reads at the nearest
   supported step. An UNLOCKED card under load does not sit within a step of
   an arbitrary 85% figure for 19 minutes.

3. The run log contains `lock_clocks()`'s success branch and not its failure
   branch. The script prints one or the other and never both.

The repair is only ever False -> True. It will not flip a True to False, and
it refuses if the column is not uniformly False, because a mixed column means
something other than the wholesale-overwrite bug happened.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import provenance                                # noqa: E402

# The pin `lock_clocks()` would have requested on this card.
L4_MAX_SM_MHZ = 2040
CLOCK_QUANTUM_MHZ = 15          # NVIDIA supported-clock granularity
SUCCESS_LINE = "clocks locked -- stamped on every row"
FAILURE_LINE = "CLOCK LOCK FAILED"


class PreconditionFailed(Exception):
    pass


def check(df: pd.DataFrame, log_text: str) -> list[dict]:
    """The three corroborations plus the two sanity conditions.

    Returns one record per assertion, with the value it actually read, so the
    audit file shows what was checked rather than that something was.
    """
    checks: list[dict] = []

    def record(name, ok, read, expected):
        checks.append(dict(assertion=name, passed=bool(ok),
                           read=read, expected=expected))
        return ok

    gpus = sorted(map(str, df.gpu_name.unique()))
    record("gpu is the card this rule is calibrated for",
           gpus == ["NVIDIA L4"], gpus, ["NVIDIA L4"])

    current = sorted(map(bool, df.clocks_locked.unique()))
    record("clocks_locked is uniformly False (the overwrite signature)",
           current == [False], current, [False])

    persist = sorted(map(str, df.persistence_mode.unique()))
    record("persistence_mode Enabled on every row (set only inside lock_clocks)",
           persist == ["Enabled"], persist, ["Enabled"])

    requested = int(L4_MAX_SM_MHZ * 0.85)
    clocks = sorted(int(c) for c in df.sm_clock_mhz.unique())
    within = (len(clocks) == 1
              and abs(clocks[0] - requested) <= CLOCK_QUANTUM_MHZ)
    record(f"sm_clock pinned within {CLOCK_QUANTUM_MHZ} MHz of the "
           f"requested {requested}", within, clocks, requested)

    record("run log carries lock_clocks()'s success branch",
           SUCCESS_LINE in log_text, SUCCESS_LINE in log_text, True)
    record("run log does NOT carry its failure branch",
           FAILURE_LINE not in log_text, FAILURE_LINE in log_text, False)

    return checks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="results/stage5/phases.parquet")
    ap.add_argument("--log", required=True,
                    help="the run log, for corroboration 3")
    ap.add_argument("--audit", default=None,
                    help="where the audit record goes "
                         "(default: <parquet>.repair.json)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    parquet = Path(args.parquet)
    df = pd.read_parquet(parquet)
    log_text = Path(args.log).read_text()

    checks = check(df, log_text)
    width = max(len(c["assertion"]) for c in checks)
    print(f"preconditions for repairing {parquet}:")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['assertion']:<{width}}"
              f"  read={c['read']!r}")

    failed = [c for c in checks if not c["passed"]]
    if failed:
        print(f"\nREFUSING -- {len(failed)} precondition(s) failed. The rule "
              f"does not apply to this artefact and nothing was written.")
        return 1

    if args.dry_run:
        print("\n--dry-run: all preconditions hold; nothing written.")
        return 0

    df["clocks_locked"] = True
    df.to_parquet(parquet, index=False)

    audit_path = Path(args.audit) if args.audit else \
        parquet.with_suffix(parquet.suffix + ".repair.json")
    audit_path.write_text(json.dumps({
        "repaired_file": str(parquet),
        "column": "clocks_locked",
        "from": False,
        "to": True,
        "rows": int(len(df)),
        "reason": "provenance.capture()'s default overwrote the measured "
                  "value; see docs/silent_failure_patterns.md #23",
        "rule": "False -> True only, under all preconditions below",
        "preconditions": checks,
        "log_consulted": str(args.log),
        "repaired_by": "scripts/repair_stage5_stamp.py",
        "repaired_at_commit": provenance.capture().git_commit,
    }, indent=2) + "\n")

    print(f"\nrepaired {len(df)} rows: clocks_locked False -> True")
    print(f"audit record written to {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
