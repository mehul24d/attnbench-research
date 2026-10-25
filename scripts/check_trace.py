#!/usr/bin/env python3
"""Trace checks on a ground-truth trace (T2.7): copy-kernel concurrency,
managed-memory use, and page faults. What each source may claim:
`attnbench/analysis/trace_checks.py`.

Subcommands:

  kineto TRACE.json --out DIR
      copy-kernel concurrency and managed-memory use from a PyTorch profiler
      trace. Claims at most "no managed-memory use".
  nsys-faults EXPORT.sqlite --control CONTROL.sqlite --arms A,B --out DIR
      page faults from an Nsight Systems sqlite export of a separate traced
      pass (never a timing run), with a positive control from the same
      session (`scripts/um_fault_control.py` under the same nsys options).
      Claims "no page faults" only for the arms named, and only if the
      control faulted.
  nsys-available
      prints the nsys binary and its version; exits 1 if there is none.

Each check writes one analysis-stamped row (`trace_check_<source>.parquet`)
and prints its claim. The nsys options both passes must use are printed by
`nsys-available`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import provenance                       # noqa: E402
from attnbench.analysis import trace_checks as tc      # noqa: E402


def _write(result: dict, out: Path, source: str) -> Path:
    import pandas as pd
    row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
           for k, v in result.items()}
    df = pd.DataFrame([row])
    provenance.stamp_analysis(df, f"check_trace.{source}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"trace_check_{source}.parquet"
    df.to_parquet(path, index=False)
    return path


def cmd_kineto(args) -> int:
    r = tc.check_kineto(args.trace)
    path = _write(r, Path(args.out), "kineto")
    print(f"{path}: copy-kernel concurrency {'FLAGGED' if r['copy_kernel_concurrency'] else 'none'} "
          f"({r['cross_stream_overlaps']} cross-stream overlaps, max "
          f"{r['cross_stream_overlap_us_max']:.3f} us; {r['same_stream_overlaps']} "
          f"same-stream timestamp overlaps); {r['claim']}")
    return 1 if r["copy_kernel_concurrency"] or r["managed_memory_use"] else 0


def cmd_nsys_faults(args) -> int:
    arms = [a for a in args.arms.split(",") if a]
    r = tc.check_page_faults(args.export, args.control, arms=arms)
    path = _write(r, Path(args.out), "nsys")
    print(f"{path}: {r['claim']}")
    return 0 if r["page_faults"] == 0 else 1


def cmd_nsys_available(args) -> int:
    nsys = tc.nsys_available()
    if nsys is None:
        print("nsys not found on PATH, under /usr/local/cuda*/bin or /opt/nvidia")
        return 1
    v = subprocess.run([nsys, "--version"], capture_output=True, text=True)
    print(f"{nsys}\n{v.stdout.strip()}")
    print("profile both passes with: nsys profile " + " ".join(tc.NSYS_PROFILE_FLAGS))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("kineto")
    s.add_argument("trace")
    s.add_argument("--out", required=True)
    s = sub.add_parser("nsys-faults")
    s.add_argument("export")
    s.add_argument("--control", required=True)
    s.add_argument("--arms", required=True)
    s.add_argument("--out", required=True)
    sub.add_parser("nsys-available")
    args = ap.parse_args()
    return {"kineto": cmd_kineto, "nsys-faults": cmd_nsys_faults,
            "nsys-available": cmd_nsys_available}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
