"""Gate: the re-run's dense arm must reproduce the banked dense arm exactly.

Exit 0 only if every new dense row matched its banked counterpart on
prediction, expected answer and tokenized length, and at least --min-checked
examples were compared. Any other outcome exits non-zero, so chaining it with
`&&` before the sparse invocation means a failed canary measures no sparse
row. The verdict, including how many examples were checked, is written to
--record whether it passes or fails.

    python scripts/check_dense_canary.py \\
        --new results/s1a/accuracy_band2048.parquet \\
        --banked canary_ref/stage3_s1.parquet \\
        --seq-len 2048 --min-checked 300 --record results/s1a/canary_2048.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.accuracy.dense_canary import (DenseCanaryFailed, check,  # noqa: E402
                                             enforce)


def _band(df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    # example ids carry the nominal band ("niah_single_2048_17"); the
    # context_length column is the real tokenized length, below the band.
    return df[df["example_id"].str.rsplit("_", n=2).str[-2] == str(seq_len)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--banked", required=True, nargs="+")
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--backend", default="sdpa_flash")
    ap.add_argument("--min-checked", type=int, required=True,
                    help="examples that must be compared (n x tasks); fewer fails")
    ap.add_argument("--record", required=True)
    args = ap.parse_args()

    record = {"seq_len": args.seq_len, "new": args.new, "banked": args.banked,
              "min_checked": args.min_checked}
    rc = 1
    try:
        new = _band(pd.read_parquet(args.new), args.seq_len)
        banked = _band(pd.concat([pd.read_parquet(p) for p in args.banked],
                                 ignore_index=True), args.seq_len)
        res = check(new, banked, backend=args.backend, min_checked=args.min_checked)
        record.update(res.to_dict())
        enforce(res, min_checked=args.min_checked)
        record["verdict"] = "PASS"
        rc = 0
        print(f"DENSE CANARY PASS: {res.n_checked}/{res.n_new_dense} examples "
              f"identical at {args.seq_len} (required {args.min_checked})")
    except (DenseCanaryFailed, OSError, KeyError, ValueError) as e:
        record["verdict"] = "FAIL"
        record["error"] = f"{type(e).__name__}: {e}"
        print(f"DENSE CANARY FAIL at {args.seq_len}: {e}", file=sys.stderr)
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(json.dumps(record, indent=1, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
