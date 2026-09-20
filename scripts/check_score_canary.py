"""Gate: the re-run's importance scores must reproduce the banked ones.

The dense canary checks that the ENVIRONMENT reproduces -- same model, same
tokenization, same greedy decode. It cannot check the scoring pass, because a
dense cell never computes importance scores (tests/test_generation_wiring.py
asserts exactly that). So a change to the scoring path is invisible to it, and
a re-run built to isolate a MASK-rule change would silently attribute the
scoring change's flips to the mask rule too.

Audit item S7 needs that separation. Between the banked 16384 run and HEAD the
mask builder changed (5cc3a40, the tie-break jitter stream) and so did the
scoring path (93d821c, which pins allow_tf32=False). The second is expected to
be a no-op -- torch already defaults matmul TF32 off on this card -- but
"expected to be a no-op" is the kind of assumption this project keeps finding
was wrong, and here it is free to check: the score cache stores exactly the
fp16 tensors the 0.75 and 0.9 arms rank from, keyed by (model, task, example,
seq_len), so the banked entries and the new ones are directly comparable.

Identical tensors -> every flip in the comparison belongs to the mask rule.
Different tensors -> the comparison is confounded and says so.

    python scripts/check_score_canary.py \\
        --rows results/s7_jitter/accuracy.parquet \\
        --new-cache results/accuracy/score_cache \\
        --banked-cache /tmp/banked_score_cache \\
        --seq-len 16384 --record results/s7_jitter/score_canary_16384.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.accuracy.score_cache import cache_key  # noqa: E402

MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"


def _band(df: pd.DataFrame, seq_len: int) -> pd.DataFrame:
    return df[df["example_id"].str.rsplit("_", n=2).str[-2] == str(seq_len)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True,
                    help="accuracy parquet naming the examples to check; the "
                         "cache key is derived from its own (task, example_id, "
                         "context_length), not from a list written by hand")
    ap.add_argument("--new-cache", required=True)
    ap.add_argument("--banked-cache", required=True)
    ap.add_argument("--fetch-from", default=None,
                    help="GCS prefix holding the banked cache. Only the keys "
                         "--rows asks for are fetched, in ONE gsutil call with "
                         "the URIs as arguments -- not piped on stdin, which "
                         "silently transferred 2 of 200 objects when this was "
                         "first tried.")
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--tasks", default=None,
                    help="comma-separated subset, when --rows holds tasks the "
                         "caches were not asked to cover")
    ap.add_argument("--min-checked", type=int, default=1,
                    help="entries that must be compared; fewer fails, so an "
                         "empty cache dir cannot pass as 'no differences'")
    ap.add_argument("--record", required=True)
    args = ap.parse_args()

    rows = _band(pd.read_parquet(args.rows), args.seq_len)
    # Sparse rows only: a dense row has no scores, so asking for its key would
    # invent an entry neither cache was ever supposed to hold.
    rows = rows[rows["backend"] == "block_sparse"]
    if args.tasks:
        rows = rows[rows["task"].isin(args.tasks.split(","))]
    want = rows.drop_duplicates(["task", "example_id"])[
        ["task", "example_id", "context_length"]]

    new_dir, banked_dir = Path(args.new_cache), Path(args.banked_cache)

    keys = [cache_key(args.model, r.task, r.example_id, int(r.context_length))
            for r in want.itertuples(index=False)]

    if args.fetch_from:
        banked_dir.mkdir(parents=True, exist_ok=True)
        uris = [f"{args.fetch_from.rstrip('/')}/{k}.pt" for k in keys
                if not (banked_dir / f"{k}.pt").exists()]
        if uris:
            print(f"fetching {len(uris)} banked score tensors from "
                  f"{args.fetch_from}")
            r = subprocess.run(["gsutil", "-q", "-m", "cp", *uris, str(banked_dir)])
            if r.returncode != 0:
                print(f"gsutil cp exited {r.returncode}; comparing what "
                      f"arrived and reporting the rest as missing",
                      file=sys.stderr)
        got = len(list(banked_dir.glob("*.pt")))
        print(f"banked cache: {got} tensors in {banked_dir}")

    checked = identical = 0
    missing_new: list[str] = []
    missing_banked: list[str] = []
    differing: list[dict] = []

    for r, key in zip(want.itertuples(index=False), keys):
        pn, pb = new_dir / f"{key}.pt", banked_dir / f"{key}.pt"
        if not pn.exists():
            missing_new.append(f"{r.example_id}:{key}")
            continue
        if not pb.exists():
            missing_banked.append(f"{r.example_id}:{key}")
            continue
        a = torch.load(pn, map_location="cpu")
        b = torch.load(pb, map_location="cpu")
        checked += 1
        if a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b):
            identical += 1
        else:
            d = (a.float() - b.float()).abs() if a.shape == b.shape else None
            differing.append({
                "example_id": r.example_id, "key": key,
                "shape_new": list(a.shape), "shape_banked": list(b.shape),
                "dtype_new": str(a.dtype), "dtype_banked": str(b.dtype),
                "n_elem_differing": int((a != b).sum()) if a.shape == b.shape else None,
                "max_abs_diff": float(d.max()) if d is not None else None,
            })

    record = {
        "seq_len": args.seq_len, "model": args.model, "rows": args.rows,
        "tasks": args.tasks,
        "new_cache": str(new_dir), "banked_cache": str(banked_dir),
        "n_examples": int(len(want)), "n_checked": checked,
        "n_identical": identical, "n_differing": len(differing),
        "missing_from_new": missing_new[:20],
        "n_missing_from_new": len(missing_new),
        "missing_from_banked": missing_banked[:20],
        "n_missing_from_banked": len(missing_banked),
        "differing": differing[:20],
        "min_checked": args.min_checked,
    }

    ok = (checked >= args.min_checked and not differing
          and not missing_new and not missing_banked)
    record["verdict"] = "PASS" if ok else "FAIL"
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(json.dumps(record, indent=1, default=str))

    if ok:
        print(f"SCORE CANARY PASS: {identical}/{checked} score tensors "
              f"bit-identical to the banked run at {args.seq_len} "
              f"(required {args.min_checked}). The scoring pass is unchanged, "
              f"so mask-rule flips are attributable to the mask rule.")
        return 0
    print(f"SCORE CANARY FAIL at {args.seq_len}: {identical}/{checked} "
          f"identical, {len(differing)} differing, {len(missing_new)} missing "
          f"from the new cache, {len(missing_banked)} from the banked one. "
          f"The scoring pass changed too -- any accuracy comparison built on "
          f"these rows is NOT attributable to the mask rule alone.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
