"""Flag every banked accuracy row whose decode ran past the model's position limit.

Both models this study has run (Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct)
have `max_position_embeddings = 32768` in `config.json` (checked 2026-10-03).

**Why `context_length` is the prompt length.** `generation.generate_one`
tokenises `example.context` with no chat template (`generation.py:109`), and
`run_accuracy.count_tokens` is `len(tokenizer(text).input_ids)`, the same
call. So for every row, the prefill length P is `context_length`.

**Which positions are used.** Prefill uses positions 0..P-1. The decode loop
(`model.py`, `generate`) feeds generated token k (1-based) at position
P + k - 1, and breaks before feeding the last one. So the highest position
any forward sees is P + n_generated - 2.

Two flags, reported separately:

- `over_user_def`: P + n_generated > LIMIT. This is the plain "prompt plus
  generation exceeds 32,768" reading.
- `over_strict`: P + n_generated - 2 >= LIMIT. At least one forward ran at a
  position index the model was not trained on.

Copies of the same parquet under session directories are deduplicated by
content hash and reported once, with every path listed.

    python scripts/flag_positions_over_limit.py [--out results/positions_over_limit]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

LIMIT = 32768
NEEDED = {"example_id", "context_length", "n_generated"}


def canonical_files(root: Path) -> dict[str, list[Path]]:
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(root.rglob("*.parquet")):
        try:
            cols = set(pd.read_parquet(p, columns=None).columns)
        except Exception:
            continue
        if not NEEDED <= cols:
            continue
        by_hash[hashlib.sha256(p.read_bytes()).hexdigest()].append(p)
    return by_hash


def flag(df: pd.DataFrame) -> pd.DataFrame:
    p = df["context_length"].astype(int)
    g = df["n_generated"].astype(int)
    out = df.assign(prompt_plus_gen=p + g,
                    max_position=(p + g - 2).clip(lower=p - 1))
    out["over_user_def"] = out["prompt_plus_gen"] > LIMIT
    out["over_strict"] = out["max_position"] >= LIMIT
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/positions_over_limit")
    args = ap.parse_args(argv)

    files = canonical_files(Path(args.results))
    rows, summary = [], []
    keep = ["example_id", "task", "backend", "score_source", "sparsity",
            "context_length", "n_generated", "prompt_plus_gen", "max_position",
            "over_user_def", "over_strict", "correct", "git_commit"]
    for digest, paths in sorted(files.items(), key=lambda kv: str(kv[1][0])):
        df = flag(pd.read_parquet(paths[0]))
        canon = str(paths[0])
        summary.append(dict(
            file=canon, copies=len(paths), sha256=digest[:12], rows=len(df),
            max_context_length=int(df.context_length.max()),
            over_user_def=int(df.over_user_def.sum()),
            over_strict=int(df.over_strict.sum()),
            commits=",".join(sorted({str(c)[:7] for c in df.get("git_commit", [])}))))
        hit = df[df.over_user_def]
        if len(hit):
            hit = hit[[c for c in keep if c in hit.columns]].copy()
            hit.insert(0, "file", canon)
            rows.append(hit)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = pd.DataFrame(summary)
    s.to_csv(out / "summary.csv", index=False)
    flagged = pd.concat(rows) if rows else pd.DataFrame()
    flagged.to_csv(out / "flagged_rows.csv", index=False)
    with open(out / "copies.txt", "w") as f:
        for paths in files.values():
            if len(paths) > 1:
                f.write("  ==  ".join(map(str, paths)) + "\n")

    # Real tokens minus the band's budget, one row per distinct example per
    # file. Budget is the band in `example_id` (`<task>_<budget>_<i>`).
    deltas = []
    for paths in files.values():
        d = pd.read_parquet(paths[0], columns=["example_id", "task", "context_length"])
        d["budget"] = d.example_id.str.extract(r"_(\d{3,6})_\d+$")[0].astype(float)
        d = d.dropna(subset=["budget"]).drop_duplicates("example_id")
        d["delta"] = d.context_length - d.budget
        deltas.append(d)
    if deltas:
        dd = pd.concat(deltas)
        # `over_budget` counts (file, example) pairs, and an example recurs
        # across files; `over_budget_distinct` counts each example once
        # (added 2026-10-03: 655 pairs are 182 examples).
        distinct = (dd[dd.delta > 0].drop_duplicates(["example_id", "context_length"])
                    .groupby(["budget", "task"]).size().rename("over_budget_distinct"))
        (dd.groupby(["budget", "task"]).delta
           .agg(n="size", min="min", median="median", max="max",
                over_budget=lambda s: int((s > 0).sum()))
           .join(distinct).fillna({"over_budget_distinct": 0})
           .astype({"over_budget_distinct": int})
           .to_csv(out / "sizing_delta_by_band_task.csv"))
        print(f"over budget: {int((dd.delta > 0).sum())} file-example pairs, "
              f"{int(distinct.sum())} distinct examples")

    pd.set_option("display.width", 250)
    print(s[s.over_user_def > 0].to_string(index=False))
    print(f"\n{len(files)} distinct accuracy parquets "
          f"({sum(map(len, files.values()))} paths); "
          f"{int(s.over_user_def.sum())} rows with P+gen > {LIMIT}, "
          f"{int(s.over_strict.sum())} with a forward at position >= {LIMIT}, "
          f"out of {int(s.rows.sum())}. Written to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
