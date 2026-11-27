"""EXPLORATORY: T1.1's qa_1/32768 non-inferiority bound recomputed on the
rows inside Qwen2.5's 32,768 positions (adviser's task of 22 Nov 2026,
item 5; docs/paper_claims_audit.md, section (b)). CPU only, from the banked
T1.1 Session B parquets. It does not replace the certified value: the
certified cell (50 vs 46 of 100, bound -6.0) is computed over all 100 rows
and stays as it is.

Exclusion rule (the audit's plan default): for each calibration arm, drop
the union of example ids that decode past the limit (positions_over_limit
> 0) in the dense file or in that arm's file, from both, so the test stays
paired. Everything else is the certified path: run_t4_noninferiority.load,
run_exact_ni, t4_pilot's margin, alpha and bands.

    python scripts/t11_qa1_within_limit.py [--results DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

import run_t4_noninferiority as r  # noqa: E402
from attnbench.accuracy import t4_pilot  # noqa: E402
from attnbench.analysis.exact_noninferiority import arm_label, run_exact_ni  # noqa: E402

TABLES = ("dense", "authors", "authors_full", "ruler_heldout")
DEFAULT = ROOT / "results" / "t11b_session_20261025" / "results"
TASK, BAND = "qa_1", 32768
CERTIFIED = {"xattention_inline@cal=authors": -6.0}   # Section 2; the claim arm


def files(results: Path) -> list:
    return [results / f"t4_xattn_cal_{t}_20261025" / "accuracy.parquet" for t in TABLES]


def over_limit_union(dense: pd.DataFrame, arm_rows: pd.DataFrame) -> set:
    """Ids past the limit in the dense file or in the arm's file."""
    return (set(dense.example_id[dense.positions_over_limit > 0])
            | set(arm_rows.example_id[arm_rows.positions_over_limit > 0]))


def recompute(df: pd.DataFrame) -> pd.DataFrame:
    kw = dict(bands=t4_pilot.PILOT_BANDS, margin_pts=t4_pilot.MARGIN_PTS, alpha=t4_pilot.ALPHA)
    df = df.assign(band=[t4_pilot_band(int(x)) for x in df.context_length])
    q = df[(df.task == TASK) & (df.band == BAND)]
    if q.positions_over_limit.isna().any():
        raise ValueError("a qa_1/32768 row has no positions_over_limit")
    dense = q[q.backend_role == "dense_reference"]
    sparse = q[q.backend_role == "block_sparse"]
    out = []
    for arm, g in sparse.groupby(sparse.apply(arm_label, axis=1), sort=True):
        full = run_exact_ni(dense, g, **kw).iloc[0]
        drop = over_limit_union(dense, g)
        inside = run_exact_ni(dense[~dense.example_id.isin(drop)],
                              g[~g.example_id.isin(drop)], **kw).iloc[0]
        for label, res in (("all rows (certified path)", full), ("inside the limit", inside)):
            out.append({"arm": arm, "rows": label, "excluded": 0 if res is full else len(drop),
                        **{k: res[k] for k in ("n", "dense_correct", "sparse_correct",
                                               "b_sparse_only", "c_dense_only", "diff_pts",
                                               "lower_pts", "non_inferior")}})
    return pd.DataFrame(out)


def t4_pilot_band(context_length: int) -> int:
    from attnbench.analysis.matched import band_for
    return band_for(context_length, t4_pilot.PILOT_BANDS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=DEFAULT)
    args = ap.parse_args(argv)
    df = r.load([str(p) for p in files(args.results)], allow_mixed_commits=False)
    res = recompute(df)
    for arm, want in CERTIFIED.items():
        got = res[(res.arm == arm) & (res.rows.str.startswith("all"))].lower_pts.iloc[0]
        if round(got, 1) != want:
            raise SystemExit(f"the all-rows bound for {arm} is {got:.2f}, not the certified {want}")
    print(f"EXPLORATORY (not a replacement for the certified value): {TASK}/{BAND}, "
          f"margin {t4_pilot.MARGIN_PTS} pts, one-sided alpha {t4_pilot.ALPHA}, "
          f"commit {df.git_commit.iloc[0][:7]}")
    with pd.option_context("display.width", 200):
        print(res.round({"diff_pts": 1, "lower_pts": 2}).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
