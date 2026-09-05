#!/usr/bin/env python3
"""Stage 2 cross-architecture analysis: Ada (L4) against Ampere (A100).

The first point in this project where the hardware-conditional question can be
asked with data on both sides, rather than one architecture and an argument.

The pipeline, in the order the guards have to run:

  1. Load every segment. Four exist, measured on four machines at four
     commits.
  2. `code_identity.restrict_to_reference_code` -- drop rows whose backend was
     timed by code that differs from the reference commit. This is what makes
     the mixed-commit join legitimate rather than merely permitted: on the
     real data it removes 172 seg1 rows (flex, naive) and keeps everything
     else, including the other 90 flex and 112 naive rows measured by the
     reference code.
  3. `cross_arch.load_segments` with `allow_mixed_commits=True` -- the flag is
     passed only because step 2 has already answered, per backend, the
     question the flag switches off wholesale.
  4. Environment check: driver and torch pinned identically across hosts. Code
     identity says our code did not move; this says the compiled kernel did
     not either.
  5. `speedup_within_host` -- ratios formed only inside one machine.
  6. `compare_across_architectures` -- ratios compared across architectures,
     with flip materiality (`flips_materially`) separating the reversals that
     clear measurement noise from the ones that do not.
  7. Every aggregate goes through `composition.aggregate`, which refuses a
     median over bands whose batch composition differs. This is not
     theoretical: the marginal median of GLA's latency on the L4 FALLS between
     4096 and 8192 purely because the larger batches OOM'd out of the band.

Outputs go to --out as parquet, plus a report on stdout. No GPU required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from attnbench.analysis import composition, cross_arch  # noqa: E402
from attnbench.analysis.code_identity import (  # noqa: E402
    backends_with_drift, restrict_to_reference_code)
from attnbench.analysis.crossover import (  # noqa: E402
    crossover_table, matched_cells)

# The segments this study has produced, newest last. Listed rather than
# globbed: `results/` also holds aborted runs, warm-up probes and a duplicate
# of seg1 under a second name, and a glob would sweep those into the join
# where `_check_no_duplicate_cells` would then reject the whole thing.
DEFAULT_SEGMENTS = (
    "results/stage2/segment_20260903_seg1/results/stage2_seg1/sweep.parquet",
    "results/stage2/seg2sweep_20260904/results/stage2/seg2sweep/sweep/sweep.parquet",
    "results/stage2/seg3_20260905/results/stage2/seg3/sweep/sweep.parquet",
    "results/a100/sweep_a100.parquet/sweep.parquet",
)

# The A100 segment's commit. Reference because it is the newest, so choosing it
# maximises retained rows: any earlier segment measured by identical code is
# kept, and only genuinely superseded code is dropped.
DEFAULT_REFERENCE = "d2d8ceb6aa26f983e53c80dd5c44010902530623"

# sdpa_flash, not sdpa_math. The baseline must exist on EVERY host or
# speedup_within_host refuses the whole join, and sdpa_math was not measured on
# the seg3 machine. sdpa_flash is also the more useful reference: it is what a
# practitioner gets from torch.nn.functional.scaled_dot_product_attention
# without asking for anything.
DEFAULT_BASELINE = "sdpa_flash"

ENV_COLUMNS = ("driver", "torch", "torch_cuda", "triton")


def load(segments, reference: str) -> tuple[pd.DataFrame, list]:
    df = cross_arch.load_segments(segments, allow_mixed_commits=True)
    df, restrictions = restrict_to_reference_code(
        df, repo=REPO, reference_commit=reference)
    residual = backends_with_drift(df, repo=REPO)
    if residual:
        raise SystemExit(
            "code drift survived the restriction: "
            + "; ".join(d.describe() for d in residual))
    return df, restrictions


def environment_report(df: pd.DataFrame) -> tuple[str, bool]:
    """Whether the compiled-kernel environment is common across hosts.

    Code identity establishes that OUR code did not move between segments. It
    says nothing about torch, the driver or triton, any of which changes the
    kernel that actually ran without touching a line of this repository. A
    cross-architecture claim needs both, and only one of them is checked by
    the code-identity module -- stated there, checked here.
    """
    lines, uniform = [], True
    for col in ENV_COLUMNS:
        if col not in df.columns:
            continue
        values = sorted({str(v) for v in df[col].dropna().unique()})
        lines.append(f"  {col:12s} {', '.join(values)}")
        if len(values) > 1:
            uniform = False
    return "\n".join(lines), uniform


def clock_report(df: pd.DataFrame) -> str:
    """What the clock provenance can and cannot support.

    `clocks_locked` is False on every row in the dataset, on both
    architectures -- which is the symmetric case, and the less damaging one:
    both halves of every ratio carry the same run-to-run variance, so a reader
    discounts them uniformly rather than mistaking a difference in measurement
    quality for a difference in hardware.

    `sm_clock_mhz` cannot rescue it. It is a single sample taken when the
    provenance stamp is captured, not a statistic over the timed region: the
    L4 hosts all read 210 MHz, which is that card's idle clock, not the
    ~2040 MHz it runs a kernel at. One sample cannot distinguish a locked
    clock from a clock that happened to be at that value, so deriving
    `clocks_locked` from this column would manufacture confidence rather than
    measure it. Fixing that needs sampling during the run, on hardware.
    """
    g = df.groupby(["gpu_name", "host"])[["sm_clock_mhz", "clocks_locked"]]
    return g.agg(lambda s: sorted({str(v) for v in s})).to_string()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segment", action="append", default=None,
                    help="results file (repeatable); defaults to the four Stage 2 segments")
    ap.add_argument("--reference-commit", default=DEFAULT_REFERENCE)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "cross_arch")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every guard and print the report, write nothing")
    args = ap.parse_args()

    segments = [REPO / s for s in (args.segment or DEFAULT_SEGMENTS)]
    df, restrictions = load(segments, args.reference_commit)

    print("=" * 74)
    print("Stage 2 cross-architecture analysis")
    print("=" * 74)

    print(f"\n## Segments ({len(segments)})")
    for gpu, hosts in cross_arch.hosts_by_architecture(df).items():
        print(f"  {gpu}: {len(hosts)} host(s) -- {', '.join(hosts)}")

    print("\n## Code identity (which rows were timed by the reference code)")
    if restrictions:
        for r in restrictions:
            print(f"  {r.describe()}")
    else:
        print("  every backend identical at every commit")
    print(f"  {len(df)} row(s) retained")

    env, uniform = environment_report(df)
    print("\n## Environment" + ("  [uniform]" if uniform else "  [NOT UNIFORM]"))
    print(env)
    if not uniform:
        print("  A version difference changes the compiled kernel without "
              "changing this repository. Ratios below span it.")

    print("\n## Clock control")
    print(clock_report(df))
    print("  Both architectures measured unlocked -- symmetric, so the "
          "variance discounts uniformly.")

    speedups = cross_arch.speedup_within_host(df, baseline_backend=args.baseline)
    comparisons = cross_arch.compare_across_architectures(speedups)
    material = [c for c in comparisons if c.flips_materially()]
    nominal = [c for c in comparisons if c.flips() and not c.flips_materially()]

    print(f"\n## Cross-architecture comparisons (baseline {args.baseline})")
    print(f"  {len(speedups)} within-host ratios -> {len(comparisons)} comparisons")
    print(f"  {len(material) + len(nominal)} flip; {len(material)} clear the "
          f"5% noise floor")
    if nominal:
        print(f"  {len(nominal)} flip only nominally -- the weaker side sits "
              f"within measurement noise of parity, and every row in this "
              f"dataset was measured on an unlocked card. Not reportable.")

    meta = df.drop_duplicates("config_key").set_index("config_key")
    print("\n### Material flips")
    if not material:
        print("  none")
    for c in sorted(material, key=lambda c: -c.flip_margin):
        m = meta.loc[c.config_key]
        arch = "  ".join(f"{k.replace('NVIDIA ', '')}={v:.3f}"
                         for k, v in sorted(c.speedup_by_architecture.items()))
        print(f"  {c.backend:8s} seq={int(m.seq_len):<6d} batch={int(m.batch):<3d} "
              f"{arch}   margin={c.flip_margin:.3f}")

    cross = crossover_table(df)
    print("\n## GLA vs FA2 crossover, per cell")
    for gpu, group in cross.groupby("gpu_name"):
        print(f"\n  {gpu}")
        print("    " + group.drop(columns=["gpu_name"]).to_string(
            index=False).replace("\n", "\n    "))

    matched = matched_cells(cross)
    print("\n## GLA vs FA2, cells measured on BOTH architectures")
    print(f"  {len(matched)} shared cell(s) of {len(cross.groupby(['seq_len', 'batch']))} total")
    print("    " + matched.to_string(index=False).replace("\n", "\n    "))
    disagree = matched[matched["disagrees"]]
    if disagree.empty:
        print("  The two architectures agree on the winner in every shared cell.")
    else:
        cells = ", ".join(f"seq={int(r.seq_len)}/batch={int(r.batch)}"
                          for r in disagree.itertuples())
        print(f"  The winner DIFFERS in {len(disagree)} shared cell(s): {cells}.")
        print("  That is the hardware-conditional result, at matched batch.")

    print("\n## Composition guard")
    try:
        composition.aggregate(df[df["backend"] == "gla"], group_by="seq_len",
                              value="latency_ms_p50")
        print("  a marginal median over seq_len is admissible for gla")
    except composition.CompositionMismatch as exc:
        first = str(exc).splitlines()[0]
        print(f"  REFUSED as expected: {first}")
        print("  Marginals over seq_len are not reportable for this dataset; "
              "the per-cell tables above are.")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "backend": c.backend, "config_key": c.config_key,
        "seq_len": int(meta.loc[c.config_key].seq_len),
        "batch": int(meta.loc[c.config_key].batch),
        **{f"speedup_{k.replace('NVIDIA ', '').replace(' ', '_')}": v
           for k, v in c.speedup_by_architecture.items()},
        "flips": c.flips(), "flip_margin": c.flip_margin,
        "flips_materially": c.flips_materially(),
        "caveat": c.caveat(),
    } for c in comparisons]).to_parquet(args.out / "comparisons.parquet")
    cross.to_parquet(args.out / "crossover.parquet")
    matched.to_parquet(args.out / "crossover_matched.parquet")
    pd.DataFrame([{
        "host": s.host, "gpu_name": s.gpu_name, "backend": s.backend,
        "config_key": s.config_key, "latency_ms": s.latency_ms,
        "baseline_latency_ms": s.baseline_latency_ms, "speedup": s.speedup,
        "clocks_locked": s.clocks_locked,
    } for s in speedups]).to_parquet(args.out / "speedups.parquet")
    print(f"\nWrote comparisons/crossover/speedups to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
