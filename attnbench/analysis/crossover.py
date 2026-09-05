"""Where one backend overtakes another, and whether that point moves with the
hardware.

The study's central hardware-conditional question in its most direct form:
FlashAttention-2's cost is quadratic in sequence length and gated linear
attention's is linear, so GLA must win eventually. The question is *when*, and
whether "when" is a property of the algorithm or of the card.

**The claim has to be made per cell.** Both halves of it are vulnerable to the
composition artefact documented in `analysis.composition`: OOM removes the
large batches from the long-context bands, and it removes different batches on
different cards, so a marginal "median latency by seq_len" comparison would be
comparing a three-batch band on one architecture against a one-batch band on
the other. `matched_cells` is the discipline made explicit -- it keeps only
(seq_len, batch) cells measured on every architecture, so a disagreement about
the winner is a disagreement about hardware rather than about which cells
survived.

On the 2026-09-05 join that leaves 11 shared cells of 17, and the winner
differs in exactly one of them.
"""

from __future__ import annotations

import pandas as pd

from .cross_arch import ratio_resolution

DEFAULT_PAIR = ("gla", "fa2")


class CrossoverError(RuntimeError):
    """A crossover table that cannot be formed from the given rows."""


def crossover_table(df: pd.DataFrame, *, backends: tuple[str, str] = DEFAULT_PAIR,
                    latency_column: str = "latency_ms_p50") -> pd.DataFrame:
    """Per-(gpu_name, seq_len, batch) latency for two backends, plus their ratio.

    `{b}_over_{a}` is greater than 1 exactly when `a` is the faster backend --
    it is b's cost expressed in units of a's, so with the default pair it
    reads as "how many times more expensive FA2 is than GLA here".

    Cells where either backend is absent are dropped rather than filled: a
    missing backend at a shape is almost always an OOM, and an OOM is not a
    slow measurement. Carrying it forward as NaN and letting a later
    `dropna` handle it would work too, but it puts the decision somewhere a
    reader of the output cannot see it.
    """
    a, b = backends
    missing = [x for x in backends if x not in set(df["backend"])]
    if missing:
        raise CrossoverError(
            f"backend(s) {missing} are not present in these rows, so no "
            f"crossover between {a!r} and {b!r} can be formed")
    if latency_column not in df.columns:
        raise CrossoverError(f"no {latency_column!r} column")

    sub = df[df["backend"].isin(backends)]
    if "ok" in sub.columns:
        sub = sub[sub["ok"].astype(bool)]
    wide = sub.pivot_table(index=["gpu_name", "seq_len", "batch"],
                           columns="backend", values=latency_column,
                           aggfunc="median")
    wide = wide.dropna(how="any").reset_index()
    wide[f"{b}_over_{a}"] = wide[b] / wide[a]
    wide["winner"] = [a if x > 1 else b for x in wide[f"{b}_over_{a}"]]

    # Whether the instrument can tell this winner from a tie.
    #
    # A ratio built on a 2 ms kernel cannot resolve a 20% difference: two L4
    # hosts in this study, same driver and same config, disagree with each
    # other by up to 27.2% below 5 ms. `resolvable` is what separates "GLA is
    # faster here" from "GLA measured faster here once".
    #
    # This is more forgiving than it sounds for a crossover specifically. The
    # ratios here run from 0.1 to 3.7, so most cells clear their band easily;
    # it is the cells NEAR PARITY that fail, and a crossover's neighbourhood
    # is exactly where those live. The bar bites hardest precisely where the
    # claim is most interesting -- which is the point.
    wide["min_latency_ms"] = wide[[a, b]].min(axis=1)
    wide["resolution"] = wide["min_latency_ms"].map(ratio_resolution)
    wide["margin"] = (wide[f"{b}_over_{a}"] - 1.0).abs()
    wide["resolvable"] = wide["margin"] > wide["resolution"]
    return wide


def matched_cells(cross: pd.DataFrame, *,
                  backends: tuple[str, str] = DEFAULT_PAIR) -> pd.DataFrame:
    """Restrict a crossover table to cells present on EVERY architecture.

    Adds one `winner_<arch>` column per architecture and a `disagrees` flag.
    `disagrees` is the hardware-conditional result: the same two kernels at
    the same shape, with a different winner depending only on the card.

    Reading a disagreement off the unrestricted table would work by accident
    here and not in general -- the A100 has three batches at 8192 where the L4
    has one, so an unmatched comparison weighs three cells against one and
    calls the difference hardware.
    """
    a, b = backends
    ratio = f"{b}_over_{a}"
    if ratio not in cross.columns:
        raise CrossoverError(f"no {ratio!r} column; pass the same backends "
                             f"used to build the table")

    piv = cross.pivot_table(index=["seq_len", "batch"], columns="gpu_name",
                            values=ratio).dropna(how="any")
    arch_columns = list(piv.columns)
    if len(arch_columns) < 2:
        raise CrossoverError(
            f"only {len(arch_columns)} architecture(s) have any shared cell; "
            f"a crossover comparison needs at least two")
    for col in arch_columns:
        piv[f"winner_{_short(col)}"] = [a if v > 1 else b for v in piv[col]]
    winner_columns = [f"winner_{_short(c)}" for c in arch_columns]
    piv["disagrees"] = piv[winner_columns].nunique(axis=1) > 1

    # A cell only counts as a disagreement if BOTH sides could resolve their
    # own verdict. Otherwise the "disagreement" may be one card measuring a
    # tie twice and landing on opposite sides of 1.0.
    res = cross.pivot_table(index=["seq_len", "batch"], columns="gpu_name",
                            values="resolvable", aggfunc="min")
    piv["both_resolvable"] = res.reindex(piv.index).all(axis=1)
    return piv.reset_index()


def _short(gpu_name) -> str:
    """`NVIDIA A100-SXM4-80GB` -> `A100-SXM4-80GB`. Column labels only; the
    full name stays in the data."""
    return str(gpu_name).replace("NVIDIA ", "").replace(" ", "_")


def crossover_point(cross: pd.DataFrame, *, gpu_name: str, batch: int,
                    backends: tuple[str, str] = DEFAULT_PAIR,
                    resolvable_only: bool = False):
    """Lowest seq_len at which `backends[0]` first wins, on one card at one
    batch, or None if it never does within the measured range.

    `None` is a result, not a gap: "GLA had not overtaken FA2 anywhere we
    measured" is a different statement from "GLA overtakes FA2 above the range"
    and this function makes neither. The caller has the measured range and can
    say which.
    """
    a, b = backends
    ratio = f"{b}_over_{a}"
    rows = cross[(cross["gpu_name"] == gpu_name) & (cross["batch"] == batch)]
    if resolvable_only and "resolvable" in rows.columns:
        rows = rows[rows["resolvable"]]
    wins = rows[rows[ratio] > 1].sort_values("seq_len")
    return int(wins["seq_len"].iloc[0]) if not wins.empty else None


def advantage_shift(matched: pd.DataFrame, *, reference: str, comparison: str,
                    ) -> pd.DataFrame:
    """Per-cell ratio-of-ratios: how much larger `comparison`'s advantage is
    than `reference`'s, at identical (seq_len, batch).

    **Why this is the stronger form of the claim.** A crossover point is where
    one curve crosses 1.0, so pinning it depends on the cells nearest parity --
    the cells whose margin is smallest relative to measurement noise, and the
    ones an instrument resolves worst. On the 2026-09-06 data the matched cell
    that would pin it missed its resolution bar by 0.007.

    The direction and size of the gap between the two architectures does not
    have that problem. It is measured at every matched cell, including the
    ones far from parity where the ratios are large and the noise is
    proportionally small, and a consistent sign across many cells is evidence
    that no single cell has to carry.

    Returns the matched frame with `advantage_shift` (comparison / reference)
    and `comparison_ahead` (whether it exceeds 1.0) added.
    """
    for col in (reference, comparison):
        if col not in matched.columns:
            raise CrossoverError(
                f"{col!r} is not a column of the matched table; pass the "
                f"gpu_name values as they appear there, e.g. "
                f"{[c for c in matched.columns if 'NVIDIA' in str(c)]}")
    out = matched.copy()
    out["advantage_shift"] = out[comparison] / out[reference]
    out["comparison_ahead"] = out["advantage_shift"] > 1.0
    return out


def sign_consistency(shifted: pd.DataFrame, *, min_seq_len: int = 0) -> dict:
    """How many matched cells agree on the direction of the shift.

    A sign summary rather than a mean: the per-cell magnitudes are ratios of
    ratios and their errors compound, while the SIGN is robust to that. `k` of
    `n` cells pointing the same way is the claim; the magnitude range is
    context for it.

    `min_seq_len` excludes short cells without deleting them from the record.
    The exclusion is a real one on this dataset -- at 1024/batch 1 the A100
    runs a 0.09 ms kernel, the smallest measurement in the study -- and it is
    a parameter rather than a hardcoded cut so a reader sees both numbers.
    """
    sub = shifted[shifted["seq_len"] >= min_seq_len]
    if sub.empty:
        raise CrossoverError(f"no matched cells at seq_len >= {min_seq_len}")
    ahead = int(sub["comparison_ahead"].sum())
    return {
        "n": len(sub),
        "ahead": ahead,
        "against": len(sub) - ahead,
        "min_shift": float(sub["advantage_shift"].min()),
        "max_shift": float(sub["advantage_shift"].max()),
        "median_shift": float(sub["advantage_shift"].median()),
        "min_seq_len": min_seq_len,
    }
