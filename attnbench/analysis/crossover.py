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
    return piv.reset_index()


def _short(gpu_name) -> str:
    """`NVIDIA A100-SXM4-80GB` -> `A100-SXM4-80GB`. Column labels only; the
    full name stays in the data."""
    return str(gpu_name).replace("NVIDIA ", "").replace(" ", "_")


def crossover_point(cross: pd.DataFrame, *, gpu_name: str, batch: int,
                    backends: tuple[str, str] = DEFAULT_PAIR):
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
    wins = rows[rows[ratio] > 1].sort_values("seq_len")
    return int(wins["seq_len"].iloc[0]) if not wins.empty else None
