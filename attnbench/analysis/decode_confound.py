"""The decode-kernel confound in Stage 3's end-to-end totals, and what
correcting for it does to Stage 6's dominance result.

What the confound is
--------------------
`block_sparse` has no decode path, so `grid_configs.decode_backend_for` falls
back to `DENSE_DECODE_BACKEND = "sdpa_math"`. The dense arm is `sdpa_flash`
and decodes through **itself**. Sparsity is applied during prefill only
(decision C), so decode is dense in both arms -- but through *different
kernels*, on the phase that dominates the bill at batch 1.

Stage 5 measured the gap: +23.5% / +22.1% / +62.4% per decode token at
2048 / 4096 / 8192, which is 18% / 16% / 29% of the sparse arm's end-to-end
total. `grid_configs.py` names this exact hazard in its own docstring --
"a row labelled `sdpa_math` and one labelled `sdpa_flash` are the same class
and different kernels, which is exactly the confound `SDPABackend` exists to
remove" -- and the decode fallback reintroduces it *between arms*. The choice
was recorded on every row the whole time. Nothing compared the two arms'
decode backends until the phases were measured apart, which is the case for
Stage 5 in one sentence: **an end-to-end number cannot surface a confound
that lives inside it.**

The second confound, which was already in the measured result
--------------------------------------------------------------
Correcting only the decode kernel gives 0/31 dominated, and that number is
wrong. On `vt` the arms do not generate the same number of tokens -- dense
35.2-38.8, sparse 27.7-34.6 -- so their mean end-to-end latencies are not
comparable in the first place. An arm that stops earlier finishes sooner for
reasons that have nothing to do with how fast attention is.

`niah_single` is clean: every arm generates exactly 14.0 tokens.

So there are three quantities here, and only the third answers "is sparse
attention faster":

| quantity | decode kernel | generation length | what it is |
|---|---|---|---|
| `measured` | different | different | what Stage 3 banked; what Stage 6 ranked |
| `decode_corrected` | matched | **different** | intermediate. Do not report alone -- inflated on `vt` |
| `normalized` | matched | matched | the unconfounded comparison |

THIS IS DERIVED, NOT MEASURED
-----------------------------
`normalized` is a model: `prefill(band, sparsity) + (n-1) * decode_dense(band)`,
built from Stage 5's phase measurements. Those were taken on random token ids
at exactly the band length, not on the RULER prompts (which range 4000-8196
inside the 8192 band), with n=10 reps rather than Stage 3's n=900. It
reconstructs what the arms *would* have cost under a matched decode kernel;
it is not a measurement of those operating points. Every consumer gets it
under a name that says so, and the measured column travels beside it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd


class PhaseEraMismatch(ValueError):
    """The phases feeding the penalty are from a different decode era than
    the rows being corrected."""


# The measured separation between the two decode eras, from the banked
# Stage 5 phases (block_sparse decode_step minus dense decode_step):
#
#   sdpa_math era  (results/stage5/)            +8.1 .. +21.8 ms/token
#   sdpa_flash era (results/stage5_ols/,
#                   _flashdecode/, _32768/)     +0.29 .. +0.75 ms/token
#
# Anything under this is same-kernel jitter, not a kernel difference. The
# two populations are more than an order of magnitude apart, so the exact
# cut is not load-bearing -- what matters is that the check exists.
SAME_KERNEL_PENALTY_MS = 2.0


@dataclass(frozen=True)
class PhaseModel:
    """Stage 5's measured phases, in the form the correction needs."""

    prefill_ms: dict[tuple[int, Optional[float]], float]
    decode_step_ms: dict[tuple[int, Optional[float]], float]
    dense_backend: str

    def dense_decode(self, band: int) -> float:
        return self.decode_step_ms[(band, None)]

    def decode_penalty(self, band: int, sparsity: Optional[float]) -> float:
        """ms per generated token that the sparse arm pays purely for
        decoding through `sdpa_math` instead of `sdpa_flash`. Zero for the
        dense arm, which decodes through itself."""
        if sparsity is None:
            return 0.0
        return self.decode_step_ms[(band, sparsity)] - self.dense_decode(band)

    def normalized_total(self, band: int, sparsity: Optional[float],
                         n_generated: float) -> float:
        """What this arm costs at `n_generated` tokens with the dense decode
        kernel -- the only term sparsity is allowed to change is prefill,
        which is the whole point of prefill-only sparsity."""
        # (n - 1): the prefill forward emits the first token's logits, so n
        # tokens cost one prefill plus n-1 decode steps. See
        # phase_timing.reconcile and silent_failure_patterns #27.
        return (self.prefill_ms[(band, sparsity)]
                + (n_generated - 1) * self.dense_decode(band))


def phase_model_from(phases: pd.DataFrame, dense_backend: str = "sdpa_flash"
                     ) -> PhaseModel:
    """Build the model from a Stage 5 `phases.parquet`."""
    def table(phase):
        out = {}
        for r in phases[phases.phase == phase].itertuples():
            sp = None if pd.isna(r.sparsity) else float(r.sparsity)
            out[(int(r.context_length), sp)] = float(r.ms_mean)
        return out

    prefill, decode = table("prefill"), table("decode_step")
    missing = [k for k in prefill if k not in decode]
    if missing:
        # A prefill with no decode step cannot be normalized, and silently
        # dropping it would shrink the comparison without saying so.
        raise ValueError(f"phases lack a decode_step for {sorted(missing)}")
    for band in {b for b, _ in decode}:
        if (band, None) not in decode:
            raise ValueError(
                f"no dense decode_step at band {band}; the correction has "
                f"nothing to normalize against")
    return PhaseModel(prefill_ms=prefill, decode_step_ms=decode,
                      dense_backend=dense_backend)


@dataclass(frozen=True)
class CorrectedPoint:
    """One Stage 6 operating point, carrying all three quantities.

    `measured` is what was banked. The other two are derived, and named so
    that a reader of the parquet cannot mistake one for the other.
    """

    task: str
    context_length: int
    epsilon: float
    backend: str
    sparsity: Optional[float]
    margin: float
    is_dense_reference: bool

    n_generated_own: float          # what this arm actually generated
    n_generated_common: float       # the dense arm's, used for normalization
    decode_penalty_ms: float        # per token, what the correction removed

    measured_ms: float
    decode_corrected_ms: float
    normalized_ms: float

    dominated_measured: bool
    dominated_normalized: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _dominates(a_lat: float, a_margin: float,
               b_lat: float, b_margin: float) -> bool:
    """Identical rule to `pareto._dominates`, restated on scalars so this
    module can apply it to a corrected latency without rebuilding
    OperatingPoints. Kept literally in step with it -- a test asserts the
    two agree on the same inputs rather than trusting the copy."""
    at_least_as_good = a_lat <= b_lat and a_margin >= b_margin
    strictly_better = a_lat < b_lat or a_margin > b_margin
    return at_least_as_good and strictly_better


def _assert_phase_era_matches(model: PhaseModel, pareto: pd.DataFrame,
                              arms_share_decode_backend: bool) -> None:
    penalties = {}
    for r in pareto.itertuples():
        if r.is_dense_reference:
            continue
        sp = None if pd.isna(r.sparsity) else float(r.sparsity)
        band = int(r.context_length)
        if (band, sp) in model.decode_step_ms:
            penalties[(band, sp)] = model.decode_penalty(band, sp)
    if not penalties:
        return
    worst = max(abs(v) for v in penalties.values())
    looks_matched = worst < SAME_KERNEL_PENALTY_MS

    if looks_matched and not arms_share_decode_backend:
        raise PhaseEraMismatch(
            f"the rows being corrected decoded through different kernels per "
            f"arm, but the largest decode penalty these phases can supply is "
            f"{worst:.3f} ms/token (< {SAME_KERNEL_PENALTY_MS} ms). These "
            f"phases were measured after DENSE_DECODE_BACKEND changed to "
            f"sdpa_flash, so both their arms already decode through the same "
            f"kernel and they cannot price a confound the rows still carry. "
            f"`decode_corrected_ms` would be a no-op wearing the name of a "
            f"correction. Use phases from the same era as the rows "
            f"(results/stage5/ for sdpa_math-era rows).")
    if not looks_matched and arms_share_decode_backend:
        raise PhaseEraMismatch(
            f"the rows being corrected all decoded through one kernel, so "
            f"there is no decode-kernel penalty to remove -- but these "
            f"phases carry one of {worst:.3f} ms/token. They are from the "
            f"sdpa_math era and the rows are not. Subtracting this would "
            f"invent a correction for a confound that is not in the data.")


def correct(pareto: pd.DataFrame, phases: pd.DataFrame,
            n_generated: dict, dense_backend: str = "sdpa_flash", *,
            arms_share_decode_backend: bool) -> list[CorrectedPoint]:
    """Recompute every Stage 6 point under a matched decode kernel and a
    matched generation length.

    `n_generated` maps (backend, task, band, sparsity) -> mean tokens
    generated, from the same Stage 3 rows Stage 6's latency came from.

    `arms_share_decode_backend` says whether the Stage 3 rows behind
    `pareto` were themselves confounded -- get it from
    `decode_backend_guard.cross_arm_cells` on those rows, never by assuming.
    It is required rather than defaulted because the whole failure below is
    a caller who did not think about it.

    Why this argument exists
    ------------------------
    `decode_corrected_ms` subtracts a penalty measured by `phases`. Nothing
    tied the era of `phases` to the era of the rows, and the two banked sets
    are three days and one `DENSE_DECODE_BACKEND` change apart. Feeding
    `results/stage5_ols/phases.parquet` (both arms already on `sdpa_flash`)
    to the `results/stage3_s1b/` rows (sparse arm on `sdpa_math`) gives a
    penalty of 0.29-0.75 ms instead of 8-22 ms: the column is then a no-op
    that *looks* like a correction, and the confound it names survives it
    untouched. That is what `results/stage6/decode_corrected.parquet` was
    from 2026-09-08 until this check.

    So the correction now checks itself, in both directions: a confounded
    set demands a real penalty, and a matched set demands a near-zero one.
    """
    model = phase_model_from(phases, dense_backend)
    _assert_phase_era_matches(model, pareto, arms_share_decode_backend)
    out: list[CorrectedPoint] = []

    for (task, band, eps), cell in pareto.groupby(
            ["task", "context_length", "epsilon"]):
        dense_rows = cell[cell.is_dense_reference]
        if dense_rows.empty:
            raise ValueError(
                f"cell {(task, band, eps)} has no dense reference point; "
                f"pareto.compute_pareto_frontiers seeds one into every cell, "
                f"so its absence means the frame was filtered after the fact")
        d = dense_rows.iloc[0]
        band = int(band)
        n_common = float(n_generated[(d.backend, task, band, None)])
        dense_norm = model.normalized_total(band, None, n_common)

        for r in cell.itertuples():
            sp = None if pd.isna(r.sparsity) else float(r.sparsity)
            n_own = float(n_generated[(r.backend, task, band, sp)])
            penalty = model.decode_penalty(band, sp)
            dec_corr = float(r.latency_ms) - (n_own - 1) * penalty
            norm = model.normalized_total(band, sp, n_common)
            out.append(CorrectedPoint(
                task=task, context_length=band, epsilon=float(eps),
                backend=r.backend, sparsity=sp, margin=float(r.margin),
                is_dense_reference=bool(r.is_dense_reference),
                n_generated_own=n_own, n_generated_common=n_common,
                decode_penalty_ms=penalty,
                measured_ms=float(r.latency_ms),
                decode_corrected_ms=dec_corr, normalized_ms=norm,
                dominated_measured=bool(r.dominated_by_dense),
                dominated_normalized=(
                    False if r.is_dense_reference else
                    _dominates(dense_norm, float(d.margin),
                               norm, float(r.margin))),
            ))
    return out


def to_dataframe(points: list[CorrectedPoint]) -> pd.DataFrame:
    return pd.DataFrame([p.to_dict() for p in points])


def movement(points: list[CorrectedPoint]) -> dict[str, list[CorrectedPoint]]:
    """Which points change dominance status, in both directions.

    Both directions on purpose. A correction that only ever frees points is
    a correction nobody checked: this one moves three `vt` points INTO the
    dominated set, because they looked fast only by generating fewer tokens.
    """
    sparse = [p for p in points if not p.is_dense_reference]
    return {
        "freed": [p for p in sparse
                  if p.dominated_measured and not p.dominated_normalized],
        "newly_dominated": [p for p in sparse
                            if not p.dominated_measured and p.dominated_normalized],
    }
