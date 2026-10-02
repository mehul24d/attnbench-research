"""Exact paired non-inferiority for binary accuracy (audit T4).

`matched.py`'s paired bootstrap is degenerate where the T4 audit pointed:
at a ceiling (or a floor) every resample has zero variance, and the bound it
returns is not a bound. This module answers the same question -- is the
sparse arm's accuracy at least dense's minus a margin, on the same examples
-- with an interval whose coverage is guaranteed at every n and every true
rate, so it means the same thing at 6% as at 60%.

**The interval.** Each example falls into one of four cells by (dense
correct, sparse correct). Let b count sparse-right/dense-wrong, c count
dense-right/sparse-wrong, out of n pairs. Then

    diff = p_sparse - p_dense = p_b - p_c

and b ~ Binomial(n, p_b), c ~ Binomial(n, p_c) marginally (they are two
cells of one multinomial). With one-sided Clopper-Pearson bounds at alpha/2
each,

    P(p_b >= L_b  and  p_c <= U_c) >= 1 - alpha        (union bound)

so `L_b - U_c` is a lower confidence bound for diff with coverage at least
1 - alpha, exactly -- no asymptotics, no resampling. It is conservative
(the union bound and Clopper-Pearson both over-cover), which is the right
direction for a claim of "no loss": it can fail to certify a true
non-inferiority, never certify a false one at more than rate alpha.

**What it can certify at small n.** Even with zero discordant pairs, U_c is
not zero: at n=50 and alpha=0.025 the best attainable bound is -8.4 points.
`attainable_lower_bound` reports this so a plan states its reachable margin
before any data exists (docs/t4_sparse_pilot.md).

Pairing is on (task, example_id) and is total or refused: an example
present in one arm and not the other is a different population, and silently
dropping it is how a paired test turns into an unpaired one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from scipy.stats import beta

from .matched import band_for


def cp_lower(k: int, n: int, a: float) -> float:
    """One-sided Clopper-Pearson lower bound at level a (P(p < L) <= a)."""
    return 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))


def cp_upper(k: int, n: int, a: float) -> float:
    """One-sided Clopper-Pearson upper bound at level a (P(p > U) <= a)."""
    return 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))


def paired_lower_bound(b: int, c: int, n: int, alpha: float = 0.025) -> float:
    """Lower confidence bound for p_sparse - p_dense, as a fraction."""
    if not (0 <= b and 0 <= c and b + c <= n):
        raise ValueError(f"impossible discordant counts b={b}, c={c} of n={n}")
    return cp_lower(b, n, alpha / 2) - cp_upper(c, n, alpha / 2)


def attainable_lower_bound(n: int, alpha: float = 0.025) -> float:
    """The best bound n pairs can give: no discordant pairs at all."""
    return paired_lower_bound(0, 0, n, alpha)


@dataclass(frozen=True)
class ExactNIResult:
    task: str
    band: int
    arm: str
    n: int
    dense_correct: int
    sparse_correct: int
    b_sparse_only: int
    c_dense_only: int
    diff_pts: float
    lower_pts: float
    margin_pts: float
    alpha: float
    non_inferior: bool
    dense_ci_pts: tuple
    sparse_ci_pts: tuple

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dense_ci_lo"], d["dense_ci_hi"] = d.pop("dense_ci_pts")
        d["sparse_ci_lo"], d["sparse_ci_hi"] = d.pop("sparse_ci_pts")
        return d


def _two_sided_cp(k: int, n: int, alpha: float) -> tuple:
    return (100 * cp_lower(k, n, alpha / 2), 100 * cp_upper(k, n, alpha / 2))


def paired_exact_ni(dense: np.ndarray, sparse: np.ndarray, *, margin_pts: float,
                    alpha: float = 0.025, task: str = "", band: int = 0,
                    arm: str = "") -> ExactNIResult:
    """One (task, band, arm) test on aligned boolean arrays."""
    dense = np.asarray(dense, dtype=bool)
    sparse = np.asarray(sparse, dtype=bool)
    if dense.shape != sparse.shape or dense.ndim != 1 or len(dense) == 0:
        raise ValueError(f"need two aligned non-empty 1-D arrays, got "
                         f"{dense.shape} and {sparse.shape}")
    n = len(dense)
    b = int((sparse & ~dense).sum())
    c = int((dense & ~sparse).sum())
    lower = 100 * paired_lower_bound(b, c, n, alpha)
    return ExactNIResult(
        task=task, band=band, arm=arm, n=n,
        dense_correct=int(dense.sum()), sparse_correct=int(sparse.sum()),
        b_sparse_only=b, c_dense_only=c,
        diff_pts=100 * (b - c) / n, lower_pts=lower, margin_pts=margin_pts,
        alpha=alpha, non_inferior=lower > -margin_pts,
        # two-sided at 2*alpha, i.e. 95% for alpha=0.025: the dense pilot's
        # intervals, so the two documents quote one kind of interval
        dense_ci_pts=_two_sided_cp(int(dense.sum()), n, 2 * alpha),
        sparse_ci_pts=_two_sided_cp(int(sparse.sum()), n, 2 * alpha))


# Score sources whose rows choose their own blocks from a threshold, so the
# arm is named by the threshold and the sparsity column is empty.
THRESHOLD_ARMS = frozenset({"xattention_inline"})


def arm_label(row: pd.Series) -> str:
    """`<score_source>@<sparsity>` for a fixed-sparsity row, and
    `<score_source>@<xattn_threshold>` for an arm that selects its own
    blocks (XAttention), whose sparsity is an outcome, not a setting."""
    if row["score_source"] in THRESHOLD_ARMS:
        setting = row.get("xattn_threshold")
        if setting is None or pd.isna(setting):
            raise ValueError(f"a {row['score_source']} row has no xattn_threshold: "
                             f"its arm is unknown")
        return f"{row['score_source']}@{float(setting):g}"
    return f"{row['score_source']}@{row['sparsity']:g}"


def run_exact_ni(dense_rows: pd.DataFrame, sparse_rows: pd.DataFrame, *,
                 bands, margin_pts: float, alpha: float = 0.025) -> pd.DataFrame:
    """Every (task, band, arm) in `sparse_rows` against `dense_rows`.

    Arms are told apart by score_source AND sparsity, never by backend: the
    oracle and the inline estimator are both `block_sparse`.
    """
    if dense_rows["backend_role"].ne("dense_reference").any():
        raise ValueError("dense_rows holds non-dense rows")
    if sparse_rows["score_source"].isna().any():
        raise ValueError("a sparse row has no score_source: its arm is unknown")
    dense = dense_rows.assign(band=[band_for(int(x), bands) for x in dense_rows.context_length])
    if dense.duplicated(["task", "example_id"]).any():
        raise ValueError("dense rows repeat an example: two dense arms are mixed")
    sparse = sparse_rows.assign(
        band=[band_for(int(x), bands) for x in sparse_rows.context_length],
        arm=sparse_rows.apply(arm_label, axis=1))

    out = []
    for (task, band, arm), grp in sparse.groupby(["task", "band", "arm"], sort=True):
        d = dense[(dense.task == task) & (dense.band == band)].set_index("example_id")
        if grp.example_id.duplicated().any():
            raise ValueError(f"{task}/{band}/{arm}: an example appears twice")
        s = grp.set_index("example_id")
        if set(d.index) != set(s.index):
            raise ValueError(
                f"{task}/{band}/{arm}: dense and sparse examples differ "
                f"({len(set(d.index) ^ set(s.index))} unpaired). A paired test "
                f"on the intersection would be a different population.")
        ids = sorted(s.index)
        row = paired_exact_ni(d.loc[ids, "correct"].to_numpy(),
                              s.loc[ids, "correct"].to_numpy(),
                              margin_pts=margin_pts, alpha=alpha,
                              task=task, band=int(band), arm=arm).to_dict()
        # Descriptive only: how sparse a self-selecting arm actually was.
        row["mean_realised_density"] = (
            float(s["realised_density"].mean()) if "realised_density" in s.columns
            and s["realised_density"].notna().all() else float("nan"))
        out.append(row)
    return pd.DataFrame(out)

