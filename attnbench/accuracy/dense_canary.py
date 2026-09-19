"""The dense canary: does a re-run's dense arm reproduce the banked one,
example by example, before any sparse row is measured?

The dense arm builds no mask and consults no scores, so nothing a mask-rule
change touches can reach it. If it produces different text on the same
example ids, something about the environment differs -- model, tokenizer,
kernels, example generation -- and every sparse number from that run is
suspect before anyone has looked at it. So this is a gate, not a report: it
compares PREDICTIONS (not scores, which can agree while text differs), and a
single mismatch fails.

Identity is (task, example_id). `expected` and `context_length` must match
too: they prove the example is the same example, generated and tokenized the
same way, rather than one that happens to share an id.

It refuses to pass vacuously. The number of examples compared is returned and
recorded, and a check that compared fewer than required -- including zero --
fails. An anti-vacuity guard is the defect shape this project has found
repeatedly (silent_failure_patterns.md #3, #5, #25, #28, #34, #36), so the
count is a first-class output, not a log line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

KEY = ("task", "example_id")
MUST_MATCH = ("predicted", "expected", "context_length")


class DenseCanaryFailed(RuntimeError):
    """The dense arm did not reproduce, or the check could not be made."""


@dataclass
class CanaryResult:
    backend: str
    n_new_dense: int
    n_checked: int
    mismatches: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.n_checked > 0 and self.n_checked == self.n_new_dense \
            and not self.mismatches

    def to_dict(self) -> dict:
        return {"backend": self.backend, "n_new_dense": self.n_new_dense,
                "n_checked": self.n_checked, "n_mismatches": len(self.mismatches),
                "passed": self.passed, "mismatches": self.mismatches[:20]}


def _dense(df: pd.DataFrame, backend: str, where: str) -> pd.DataFrame:
    missing = [c for c in (*KEY, *MUST_MATCH, "backend") if c not in df.columns]
    if missing:
        raise DenseCanaryFailed(f"{where} lacks columns {missing}")
    d = df[df["backend"] == backend]
    if "sparsity" in d.columns:
        d = d[d["sparsity"].isna()]
    dup = d.duplicated(list(KEY), keep=False)
    if dup.any():
        # Two rows for one example. If they agree it is a harmless resume
        # overlap; if they disagree the file does not know its own answer.
        g = d[dup].groupby(list(KEY))["predicted"].nunique()
        if (g > 1).any():
            raise DenseCanaryFailed(
                f"{where} has {int((g > 1).sum())} example(s) with conflicting "
                f"dense predictions; it cannot serve as a reference")
        d = d.drop_duplicates(list(KEY))
    return d


def check(new: pd.DataFrame, banked: pd.DataFrame, *, backend: str,
          min_checked: int) -> CanaryResult:
    """Compare every new dense row with its banked counterpart.

    Every new dense row must find a banked partner: an unmatched row counts
    against `n_checked`, so a run that silently changed its example ids fails
    rather than checking the overlap and passing. `min_checked` is the floor
    the caller expects (the run's n x tasks); fewer is a failure.
    """
    if min_checked < 1:
        raise ValueError("min_checked must be >= 1; a canary that may check "
                         "nothing is not a canary")
    n = _dense(new, backend, "new run")
    b = _dense(banked, backend, "banked reference")
    joined = n.merge(b, on=list(KEY), how="inner", suffixes=("_new", "_banked"))
    res = CanaryResult(backend=backend, n_new_dense=len(n), n_checked=len(joined))
    for _, r in joined.iterrows():
        diff = {c: (r[f"{c}_new"], r[f"{c}_banked"]) for c in MUST_MATCH
                if str(r[f"{c}_new"]) != str(r[f"{c}_banked"])}
        if diff:
            res.mismatches.append({"task": r["task"], "example_id": r["example_id"],
                                   **{c: {"new": str(v[0]), "banked": str(v[1])}
                                      for c, v in diff.items()}})
    return res


def enforce(res: CanaryResult, *, min_checked: int) -> None:
    """Raise unless the canary passed on at least `min_checked` examples."""
    if res.n_checked < min_checked:
        raise DenseCanaryFailed(
            f"dense canary compared {res.n_checked} example(s), required "
            f"{min_checked} ({res.n_new_dense} new dense rows). A canary that "
            f"checks less than it should is not evidence of anything.")
    if res.n_checked != res.n_new_dense:
        raise DenseCanaryFailed(
            f"{res.n_new_dense - res.n_checked} new dense row(s) have no banked "
            f"counterpart; the run's example ids differ from the reference")
    if res.mismatches:
        m = res.mismatches[0]
        raise DenseCanaryFailed(
            f"dense arm did not reproduce: {len(res.mismatches)} of "
            f"{res.n_checked} examples differ (first: {m}). The environment "
            f"differs from the banked run; no sparse row should be measured.")
