"""Results tables for the T3.x outputs: the P1 session (T3.1) and the
numbers the Gate A decision reads (T3.3).

  - `cell_table`: one row per committed cell that was measured, with the
    measured statistic and, per predictor, the committed prediction, its
    relative error and whether that error is inside the noise floor. The
    numbers are `predict.residuals`', not recomputed here.
  - `predictor_summary`: per predictor (and, if asked, per arm class, which
    H1 reports separately), the count, the count inside the floor, the
    median and the largest absolute relative error and the signed mean.
    Gate A asks whether the recursion beats the sum and the step form on
    P1 (Section 7); the table reports, the decision is the researcher's.
  - Committed cells with no measurement, and measured cells with no
    prediction, are listed by `unmatched`, never dropped silently.

The noise floor is a lock value with no default (Section 5.6); a table made
before the lock with a provisional floor should say so where it is shown.
T3.2's between-session table reads `lock.floors`, which is on the C6 branch,
and is left to the integration.

Times are seconds; relative errors are fractions. CSV keeps full precision;
Markdown rounds for reading (`MD_FORMAT`).
"""

from __future__ import annotations

import csv
import io
from typing import Iterable, Mapping, Optional, Sequence

from ..predict import residuals as R
from ..predict.commit import COMPUTED, CommittedPrediction
from .results import check_loaded

CELL_FIELDS = ("model", "code_path", "arm", "density", "context_length", "batch",
               "n_synchronisations", "arm_class", "measured_s")
SUMMARY_FIELDS = ("n", "n_inside_floor", "median_abs_relative", "mean_relative",
                  "max_abs_relative")
MD_FORMAT = {"_s": "{:.6f}", "relative": "{:+.4f}", "abs_relative": "{:.4f}"}


class TableError(ValueError):
    """Inputs a results table cannot be made from."""


def _residual_set(cp, measured, *, dense_arms, relative_floor, predictors):
    check_loaded(cp)
    return R.residuals(cp, measured, statistic=cp.statistic, dense_arms=dense_arms,
                       relative_floor=relative_floor, predictors=list(predictors))


def cell_table(cp: CommittedPrediction, measured: Mapping, *, dense_arms: Iterable[str],
               relative_floor: float,
               predictors: Sequence[str] = COMPUTED) -> tuple[tuple[str, ...], list[dict]]:
    """(columns, rows). `measured` is {cell key: seconds}
    (`results.measured_cells`)."""
    predictors = tuple(predictors)
    if not predictors or len(set(predictors)) != len(predictors):
        raise TableError(f"name each predictor once, got {predictors}")
    rs = _residual_set(cp, measured, dense_arms=dense_arms, relative_floor=relative_floor,
                       predictors=predictors)
    rows: dict = {}
    for r in rs.residuals:
        k = (r.model, r.code_path, r.arm, r.density, r.context_length, r.batch)
        row = rows.setdefault(k, {
            "model": r.model, "code_path": r.code_path, "arm": r.arm, "density": r.density,
            "context_length": r.context_length, "batch": r.batch,
            "n_synchronisations": r.n_synchronisations, "arm_class": r.arm_class,
            "measured_s": r.measured})
        row[f"{r.predictor}_s"] = r.predicted
        row[f"{r.predictor}_relative"] = r.relative
        row[f"{r.predictor}_inside_floor"] = r.inside_floor
    cols = CELL_FIELDS + tuple(f"{p}_{x}" for p in predictors
                               for x in ("s", "relative", "inside_floor"))
    return cols, [rows[k] for k in sorted(rows)]


def predictor_summary(cp: CommittedPrediction, measured: Mapping, *,
                      dense_arms: Iterable[str], relative_floor: float,
                      by_arm_class: bool,
                      predictors: Sequence[str] = COMPUTED) -> tuple[tuple[str, ...], list[dict]]:
    rs = _residual_set(cp, measured, dense_arms=dense_arms, relative_floor=relative_floor,
                       predictors=predictors)
    if not rs.residuals:
        raise TableError("no committed cell was measured")
    by = ("predictor", "arm_class") if by_arm_class else ("predictor",)
    order = {p: i for i, p in enumerate(predictors)}
    out = []
    for g in R.summarise(rs.residuals, by):
        row = dict(zip(by, g.group))
        row.update({f: getattr(g, f) for f in SUMMARY_FIELDS})
        out.append(row)
    out.sort(key=lambda r: (order[r["predictor"]],) + tuple(r[f] for f in by[1:]))
    return by + SUMMARY_FIELDS, out


def unmatched(cp: CommittedPrediction, measured: Mapping, *, dense_arms: Iterable[str],
              relative_floor: float) -> dict:
    rs = _residual_set(cp, measured, dense_arms=dense_arms, relative_floor=relative_floor,
                       predictors=COMPUTED)
    return {"unmeasured": [list(k) for k in rs.unmeasured],
            "unpredicted": [list(k) for k in rs.unpredicted]}


def to_csv(columns: Sequence[str], rows: Sequence[Mapping]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    w.writeheader()
    for r in rows:
        if set(r) != set(columns):
            raise TableError(f"a row with fields {sorted(r)}, expected {list(columns)}")
        w.writerow({k: repr(v) if isinstance(v, float) else v for k, v in r.items()})
    return buf.getvalue()


def _md(col: str, v) -> str:
    if isinstance(v, bool) or not isinstance(v, float):
        return str(v)
    for suffix in ("abs_relative", "relative", "_s"):
        if col.endswith(suffix):
            return MD_FORMAT[suffix].format(v)
    return f"{v:g}"


def to_markdown(columns: Sequence[str], rows: Sequence[Mapping],
                caption: Optional[str] = None) -> str:
    lines = [f"**{caption}**", ""] if caption else []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_md(c, r[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"
