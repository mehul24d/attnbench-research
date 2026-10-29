"""What a figure or table reads from banked results, checked once.

  - A committed prediction is read only as `predict.commit.load_committed`
    returns it, with its sync time set. `verify_bytes` alone checks the
    hash but not the K9 order (audit C2, finding 3), so a prediction
    without `synced_at` is refused here.
  - A ground-truth result (`ground_truth.time_arms`) is read only when it is
    untraced and complete. Its rows are keyed by arm alone (they carry no
    density, length or batch; audit C3, finding 2), so the caller states
    which committed cell each arm's row measures, and the arm must be the
    cell's arm.
  - Ground truth is in milliseconds and predictions in seconds;
    `ms_to_seconds` and `seconds_to_ms` are the only conversions.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence, Union

from ..gt_schema import RowFieldError, check_row
from ..predict.commit import CommittedPrediction, cell_key

MS_PER_S = 1000.0


class ResultsError(ValueError):
    """Banked results a figure or table refuses to read."""


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ms_to_seconds(ms: float) -> float:
    return ms / MS_PER_S


def seconds_to_ms(s: float) -> float:
    return s * MS_PER_S


def check_loaded(cp: CommittedPrediction) -> CommittedPrediction:
    if not isinstance(cp, CommittedPrediction):
        raise ResultsError(f"expected a CommittedPrediction, got {type(cp).__name__}")
    if cp.synced_at is None:
        raise ResultsError("this prediction has no sync time: read it with "
                           "load_committed, which checks it was synced before the "
                           "timed workload (K9), not with verify_bytes alone")
    return cp


def check_result(result: Mapping, name: str = "the result") -> Mapping:
    if result.get("traced") is not False:
        raise ResultsError(f"{name} is traced or does not say; ground truth is untraced")
    if result.get("complete") is not True:
        raise ResultsError(f"{name} is not complete (too few clean rounds)")
    return result


def measured_seconds(result: Mapping, arm: str, statistic: str,
                     name: str = "the result") -> float:
    """The arm's `statistic` (median, p90 or p99) from one result, in seconds."""
    check_result(result, name)
    rows = [r for r in result.get("rows", ()) if r.get("arm") == arm]
    if len(rows) != 1:
        raise ResultsError(f"{name} has {len(rows)} rows for arm {arm!r}")
    try:
        check_row(rows[0], name)
    except RowFieldError as e:
        raise ResultsError(str(e)) from e
    v = rows[0].get(f"prefill_ms_{statistic}")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise ResultsError(f"{name}: arm {arm!r} has no positive {statistic} ({v!r})")
    return ms_to_seconds(float(v))


def measured_cells(results: Sequence[tuple[str, Mapping, Mapping[str, Sequence]]], *,
                   statistic: str) -> dict:
    """{cell key: seconds} from (name, result, {arm: cell}) triples, where a
    cell is (model, code path, arm, density, context length, batch)."""
    out: dict = {}
    for name, result, arms in results:
        for arm, cell in arms.items():
            k = cell_key(*cell)
            if k.arm != arm:
                raise ResultsError(f"{name}: arm {arm!r}'s row is stated to measure a "
                                   f"cell of arm {k.arm!r}")
            if k in out:
                raise ResultsError(f"cell {k} is measured twice")
            out[k] = measured_seconds(result, arm, statistic, name)
            r = [r for r in result["rows"] if r.get("arm") == arm][0]
            if r["context_length"] != k.context_length:
                raise ResultsError(f"{name}: arm {arm!r}'s row was measured at context length "
                                   f"{r['context_length']}, not the stated {k.context_length}")
            if r["batch"] != k.batch:
                raise ResultsError(f"{name}: arm {arm!r}'s row was measured at batch "
                                   f"{r['batch']}, not the stated {k.batch}")
    if not out:
        raise ResultsError("no measured cells")
    return out
