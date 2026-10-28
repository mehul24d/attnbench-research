"""What a fitted baseline may be trained on (plan Section 5.4, "Fitting the
baselines"), shared by the fitted sum (T2.9) and the learned predictor
(T2.10).

A committed prediction for a pairing is trained only on pairings whose
ground truth was banked before the prediction is written. Which pairings
those are is a fact about time that this module cannot see. The caller
(T2.11's prediction commit) passes them as `banked`, and every training set
records them so the prediction file can name them. A full
leave-one-pairing-out fit, trained on every other pairing including those
run later, is the retrospective secondary analysis and is labelled as such.

The rows hold components priced by `recursion.price`, so a fitted baseline
reads the same h, l, d and s_0 as the recursion.

**Which statistic.** Section 5.3's ground truth reports the median, p90 and
p99 over repeats and names none of them as the one predicted, so rows carry
the median unless they say otherwise (researcher's decision, 25 Oct 2026).
The statistic is a field on every row, and a training set refuses rows that
mix statistics: a fit across them would average a median with a tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, NamedTuple, Optional

from .intensity import ModelShape
from .recursion import CostModelError, PricedSequence

COMMITTED = "committed"
RETROSPECTIVE = "retrospective"
Kind = Literal["committed", "retrospective"]

STATISTICS = ("median", "p90", "p99")
GROUND_TRUTH_STATISTIC = "median"


@dataclass(frozen=True)
class HardwareFeatures:
    """Pairing-level link microbenchmark figures (T2.5), bytes per second."""

    pinned_bandwidth: float
    pageable_bandwidth: float


class CellKey(NamedTuple):
    """What identifies a committed prediction inside one pairing's file
    (plan Section 5.3 step 6; Appendix A, 25 Oct 2026): the model, the code
    path, the arm, the density, the context length and the batch.

    A named tuple, in the plan's order, so no consumer reads a field by
    position. `model` and `code_path` are labels: two cells that differ only
    in either are different cells and are never pooled (rule 3).
    """

    model: str
    code_path: str
    arm: str
    density: float
    context_length: int
    batch: int


def make_cell_key(model: str, code_path: str, arm: str, density: float,
                  context_length: int, batch: int) -> CellKey:
    """A `CellKey` with its fields normalised, so equal cells hash equal
    (`0.5` and `1/2`, `4` and `np.int64(4)`). Labels must be non-empty
    strings."""
    for name, v in (("model", model), ("code_path", code_path), ("arm", arm)):
        if not isinstance(v, str) or not v.strip():
            raise CostModelError(
                f"a cell's {name} is a non-empty label, got {v!r}")
    return CellKey(model, code_path, arm, float(density), int(context_length),
                   int(batch))


def partner_of(key: CellKey) -> tuple:
    """What a sparse cell is compared with its dense cell on: model, context
    length and batch. The code path is not matched: it names the
    implementation, which differs between the dense arm and a sparse one by
    construction (a sparse arm on two code paths has one dense partner). The
    dispatcher decides per this triple too (plan Section 5.9)."""
    return (key.model, key.context_length, key.batch)


@dataclass(frozen=True)
class Cell:
    """One arm at one grid cell on one pairing, with its components.

    Everything a predictor may read before the workload is timed: the
    priced operation sequence for this cell (components measured on the
    target, or transferred under H2), the pairing's link figures, and the
    model's shape for the analytic arithmetic intensities (T2.10).

    `model_name` and `code_path` are part of the cell's key (`key`) and are
    required: a cell with no model or code path would be pooled with every
    other. They are labels and never features of a fitted baseline. The
    shape in `model` is a different thing: numbers for the arithmetic
    intensities, no identifier.
    """

    pairing: str
    model_name: str
    code_path: str
    arm: str
    context_length: int
    density: float
    batch: int
    priced: PricedSequence
    hardware: HardwareFeatures
    model: Optional[ModelShape] = None

    @property
    def key(self) -> CellKey:
        return make_cell_key(self.model_name, self.code_path, self.arm,
                             self.density, self.context_length, self.batch)


@dataclass(frozen=True)
class GroundTruthRow:
    """A cell and its measured untraced end-to-end prefill, in seconds, as
    the named statistic over repeats."""

    cell: Cell
    measured: float
    statistic: str = GROUND_TRUTH_STATISTIC

    @property
    def pairing(self) -> str:
        return self.cell.pairing


@dataclass(frozen=True)
class TrainingSet:
    """Rows a baseline is fitted on, the pairings they come from, and
    whether this is a committed or a retrospective fit."""

    target: str
    kind: Kind
    trained_on: tuple[str, ...]
    rows: tuple[GroundTruthRow, ...]
    statistic: str = GROUND_TRUTH_STATISTIC


def _check_rows(rows: Iterable[GroundTruthRow]) -> list[GroundTruthRow]:
    rows = list(rows)
    for r in rows:
        if r.statistic not in STATISTICS:
            raise CostModelError(
                f"statistic must be one of {STATISTICS}, got {r.statistic!r} "
                f"for {r.cell.pairing}/{r.cell.arm}")
        m = r.measured
        if (isinstance(m, bool) or not isinstance(m, (int, float))
                or not math.isfinite(m) or m <= 0):
            raise CostModelError(
                f"measured time must be a positive finite number of "
                f"seconds; got {m!r} for {r.cell.pairing}/{r.cell.arm}")
    return rows


def statistic_of(rows: Iterable[GroundTruthRow]) -> str:
    """The one statistic these rows share; refuses a mix."""
    found = sorted({r.statistic for r in rows})
    if len(found) != 1:
        raise CostModelError(
            f"rows mix ground-truth statistics {found}; fit on one")
    return found[0]


def committed_training_set(rows: Iterable[GroundTruthRow], target: str,
                           banked: Iterable[str]) -> TrainingSet:
    """Rows from the pairings banked before the prediction for `target`.

    Refuses `target` among the banked pairings (its own ground truth is
    never training data for its committed prediction), an empty `banked`
    (on P1 no fitted baseline exists; Gate A uses the sum and the step
    form), and a banked pairing with no rows (training silently on fewer
    pairings than the prediction file names).
    """
    rows = _check_rows(rows)
    banked = tuple(sorted(set(banked)))
    if target in banked:
        raise CostModelError(
            f"{target} is listed as banked before its own prediction; its "
            f"ground truth cannot train its committed prediction")
    if not banked:
        raise CostModelError(
            f"no pairing is banked before {target}'s prediction, so there is "
            f"nothing to fit; on P1 Gate A uses the sum and the step form")
    present = {r.pairing for r in rows}
    missing = [p for p in banked if p not in present]
    if missing:
        raise CostModelError(
            f"banked pairing(s) {missing} have no ground-truth rows here; a "
            f"committed fit must use every pairing it names")
    chosen = tuple(r for r in rows if r.pairing in banked)
    return TrainingSet(target=target, kind=COMMITTED, trained_on=banked,
                       rows=chosen, statistic=statistic_of(chosen))


def retrospective_training_set(rows: Iterable[GroundTruthRow],
                               target: str) -> TrainingSet:
    """Leave-one-pairing-out: every pairing except `target`, labelled
    retrospective because a forecaster could not have had later ones."""
    rows = _check_rows(rows)
    chosen = tuple(r for r in rows if r.pairing != target)
    if not chosen:
        raise CostModelError(f"no pairing other than {target} has rows")
    return TrainingSet(target=target, kind=RETROSPECTIVE,
                       trained_on=tuple(sorted({r.pairing for r in chosen})),
                       rows=chosen, statistic=statistic_of(chosen))
