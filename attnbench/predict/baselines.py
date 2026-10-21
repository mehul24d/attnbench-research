"""Baselines of plan Section 5.4 built from the recursion's own components
(T2.9). Each reads a `recursion.PricedSequence`, so no baseline can be
priced from different numbers than the recursion it is compared with.

  - `sum_as_written`: sum(h) + sum(l) + sum(d), no overlap. As the plan
    writes it, it leaves out s_0. A synchronisation has l = d = 0 and, once
    its host cost is s_fixed, h = 0 too, so this Sum charges nothing at all
    for a synchronisation.
  - `sum_plus_s0`: the same plus sum(s_0). Not in the plan; built so the
    effect of that omission can be read, and labelled separately.
  - `step_form`: the KernelSight-style step form, max(sum h, sum l + sum d)
    per layer plus every s_0 as a serial tail. A step is a maximal run of
    consecutive kernels and copies with the same `layer` label, so a
    chunked prefill that visits layer 3 twice has two steps for it.
    Synchronisations carry no h, l or d, so they never split a step and
    enter only through the tail.
  - `FittedSum`: a·sum(h) + b·sum(l) + c·sum(d), a, b, c >= 0, fitted on
    the rows `training` selects (committed or retrospective). No s_0 term,
    as written.

HDBI (TaxBreak's diagnostic) is not here: the plan names it without
defining it, and its definition has to come from the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import nnls

from .recursion import CostModelError, PricedSequence
from .training import TrainingSet

LABELS = {
    "sum": "Sum, as written: sum h + sum l + sum d (no s_0)",
    "sum_plus_s0": "Sum + s_0: sum h + sum l + sum d + sum s_0 (not in the plan)",
    "step_form": "KernelSight-style step form: per-layer max(sum h, sum l + "
                 "sum d) + sum s_0",
    "fitted_sum": "Fitted sum, as written: a sum h + b sum l + c sum d, "
                  "a, b, c >= 0 (no s_0)",
}

FITTED_TERMS = ("host", "link", "device")
Loss = Literal["relative", "absolute"]


def sum_as_written(priced: PricedSequence) -> float:
    return float(sum(priced.host) + sum(priced.link) + sum(priced.device))


def sum_plus_s0(priced: PricedSequence) -> float:
    return sum_as_written(priced) + float(sum(priced.sync_cost))


def steps(priced: PricedSequence) -> list[tuple[int, float, float]]:
    """(layer, sum h, sum l + sum d) for each step, in order."""
    out: list[list] = []
    for k in range(len(priced)):
        if priced.is_sync[k]:
            continue
        layer = priced.layers[k]
        if layer is None:
            raise CostModelError(
                f"operation {k + 1} is a {priced.kinds[k].value} with no layer "
                f"label; the step form needs every kernel and copy assigned "
                f"to a step (label work outside the layers with its own "
                f"layer value)")
        if not out or out[-1][0] != layer:
            out.append([layer, 0.0, 0.0])
        out[-1][1] += priced.host[k]
        out[-1][2] += priced.link[k] + priced.device[k]
    return [tuple(s) for s in out]


def step_form(priced: PricedSequence) -> float:
    body = sum(max(h, ld) for _, h, ld in steps(priced))
    return float(body + sum(priced.sync_cost))


def fitted_terms(priced: PricedSequence) -> tuple[float, float, float]:
    return (float(sum(priced.host)), float(sum(priced.link)),
            float(sum(priced.device)))


@dataclass(frozen=True)
class FittedSum:
    a: float
    b: float
    c: float
    loss: str
    target: str
    kind: str
    trained_on: tuple[str, ...]
    n_rows: int
    # Terms that are zero on every training row, so their coefficient says
    # nothing (nnls returns 0 for them). A pairing whose arms have copies,
    # predicted from rows with none, gets b = 0 and charges its copies
    # nothing.
    unidentified: tuple[str, ...]

    def predict(self, priced: PricedSequence) -> float:
        H, L, D = fitted_terms(priced)
        return self.a * H + self.b * L + self.c * D


def fit_fitted_sum(training: TrainingSet, *, loss: Loss) -> FittedSum:
    """Non-negative least squares for a, b, c.

    `loss` has no default: "relative" minimises sum((pred - y) / y)^2,
    which matches H1's absolute relative error; "absolute" minimises
    sum(pred - y)^2 in seconds, which the longest contexts dominate. The
    plan does not say which, so the caller states it.
    """
    if loss not in ("relative", "absolute"):
        raise CostModelError(f"loss must be 'relative' or 'absolute', got {loss!r}")
    if not training.rows:
        raise CostModelError("an empty training set fits nothing")
    A = np.array([fitted_terms(r.cell.priced) for r in training.rows])
    y = np.array([r.measured for r in training.rows], dtype=float)
    if loss == "relative":
        A_fit, y_fit = A / y[:, None], np.ones_like(y)
    else:
        A_fit, y_fit = A, y
    coef, _ = nnls(A_fit, y_fit)
    if not all(math.isfinite(float(x)) for x in coef):
        raise CostModelError(f"the fit returned non-finite coefficients {coef}")
    unidentified = tuple(name for name, col in zip(FITTED_TERMS, A.T)
                         if not np.any(col))
    return FittedSum(a=float(coef[0]), b=float(coef[1]), c=float(coef[2]),
                     loss=loss, target=training.target, kind=training.kind,
                     trained_on=training.trained_on,
                     n_rows=len(training.rows), unidentified=unidentified)
