"""Baselines of plan Section 5.4 built from the recursion's own components
(T2.9). Each reads a `recursion.PricedSequence`, so no baseline can be
priced from different numbers than the recursion it is compared with.

  - `sum_as_written`: sum(h) + sum(l) + sum(d), no overlap. As the plan
    writes it, it leaves out s_0. A synchronisation has h = l = d = 0, so
    this Sum charges nothing at all for a synchronisation.
  - `sum_plus_s0`: the same plus sum(s_0). Not in the plan; labelled
    separately.
  - `step_form`: the KernelSight-style step form, max(sum h, sum l + sum d)
    per layer plus every s_0 as a serial tail. A step is a maximal run of
    consecutive kernels and copies with the same `layer` label, so a
    chunked prefill that visits layer 3 twice has two steps for it.
    Synchronisations carry no h, l or d, so they never split a step and
    enter only through the tail.
  - `fit_fitted_sum`: a·sum(h) + b·sum(l) + c·sum(d), a, b, c >= 0, as
    written (no s_0). With `add_s0=True` it is the fitted sum + s_0,
    labelled separately: a, b and c are fitted to what is left after the
    measured sum(s_0), which is then added back with coefficient 1, never
    fitted (researcher's decision, 25 Oct 2026).

**The loss** is the plan's headline error metric, absolute relative error
|predicted - measured| / measured (H1; "Reported for every prediction",
Section 5.4), summed over the training rows and minimised as a linear
program (researcher's decision, 25 Oct 2026). Least squares on relative
error would minimise its square instead, which weights the worst rows more
than the metric the fit is scored on.

Every fit reports the 2-norm condition number of its design matrix (the
rows of sum h, sum l, sum d, each divided by the row's measured time, as
the loss weights them), over the identified terms only. A large condition
number says the training rows cannot tell the terms apart, so a, b and c
can trade against each other with little change in the fit.

HDBI is in `hdbi.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .recursion import CostModelError, PricedSequence
from .training import TrainingSet

LABELS = {
    "sum": "Sum, as written: sum h + sum l + sum d (no s_0)",
    "sum_plus_s0": "Sum + s_0: sum h + sum l + sum d + sum s_0 (not in the plan)",
    "step_form": "KernelSight-style step form: per-layer max(sum h, sum l + "
                 "sum d) + sum s_0",
    "fitted_sum": "Fitted sum, as written: a sum h + b sum l + c sum d, "
                  "a, b, c >= 0 (no s_0)",
    "fitted_sum_plus_s0": "Fitted sum + s_0: a sum h + b sum l + c sum d + "
                          "sum s_0, s_0 measured and not fitted (not in the plan)",
}

FITTED_TERMS = ("host", "link", "device")
LOSS = "sum of absolute relative errors"


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
    add_s0: bool
    target: str
    kind: str
    trained_on: tuple[str, ...]
    statistic: str
    n_rows: int
    # The summed absolute relative error on the training rows.
    training_loss: float
    # 2-norm condition number of the weighted design over the identified
    # terms; inf when those columns are linearly dependent.
    condition_number: float
    # Terms that are zero on every training row, so their coefficient says
    # nothing (fixed at 0). A pairing whose arms have copies, predicted from
    # rows with none, gets b = 0 and charges its copies nothing.
    unidentified: tuple[str, ...]

    @property
    def label(self) -> str:
        return "fitted_sum_plus_s0" if self.add_s0 else "fitted_sum"

    def predict(self, priced: PricedSequence) -> float:
        H, L, D = fitted_terms(priced)
        tail = float(sum(priced.sync_cost)) if self.add_s0 else 0.0
        return self.a * H + self.b * L + self.c * D + tail


def _condition_number(A_weighted: np.ndarray, identified: list[int]) -> float:
    if not identified:
        return math.inf
    sub = A_weighted[:, identified]
    if np.linalg.matrix_rank(sub) < len(identified):
        return math.inf
    return float(np.linalg.cond(sub))


def fit_fitted_sum(training: TrainingSet, *, add_s0: bool = False) -> FittedSum:
    """Minimise sum |a H + b L + c D (+ S) - y| / y over a, b, c >= 0.

    As a linear program in (a, b, c, t_1..t_n): minimise sum t_i subject to
    t_i >= +(x_i . beta - r_i) and t_i >= -(x_i . beta - r_i), where x_i is
    (H, L, D) / y_i and r_i is (y_i - S_i) / y_i (S_i = 0 without s_0).
    """
    if not training.rows:
        raise CostModelError("an empty training set fits nothing")
    A = np.array([fitted_terms(r.cell.priced) for r in training.rows])
    y = np.array([r.measured for r in training.rows], dtype=float)
    S = np.array([float(sum(r.cell.priced.sync_cost)) if add_s0 else 0.0
                  for r in training.rows])
    X = A / y[:, None]
    rhs = (y - S) / y
    n = len(y)
    identified = [j for j in range(3) if np.any(A[:, j])]

    # Variables: beta (3), then t (n).
    c_obj = np.concatenate([np.zeros(3), np.ones(n)])
    eye = np.eye(n)
    A_ub = np.vstack([np.hstack([X, -eye]), np.hstack([-X, -eye])])
    b_ub = np.concatenate([rhs, -rhs])
    bounds = [(0, None) if j in identified else (0, 0) for j in range(3)] \
        + [(0, None)] * n
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        raise CostModelError(f"the fitted-sum program did not solve: {res.message}")
    beta = res.x[:3]
    return FittedSum(
        a=float(beta[0]), b=float(beta[1]), c=float(beta[2]), add_s0=add_s0,
        target=training.target, kind=training.kind,
        trained_on=training.trained_on, statistic=training.statistic,
        n_rows=n, training_loss=float(np.abs(X @ beta - rhs).sum()),
        condition_number=_condition_number(X, identified),
        unidentified=tuple(FITTED_TERMS[j] for j in range(3)
                           if j not in identified))
