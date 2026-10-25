"""T2.19: the constant-host-cost and nearest-measured-pairing baselines.

The pairing table is read back from the plan's Section 5.2, not from
`PAIRINGS`. The nearest-pairing rule is checked against a scorer written a
different way (2 points for the same GPU, 1 for the same host platform).
"""

from __future__ import annotations

import itertools
import random
import re
from pathlib import Path

import pytest

from attnbench.predict.naive import (
    PAIRINGS, Nearest, Pairing, constant_host_cost, median_launch_cost,
    nearest_pairing, nearest_pairing_predictions, with_constant_host_cost)
from attnbench.predict.recursion import (
    CostModelError, close_prefill, copy_op, kernel_op, price, recurse, sync_op)
from attnbench.predict.training import Cell, GroundTruthRow, HardwareFeatures

from test_predict_recursion import COSTS

PLAN = Path(__file__).resolve().parents[1] / "docs" / "RESEARCH_PLAN.md"
HW = HardwareFeatures(pinned_bandwidth=1024, pageable_bandwidth=512)


def _plan_pairings():
    text = PLAN.read_text()
    sec = text.split("### 5.2 Hardware pairings", 1)[1].split("**Run order", 1)[0]
    out = {}
    for line in sec.splitlines():
        m = re.match(r"\| (P\d)( \(optional\))? \| ([^|]+) \| ([^|]+) \|", line)
        if m:
            host = re.sub(r"\s*\(.*\)$", "", m.group(3).strip())
            out[m.group(1)] = (host, m.group(4).strip())
    return out


def test_the_pairing_table_is_section_5_2s():
    plan = _plan_pairings()
    assert len(plan) == 7
    assert {p: (v.host_platform, v.gpu) for p, v in PAIRINGS.items()} == plan
    assert all(v.name == p for p, v in PAIRINGS.items())


# --------------------------------------------------------------------------
# Nearest measured pairing
# --------------------------------------------------------------------------

# The planned run order (Section 5.2): P1, P2, P4, then P3, P5, P6, P7.
RUN_ORDER = ("P1", "P2", "P4", "P3", "P5", "P6", "P7")
# Worked from the table by hand.
PLANNED = {
    "P2": ("P1", "same host platform"),   # no A100 yet; Cascade Lake
    "P4": (None, "none"),                 # first H100, first Sapphire Rapids
    "P3": ("P1", "same GPU"),             # L4
    "P5": ("P3", "same host platform"),   # no L40S; AMD EPYC 3rd gen
    "P6": ("P4", "same GPU"),             # H100, ahead of P3/P5's host
    "P7": (None, "none"),                 # A10G; EPYC 2nd gen is new
}


@pytest.mark.parametrize("target", list(PLANNED))
def test_nearest_pairing_in_the_planned_run_order(target):
    banked = RUN_ORDER[:RUN_ORDER.index(target)]
    n = nearest_pairing(target, banked)
    assert (n.source, n.rule) == PLANNED[target]


def _score_oracle(target, banked, table):
    t = table[target]
    scored = [(2 if table[b].gpu == t.gpu else
               1 if table[b].host_platform == t.host_platform else 0, b)
              for b in banked]
    best = max([s for s, _ in scored], default=0)
    if best == 0:
        return (None, "none")
    hits = [b for s, b in scored if s == best]
    if len(hits) > 1:
        return "tie"
    return (hits[0], "same GPU" if best == 2 else "same host platform")


def test_nearest_pairing_matches_an_independent_scorer_on_every_banked_set():
    names = sorted(PAIRINGS)
    for target in names:
        others = [p for p in names if p != target]
        for k in range(len(others) + 1):
            for banked in itertools.combinations(others, k):
                want = _score_oracle(target, banked, PAIRINGS)
                if want == "tie":
                    with pytest.raises(CostModelError, match="tie-break"):
                        nearest_pairing(target, banked)
                else:
                    n = nearest_pairing(target, banked)
                    assert (n.source, n.rule) == want, (target, banked)


def test_a_tie_is_refused():
    table = {"A": Pairing("A", "h1", "g1"), "B": Pairing("B", "h2", "g1"),
             "C": Pairing("C", "h3", "g1")}
    with pytest.raises(CostModelError, match="tie-break"):
        nearest_pairing("A", ["B", "C"], table)
    assert nearest_pairing("A", ["B"], table) == Nearest("A", "B", "same GPU")


def test_nearest_refuses_the_target_and_unknown_pairings():
    with pytest.raises(CostModelError, match="banked before its own"):
        nearest_pairing("P2", ["P1", "P2"])
    with pytest.raises(CostModelError, match="unknown pairing"):
        nearest_pairing("P2", ["P9"])


def cell(pairing, arm="xattn", L=4096, density=0.5, batch=1):
    return Cell(pairing=pairing, arm=arm, context_length=L, density=density,
                batch=batch, priced=price(close_prefill([kernel_op(1, 1, layer=0)]),
                                          COSTS), hardware=HW)


def test_nearest_predictions_are_the_source_pairings_measured_times_unscaled():
    rows = [GroundTruthRow(cell("P1"), 0.25),
            GroundTruthRow(cell("P1", L=8192), 0.5),
            GroundTruthRow(cell("P4"), 9.0)]       # not the source for P2
    keys = [("xattn", 4096, 0.5, 1), ("xattn", 8192, 0.5, 1),
            ("xattn", 16384, 0.5, 1)]
    near, values = nearest_pairing_predictions("P2", keys, rows, ["P1", "P4"])
    assert near == Nearest("P2", "P1", "same host platform")
    assert values == {keys[0]: 0.25, keys[1]: 0.5, keys[2]: None}


def test_nearest_predictions_when_nothing_is_near():
    near, values = nearest_pairing_predictions(
        "P4", [("xattn", 4096, 0.5, 1)], [GroundTruthRow(cell("P1"), 1.0)],
        ["P1", "P2"])
    assert near.source is None and values == {("xattn", 4096, 0.5, 1): None}


def test_nearest_predictions_refuse_bad_source_rows():
    k = [("xattn", 4096, 0.5, 1)]
    with pytest.raises(CostModelError, match="no ground-truth rows"):
        nearest_pairing_predictions("P2", k, [], ["P1"])
    with pytest.raises(CostModelError, match="two rows"):
        nearest_pairing_predictions("P2", k, [GroundTruthRow(cell("P1"), 1.0)] * 2,
                                    ["P1"])
    with pytest.raises(CostModelError, match="mix"):
        nearest_pairing_predictions(
            "P2", k, [GroundTruthRow(cell("P1"), 1.0),
                      GroundTruthRow(cell("P1", L=8192), 1.0, "p90")], ["P1"])


# --------------------------------------------------------------------------
# Constant host cost
# --------------------------------------------------------------------------

def test_constant_host_cost_hand_computed():
    # Ops: K(h5,d1) K(h0.5,d3) S(item, 4 B; s0 = 1 + 4/4 = 2) K(h2,d1), close.
    # With the constant 1 every kernel's h is 1 and the syncs keep h = 0:
    #   K: H1 = 1, D1 = 2.   K: H2 = 2, D2 = max(2, 2) + 3 = 5.
    #   S: max(2, 5) = 5, + 2: H3 = D3 = 7.   K: H4 = 8, D4 = max(8, 7) + 1 = 9.
    #   closing: max(8, 9) = 9, + 1 = 10.
    ops = close_prefill([kernel_op(5, 1), kernel_op(0.5, 3),
                         sync_op("item", payload_bytes=4), kernel_op(2, 1)])
    p = price(ops, COSTS)
    c = with_constant_host_cost(p, 1)
    assert c.host == (1, 1, 0, 1, 0)
    assert (c.link, c.device, c.sync_cost) == (p.link, p.device, p.sync_cost)
    assert constant_host_cost(p, 1, None).end_to_end == 10
    # The recursion itself, with the measured host costs: H 5, 5.5; D 6, 9;
    # S at max(5.5, 9) = 9, + 2 = 11; K: H 13, D 14; closing 14 + 1 = 15.
    assert recurse(p, None).end_to_end == 15


def test_constant_host_cost_replaces_host_compute_too():
    # A copy whose h holds CPU mask building (h 7) is charged the constant.
    p = price(close_prefill([copy_op(7, 1), kernel_op(1, 1)]), COSTS)
    assert with_constant_host_cost(p, 0.5).host == (0.5, 0.5, 0)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "1"])
def test_constant_must_be_a_positive_time(bad):
    p = price(close_prefill([kernel_op(1, 1)]), COSTS)
    with pytest.raises(CostModelError, match="constant host cost"):
        with_constant_host_cost(p, bad)


def test_median_launch_cost():
    assert median_launch_cost([3e-6, 1e-6, 2e-6]) == 2e-6
    assert median_launch_cost([1, 2, 3, 4]) == 2.5
    assert median_launch_cost([1, 2, 10]) == 2          # the mean is 4.33
    with pytest.raises(CostModelError, match="no null-kernel"):
        median_launch_cost([])
    with pytest.raises(CostModelError, match="positive"):
        median_launch_cost([1e-6, 0])


@pytest.mark.parametrize("seed", range(30))
def test_constant_host_cost_equals_the_recursion_when_every_h_is_the_constant(seed):
    rng = random.Random(4400 + seed)
    c = rng.choice([0.5, 1, 2])
    ops = []
    for _ in range(rng.randint(1, 20)):
        r = rng.random()
        ops.append(kernel_op(c, rng.randint(0, 6)) if r < 0.6 else
                   copy_op(c, rng.randint(0, 3)) if r < 0.8 else
                   sync_op("bool", payload_bytes=rng.choice([0, 4])))
    p = price(close_prefill(ops), COSTS)
    Q = rng.choice([None, 1, 2, 4])
    assert constant_host_cost(p, c, Q).end_to_end == recurse(p, Q).end_to_end
