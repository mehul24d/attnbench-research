"""EXPLORATORY parameter-free pay condition (paper-quality wave, 28 Nov 2026,
item 11): hand-worked fixtures."""

import pytest

from attnbench.exploratory import pay_bound as PB
from attnbench.predict.recursion import OpKind, PricedSequence


def _seq(kinds, syncs):
    n = len(kinds)
    z = tuple(0.0 for _ in range(n))
    return PricedSequence(host=z, link=z, device=z, sync_cost=z, is_sync=tuple(syncs),
                          kinds=tuple(kinds), layers=tuple(None for _ in range(n)),
                          mechanisms=tuple(None for _ in range(n)),
                          payload_bytes=tuple(0 for _ in range(n)))


def test_counts_leave_out_the_closing_sync():
    K, C, S = OpKind.KERNEL, OpKind.COPY, OpKind.SYNC
    s = _seq([K, S, K, C, S], [False, True, False, False, True])
    assert PB.counts(s) == {"ops": 3, "in_step_syncs": 1}
    s = _seq([K, S, K], [False, True, False])
    assert PB.counts(s) == {"ops": 2, "in_step_syncs": 1}


def test_lower_bound_terms():
    lb = PB.lower_bound(10.0, 0.5, ops=4, in_step_syncs=2)
    assert lb == {"lb_s": 11.0, "device_term_s": 11.0, "host_term_s": 2.0, "binding": "device"}
    lb = PB.lower_bound(1.0, 0.5, ops=10, in_step_syncs=0)
    assert lb["lb_s"] == 5.0 and lb["binding"] == "host"
    with pytest.raises(ValueError):
        PB.lower_bound(-1.0, 0.5, 1, 0)


def test_verdict_is_strict():
    assert PB.verdict(9.0, 10.0) == "may pay"
    assert PB.verdict(10.0, 10.0) == "cannot pay"


def test_check_counts_violations_and_separations():
    rows = [
        {"lb_s": 5.0, "sparse_s": 6.0, "dense_s": 8.0},   # win, may pay
        {"lb_s": 9.0, "sparse_s": 7.0, "dense_s": 8.0},   # win ruled out: violation, floor above
        {"lb_s": 9.0, "sparse_s": 12.0, "dense_s": 8.0},  # loss ruled out
        {"lb_s": 4.0, "sparse_s": 12.0, "dense_s": 8.0},  # loss not ruled out
    ]
    k = PB.check(rows)
    assert (k["measured_wins"], k["measured_losses"]) == (2, 2)
    assert k["violations"] == 1 and k["losses_ruled_out"] == 1 and k["separates_any"]
    assert k["floor_above_measured_sparse"] == 1
    assert k["slack_min"] == pytest.approx(7 / 9)
    rows2 = [{**r, "other": r["lb_s"] / 10} for r in rows]
    k2 = PB.check(rows2, "other")
    assert k2["losses_ruled_out"] == 0 and not k2["separates_any"] and k2["violations"] == 0
