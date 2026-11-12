"""`attnbench.analysis.probe_flags`: T4.1's probe bounds as analysis flags,
never a stop (lock-wave rulings of 12 Nov 2026, item 4)."""

import pytest

from attnbench.analysis import probe_flags as PF

BOUNDS = {"loop_cv": 0.01, "loop_median_relative": 0.05, "null_median_relative": 0.10}


def row(loop=125e-9, cv=0.002, null=4.0e-6):
    return {PF.LOOP_MEDIAN: loop, PF.LOOP_CV: cv, PF.NULL_MEDIAN: null}


def flags(r, ref=None):
    ref = ref or PF.reference([row(124e-9, null=3.9e-6), row(126e-9, null=4.1e-6),
                               row(125e-9, null=4.0e-6)])
    return {f["check"]: f for f in PF.probe_flags(r, ref, BOUNDS)}


def test_the_reference_is_the_median_over_the_source_repeats():
    ref = PF.reference([row(124e-9, null=3.9e-6), row(130e-9, null=4.4e-6),
                        row(125e-9, null=4.0e-6)])
    assert ref == {PF.LOOP_MEDIAN: 125e-9, PF.NULL_MEDIAN: 4.0e-6}
    with pytest.raises(PF.ProbeFlagError, match="source pairing"):
        PF.reference([])


def test_a_probe_inside_every_bound_is_not_flagged():
    f = flags(row())
    assert not any(x["flagged"] for x in f.values())
    assert f["loop_median_relative"]["value"] == 0 and f["loop_cv"]["value"] == 0.002


@pytest.mark.parametrize("r,check,value", [
    (row(cv=0.011), "loop_cv", 0.011),
    (row(loop=132e-9), "loop_median_relative", 0.056),
    (row(loop=118e-9), "loop_median_relative", 0.056),
    (row(null=4.5e-6), "null_median_relative", 0.125),
    (row(null=3.5e-6), "null_median_relative", 0.125)])
def test_each_check_flags_beyond_its_own_bound_in_either_direction(r, check, value):
    f = flags(r)
    assert f[check]["flagged"] and f[check]["value"] == pytest.approx(value)
    assert f[check]["bound"] == BOUNDS[check]
    assert [k for k, x in f.items() if x["flagged"]] == [check]


def test_the_bound_itself_is_inside():
    ref = {PF.LOOP_MEDIAN: 1.0, PF.NULL_MEDIAN: 1.0}
    at = {"loop_cv": 0.25, "loop_median_relative": 0.25, "null_median_relative": 0.25}
    f = PF.probe_flags(row(cv=0.25, loop=1.25, null=0.75), ref, at)
    assert [x["value"] for x in f] == [0.25, 0.25, 0.25]
    assert not any(x["flagged"] for x in f)


@pytest.mark.parametrize("bad,match", [
    ({**row(), PF.LOOP_CV: float("nan")}, "loop_s_per_iter_cv"),
    ({k: v for k, v in row().items() if k != PF.NULL_MEDIAN}, "null_launch_s_median"),
    ({**row(), PF.LOOP_MEDIAN: True}, "loop_s_per_iter_median")])
def test_an_unreadable_row_is_refused(bad, match):
    with pytest.raises(PF.ProbeFlagError, match=match):
        flags(bad)


def test_the_bounds_are_the_three_checks():
    with pytest.raises(PF.ProbeFlagError, match="probe bounds"):
        PF.probe_flags(row(), PF.reference([row()]), {"loop_cv": 0.01})
