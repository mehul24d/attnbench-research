"""The Holm-corrected admissible set (Section 5.7), on the exact paired
bound of `exact_noninferiority`."""

from __future__ import annotations

import pytest

from attnbench.analysis.exact_noninferiority import paired_lower_bound
from attnbench.lock.holm import Counts, HolmError, holm_admissible, ni_p_value
from attnbench.predict.dispatcher import Config, admissible_at_margin

MARGIN = 10.0


def lower_pts(c: Counts, alpha=0.025) -> float:
    return 100 * paired_lower_bound(c.b, c.c, c.n, alpha)


def test_the_p_value_is_where_the_bound_meets_minus_the_margin():
    c = Counts(b=4, c=8, n=200)
    p = ni_p_value(c, MARGIN)
    assert 0 < p < 1
    assert lower_pts(c, p * 1.001) > -MARGIN
    assert lower_pts(c, p * 0.999) <= -MARGIN


@pytest.mark.parametrize("b, c, n", [(0, 0, 50), (2, 5, 200), (4, 8, 200), (3, 9, 200),
                                     (1, 17, 200), (10, 10, 60), (0, 30, 100)])
def test_the_bound_rises_with_alpha_so_the_inversion_holds(b, c, n):
    # ni_p_value bisects on alpha; that is valid only if the bound is
    # non-decreasing in alpha.
    alphas = [1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 0.9]
    bounds = [paired_lower_bound(b, c, n, a) for a in alphas]
    assert all(lo <= hi + 1e-12 for lo, hi in zip(bounds, bounds[1:])), bounds


def test_a_test_no_alpha_can_pass_has_p_one():
    assert ni_p_value(Counts(b=0, c=60, n=100), MARGIN) == 1.0


def test_with_one_member_holm_is_the_uncorrected_rule():
    for cnt in (Counts(2, 5, 200), Counts(5, 25, 200), Counts(0, 0, 100)):
        cfg = Config("meanpool", 0.25)
        r = holm_admissible({cfg: cnt}, declared={cfg}, margin_pts=MARGIN)
        assert r.holm == r.uncorrected == admissible_at_margin({cfg: lower_pts(cnt)}, MARGIN)


def _family():
    # Three configurations with clearly ordered evidence.
    return {Config("meanpool", 0.25): Counts(b=2, c=2, n=200),     # strong
            Config("meanpool", 0.1): Counts(b=2, c=10, n=200),     # moderate
            Config("xattention", 0.1): Counts(b=1, c=17, n=200)}   # weak


def test_holm_steps_down_at_alpha_over_m_minus_k():
    fam = _family()
    r = holm_admissible(fam, declared=set(fam), margin_pts=MARGIN)
    # The order is by p, and each step's level is alpha / (m - k).
    ps = [p for _, p, _, _ in r.order]
    assert ps == sorted(ps)
    assert [lvl for _, _, lvl, _ in r.order] == pytest.approx([0.025 / 3, 0.025 / 2, 0.025])
    # Each admission is the bound itself at the step's level.
    for k, _, lvl, ok in r.order:
        if ok:
            assert lower_pts(fam[k], lvl) > -MARGIN
    # Holm never admits more than the uncorrected rule.
    assert r.holm <= r.uncorrected
    assert r.uncorrected == admissible_at_margin(
        {k: lower_pts(v) for k, v in fam.items()}, MARGIN)
    # By hand: the moderate member clears at 0.025 (-9.54 points) and not at
    # 0.025 / 2 (-10.16), so the correction drops it; the weak one clears at
    # neither.
    assert r.uncorrected == {Config("meanpool", 0.25), Config("meanpool", 0.1)}
    assert r.holm == {Config("meanpool", 0.25)}


def test_holm_stops_at_the_first_failure():
    fam = _family()
    r = holm_admissible(fam, declared=set(fam), margin_pts=MARGIN)
    seen_fail = False
    for _, _, _, ok in r.order:
        if seen_fail:
            assert not ok
        seen_fail = seen_fail or not ok
    # Two members, both clearing at 0.025 alone. A (3, 10) has the smaller p
    # and fails at 0.025 / 2 (-10.03 points); B (2, 10) would clear at the
    # last step's 0.025 (-9.54), but Holm has stopped, so B is out.
    a, b = Config("meanpool", 0.1), Config("meanpool", 0.2)
    two = {a: Counts(3, 10, 200), b: Counts(2, 10, 200)}
    r2 = holm_admissible(two, declared=[a, b], margin_pts=MARGIN)
    assert [k for k, _, _, _ in r2.order] == [a, b]
    assert lower_pts(two[b], 0.025) > -MARGIN
    assert r2.uncorrected == {a, b} and r2.holm == frozenset()


def test_the_correction_shrinks_the_set_on_a_borderline_member():
    # Five near-identical borderline members: each clears alone at 0.025, and
    # not at 0.025 / 5.
    cnt = Counts(b=3, c=9, n=200)
    assert lower_pts(cnt, 0.025) > -MARGIN
    assert lower_pts(cnt, 0.005) <= -MARGIN
    fam = {Config("meanpool", d): cnt for d in (0.1, 0.2, 0.3, 0.4, 0.5)}
    r = holm_admissible(fam, declared=set(fam), margin_pts=MARGIN)
    assert len(r.uncorrected) == 5 and r.holm == frozenset()


@pytest.mark.parametrize("kw, match", [
    (dict(family={}), "empty"),
    (dict(family={("meanpool", 0.1): Counts(1, 1, 10)}), "Config"),
    (dict(family={Config("m", 0.1): (1, 1, 10)}), "Counts"),
    (dict(family={Config("m", 0.1): Counts(1, 1, 10)}, margin_pts=0), "margin"),
    (dict(family={Config("m", 0.1): Counts(1, 1, 10)}, alpha=1.5), "alpha"),
])
def test_a_family_or_level_that_cannot_be_corrected_is_refused(kw, match):
    kw.setdefault("margin_pts", MARGIN)
    kw.setdefault("declared", set(kw["family"]))
    with pytest.raises(HolmError, match=match):
        holm_admissible(**kw)


def test_the_family_must_be_the_set_declared_at_lock():
    fam = _family()
    extra = Config("meanpool", 0.5)
    with pytest.raises(HolmError, match="declared with no counts"):
        holm_admissible(fam, declared=set(fam) | {extra}, margin_pts=MARGIN)
    with pytest.raises(HolmError, match="counts not declared"):
        holm_admissible(fam, declared=set(list(fam)[:2]), margin_pts=MARGIN)
    with pytest.raises(HolmError, match="declared twice"):
        holm_admissible(fam, declared=list(fam) + [next(iter(fam))], margin_pts=MARGIN)
    with pytest.raises(TypeError):
        holm_admissible(fam, margin_pts=MARGIN)              # no default family
    for bad in ("meanpool", fam):
        with pytest.raises(HolmError, match="declared"):
            holm_admissible(fam, declared=bad, margin_pts=MARGIN)


@pytest.mark.parametrize("args", [(-1, 0, 10), (5, 6, 10), (0, 0, 0), (True, 0, 10)])
def test_impossible_counts_are_refused(args):
    with pytest.raises(HolmError):
        Counts(*args)
