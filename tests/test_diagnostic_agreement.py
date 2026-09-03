"""Segment 2's Stage 1 must reproduce the 2026-09-04 diagnostic.

The load tests run against the REAL files on disk when they exist, because a
reference loader validated only against a synthetic fixture tests the fixture.
They skip rather than fail when the files are absent, so the suite still runs
on a fresh clone.
"""

from __future__ import annotations

import json

import pytest

from attnbench.analysis import diagnostic_agreement as DA

# The two fingerprints, from results on disk. Kept here as literals so a test
# failure says which number moved, not merely that two files disagree.
COMPILED = {"03091d1851d0": 0.008404, "b614e994ff01": 0.008983}
EAGER = {"03091d1851d0": 0.015746, "b614e994ff01": 0.016893}


def test_reproducing_the_compiled_numbers_agrees():
    rows = DA.compare(dict(COMPILED), COMPILED, EAGER)
    assert [r.verdict for r in rows] == ["AGREES", "AGREES"]
    DA.assert_agreement(rows)


def test_small_drift_still_agrees():
    """The claim is 'same implementation', not 'same float'. A reduction over
    random inputs moves a little between runs."""
    observed = {k: v * 1.08 for k, v in COMPILED.items()}
    rows = DA.compare(observed, COMPILED, EAGER)
    assert all(r.verdict == "AGREES" for r in rows)


def test_the_eager_fingerprint_is_named_not_merely_flagged():
    """The failure this module exists for. Reporting a generic mismatch would
    make someone diagnose the fallback from scratch a second time."""
    rows = DA.compare(dict(EAGER), COMPILED, EAGER)
    assert [r.verdict for r in rows] == ["EAGER_SIGNATURE"] * 2
    assert all("fell back again" in r.detail for r in rows)
    with pytest.raises(DA.DiagnosticAgreementError, match="EAGER_SIGNATURE"):
        DA.assert_agreement(rows)


def test_the_two_fingerprints_are_far_enough_apart_to_separate():
    """The whole approach rests on eager and compiled being distinguishable at
    REL_TOL. If a future kernel change narrowed that gap, this check would
    silently start passing both -- so assert the separation itself."""
    for key in COMPILED:
        sep = DA._rel(COMPILED[key], EAGER[key])
        assert sep > 3 * DA.REL_TOL, (
            f"{key}: eager and compiled are only {sep:.1%} apart, too close "
            f"to separate at rel_tol={DA.REL_TOL:.0%}")


def test_a_third_behaviour_is_neither_agreement_nor_the_known_failure():
    rows = DA.compare({"03091d1851d0": 0.9}, COMPILED, EAGER)
    assert rows[0].verdict == "DISAGREES"
    assert "third behaviour" in rows[0].detail


def test_a_missing_cell_is_reported_not_skipped():
    """A pipeline that stopped producing a cell has changed too, and an empty
    comparison that reports success is the vacuous-check failure mode."""
    rows = DA.compare({}, COMPILED, EAGER)
    assert [r.verdict for r in rows] == ["NOT_MEASURED"] * 2
    with pytest.raises(DA.DiagnosticAgreementError):
        DA.assert_agreement(rows)


def test_loads_the_real_diagnostic_file():
    if not DA.DEFAULT_COMPILED_REF.exists():
        pytest.skip("diagnostic results not present in this checkout")
    ref = DA.load_compiled_reference()
    assert ref, "loader returned nothing from a file that has OK rows"
    for key, expected in COMPILED.items():
        assert ref[key] == pytest.approx(expected, rel=1e-6)


def test_parses_max_abs_err_out_of_the_detail_string(tmp_path):
    """The 2026-09-04 run recorded the error only in free-text `detail`.
    Re-deriving it by hand would put a transcription error between the
    reference and the thing it references."""
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"rows": [
        {"config_key": "aaa", "verdict": "OK",
         "detail": "max_abs_err=0.008404 (tol 0.02)"},
        {"config_key": "bbb", "verdict": "COMPILE_FAIL", "detail": "nope"},
        {"config_key": "ccc", "verdict": "OK", "max_abs_err": 0.0071,
         "detail": "field wins over detail"},
    ]}))
    ref = DA.load_compiled_reference(p)
    assert ref == {"aaa": pytest.approx(0.008404), "ccc": pytest.approx(0.0071)}


def test_loads_the_real_eager_reference():
    if not DA.DEFAULT_EAGER_REF.exists():
        pytest.skip("segment 1 probe results not present in this checkout")
    ref = DA.load_eager_reference()
    assert len(ref) == 72, "segment 1 recorded 72 flex block_sparse rows"
    for key, expected in EAGER.items():
        # rel=1e-4, not 1e-6: the EAGER literals above are the 6dp values as
        # displayed, while the parquet holds full precision. The compiled
        # literals are exact because they came through a 6dp detail string.
        assert ref[key] == pytest.approx(expected, rel=1e-4)
