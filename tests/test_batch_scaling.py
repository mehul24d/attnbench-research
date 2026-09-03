"""batch_scaling schema and the speedup summary it feeds -- CPU-only,
synthetic values, no GPU or model, mirroring test_timing_probe.py's
discipline for the other half of the Stage 3 planning math.
"""

from __future__ import annotations

import pandas as pd
import pytest

from attnbench.accuracy.batch_scaling import (
    BatchScalingResult, batching_speedup, per_example_seconds)


def _row(backend="gla", batch=1, status="ok", wall=4.0, **kw):
    fields = dict(
        backend=backend, model_id="Qwen/Qwen2.5-1.5B-Instruct",
        seq_len_nominal=16384, seq_len_real=19821, batch=batch, status=status,
        wall_seconds=wall if status == "ok" else None,
        wall_seconds_per_example=(wall / batch) if status == "ok" else None,
        peak_memory_mb=4330.0 if status == "ok" else None)
    fields.update(kw)
    return BatchScalingResult(**fields)


def test_per_example_seconds_divides_by_batch():
    assert per_example_seconds(16.0, 8) == pytest.approx(2.0)


def test_oom_is_a_recorded_result_not_an_absent_one():
    """The distinction the whole schema exists to preserve: 'we measured
    that batch=12 does not fit' must be readable as different from 'batch=12
    was never tried'."""
    r = _row(batch=12, status="oom")
    assert r.status == "oom"
    assert r.wall_seconds is None
    assert r.peak_memory_mb is None


def test_ok_status_requires_a_measurement():
    with pytest.raises(ValueError):
        BatchScalingResult(backend="gla", model_id="m", seq_len_nominal=16384,
                            seq_len_real=19821, batch=1, status="ok")


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError):
        _row(status="failed")


def test_batch_below_one_is_rejected():
    # constructed directly rather than via _row, whose per-example division
    # would raise ZeroDivisionError before the validation under test runs
    with pytest.raises(ValueError):
        BatchScalingResult(backend="gla", model_id="m", seq_len_nominal=16384,
                            seq_len_real=19821, batch=0, status="oom")


def test_flat_per_example_time_reports_no_speedup():
    """The measured finding, as data: sdpa_flash at 16384 took 2.378s at
    batch=1 and 16.593s at batch=8 -- per-example time is flat, so batching
    returns ~1x and the workload is compute-bound, not bandwidth-bound."""
    rows = [_row(backend="sdpa_flash", batch=1, wall=2.378),
            _row(backend="sdpa_flash", batch=2, wall=4.028),
            _row(backend="sdpa_flash", batch=4, wall=8.142),
            _row(backend="sdpa_flash", batch=8, wall=16.593)]
    speedup = batching_speedup(rows)["sdpa_flash"]
    assert speedup == pytest.approx(1.15, abs=0.05)


def test_oom_rows_do_not_become_the_largest_batch_in_the_speedup():
    """A batch that OOM'd has no per-example time; treating it as the top of
    the range would silently compare against nothing."""
    rows = [_row(batch=1, wall=4.0), _row(batch=8, wall=32.0),
            _row(batch=12, status="oom")]
    assert batching_speedup(rows)["gla"] == pytest.approx(1.0)


def test_backend_with_a_single_fitting_batch_is_absent_not_reported_as_1x():
    """block_sparse runs at batch=1 only by construction -- reporting it as
    '1.00x, batching wins nothing' would state a result never measured."""
    rows = [_row(backend="block_sparse", batch=1, wall=9.0)]
    assert "block_sparse" not in batching_speedup(rows)


def test_warmup_row_is_excluded_from_the_speedup_ratio():
    """The regression this guards is measured, not hypothetical.

    On 2026-09-03 the probe ran GLA cold at batch=1 (5.394 s), then batch=2
    (2.797 s total). Using the cold call as the numerator reported
    "gla 3.86x -- batching helps". A repeat of the *identical* batch=1 call
    later in the same process took 1.384 s, so the 3.9x was Triton JIT
    compilation, not batching. With the warmup row excluded the ratio is
    ~1.0 and the compute-bound conclusion is what the data says.
    """
    rows = [_row(backend="gla", batch=1, wall=5.394, warmup=True),
            _row(backend="gla", batch=1, wall=1.384),
            _row(backend="gla", batch=2, wall=2.797)]

    assert batching_speedup(rows)["gla"] == pytest.approx(0.99, abs=0.02)

    # and the contaminated ratio is what the old code reported: the same
    # cold measurement, simply not flagged, becomes the numerator
    contaminated = [_row(backend="gla", batch=1, wall=5.394),
                    _row(backend="gla", batch=2, wall=2.797)]
    assert batching_speedup(contaminated)["gla"] == pytest.approx(3.86, abs=0.05)


def test_backend_with_only_a_warmup_row_is_absent():
    """Excluding warmup must not leave a single point that then gets reported
    as a ratio against itself."""
    rows = [_row(backend="gla", batch=1, wall=5.394, warmup=True),
            _row(backend="gla", batch=2, wall=2.797)]
    assert "gla" not in batching_speedup(rows)


def test_warmup_rows_are_still_recorded():
    """Excluded from the ratio, not dropped from the data -- the JIT cost is
    itself worth seeing, and a silently discarded measurement is how the
    original defect stayed invisible."""
    row = _row(backend="gla", batch=1, wall=5.394, warmup=True)
    assert row.to_dict()["warmup"] is True
    assert row.to_dict()["wall_seconds"] == pytest.approx(5.394)


def test_rows_round_trip_through_parquet(tmp_path):
    rows = [_row(batch=1, wall=4.0).to_dict(), _row(batch=12, status="oom").to_dict()]
    path = tmp_path / "batch_scaling.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert set(df["status"]) == {"ok", "oom"}
    assert df.loc[df["status"] == "ok", "wall_seconds_per_example"].iloc[0] == 4.0


def test_nominal_and_real_seq_len_are_both_recorded():
    """They differ by ~20% and conflating them is what made every Stage 3
    hour estimate low -- a row that carried only one would let that mistake
    back in at analysis time."""
    r = _row()
    assert r.seq_len_nominal == 16384 and r.seq_len_real == 19821
