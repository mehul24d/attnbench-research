"""sweep.py's resume logic: the one thing that, if broken, costs rented
GPU hours rather than free ones. Simulates the actual failure mode a rented
instance dying mid-sweep looks like -- an uncaught exception partway through
-- and confirms a second call picks up exactly where the first left off.

No GPU touched: measure_fn is a stub, so this validates the checkpoint/
resume control flow in isolation from anything CUDA.
"""

from __future__ import annotations

import pandas as pd
import pytest

from attnbench.config import AttnConfig
from attnbench.sweep import HostMismatchError, SweepCell, run_sweep
from attnbench.timing import Measurement


def _cells(n: int) -> list[SweepCell]:
    return [
        SweepCell(cfg=AttnConfig(seq_len=128 + i, batch=1, n_heads_q=2,
                                  n_heads_kv=2, head_dim=64),
                  backend_name="fake")
        for i in range(n)
    ]


def _fake_measurement() -> Measurement:
    return Measurement(ok=True, status="ok", latency_ms_p50=1.0, latency_ms_p25=1.0,
                        latency_ms_p75=1.0, peak_memory_mb=1.0,
                        issued_tflops=1.0, useful_tflops=1.0)


def _write_stage1_pass_table(path, cells):
    pd.DataFrame([
        {"backend": c.backend_name, "config_key": c.cfg.key(), "passed": True}
        for c in cells
    ]).to_parquet(path)


def test_resume_after_simulated_crash(tmp_path):
    cells = _cells(5)
    stage1_path = tmp_path / "correctness.parquet"
    _write_stage1_pass_table(stage1_path, cells)

    calls = {"n": 0}

    def crash_after_two(backend, cfg, mask):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("simulated process death (e.g. instance reclaimed)")
        return _fake_measurement()

    with pytest.raises(RuntimeError):
        run_sweep(cells, out_dir=tmp_path, stage1_path=stage1_path,
                  backend_lookup={"fake": None}, measure_fn=crash_after_two,
                  exclusivity_check=lambda: ("exclusive", ""), checkpoint_every=1)

    partial = pd.read_parquet(tmp_path / "sweep.parquet")
    assert len(partial) == 2

    calls2 = {"n": 0}

    def count_calls(backend, cfg, mask):
        calls2["n"] += 1
        return _fake_measurement()

    report = run_sweep(cells, out_dir=tmp_path, stage1_path=stage1_path,
                        backend_lookup={"fake": None}, measure_fn=count_calls,
                        exclusivity_check=lambda: ("exclusive", ""), checkpoint_every=1)

    assert calls2["n"] == 3          # only the 3 not-yet-done cells were measured
    assert report.run == 3
    assert report.skip_done == 2

    final = pd.read_parquet(tmp_path / "sweep.parquet")
    assert len(final) == 5           # not 7 -- resume didn't re-measure the first 2
    assert sorted(final["config_key"]) == sorted(c.cfg.key() for c in cells)


def _prov_as(host):
    from dataclasses import replace
    from attnbench import provenance as prov_mod
    real_prov = prov_mod.capture()
    return lambda: replace(real_prov, host=host)


def test_resume_on_same_host_appends_not_collides(tmp_path):
    """Row key includes host: a second run on the *same* machine (e.g. after
    a clean restart, not a host change) must add only the not-yet-done
    cells, never re-measure or collide against its own prior rows."""
    cells = _cells(2)
    stage1_path = tmp_path / "correctness.parquet"
    _write_stage1_pass_table(stage1_path, cells)

    run_sweep(cells, out_dir=tmp_path, stage1_path=stage1_path,
              backend_lookup={"fake": None}, measure_fn=lambda b, c, m: _fake_measurement(),
              exclusivity_check=lambda: ("exclusive", ""),
              provenance_fn=_prov_as("machine-a"))
    report = run_sweep(cells, out_dir=tmp_path, stage1_path=stage1_path,
                        backend_lookup={"fake": None}, measure_fn=lambda b, c, m: _fake_measurement(),
                        exclusivity_check=lambda: ("exclusive", ""),
                        provenance_fn=_prov_as("machine-a"))

    assert report.skip_done == 2
    df = pd.read_parquet(tmp_path / "sweep.parquet")
    assert len(df) == 2
    assert set(df["host"]) == {"machine-a"}


def test_resume_on_a_different_host_refuses(tmp_path):
    """A host change across a resume is a hard boundary, not something to
    silently append through. Without this, load_done_keys' host-scoped keys
    make the new host's cells look "not done yet" -- they'd get
    re-measured and appended next to the old host's rows in the same file,
    and a speedup ratio computed against a baseline row from a different
    physical machine is invalid (constraint 5). This is exactly the shape
    of a Spot preemption that resumes on a different instance mid-sweep."""
    cells = _cells(2)
    stage1_path = tmp_path / "correctness.parquet"
    _write_stage1_pass_table(stage1_path, cells)

    run_sweep(cells, out_dir=tmp_path, stage1_path=stage1_path,
              backend_lookup={"fake": None}, measure_fn=lambda b, c, m: _fake_measurement(),
              exclusivity_check=lambda: ("exclusive", ""),
              provenance_fn=_prov_as("machine-a"))

    before = pd.read_parquet(tmp_path / "sweep.parquet").copy()

    with pytest.raises(HostMismatchError):
        run_sweep(cells, out_dir=tmp_path, stage1_path=stage1_path,
                  backend_lookup={"fake": None}, measure_fn=lambda b, c, m: _fake_measurement(),
                  exclusivity_check=lambda: ("exclusive", ""),
                  provenance_fn=_prov_as("machine-b"))

    # refused before touching the checkpoint or calling measure_fn again
    after = pd.read_parquet(tmp_path / "sweep.parquet")
    pd.testing.assert_frame_equal(before, after)


def test_dry_run_also_refuses_on_host_change(tmp_path):
    """plan() must predict exactly what a real run would do (module
    docstring) -- including refusing. A dry-run that reports a clean plan
    right before a real run raises would defeat the whole point of
    dry-run as a fail-fast check."""
    cells = _cells(2)
    stage1_path = tmp_path / "correctness.parquet"
    _write_stage1_pass_table(stage1_path, cells)

    run_sweep(cells, out_dir=tmp_path, stage1_path=stage1_path,
              backend_lookup={"fake": None}, measure_fn=lambda b, c, m: _fake_measurement(),
              exclusivity_check=lambda: ("exclusive", ""),
              provenance_fn=_prov_as("machine-a"))

    with pytest.raises(HostMismatchError):
        run_sweep(cells, out_dir=tmp_path, stage1_path=stage1_path,
                  backend_lookup={"fake": None}, measure_fn=lambda b, c, m: _fake_measurement(),
                  exclusivity_check=lambda: ("exclusive", ""),
                  provenance_fn=_prov_as("machine-b"), dry_run=True)
