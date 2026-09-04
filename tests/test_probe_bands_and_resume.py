"""Stage 0/1 banding, checkpointing and resume. CPU-only.

Both Stage 1 attempts on 2026-09-04 lost their complete run to a late crash --
24 minutes to an OOM, then 29 to an Xid 31 MMU fault -- because `run_probe.py`
accumulated rows in a list and wrote parquet only at the very end. The second
fault's last logged configs were `seq_len=32768`, and it destroyed the 1024
through 16384 work with it: a coupling between independent measurements that
had no reason to exist.

Two properties fix that, and both are tested here rather than trusted:

  * **bands ascend, and each is banked before the next begins** -- so a fault
    at the longest length cannot cost the shorter ones
  * **resume skips what is already banked** -- so retrying a faulted band
    costs only that band

Stage 3 makes this non-optional rather than merely prudent: it is a ~10 h job
on hardware that has become unreachable three times in this project.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from attnbench import checkpoint
from attnbench.config import AttnConfig

ROOT = Path(__file__).resolve().parent.parent


def _load_run_probe():
    spec = importlib.util.spec_from_file_location(
        "run_probe_mod", ROOT / "scripts" / "run_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_probe_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


RP = _load_run_probe()


def _cfg(seq_len: int, batch: int = 1) -> AttnConfig:
    return AttnConfig(seq_len=seq_len, batch=batch, n_heads_q=32,
                      n_heads_kv=32, head_dim=128, dtype="bfloat16",
                      mask="causal", pass_kind="fwd", regime="prefill")


# --- banding ---------------------------------------------------------------

def test_bands_are_visited_shortest_first():
    configs = [_cfg(s) for s in (32768, 1024, 8192, 2048)]
    order = [band for band, _, _ in RP.band_plan(configs, ["fa2"], set())]
    assert order == [1024, 2048, 8192, 32768]


def test_a_band_is_finished_before_the_next_one_starts():
    """Interleaving would put 32768 work ahead of shorter results being
    banked, which is exactly the coupling this removes."""
    configs = [_cfg(s) for s in (1024, 8192)]
    seen = [band for band, _, _ in
            RP.band_plan(configs, ["fa2", "flex", "naive"], set())]
    assert seen == [1024, 1024, 1024, 8192, 8192, 8192]


def test_every_backend_is_planned_within_each_band():
    configs = [_cfg(1024), _cfg(8192)]
    plan = list(RP.band_plan(configs, ["fa2", "flex"], set()))
    assert {(b, n) for b, n, _ in plan} == {
        (1024, "fa2"), (1024, "flex"), (8192, "fa2"), (8192, "flex")}


def test_min_and_max_seq_select_a_band():
    mid = list(RP.probe_configs(max_seq=8192, min_seq=8192))
    assert mid, "the 8192 band should be non-empty"
    assert {c.seq_len for c in mid} == {8192}

    short = list(RP.probe_configs(max_seq=4096))
    assert {c.seq_len for c in short} <= {1024, 2048, 4096}
    assert 8192 not in {c.seq_len for c in short}


# --- resume ----------------------------------------------------------------

def test_resume_skips_banked_work_for_that_backend_only():
    configs = [_cfg(1024), _cfg(1024, batch=4)]
    done = {("fa2", configs[0].key())}
    plan = {(b, n): [c.key() for c in todo]
            for b, n, todo in RP.band_plan(configs, ["fa2", "flex"], done)}
    assert plan[(1024, "fa2")] == [configs[1].key()]
    assert len(plan[(1024, "flex")]) == 2, (
        "one backend's banked row must not suppress another's work")


def test_done_keys_round_trips(tmp_path):
    path = tmp_path / "probe.parquet"
    checkpoint.append_checkpoint(path, [
        {"backend": "fa2", "config_key": "aaa", "x": 1},
        {"backend": "flex", "config_key": "bbb", "x": 2}])
    assert checkpoint.done_keys(path, "backend", "config_key") == {
        ("fa2", "aaa"), ("flex", "bbb")}


def test_done_keys_on_a_missing_file_is_empty_not_an_error(tmp_path):
    assert checkpoint.done_keys(tmp_path / "nope.parquet",
                                "backend", "config_key") == set()


def test_an_unreadable_checkpoint_means_redo_not_skip(tmp_path):
    """The dangerous direction. A corrupt checkpoint read as 'everything is
    done' would silently produce an empty run that looks complete."""
    path = tmp_path / "corrupt.parquet"
    path.write_bytes(b"not a parquet file at all")
    assert checkpoint.done_keys(path, "backend", "config_key") == set()


def test_missing_columns_mean_redo_not_skip(tmp_path):
    path = tmp_path / "other.parquet"
    checkpoint.append_checkpoint(path, [{"something_else": 1}])
    assert checkpoint.done_keys(path, "backend", "config_key") == set()


# --- checkpoint durability -------------------------------------------------

def test_append_accumulates_across_calls(tmp_path):
    path = tmp_path / "c.parquet"
    checkpoint.append_checkpoint(path, [{"backend": "a", "config_key": "1"}])
    checkpoint.append_checkpoint(path, [{"backend": "b", "config_key": "2"}])
    assert len(pd.read_parquet(path)) == 2


def test_append_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "c.parquet"
    checkpoint.append_checkpoint(path, [{"backend": "a", "config_key": "1"}])
    assert [p.name for p in tmp_path.iterdir()] == ["c.parquet"]


def test_a_failed_write_leaves_the_previous_checkpoint_intact(tmp_path,
                                                              monkeypatch):
    """Read-concat-overwrite rewrites the WHOLE file on every append, so a
    crash mid-write would otherwise destroy every row already banked -- the
    exact failure the checkpoint exists to prevent."""
    path = tmp_path / "c.parquet"
    checkpoint.append_checkpoint(path, [{"backend": "a", "config_key": "1"}])
    before = path.read_bytes()

    real = pd.DataFrame.to_parquet

    def explode(self, where, *a, **kw):
        real(self, where, *a, **kw)          # write the temp file...
        raise OSError("simulated crash before the rename")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", explode)
    with pytest.raises(OSError):
        checkpoint.append_checkpoint(path, [{"backend": "b",
                                             "config_key": "2"}])
    monkeypatch.undo()

    assert path.read_bytes() == before, "the banked checkpoint was damaged"
    assert len(pd.read_parquet(path)) == 1


def test_append_creates_the_parent_directory(tmp_path):
    path = tmp_path / "nested" / "deeper" / "c.parquet"
    checkpoint.append_checkpoint(path, [{"backend": "a", "config_key": "1"}])
    assert path.exists()
