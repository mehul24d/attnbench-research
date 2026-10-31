"""`builder_kind` is where the TIMED mask construction runs.

An arm's declared kind is held to what the probe saw inside its timed blocks:
the host's reference builder (`masks.mask_for`) is "cpu"; the device builder and
the XAttention estimator are "gpu". The oracle builds its mask from cached
scores on the host inside the timed block (its rows count `REPS * LAYERS`
timed mask builds), so it keeps "cpu": there is no "none" value because no arm
has no timed construction (a dense arm has no mask, and says None).
"""

from __future__ import annotations

import pytest

from attnbench import ground_truth as gt
from attnbench.gt_schema import BUILDER_KINDS
from _gt_fakes import fake_densities
from test_ground_truth_timed_region import (LAYERS, REPS, _controls, _cfg, _dense, _ids,
                                            _wrapped)
from attnbench.backends.impls import NaiveAttention


def _oracle(kind):
    return gt.Arm("oracle", gt.ORACLE, NaiveAttention(), _cfg(), builder_kind=kind)


def _inline(kind):
    return gt.Arm("meanpool", gt.INLINE, NaiveAttention(), _cfg(),
                  inline_estimator="minference_meanpool", builder_kind=kind)


def _time(arm, tmp_path):
    return gt.time_arms(_wrapped(), _ids(), [arm, _dense()], reps=REPS,
                        density_of=fake_densities, controls=_controls(),
                        score_kwargs=dict(task="t", example_id="e", cache_dir=str(tmp_path)))


def test_the_oracle_builds_on_the_host_inside_the_timed_block(tmp_path):
    row = _time(_oracle("cpu"), tmp_path)["rows"][0]
    assert row["builder_kind"] == "cpu" and row["mask_builds_timed"] == REPS * LAYERS


def test_an_oracle_labelled_gpu_is_refused(tmp_path):
    with pytest.raises(gt.TimedRegionError, match=r"builder_kind='gpu'.*\['cpu'\]"):
        _time(_oracle("gpu"), tmp_path)


def test_the_device_builder_is_gpu_and_labelled_cpu_is_refused(tmp_path):
    assert _time(_inline("gpu"), tmp_path)["rows"][0]["builder_kind"] == "gpu"
    with pytest.raises(gt.TimedRegionError, match=r"builder_kind='cpu'.*\['gpu'\]"):
        _time(_inline("cpu"), tmp_path)


def test_every_observed_target_is_one_the_probe_logs():
    # A map entry the probe never logs would make the check vacuous for that arm.
    logged = {f"{m}.{p}" for m, p in gt.SCORE_TARGETS + gt.MASK_TARGETS}
    assert set(gt.BUILDER_OF_TARGET) <= logged
    assert set(gt.BUILDER_OF_TARGET.values()) <= set(BUILDER_KINDS)
