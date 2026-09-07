"""The forget gate has to be on the row, not in the commit message.

Stage 3 segment 1 wrote 900 GLA rows that looked exactly like every other
row and meant nothing, because the gate that produced them was random noise
with a 1.24-token memory horizon (docs/silent_failure_patterns.md #17). The
fix that stops it recurring is not a better default -- it is that a linear
row cannot be written without naming the mechanism that produced it.

These tests assert both halves of that, in the shape this project settled on
after the position_ids catch: assert the mechanism, and separately assert the
mechanism is observable.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pandas as pd
import pytest

from attnbench.accuracy.grid_configs import backend_instance, gate_source_of
from attnbench.accuracy.runner import AccuracyCell, run_accuracy
from attnbench.accuracy.schema import (GATED_BACKENDS, AccuracyResult,
                                        Generated, GateSource)
from attnbench.backends.linear import GATE_SOURCES
from attnbench.config import AttnConfig
from attnbench import backends


# --------------------------------------------------------------------------
# The two vocabularies must be one vocabulary.
# --------------------------------------------------------------------------

def test_schema_literal_matches_the_backend_that_implements_it():
    """A gate the backend accepts but the schema does not list would write a
    value no reader can interpret; the reverse advertises a mechanism that
    cannot run. Kept as one fact rather than two copies."""
    assert set(get_args(GateSource)) == set(GATE_SOURCES)


def test_every_backend_with_a_gate_is_named_in_gated_backends():
    """The observability half.

    `GATED_BACKENDS` is a name set, so nothing stops a future gated backend
    (Gated DeltaNet is planned in backends/linear.py's docstring) from being
    registered without being added to it -- and its rows would then be
    allowed through with no gate recorded, which is exactly #17 again under a
    different name. This is the test that fails when that happens.
    """
    has_gate = set()
    for name, cls in backends.all_backends().items():
        try:
            instance = cls()
        except TypeError:
            continue      # needs constructor args; covered by name below
        if gate_source_of(instance) is not None:
            has_gate.add(name)
    assert has_gate == set(GATED_BACKENDS), (
        f"backends carrying a gate_source: {sorted(has_gate)}; "
        f"schema.GATED_BACKENDS: {sorted(GATED_BACKENDS)}")


def test_named_gated_backends_actually_have_a_gate():
    """The converse. A name in the set that no longer has a gate would make
    every row from it unwritable -- a run that fails at cell 1 of 4500."""
    for name in GATED_BACKENDS:
        assert gate_source_of(backend_instance(name)) is not None


# --------------------------------------------------------------------------
# The row refuses to exist without the caveat.
# --------------------------------------------------------------------------

def _row(**overrides):
    fields = dict(
        backend="sdpa_flash", backend_role="dense_reference", config_key="k",
        task="niah_single", example_id="e0", context_length=2048,
        mask_source=None, sparsity=None, score_source=None,
        haystack_mode="noise", predicted="1234567", expected="1234567",
        score=100.0, correct=True)
    fields.update(overrides)
    return AccuracyResult(**fields)


def test_a_gated_row_without_a_gate_source_is_refused():
    with pytest.raises(ValueError, match="must record which one ran"):
        _row(backend="gla", backend_role="linear")


def test_a_gated_row_with_a_gate_source_is_fine():
    assert _row(backend="gla", backend_role="linear",
                gate_source="ungated").gate_source == "ungated"


def test_an_ungated_row_claiming_a_gate_is_refused():
    """The direction that is easy to forget. A value here would make a reader
    believe the dense arm was configured, and would break the one query the
    column exists for."""
    with pytest.raises(ValueError, match="has no forget gate"):
        _row(gate_source="ungated")


def test_backend_instance_refuses_a_gate_for_a_backend_without_one():
    """Accepted-and-ignored is how a run believes it configured something it
    did not."""
    with pytest.raises(ValueError, match="has no forget gate"):
        backend_instance("sdpa_flash", gate_source="ungated")


def test_omitting_the_gate_leaves_the_backend_on_the_refusing_default():
    """The absence of a choice must not resolve to a choice."""
    assert gate_source_of(backend_instance("gla")) == "learned"


# --------------------------------------------------------------------------
# End to end: it reaches the parquet, and its absence stops the run.
# --------------------------------------------------------------------------

_CFG = AttnConfig(seq_len=2048, batch=1, n_heads_q=4, n_heads_kv=2,
                  head_dim=64, mask="causal")


class _Example:
    task = "niah_single"
    example_id = "e0"
    context_length = 2048
    answer = ["1234567"]
    sizing = "exact"
    context = "irrelevant"


def _run(tmp_path: Path, gen: Generated) -> pd.DataFrame:
    cells = [AccuracyCell(cfg=_CFG, backend_name="gla", task="niah_single",
                          example_id="e0")]
    run_accuracy(cells, out_dir=tmp_path,
                 examples_by_id={("niah_single", "e0"): _Example()},
                 generate_fn=lambda cfg, name, ex: gen,
                 provenance_fn=lambda: _Prov())
    return pd.read_parquet(tmp_path / "accuracy.parquet")


class _Prov:
    git_commit = "0" * 40
    git_dirty = False

    def to_dict(self):
        return {"git_commit": self.git_commit, "git_dirty": self.git_dirty}


def test_gate_source_reaches_the_parquet_column(tmp_path):
    df = _run(tmp_path, Generated(text="1234567", gate_source="ungated"))
    assert df.iloc[0]["gate_source"] == "ungated"
    assert df.iloc[0]["backend"] == "gla"


def test_a_generate_fn_that_forgets_the_gate_stops_the_run(tmp_path):
    """The point of the invariant. Before it, this wrote a row -- 900 of
    them -- indistinguishable from a meaningful one. Now it raises at the
    first cell, which costs one cell of GPU time instead of a band."""
    with pytest.raises(ValueError, match="must record which one ran"):
        _run(tmp_path, Generated(text="1234567"))
    assert not (tmp_path / "accuracy.parquet").exists()


def test_the_column_distinguishes_the_two_gates(tmp_path):
    """Assert the mechanism is observable: a column that recorded the same
    value regardless of what ran would pass every test above and carry no
    information."""
    a = _run(tmp_path / "a", Generated(text="x", gate_source="ungated"))
    b = _run(tmp_path / "b", Generated(text="x", gate_source="synthetic"))
    assert a.iloc[0]["gate_source"] != b.iloc[0]["gate_source"]


# --------------------------------------------------------------------------
# A DROP verdict has to be expressible, or the rule decides nothing.
# --------------------------------------------------------------------------

def test_the_gla_arm_can_be_dropped_from_the_grid():
    """What a DROP verdict does.

    Found in Phase 0 of the S2 booking: `build_configs_by_backend` included
    `gla` unconditionally, so the pre-registered rule could return DROP and
    the pipeline had no way to act on it. And because gla's default gate
    REFUSES, an unactioned DROP is not a quiet skip -- it raises on the first
    gla cell, which is the LAST 900 cells of a 4-hour band, taking every
    chained band after it.
    """
    from attnbench.accuracy.config import load_grid
    from attnbench.accuracy.grid_configs import build_configs_by_backend

    grid = load_grid("configs/accuracy/stage3_grid.yaml")
    kept = build_configs_by_backend(grid, include_sage=False, seq_lens=(4096,))
    dropped = build_configs_by_backend(grid, include_sage=False,
                                        seq_lens=(4096,), include_gla=False)

    assert "gla" in kept
    assert "gla" not in dropped
    # The rest of the study is untouched by the verdict.
    assert set(kept) - {"gla"} == set(dropped)
    assert dropped["block_sparse"] and dropped[grid.dense_backend]


def test_the_hours_reporter_cannot_enumerate_a_category_it_cannot_price():
    """Regression for KeyError: 'decode' (2026-09-07, on billed hardware).

    `total_grid_flops_by_category` grew a "decode" key; `corrected_grid_hours`
    deliberately RAISES rather than price a bandwidth-bound phase from TFLOPS,
    so the priced dicts are built from a stripped copy. The reporter still
    enumerated the unstripped one and indexed the stripped result.

    Asserted as a general property rather than about "decode" specifically:
    whatever categories a future version adds, the set the reporter walks must
    be a subset of the set it can look up.
    """
    from attnbench.accuracy.timing_probe import corrected_grid_hours

    total_flops = {"scoring": 10**15, "measured": 10**15, "decode": 10**15}
    prefill_only = {k: v for k, v in total_flops.items() if k != "decode"}
    hours = corrected_grid_hours(prefill_only, {c: 15.0 for c in prefill_only})

    assert set(prefill_only) <= set(hours), (
        "every category the reporter walks must be priceable")
    assert "decode" not in hours, (
        "decode must NOT acquire a TFLOPS price -- the raise is the point")
