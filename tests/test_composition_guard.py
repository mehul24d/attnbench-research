"""A median across differently-composed cells must not be computable by
accident.

The fixture below is not synthetic in shape: `GLA_L4` reproduces the real
per-(seq_len, batch) medians measured for the `gla` backend on an L4 across
Stage 2 segments seg1/seg2sweep/seg3, transcribed here because `results/` is
gitignored and a regression test that vanishes on a fresh clone is not a
regression test.

Those numbers are the 2026-09-05 artefact in its original form:

    per-cell, batch=1:   0.96 -> 2.56 -> 5.08 -> 10.97 -> 20.55 -> 41.90 ms
    marginal median:     5.02 -> 9.93 -> 19.81 -> 10.97 -> 20.55 -> 101.20 ms
                                              ^^^^^^^^^^ cost FALLS 45%

Nothing was mismeasured. GLA is close to perfectly linear per cell -- almost
exactly 2x per doubling, which is what a linear-attention kernel should do.
The marginal median inverts at 8192 purely because batches 4 and 16 OOM'd
there, so the band's surviving composition changed from {1,4,16} to {1}.

The test that matters is `test_the_real_gla_artefact_is_refused`: had this
guard existed, the sentence "GLA's cost falls from 4096 to 8192" could not
have been written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attnbench.analysis.composition import (
    Composition,
    CompositionMismatch,
    aggregate,
    check_composition,
    composition,
    composition_report,
    compositions_by_group,
    matched_subset,
    max_pairwise_distance,
)

# (seq_len, batch) -> latency_ms_p50, measured on an NVIDIA L4. NaN cells are
# the OOM attrition -- omitted here exactly as they are absent from the data.
GLA_L4 = {
    (1024, 1): 0.962048, (1024, 4): 5.019624, (1024, 16): 19.492352,
    (2048, 1): 2.564096, (2048, 4): 9.934848, (2048, 16): 39.091713,
    (4096, 1): 5.080576, (4096, 4): 19.807743, (4096, 16): 78.840832,
    (8192, 1): 10.971904,
    (16384, 1): 20.550655,
    (32768, 1): 41.902063, (32768, 4): 164.914688,
}


def gla_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"seq_len": s, "batch": b, "latency_ms_p50": v, "backend": "gla"}
         for (s, b), v in GLA_L4.items()])


def balanced_frame(seq_lens=(1024, 2048, 4096), batches=(1, 4, 16)) -> pd.DataFrame:
    return pd.DataFrame(
        [{"seq_len": s, "batch": b, "latency_ms_p50": float(s * b) / 1000.0}
         for s in seq_lens for b in batches])


# --- the incident itself ---------------------------------------------------

def test_the_marginal_median_really_does_invert():
    """Guard against the fixture drifting into something that no longer
    reproduces the bug -- a regression test whose regression has quietly
    stopped happening is worse than none, because it still reports green."""
    df = gla_frame()
    marginal = df.groupby("seq_len")["latency_ms_p50"].median()
    assert marginal[8192] < marginal[4096], "fixture no longer reproduces the inversion"

    per_cell = df[df["batch"] == 1].set_index("seq_len")["latency_ms_p50"]
    assert per_cell.is_monotonic_increasing, "per-cell truth is monotone; only the marginal inverts"


def test_the_real_gla_artefact_is_refused():
    df = gla_frame()
    with pytest.raises(CompositionMismatch) as exc:
        aggregate(df, group_by="seq_len", value="latency_ms_p50")
    msg = str(exc.value)
    assert "seq_len" in msg
    assert "batch" in msg
    # The refusal must say WHICH levels went missing, not just that something
    # differs -- "composition differs" alone gets suppressed, not investigated.
    assert "16" in msg


def test_the_repair_produces_the_monotone_truth():
    """matched_subset restricts to batch=1, the only level present in every
    band, and the aggregate over it recovers the physical result."""
    df = gla_frame()
    out = aggregate(df, group_by="seq_len", value="latency_ms_p50",
                    on_mismatch="match").sort_values("seq_len")
    assert out["latency_ms_p50"].is_monotonic_increasing
    assert (out["facet_levels"].apply(lambda lv: lv == [1])).all()
    assert out["composition_matched"].all()


def test_annotating_carries_the_caveat_into_the_table():
    """The 'at minimum, report it' path: the wrong number is still wrong, but
    it can no longer be quoted without its composition."""
    df = gla_frame()
    out = aggregate(df, group_by="seq_len", value="latency_ms_p50",
                    on_mismatch="annotate").sort_values("seq_len")
    assert not out["latency_ms_p50"].is_monotonic_increasing   # still the artefact
    assert not out["composition_matched"].all()
    assert (out["composition_tvd"] > 0).all()
    row_8192 = out[out["seq_len"] == 8192].iloc[0]
    assert row_8192["facet_counts"] == "{1:1}"
    assert row_8192["n"] == 1


# --- Composition arithmetic ------------------------------------------------

def test_identical_composition_has_zero_distance():
    a = composition(balanced_frame(seq_lens=(1024,)), "batch")
    b = composition(balanced_frame(seq_lens=(2048,)), "batch")
    assert a.distance(b) == 0.0
    assert a.missing_versus(b) == []


def test_disjoint_levels_are_distance_one():
    a = Composition("batch", {1: 5})
    b = Composition("batch", {16: 5})
    assert a.distance(b) == pytest.approx(1.0)
    assert a.missing_versus(b) == [16]


def test_distance_degrades_smoothly_rather_than_as_a_set_test():
    """Losing one row of a hundred must not read the same as losing every
    large batch; a set comparison cannot tell those apart, which is why the
    metric is TVD."""
    full = Composition("batch", {1: 50, 4: 50})
    one_row_short = Composition("batch", {1: 50, 4: 49})
    half_gone = Composition("batch", {1: 50})
    assert 0 < full.distance(one_row_short) < 0.02
    assert full.distance(half_gone) == pytest.approx(0.5)


def test_reweighting_alone_is_still_a_mismatch():
    """Same level set, different weights: a median over ten batch-1 rows and
    one batch-16 row is not comparable to the reverse, even though nothing
    is missing."""
    df = pd.DataFrame(
        [{"g": "a", "batch": 1, "v": 1.0}] * 10 + [{"g": "a", "batch": 16, "v": 100.0}]
        + [{"g": "b", "batch": 1, "v": 1.0}] + [{"g": "b", "batch": 16, "v": 100.0}] * 10)
    with pytest.raises(CompositionMismatch):
        aggregate(df, group_by="g", value="v", facet="batch")


def test_max_pairwise_takes_the_worst_pair_not_the_average():
    comps = {"a": Composition("batch", {1: 1}),
             "b": Composition("batch", {1: 1}),
             "c": Composition("batch", {16: 1})}
    (pair, dist) = max_pairwise_distance(comps)
    assert dist == pytest.approx(1.0)
    assert set(pair) in ({"a", "c"}, {"b", "c"})


def test_a_single_group_has_no_pairwise_distance():
    assert max_pairwise_distance({"only": Composition("batch", {1: 1})}) is None


# --- the balanced case must stay usable ------------------------------------

def test_a_balanced_aggregate_is_allowed_and_still_annotated():
    """A guard that only ever refuses gets removed. The common case must pass
    -- and still carry its composition, so the annotation is not something a
    reader learns to associate only with bad tables."""
    out = aggregate(balanced_frame(), group_by="seq_len", value="latency_ms_p50")
    assert len(out) == 3
    assert out["composition_matched"].all()
    assert (out["composition_tvd"] == 0.0).all()
    assert (out["n"] == 3).all()
    assert out["facet_counts"].iloc[0] == "{1:1, 4:1, 16:1}"


def test_there_is_no_way_to_get_a_bare_number_out():
    """Every exit carries provenance columns. This is the structural property
    the module rests on: a table built from `aggregate` cannot lose the
    composition of its own averages the way a hand-written groupby did."""
    required = {"n", "facet", "facet_levels", "facet_counts",
                "composition_tvd", "composition_matched", "how"}
    for mode in ("raise", "annotate", "match"):
        out = aggregate(balanced_frame(), group_by="seq_len",
                        value="latency_ms_p50", on_mismatch=mode)
        assert required <= set(out.columns), f"{mode} lost provenance columns"


def test_multi_column_group_by_round_trips_its_keys():
    df = balanced_frame()
    df["backend"] = "gla"
    out = aggregate(df, group_by=["backend", "seq_len"], value="latency_ms_p50")
    assert set(out.columns) >= {"backend", "seq_len"}
    assert sorted(out["seq_len"]) == [1024, 2048, 4096]


# --- refusals that must not be silent --------------------------------------

def test_a_missing_facet_column_refuses_rather_than_skipping_the_check():
    """The dangerous default: no `batch` column could plausibly mean 'nothing
    to check'. It means the check cannot run, which is not the same thing."""
    df = balanced_frame().drop(columns=["batch"])
    with pytest.raises(CompositionMismatch, match="batch"):
        aggregate(df, group_by="seq_len", value="latency_ms_p50")


def test_unknown_aggregation_and_mode_names_refuse():
    df = balanced_frame()
    with pytest.raises(CompositionMismatch, match="how="):
        aggregate(df, group_by="seq_len", value="latency_ms_p50", how="p50")
    with pytest.raises(CompositionMismatch, match="on_mismatch="):
        aggregate(df, group_by="seq_len", value="latency_ms_p50", on_mismatch="ignore")
    with pytest.raises(CompositionMismatch, match="latency"):
        aggregate(df, group_by="seq_len", value="latency", on_mismatch="annotate")


def test_no_shared_level_refuses_instead_of_returning_empty():
    """An empty frame silently aggregates to nothing, which downstream reads
    as 'no data' rather than 'these groups have no cell in common'."""
    df = pd.DataFrame([{"g": "a", "batch": 1, "v": 1.0},
                       {"g": "b", "batch": 16, "v": 2.0}])
    with pytest.raises(CompositionMismatch, match="no batch level"):
        matched_subset(df, group_by="g", facet="batch")


def test_max_distance_must_be_opted_into_at_the_call_site():
    df = gla_frame()
    with pytest.raises(CompositionMismatch):
        check_composition(df, group_by="seq_len", facet="batch")
    # Loose enough to admit it, stated in the caller's code where it is visible.
    check_composition(df, group_by="seq_len", facet="batch", max_distance=1.0)


# --- NaN handling ----------------------------------------------------------

def test_a_nan_facet_level_is_reported_not_dropped():
    """`sparsity` is NaN for every dense row. A facet with NaN levels must
    still be countable, because 'some rows have no sparsity' is itself a
    composition difference between a dense and a sparse group."""
    df = pd.DataFrame([{"g": "a", "sparsity": np.nan, "v": 1.0},
                       {"g": "a", "sparsity": 0.5, "v": 2.0},
                       {"g": "b", "sparsity": 0.5, "v": 3.0}])
    comps = compositions_by_group(df, group_by="g", facet="sparsity")
    assert comps["a"].n == 2
    assert comps["b"].n == 1
    assert comps["a"].distance(comps["b"]) == pytest.approx(0.5)
    with pytest.raises(CompositionMismatch):
        aggregate(df, group_by="g", value="v", facet="sparsity")


def test_two_groups_of_all_nan_are_identically_composed():
    """The assertion that locks the sentinel fix. Without normalising NaN to
    a single key, these two groups score TVD=1.0 -- maximally different --
    because the two NaN dict keys never compare equal. That over-refuses, and
    the same equality failure makes matched_subset drop shared rows."""
    df = pd.DataFrame([{"g": "a", "sparsity": np.nan, "v": 1.0},
                       {"g": "b", "sparsity": np.nan, "v": 2.0}])
    comps = compositions_by_group(df, group_by="g", facet="sparsity")
    assert comps["a"].distance(comps["b"]) == 0.0
    assert comps["a"].missing_versus(comps["b"]) == []
    out = aggregate(df, group_by="g", value="v", facet="sparsity")
    assert out["composition_matched"].all()


def test_matched_subset_keeps_a_shared_nan_level():
    """isin([nan]) is False for NaN -- the subset would silently drop rows
    that are genuinely shared. Handled explicitly, and asserted here because
    it is invisible on inspection."""
    df = pd.DataFrame([{"g": "a", "sparsity": np.nan, "v": 1.0},
                       {"g": "a", "sparsity": 0.5, "v": 2.0},
                       {"g": "b", "sparsity": np.nan, "v": 3.0},
                       {"g": "b", "sparsity": np.nan, "v": 4.0}])
    sub = matched_subset(df, group_by="g", facet="sparsity")
    assert len(sub) == 3
    assert sub["sparsity"].isna().all()


# --- the report a human reads ----------------------------------------------

def test_the_report_names_groups_levels_and_the_worst_pair():
    text = composition_report(gla_frame(), group_by="seq_len", facet="batch")
    assert "worst pair" in text
    assert "TVD=" in text
    assert "present in one group and not the other" in text
    for level in ("1", "4", "16"):
        assert level in text
