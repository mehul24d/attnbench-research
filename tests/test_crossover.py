"""GLA overtakes FA2 somewhere; the study's claim is that "somewhere" depends
on the card.

The fixture reproduces the real 2026-09-05 join: FA2/GLA latencies at matched
(seq_len, batch) on an L4 and an A100, with the same OOM attrition the real
data has -- the A100 covers three batches at 8192 and 16384 where the L4
covers one, and only the L4 reaches 32768.

That asymmetry is the point. The winner really does differ at 8192, and it
would ALSO appear to differ if one read it off the unrestricted table, where
three A100 cells face one L4 cell. `matched_cells` is what makes the first
reading legitimate rather than lucky.
"""

from __future__ import annotations

import pandas as pd
import pytest

from attnbench.analysis.crossover import (
    CrossoverError,
    crossover_table,
    crossover_point,
    matched_cells,
)

A100 = "NVIDIA A100-SXM4-80GB"
L4 = "NVIDIA L4"

# (gpu, seq_len, batch) -> (fa2_ms, gla_ms). Real measured medians.
CELLS = {
    (A100, 1024, 1): (0.093696, 0.880128),
    (A100, 1024, 4): (0.242176, 1.104384),
    (A100, 2048, 1): (0.251392, 0.870400),
    (A100, 4096, 1): (0.768512, 1.114112),
    (A100, 8192, 1): (2.732032, 2.160128),      # gla wins
    (A100, 8192, 4): (10.678528, 8.359936),
    (A100, 16384, 1): (10.621696, 4.282112),
    (L4, 1024, 1): (0.180992, 0.962048),
    (L4, 1024, 4): (0.610560, 5.019624),
    (L4, 2048, 1): (0.535552, 2.564096),
    (L4, 4096, 1): (1.997568, 5.080576),
    (L4, 8192, 1): (9.193472, 10.971904),       # fa2 still wins
    (L4, 16384, 1): (35.590143, 20.550655),
    (L4, 32768, 1): (153.103363, 41.902063),    # A100 never measured here
}


def frame() -> pd.DataFrame:
    rows = []
    for (gpu, seq, batch), (fa2, gla) in CELLS.items():
        rows.append({"gpu_name": gpu, "seq_len": seq, "batch": batch,
                     "backend": "fa2", "latency_ms_p50": fa2, "ok": True})
        rows.append({"gpu_name": gpu, "seq_len": seq, "batch": batch,
                     "backend": "gla", "latency_ms_p50": gla, "ok": True})
    return pd.DataFrame(rows)


# --- the result --------------------------------------------------------------

def test_the_winner_differs_in_exactly_one_matched_cell():
    matched = matched_cells(crossover_table(frame()))
    disagree = matched[matched["disagrees"]]
    assert len(disagree) == 1
    row = disagree.iloc[0]
    assert (int(row["seq_len"]), int(row["batch"])) == (8192, 1)
    assert row["winner_A100-SXM4-80GB"] == "gla"
    assert row["winner_L4"] == "fa2"


def test_the_crossover_point_moves_one_doubling_earlier_on_ampere():
    cross = crossover_table(frame())
    assert crossover_point(cross, gpu_name=A100, batch=1) == 8192
    assert crossover_point(cross, gpu_name=L4, batch=1) == 16384


def test_matched_cells_drops_the_unshared_shapes():
    """8192/4 exists only on the A100 and 32768/1 only on the L4; both must
    leave the matched table, or the comparison weighs cells that have no
    counterpart."""
    matched = matched_cells(crossover_table(frame()))
    shapes = set(zip(matched["seq_len"], matched["batch"]))
    assert (8192, 4) not in shapes
    assert (32768, 1) not in shapes
    assert (8192, 1) in shapes


def test_a_backend_that_never_wins_has_no_crossover_point():
    """None is a result -- 'not overtaken anywhere we measured' -- and must
    not be confused with a crossover just above the range."""
    df = frame()
    df = df[~((df["gpu_name"] == L4) & (df["seq_len"] >= 16384))]
    assert crossover_point(crossover_table(df), gpu_name=L4, batch=1) is None


# --- the ratio's orientation, which is easy to invert silently --------------

def test_the_ratio_reads_as_fa2_cost_in_units_of_gla():
    cross = crossover_table(frame())
    cell = cross[(cross["gpu_name"] == A100) & (cross["seq_len"] == 16384)].iloc[0]
    assert cell["fa2_over_gla"] == pytest.approx(10.621696 / 4.282112)
    assert cell["fa2_over_gla"] > 1 and cell["winner"] == "gla"

    slow_gla = cross[(cross["gpu_name"] == A100) & (cross["seq_len"] == 1024)].iloc[0]
    assert slow_gla["fa2_over_gla"] < 1 and slow_gla["winner"] == "fa2"


# --- refusals ----------------------------------------------------------------

def test_a_missing_backend_refuses_rather_than_returning_an_empty_table():
    """An empty crossover table reads downstream as 'the backends never
    crossed', which is a claim; 'this backend was not measured' is not."""
    df = frame()
    df = df[df["backend"] != "gla"]
    with pytest.raises(CrossoverError, match="gla"):
        crossover_table(df)


def test_one_architecture_alone_cannot_produce_a_comparison():
    df = frame()
    df = df[df["gpu_name"] == A100]
    with pytest.raises(CrossoverError, match="at least two"):
        matched_cells(crossover_table(df))


def test_failed_rows_are_excluded_from_the_median():
    """An `ok=False` row carries whatever latency the failure path left
    behind; including it would put a non-measurement into a median."""
    df = frame()
    poison = df[(df["gpu_name"] == A100) & (df["seq_len"] == 16384)].copy()
    poison["latency_ms_p50"] = 0.0001
    poison["ok"] = False
    cross = crossover_table(pd.concat([df, poison], ignore_index=True))
    cell = cross[(cross["gpu_name"] == A100) & (cross["seq_len"] == 16384)].iloc[0]
    assert cell["fa2_over_gla"] == pytest.approx(10.621696 / 4.282112)
