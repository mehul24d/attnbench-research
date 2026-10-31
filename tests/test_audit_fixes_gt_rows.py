"""Fix for audit C3, finding 2: every ground-truth row carries its context
length, batch, mask density, executed density and builder kind, and the
readers refuse a row without them. Written before the fix. torch-free; the
harness half (`time_arms` writing the fields) is in
test_audit_fixes_ground_truth.py.
"""

from __future__ import annotations

import pytest

from attnbench import gt_schema as G
from attnbench.figures import paper_inputs as PI
from attnbench.figures import results as RS

MODEL, CP = "m", "L4|xattn:torch|torch=2.8.0|O0"


def row(arm="xattn", kind="in_backend", **over):
    r = {"arm": arm, "arm_kind": kind, "deployable": kind in ("inline", "in_backend"),
         "prefill_ms_median": 20000.0, "prefill_ms_p90": 20000.0, "prefill_ms_p99": 20000.0,
         "context_length": 4096, "batch": 1, "mask_density": 0.25,
         "executed_density": 0.3, "builder_kind": "gpu"}
    if kind == "dense":
        r.update(mask_density=None, executed_density=1.0, builder_kind=None)
    r.update(over)
    return r


def result(*rows):
    return {"traced": False, "complete": True, "rows": list(rows)}


def test_the_fields_and_the_builder_kinds_are_c10s():
    assert G.ROW_FIELDS == ("context_length", "batch", "mask_density",
                            "executed_density", "builder_kind")
    assert G.BUILDER_KINDS == ("cpu", "gpu")
    assert PI.BUILDER_KINDS is G.BUILDER_KINDS


def test_complete_rows_pass():
    G.check_row(row())
    G.check_row(row("dense", "dense"))
    G.check_row(row("oracle", "oracle", builder_kind="cpu"))


@pytest.mark.parametrize("field", ["context_length", "batch", "mask_density",
                                   "executed_density", "builder_kind", "arm_kind"])
def test_a_row_missing_a_field_is_refused(field):
    r = row()
    del r[field]
    with pytest.raises(G.RowFieldError, match=field):
        G.check_row(r)


@pytest.mark.parametrize("over, match", [
    ({"context_length": 0}, "context_length"), ({"context_length": 4096.0}, "context_length"),
    ({"batch": True}, "batch"), ({"batch": 0}, "batch"),
    ({"mask_density": None}, "mask_density"), ({"mask_density": 1.5}, "mask_density"),
    ({"executed_density": 0.0}, "executed_density"),
    ({"builder_kind": None}, "builder_kind"), ({"builder_kind": "tpu"}, "builder_kind"),
])
def test_a_sparse_row_with_a_bad_field_is_refused(over, match):
    with pytest.raises(G.RowFieldError, match=match):
        G.check_row(row(**over))


@pytest.mark.parametrize("over, match", [
    ({"mask_density": 0.5}, "mask_density"), ({"executed_density": 0.9}, "executed_density"),
    ({"builder_kind": "gpu"}, "builder_kind"),
])
def test_a_dense_row_has_no_mask_no_builder_and_executes_every_block(over, match):
    with pytest.raises(G.RowFieldError, match=match):
        G.check_row(row("dense", "dense", **over))


def test_the_reader_refuses_a_row_without_the_fields():
    bare = {k: v for k, v in row().items() if k not in G.ROW_FIELDS}
    with pytest.raises(RS.ResultsError, match="context_length"):
        RS.measured_cells([("gt", result(bare, row("dense", "dense")),
                            {"xattn": (MODEL, CP, "xattn", 0.25, 4096, 1)})],
                          statistic="median")


@pytest.mark.parametrize("cell, match", [
    ((MODEL, CP, "xattn", 0.25, 8192, 1), "context length"),
    ((MODEL, CP, "xattn", 0.25, 4096, 4), "batch"),
])
def test_the_reader_refuses_a_stated_cell_the_row_contradicts(cell, match):
    with pytest.raises(RS.ResultsError, match=match):
        RS.measured_cells([("gt", result(row(), row("dense", "dense")), {"xattn": cell})],
                          statistic="median")


def test_the_reader_takes_a_complete_row():
    out = RS.measured_cells([("gt", result(row(), row("dense", "dense")),
                              {"xattn": (MODEL, CP, "xattn", 0.25, 4096, 1)})],
                            statistic="median")
    assert list(out.values()) == [20.0]
