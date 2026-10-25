"""`scripts/measure_mask_h2d_tax.py` builds summary.json in `summary()`.

The fourth audit pass (`ce2ad91`) moved the analysis into `analyse()` and
removed `p50s` from `main()`, but the summary still read it (line 375 on
25 Oct 2026). Every GPU run since would have written mask_h2d.parquet and
then raised NameError before summary.json. Nothing caught it, because the
only code that ran on CPU was `analyse()`.

Two tests: the summary on synthetic rows, on CPU; and a static check that
every global name `main()` and `summary()` read exists in the module, with a
break-test that puts the old bug back.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import symtable
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "measure_mask_h2d_tax.py"


def _module():
    spec = importlib.util.spec_from_file_location("_s11_summary", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_s11_summary"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows():
    rows = []
    for band, copy, full in ((1024, 20.0, 31.0), (8192, 22.5, 33.0), (32768, 28.0, 39.5)):
        rows.append({"band": band, "n_blocks": band // 64, "bytes": band * 4,
                     "copy_p50_us": copy, "full_call_p50_us": full,
                     "batched_per_call_us": copy - 4.0,
                     "batched_full_per_call_us": full - 4.0})
    return rows


def test_the_summary_is_built_on_cpu_and_serialises():
    mod = _module()
    rows = _rows()
    a = mod.analyse(rows, 18.0)
    s = mod.summary(rows, {"p50_us": 18.0}, {"p50_us": 9.0}, a, "fake-gpu")
    json.dumps(s)
    assert s["measured_p50_span_us"] == [20.0, 28.0]
    assert s["ratio_cost"] == round(28.0 / 20.0, 3)
    assert s["marginal_ratio"] == round((28.0 - 18.0) / (20.0 - 18.0), 3)
    assert s["gpu"] == "fake-gpu"
    assert "over_floor_us" not in s["analysis"]
    assert set(s) == {"rows", "floor", "sync_only", "ratio_cost", "ratio_size",
                      "marginal_ratio", "unresolvable_bands", "verdict", "gpu",
                      "assumed_range_us", "analysis", "measured_p50_span_us"}


def undefined_globals(source: str, namespace: dict, functions=("main", "summary")) -> dict:
    """For each named function, the global names it reads that neither the
    module namespace nor builtins defines."""
    top = symtable.symtable(source, str(SCRIPT), "exec")
    out = {}
    for child in top.get_children():
        if child.get_name() not in functions:
            continue
        missing = sorted(s.get_name() for s in child.get_symbols()
                         if s.is_global() and s.is_referenced()
                         and s.get_name() not in namespace
                         and not hasattr(builtins, s.get_name()))
        out[child.get_name()] = missing
    return out


def test_main_and_summary_read_no_undefined_global():
    mod = _module()
    found = undefined_globals(SCRIPT.read_text(), vars(mod))
    assert set(found) == {"main", "summary"}, found
    assert found == {"main": [], "summary": []}


def test_the_check_notices_the_old_bug():
    mod = _module()
    src = SCRIPT.read_text()
    mutated = src.replace('    p50s = [r["copy_p50_us"] for r in rows]\n', "", 1)
    assert mutated != src
    assert undefined_globals(mutated, vars(mod))["summary"] == ["p50s"]
