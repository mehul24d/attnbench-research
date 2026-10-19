#!/usr/bin/env python3
"""Where `torch.compile` with default settings splits the graph, on a CPU.

**What this banks.** The plan (Section 1 and T2.21) states that under
`torch.compile` with default settings an `assert` on a tensor, `.item()` and
`nonzero()` each break the graph, while a fixed-size `topk` does not. That was
checked on a CPU with PyTorch 2.8 on 6 Oct 2026 and never banked. This script
reproduces the check so the statement rests on a file.

It uses PyTorch only. No estimator's code is imported or copied: each pattern
is a short function written here, with work on both sides of the pattern so
that a split leaves two graphs.

**`assert` is checked in four forms**, because whether it splits depends on
the form. Dynamo rewrites `assert <tensor>` into `torch._assert` by default
(`rewrite_assert_with_torch_assert`), which stays in the graph; an assert
whose condition first turns the tensor into a Python bool (`not`, `bool()`)
cannot be rewritten that way. Which form a given library uses is not
something this script knows.

**How a split is counted, two ways, which must agree:**

  - `torch.compile(fn)` with every setting at its default (the inductor
    backend), then Dynamo's own counters: the number of graphs compiled and
    the graph-break reasons;
  - `torch._dynamo.explain(fn)`: graph count, break count and reasons.

The compiled output is also compared with eager output. As a third record,
not a vote, `torch.compile(fn, fullgraph=True, backend="eager")` is run and
whether it raises is kept: `fullgraph=True` is not a default setting, and
later PyTorch versions trace some patterns under it that they split by
default.

**What would falsify the plan's statement:** any of patterns (a) to (c)
compiling to one graph, or pattern (d) compiling to more than one. The script
records the outcome; it does not fail on a disagreement.

Usage (from the repo root, CPU only):
    python3 scripts/check_compile_graph_breaks.py --out results/compile_graph_breaks_cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import provenance                                  # noqa: E402


def assert_tensor(x):
    y = x * 2
    assert (y >= 0).all()
    return y + 1


def assert_tensor_with_message(x):
    y = x * 2
    assert (y >= 0).all(), "negative value"
    return y + 1


def assert_not_tensor(x):
    y = x * 2
    assert not (y < 0).any()
    return y + 1


def assert_bool_of_tensor(x):
    y = x * 2
    assert bool((y >= 0).all())
    return y + 1


def item(x):
    y = x * 2
    s = y.sum().item()
    return y + s


def nonzero(x):
    y = x * 2
    idx = (y > 1).nonzero()
    return y.sum() + idx.shape[0]


def fixed_size_topk(x):
    y = x * 2
    vals, idx = torch.topk(y, k=4, dim=-1)
    return y.gather(-1, idx) + vals


# (key, function, the plan's statement: does it split?)
PATTERNS = (
    ("a1_assert_tensor", assert_tensor, True),
    ("a2_assert_tensor_with_message", assert_tensor_with_message, True),
    ("a3_assert_not_tensor", assert_not_tensor, True),
    ("a4_assert_bool_of_tensor", assert_bool_of_tensor, True),
    ("b_item", item, True),
    ("c_nonzero", nonzero, True),
    ("d_fixed_size_topk", fixed_size_topk, False),
)


def _first_line(e: BaseException) -> str:
    text = str(e).strip().splitlines()
    return f"{type(e).__name__}: {text[0] if text else ''}"


def _default_compile(fn, x):
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    counters.clear()
    try:
        out = torch.compile(fn)(x)
    except Exception as e:                       # recorded, not raised
        return {"error": _first_line(e)}
    return {
        "graphs": int(counters["stats"]["unique_graphs"]),
        "break_reasons": sorted(counters["graph_break"].keys()),
        "matches_eager": bool(torch.allclose(out, fn(x))),
    }


def _explain(fn, x):
    torch._dynamo.reset()
    try:
        e = torch._dynamo.explain(fn)(x)
    except Exception as err:
        return {"error": _first_line(err)}
    return {
        "graphs": int(e.graph_count),
        "breaks": int(e.graph_break_count),
        "break_reasons": [str(r.reason).strip().splitlines()[0]
                          for r in e.break_reasons],
    }


def _fullgraph(fn, x):
    torch._dynamo.reset()
    try:
        torch.compile(fn, fullgraph=True, backend="eager")(x)
    except Exception as e:
        return {"raised": True, "error": _first_line(e)}
    return {"raised": False}


def check(fn, x) -> dict:
    d = _default_compile(fn, x)
    ex = _explain(fn, x)
    fg = _fullgraph(fn, x)
    ok = "error" not in d and "error" not in ex
    votes = {d.get("graphs", 0) > 1, ex.get("breaks", 0) > 0}
    agree = ok and len(votes) == 1
    return {
        "default_compile": d, "explain": ex, "fullgraph_eager": fg,
        "split": votes.pop() if agree else None,
        "methods_agree": agree,
    }


def dynamo_defaults() -> dict:
    cfg = torch._dynamo.config
    names = ("capture_scalar_outputs", "capture_dynamic_output_shape_ops",
             "rewrite_assert_with_torch_assert")
    return {n: getattr(cfg, n) for n in names if hasattr(cfg, n)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--host-label", default=None,
                    help="store this in the stamp's host field instead of "
                         "the machine's network name")
    args = ap.parse_args()

    torch.manual_seed(0)
    x = torch.rand(4, 64)
    results = {}
    for key, fn, expected in PATTERNS:
        r = check(fn, x)
        r["plan_says_split"] = expected
        r["agrees_with_plan"] = r["split"] is expected
        results[key] = r
        print(f"{key:30s} split={r['split']!s:5s} "
              f"methods_agree={r['methods_agree']!s:5s} "
              f"agrees_with_plan={r['agrees_with_plan']}", flush=True)

    stamp = provenance.capture().to_dict()
    if args.host_label:
        stamp["host"] = args.host_label
    stamp["host_is_label"] = bool(args.host_label)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"torch-{torch.__version__.split('+')[0]}_cpu.json"
    path.write_text(json.dumps({
        "provenance": stamp,
        "device": "cpu",
        "input_shape": list(x.shape),
        "dynamo_defaults": dynamo_defaults(),
        "patterns": results,
        "all_agree_with_plan": all(r["agrees_with_plan"]
                                   for r in results.values()),
    }, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
