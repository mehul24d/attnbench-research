"""Every site that runs a kernel is classified. CPU-only, AST-based.

Twice now, the same guard has been present where someone thought about it and
absent a few lines away:

  2026-09-03  `run_probe.py` filtered the BACKEND on `claims_support` after a
              dense kernel handed a block_sparse config ignored the mask,
              "succeeded", and produced `max_abs_err=4.81` across 114/126
              cells.
  2026-09-04  `check_cross_backend` called `forward` on REFERENCES that
              answered `(False, "no block sparse")`, producing five dense
              kernels that agreed with each other to six decimal places and
              disagreed with block_sparse by ~4.9. The same bug, on the other
              side of the same comparison, eleven months of reasoning apart.

The pattern is not "we forgot once". It is that the check lives at call sites
rather than in the type system, so its coverage is whatever anyone happened to
consider. This test replaces that with an enumeration: every call to a
kernel-running method in production code must appear below, classified either
GUARDED or UNGUARDED-with-a-reason. A new call site fails until someone
decides which it is.

Auditing a third gap this way found one immediately: `check_correctness` never
asked its float64 ORACLE whether it supported the config. Naive claims every
config the study uses, so nothing was wrong -- correctness by luck of one
backend's breadth, which is what this file exists to stop relying on.

Deliberately AST-based, not grep: a commented-out call, a call inside a
string, or a method named `forward` on something that is not a backend would
all fool a text search, and a guard that can be fooled by formatting is not a
guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Methods that actually put work on the device.
KERNEL_METHODS = {"forward", "run_once", "timed_call", "reference"}

# Files whose calls are dispatch/definition rather than use.
EXCLUDED = {
    REPO / "attnbench" / "backends" / "base.py",
    REPO / "attnbench" / "backends" / "impls.py",
    REPO / "attnbench" / "backends" / "linear.py",
    REPO / "attnbench" / "backends" / "block_sparse.py",
}

# (module, enclosing function, method) -> why this site is safe.
#
# GUARDED means a claims_support check gates this call, here or in the only
# caller. UNGUARDED means it deliberately runs a config the backend may
# decline, and the reason must say why that is correct.
CLASSIFIED: dict[tuple[str, str, str], str] = {
    # --- deliberately unguarded ---------------------------------------------
    ("attnbench/gates.py", "probe", "run_once"):
        "UNGUARDED BY DESIGN. Stage 0's entire purpose is the claim/actual "
        "split: it runs configs the backend declines in order to record where "
        "documentation and behaviour disagree. Guarding here would delete the "
        "finding. The one exception is a kernel that faults the DEVICE, which "
        "is refused earlier via FAULT_REASON_PREFIX -- an Xid 31 is not a "
        "disagreement to record, it kills the process.",

    ("attnbench/accuracy/model.py", "forward", "forward"):
        "UNGUARDED BY DESIGN. Stage 3 pairs each backend with its own curated "
        "configs rather than crossing all backends against all configs (see "
        "accuracy/runner.py's docstring), so the config reaching a backend "
        "here was chosen for it. claims_support would additionally match "
        "block_sparse against plain causal configs, creating cells this "
        "study's design excludes.",

    # --- guarded -------------------------------------------------------------
    ("attnbench/gates.py", "check_correctness", "reference"):
        "GUARDED. The oracle is asked before it is run -- added 2026-09-04 "
        "after this audit found it missing.",
    ("attnbench/gates.py", "check_correctness", "forward"):
        "GUARDED by the caller: run_probe.py's Stage 1 loop filters on "
        "b.claims_support(cfg) before dispatching.",
    ("attnbench/gates.py", "check_structural", "forward"):
        "GUARDED by the caller, as check_correctness. Linear backends are "
        "additionally only reached via the family dispatch.",
    ("attnbench/gates.py", "check_cross_backend", "forward"):
        "GUARDED. The backend under test by the caller; each reference by the "
        "claims_support filter added 2026-09-04.",
    ("attnbench/timing.py", "measure", "timed_call"):
        "GUARDED in-function: measure() returns Measurement(False, "
        "'unsupported') before allocating anything.",
}


def _call_sites() -> set[tuple[str, str, str]]:
    """(module, enclosing function, method) for every kernel call in
    production code."""
    found: set[tuple[str, str, str]] = set()
    for path in sorted((REPO / "attnbench").rglob("*.py")):
        if path in EXCLUDED:
            continue
        rel = str(path.relative_to(REPO))
        tree = ast.parse(path.read_text())

        # Attach the enclosing function to every node, so a site is keyed on
        # something stable across edits rather than on a line number.
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in KERNEL_METHODS):
                    found.add((rel, fn.name, node.func.attr))
    return found


def test_every_kernel_call_site_is_classified():
    """The check that makes the next call site a decision, not an oversight."""
    unclassified = sorted(_call_sites() - set(CLASSIFIED))
    assert not unclassified, (
        "kernel call site(s) not classified in this file:\n  "
        + "\n  ".join(f"{m} :: {fn}() :: .{meth}()"
                      for m, fn, meth in unclassified)
        + "\n\nEach must be recorded as GUARDED (a claims_support check gates "
          "it) or UNGUARDED with a reason. Running a config a backend declines "
          "gives a confident answer about a different function -- it has cost "
          "this project 114 cells once and 66 cells again.")


def test_the_classification_has_no_dead_entries():
    """A stale entry would silently exempt a site that no longer matches it."""
    stale = sorted(set(CLASSIFIED) - _call_sites())
    assert not stale, f"classified sites that no longer exist: {stale}"


@pytest.mark.parametrize("site,reason", sorted(CLASSIFIED.items()))
def test_each_site_states_which_it_is(site, reason):
    assert reason.startswith(("GUARDED", "UNGUARDED")), (
        f"{site} has a reason that does not begin GUARDED or UNGUARDED")


def test_the_audit_actually_finds_things():
    """A discovery routine that returns nothing would make every assertion
    above pass vacuously -- instance 5's shape, and the reason this file
    would otherwise be worthless."""
    sites = _call_sites()
    assert len(sites) >= 5, f"only found {len(sites)} call sites; parser broken?"
    assert ("attnbench/gates.py", "check_cross_backend", "forward") in sites


def test_the_audit_would_catch_a_new_unguarded_site(tmp_path):
    """Break it and watch it fail, without editing the real tree."""
    module = tmp_path / "sneaky.py"
    module.write_text("def go(b, q, k, v, cfg):\n    return b.forward(q, k, v, cfg)\n")
    tree = ast.parse(module.read_text())
    found = {
        (fn.name, node.func.attr)
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in KERNEL_METHODS
    }
    assert ("go", "forward") in found


def test_a_commented_out_call_is_not_counted():
    """Why this is AST-based rather than grep: text search cannot tell a call
    from a mention of one, and a guard that formatting can fool is not one."""
    tree = ast.parse("def go(b):\n    # b.forward(q, k, v, cfg)\n    s = 'b.forward('\n    return s\n")
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in KERNEL_METHODS]
    assert calls == []
