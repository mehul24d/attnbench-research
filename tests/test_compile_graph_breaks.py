"""The banked CPU compile check agrees with its script and with its README.

**Why this file exists.** `results/compile_graph_breaks_cpu/` backs a
sentence in `README.md`, in plan Section 1 and in the brief: under
`torch.compile` with default settings, `.item()`, `nonzero()` and some forms
of `assert` split the graph, while a plain `assert` on a tensor and a
fixed-size `topk` do not. The claim it replaced, "an assert on a tensor
breaks the graph", was stated for two weeks with no file behind it. So the
files are the test: the README's pattern table is read back against them,
every pattern in the script must be banked for both PyTorch versions, and two
runs of one version must agree wherever they overlap.

The files are force-committed, so this runs in a fresh clone too. The last
two tests are anti-vacuity checks: one runs the script's counter live on a
pattern that must split and one that must not, the other flips a cell in the
README table and requires the comparison to notice.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DIR = REPO / "results" / "compile_graph_breaks_cpu"
README = DIR / "README.md"
SCRIPT = REPO / "scripts" / "check_compile_graph_breaks.py"
VERSIONS = ("2.8.0", "2.14.1")
NAME_RE = re.compile(r"^torch-(?P<ver>[0-9.]+)_cpu(?:_(?P<commit>[0-9a-f]{7}))?\.json$")
ROW_RE = re.compile(r"^\| \((?P<label>[a-z][0-9]?)\) .*\| (?P<v1>split|no split) \| (?P<v2>split|no split) \|$")


def _script():
    spec = importlib.util.spec_from_file_location("check_compile_graph_breaks", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _banked() -> dict[str, dict]:
    return {p.name: json.loads(p.read_text()) for p in sorted(DIR.glob("*.json"))}


def _readme_disagreements(text: str, banked: dict[str, dict]) -> list[str]:
    """Rows of the README table that do not match every banked file."""
    rows = {}
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            rows[m["label"]] = {VERSIONS[0]: m["v1"] == "split",
                                VERSIONS[1]: m["v2"] == "split"}
    problems = []
    keys = {k for d in banked.values() for k in d["patterns"]}
    for key in sorted(keys):
        label = key.split("_", 1)[0]
        if label not in rows:
            problems.append(f"{key}: no row in the README table")
            continue
        for name, d in banked.items():
            ver = d["provenance"]["torch"].split("+")[0]
            if key in d["patterns"] and d["patterns"][key]["split"] != rows[label][ver]:
                problems.append(f"{key}: README says split={rows[label][ver]} "
                                f"for {ver}, {name} says {d['patterns'][key]['split']}")
    return problems


def test_there_are_banked_files_for_both_versions():
    vers = {d["provenance"]["torch"].split("+")[0] for d in _banked().values()}
    assert vers == set(VERSIONS), vers


def test_every_banked_file_is_stamped_clean_on_a_cpu():
    for name, d in _banked().items():
        p = d["provenance"]
        m = NAME_RE.match(name)
        assert m, f"{name}: unexpected file name"
        assert m["ver"] == p["torch"].split("+")[0], name
        assert re.fullmatch(r"[0-9a-f]{40}", p["git_commit"] or ""), name
        if m["commit"]:
            assert p["git_commit"].startswith(m["commit"]), name
        assert p["git_dirty"] is False, name
        assert d["device"] == "cpu" and p["gpu_count"] == 0, name
        assert p["host_is_label"] is True, f"{name}: host is a machine name"


def test_both_counting_methods_agree_and_compiled_output_matches_eager():
    for name, d in _banked().items():
        for key, r in d["patterns"].items():
            assert r["methods_agree"] is True, f"{name} {key}"
            assert r["split"] in (True, False), f"{name} {key}"
            assert r["default_compile"]["matches_eager"] is True, f"{name} {key}"


def test_every_pattern_in_the_script_is_banked_for_both_versions():
    keys = {k for k, _, _ in _script().PATTERNS}
    for ver in VERSIONS:
        full = [n for n, d in _banked().items()
                if d["provenance"]["torch"].startswith(ver)
                and set(d["patterns"]) == keys]
        assert full, f"no banked file for {ver} covers all of {sorted(keys)}"


def test_runs_of_one_version_agree_wherever_they_overlap():
    by_ver: dict[str, list[dict]] = {}
    for d in _banked().values():
        by_ver.setdefault(d["provenance"]["torch"], []).append(d["patterns"])
    for ver, runs in by_ver.items():
        for a in runs:
            for b in runs:
                for k in set(a) & set(b):
                    assert a[k]["split"] == b[k]["split"], f"{ver} {k}"


def test_the_two_versions_agree_on_every_pattern():
    """The README's bold sentence; also what lets the plan cite one result."""
    split: dict[str, set] = {}
    seen: dict[str, set] = {}
    for d in _banked().values():
        ver = d["provenance"]["torch"]
        for k, r in d["patterns"].items():
            split.setdefault(k, set()).add(r["split"])
            seen.setdefault(k, set()).add(ver)
    for k in split:
        assert len(split[k]) == 1, f"{k}: files disagree, {split[k]}"
        assert seen[k] == set(VERSIONS), f"{k}: banked only for {seen[k]}"


def test_the_readme_table_matches_the_banked_files():
    problems = _readme_disagreements(README.read_text(), _banked())
    assert not problems, "\n".join(problems)


def test_the_readme_comparison_notices_a_flipped_cell():
    text = README.read_text()
    flipped = re.sub(r"^(\| \(e\) .*\| )no split( \| no split \|)$",
                     r"\1split\2", text, count=1, flags=re.M)
    assert flipped != text, "the (e) row was not found; the mutation is vacuous"
    assert _readme_disagreements(flipped, _banked())


def test_the_counter_tells_a_split_from_no_split_live():
    torch = pytest.importorskip("torch")
    mod = _script()
    x = torch.rand(4, 64)
    assert mod._explain(mod.item, x)["breaks"] > 0
    assert mod._explain(mod.fixed_size_topk, x)["breaks"] == 0
