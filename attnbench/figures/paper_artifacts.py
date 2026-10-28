"""Writes the paper's tables, figures and numbers from checked inputs (C10).

Into an output directory that must be new or empty (an earlier run's
artifacts are never mixed with this run's, nor overwritten):

  tables/cells.tex, predictors.tex, ablations.tex, queue.tex,
         residuals_sync.tex, decision.tex, crossover.tex (when supplied)
  figures/error_vs_length.pdf, decision_error.pdf, crossover.pdf (when supplied)
  numbers.json, numbers.tex
  manifest.json: every output's sha256, and every input's

The run is stamped with the commit it ran at, which must be clean (rule 3).
"""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from typing import Union

from ..predict.commit import CodeStamp
from . import paper_figures as PF
from . import paper_numbers as PN
from . import paper_tables as PT
from .paper_inputs import Inputs, PaperInputError
from .results import sha256_file

MANIFEST_SCHEMA = "attnbench.paper_artifacts/1"


def _iso(t) -> str:
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _write(path: Path, text: str) -> Path:
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
    return path


def build(inp: Inputs, out_dir: Union[str, Path], code: CodeStamp) -> dict:
    if not isinstance(code, CodeStamp) or code.git_dirty is not False \
            or len(code.git_commit or "") != 40:
        raise PaperInputError(f"paper artifacts are made at a clean, known commit, "
                              f"got {code!r}")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise PaperInputError(f"{out} is not empty; artifacts from an earlier run are "
                              f"never mixed with or overwritten by this one")
    # Compute everything before writing anything, so a refusal leaves no partial set.
    decision = PT.decision(inp)
    crossover = PT.crossover(inp)
    tables = {"cells": PT.tex_cells(inp), "predictors": PT.tex_predictors(inp),
              "ablations": PT.tex_ablations(inp), "queue": PT.tex_queue(inp),
              "residuals_sync": PT.tex_sync(inp), "decision": PT.tex_decision(inp, decision)}
    if crossover:
        tables["crossover"] = PT.tex_crossover(inp, crossover)
    numbers = PN.collect(inp, decision, crossover)
    cp = inp.cp
    header = {
        "schema": MANIFEST_SCHEMA,
        "generated_by": {"git_commit": code.git_commit, "git_dirty": code.git_dirty},
        "pairing": cp.pairing, "statistic": cp.statistic,
        "prediction": {"sha256": cp.sha256, "written_at": _iso(cp.written_at),
                       "synced_at": _iso(cp.synced_at),
                       "code": {"git_commit": cp.code.git_commit,
                                "git_dirty": cp.code.git_dirty}},
        "floors": {"relative": inp.relative_floor, "resolution_s": inp.resolution_floor_s,
                   "provisional": inp.floor_provisional},
        "predictors": list(inp.predictors),
        "builder_kinds": {a: {"kind": inp.builder(a),
                              "from": list((inp.builder_sources or {}).get(a, ()))}
                          for a in sorted(inp.classes) if a in {c.arm for c in cp.cells}},
        "crossover_intervals_from": None if inp.crossover is None
        else inp.crossover["source"],
        "crossover": ("computed" if crossover else
                      "not computed: the spec supplies no predicted difference intervals"
                      if inp.crossover is None else "no series with two or more lengths"),
        "inputs": [dict(f) for f in inp.files],
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir()
    (out / "figures").mkdir()
    written = [_write(out / "tables" / f"{n}.tex", t) for n, t in tables.items()]
    written.append(PF.error_vs_length(inp, out / "figures" / "error_vs_length.pdf"))
    written.append(PF.decision_error(inp, decision, out / "figures" / "decision_error.pdf"))
    if crossover:
        written.append(PF.crossover(inp, crossover, out / "figures" / "crossover.pdf"))
    written.append(_write(out / "numbers.json",
                          json.dumps(PN.as_json(numbers, extra=header), indent=1,
                                     sort_keys=True) + "\n"))
    written.append(_write(out / "numbers.tex", numbers.tex()))
    manifest = {**header, "outputs": {str(p.relative_to(out)): sha256_file(p)
                                      for p in written}}
    _write(out / "manifest.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest
