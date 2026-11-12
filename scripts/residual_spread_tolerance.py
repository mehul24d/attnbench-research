#!/usr/bin/env python3
"""T4.1's residual spread tolerance (row 10 of the 14-value table; the
second lock-wave rulings of 12 Nov 2026, item 2), derived from P1 before any
P2 data: from T3.2d alone, 0.10 if every layer's spread is inside it, else
1.5 times the largest. No layer and no cell is excluded.

A layer's spread is `layer_residual`'s: the interquartile range of its runs
over their median (`components.layer_residual`). The source is T3.2d's step
5, one `layer_residual.json` per cell (`05_host_components/<cell>/`), on the
current method (schema 3, `idle_device_gaps_bare_host`). T3.2c measured the
superseded method (schema 2) and is read as a sensitivity only: its largest
spread is reported and decides nothing.

    python3 scripts/residual_spread_tolerance.py \\
        --source <T3.2d>/05_host_components --sensitivity <T3.2c>/05_host_components
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WORKING = 0.10
FACTOR = 1.5
SOURCE_SCHEMA = "attnbench.layer_residual/3"


def largest_spread(host_dir: Path, *, schema=None) -> dict:
    """The largest layer spread over every cell under `host_dir`, with its
    cell and layer, the cell count and a digest of the files read. Refuses
    an empty directory, a cell with no layers and, when `schema` is given,
    a record on another schema."""
    files = sorted(Path(host_dir).glob("*/layer_residual.json"))
    if not files:
        raise SystemExit(f"no layer_residual.json under {host_dir}")
    top, digest = None, hashlib.sha256()
    for f in files:
        raw = f.read_bytes()
        digest.update(f.parent.name.encode() + b"\0" + hashlib.sha256(raw).digest())
        rec = json.loads(raw)
        if schema is not None and rec.get("schema") != schema:
            raise SystemExit(f"{f}: schema {rec.get('schema')!r}, not {schema}")
        if not rec.get("layers"):
            raise SystemExit(f"{f}: no layers")
        for layer in rec["layers"]:
            if top is None or layer["spread"] > top["spread"]:
                top = {"spread": layer["spread"], "cell": f.parent.name, "layer": layer["layer"]}
    return {**top, "n_cells": len(files), "files_sha256": digest.hexdigest()}


def tolerance(spread: float) -> float:
    """The working 0.10 when the largest spread is inside it, else 1.5 times
    the largest."""
    return WORKING if spread <= WORKING else FACTOR * spread


def derive(source: Path, sensitivity=None) -> dict:
    src = largest_spread(source, schema=SOURCE_SCHEMA)
    out = {"tolerance": tolerance(src["spread"]), "source": src}
    if sensitivity is not None:
        out["sensitivity"] = largest_spread(sensitivity)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--sensitivity", type=Path)
    args = ap.parse_args(argv)
    print(json.dumps(derive(args.source, args.sensitivity), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
