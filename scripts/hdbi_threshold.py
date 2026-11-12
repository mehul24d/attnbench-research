#!/usr/bin/env python3
"""T4.1's HDBI boundedness threshold, fitted at the lock (the researcher's
item 18, lock-wave rulings of 12 Nov 2026): on T3.2d, each row dropped by its
own cell's floor (`lock.hdbi_fit.fit_threshold`).

T3.2d is the lock training file's repeat a: its rows carry T3.2d's
components, and the `repeat_a_only` set holds T3.2d's own medians. A row is a
sparse cell: its HDBI is that of its own components (`predict.hdbi.hdbi`),
and its measured difference is its median minus its dense partner's
(`training.partner_of`: same model, length and batch). The floors are the
lock's table (`FloorTable.from_lock`).

    python3 scripts/hdbi_threshold.py --training results/p1_training/p1_training.json.gz \\
        --lock docs/lock_values.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from attnbench.lock.floors import FloorTable                # noqa: E402
from attnbench.lock.hdbi_fit import HdbiRow, fit_threshold  # noqa: E402
from attnbench.predict import p1_training as PT             # noqa: E402
from attnbench.predict.hdbi import hdbi                     # noqa: E402
from attnbench.predict.training import partner_of           # noqa: E402

DENSE_ARM = "dense"
FIT_SET = PT.REPEAT_A_ONLY       # T3.2d's own targets
FIT_REPEAT = "T3.2d"


def hdbi_rows(rows) -> list:
    """One `HdbiRow` per sparse cell, against the one dense cell at its
    partner point; a point without exactly one dense cell is refused."""
    dense = {}
    for r in rows:
        if r.cell.arm == DENSE_ARM:
            pt = partner_of(r.cell.key)
            if pt in dense:
                raise SystemExit(f"two dense cells at {pt}")
            dense[pt] = r
    out = []
    for r in rows:
        if r.cell.arm == DENSE_ARM:
            continue
        pt = partner_of(r.cell.key)
        if pt not in dense:
            raise SystemExit(f"no dense cell at {pt} for {tuple(r.cell.key)}")
        out.append(HdbiRow(pairing=r.cell.pairing, cell=tuple(r.cell.key),
                           hdbi=hdbi(r.cell.priced),
                           measured_difference=r.measured - dense[pt].measured,
                           context_length=r.cell.context_length, batch=r.cell.batch))
    return out


def fit(rec, lock) -> dict:
    if rec.get("features_from") != FIT_REPEAT:
        raise SystemExit(f"the training file's repeat a is {rec.get('features_from')!r}, "
                         f"not {FIT_REPEAT}")
    rows = PT.rows_from_record(rec, sensitivity=FIT_SET)
    f = fit_threshold(hdbi_rows(rows), training_pairings=[PT.PAIRING],
                      floors=FloorTable.from_lock(lock))
    return {"threshold": f.threshold, "agreements": f.agreements, "n_scored": f.n_scored,
            "n_inside_floor": f.n_inside_floor, "agreement_rate": f.agreement_rate,
            "pairings": list(f.pairings), "curve": [list(x) for x in f.curve],
            "training_file_sha256": rec["sha256"], "fit_set": FIT_SET,
            "repeat": FIT_REPEAT}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--training", type=Path, required=True)
    ap.add_argument("--lock", type=Path, required=True)
    args = ap.parse_args(argv)
    out = fit(PT.load(args.training), json.loads(args.lock.read_text(encoding="utf-8")))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
