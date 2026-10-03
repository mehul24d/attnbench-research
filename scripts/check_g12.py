#!/usr/bin/env python
"""Gate G12 of the estimator-frontier pre-registration (sec. 6), run by hand.

    check_g12.py --text-json PATH/TO/x-attention/.../text.json

Prints every held-out `qa_1` index whose question or gold document occurs in
the calibration texts, by split, and exits 1 if there is any. It reads the
authors' file where it is installed and writes none of its text anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from attnbench.accuracy import frontier_splits  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--text-json", type=Path, required=True)
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("cal", REPO / "scripts" / "calibrate_xattn_thresholds.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    raw = a.text_json.read_bytes()
    texts = [cal.strip_chat_markers(t) for t in json.loads(raw)]
    print(f"{len(texts)} texts, sha256 {hashlib.sha256(raw).hexdigest()}")
    hits = frontier_splits.g12_hits(texts)
    for split in frontier_splits.G12_SPLITS:
        n = len(frontier_splits.split_indices(split, "qa_1"))
        mine = [h for h in hits if h["split"] == split]
        q = [h["index"] for h in mine if h["question_in_texts"]]
        d = [h["index"] for h in mine if h["gold_document_in_texts"]]
        print(f"{split:15s} qa_1 n={n:3d}: question in a text {len(q):3d} {q[:6]}; "
              f"gold document in a text {len(d):3d}")
    print("G12:", "STOP" if hits else "pass")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
