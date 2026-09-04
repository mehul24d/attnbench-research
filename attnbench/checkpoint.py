"""Shared parquet checkpoint-append helper.

One append implementation for every stage that writes incremental results
(Stage 2's `sweep.py`, Stage 3's `accuracy/runner.py`) -- two independent
copies of "read existing, concat, overwrite" is exactly the kind of drift
that produces a subtly different resume behavior in one stage but not the
other.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def append_checkpoint(path: Path, rows: list[dict]) -> None:
    """Append `rows` to the parquet file at `path`, creating it if absent.

    Written to a temporary file in the same directory and renamed into place.
    `rename` within a filesystem is atomic, so a crash or a hard shutdown
    during the write leaves either the previous checkpoint or the new one --
    never a truncated file.

    That matters more than it looks: this function exists precisely so that
    work survives a crash, and read-concat-overwrite means every append
    rewrites the WHOLE file. A crash during the final write of a long run
    would destroy every row already banked, which is the exact failure the
    checkpoint is supposed to prevent. Instances in this project have become
    unreachable mid-work three times.
    """
    new_df = pd.DataFrame(rows)
    if path.exists():
        combined = pd.concat([pd.read_parquet(path), new_df], ignore_index=True)
    else:
        combined = new_df

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    combined.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def done_keys(path: Path, *columns: str) -> set[tuple]:
    """Existing `columns` tuples in a checkpoint, for skip-on-resume.

    Returns an empty set for a missing file, and also for an unreadable one:
    a checkpoint that cannot be parsed must mean "redo the work", never
    "assume it was all done".
    """
    if not path.exists():
        return set()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return set()
    if not all(c in df.columns for c in columns):
        return set()
    return set(zip(*(df[c] for c in columns)))
