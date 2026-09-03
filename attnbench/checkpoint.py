"""Shared parquet checkpoint-append helper.

One append implementation for every stage that writes incremental results
(Stage 2's `sweep.py`, Stage 3's `accuracy/runner.py`) -- two independent
copies of "read existing, concat, overwrite" is exactly the kind of drift
that produces a subtly different resume behavior in one stage but not the
other.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def append_checkpoint(path: Path, rows: list[dict]) -> None:
    """Append `rows` to the parquet file at `path`, creating it if absent."""
    new_df = pd.DataFrame(rows)
    if path.exists():
        combined = pd.concat([pd.read_parquet(path), new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path, index=False)
