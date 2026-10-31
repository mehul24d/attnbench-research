"""Shared by the figure scripts: read a committed prediction the way the
analysis does (`predict.commit.load_committed`), never by hash alone."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Sequence

from attnbench.predict import commit as C


def banked_times(pairs: Optional[Sequence[str]]) -> dict:
    out = {}
    for p in pairs or ():
        name, _, iso = p.partition("=")
        if not name or not iso:
            raise ValueError(f"--banked-at takes PAIRING=ISO-TIME, got {p!r}")
        out[name] = datetime.fromisoformat(iso)
    return out


def load(remote: str, local: str, *, started_at: Sequence[float], clock_skew_s: float,
         banked_at: Optional[Sequence[str]], store=None) -> C.CommittedPrediction:
    if clock_skew_s < 0:
        raise ValueError("--clock-skew-s is >= 0")
    return C.load_committed(store or C.GsutilStore(), remote,
                            ground_truth_started_at=C.ground_truth_start(started_at),
                            clock_skew=timedelta(seconds=clock_skew_s),
                            banked_at=banked_times(banked_at), local=local)
