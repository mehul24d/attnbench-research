"""EXPLORATORY: a lower bound on a sparse cell's end-to-end prefill time
(item 4 of the 16 Nov 2026 rulings), fitted on P1 alone.

    T_lb = max(F / (phi_c * P_peak), Bytes / (phi_m * BW_peak), n_ops * tau)

F and Bytes are the compulsory FLOPs and bytes of one prefill, counted from
the model's shape by `predict.intensity.flops_and_bytes` (every layer's
classes, plus the last-position head; estimator scoring is not counted, so
the bound stays a lower bound). P_peak and BW_peak are each GPU's
spec-sheet dense 16-bit tensor peak and memory bandwidth. n_ops is the
number of operations in the cell's priced sequence (launches, copies and
synchronisations), and tau a host cost per operation.

The fit is the envelope over P1's sparse cells: phi_c and phi_m are the
largest fractions of peak any P1 cell would have needed, and tau the
smallest time per operation any P1 cell took, so every P1 cell sits at or
above each term and at least one cell touches each. On P2 and P4 a cell
"breaks" the bound when measured below it, and "comes close" when measured
within `CLOSE` of it. A break means a term did not transfer: the GPU
reached a larger fraction of its peak than the L4 did, or the host issues
operations faster.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from attnbench.predict import intensity as IN

# Spec-sheet dense (no structured sparsity) BF16 tensor peak, FLOP/s, and
# memory bandwidth, B/s.
PEAKS = {
    "NVIDIA L4": (121e12, 300e9),
    "NVIDIA A100-SXM4-80GB": (312e12, 2039e9),
    "NVIDIA H100 80GB HBM3": (989e12, 3350e9),
}
CLOSE = 0.10


def layers(record=IN.PRIMARY_SHAPE) -> int:
    return int(json.loads(Path(record).read_text())["layers"])


def work(shape: IN.ModelShape, n_layers: int, context_length: int, batch: int,
         density: float) -> tuple[float, float]:
    """(FLOPs, bytes) of one prefill."""
    fb = IN.flops_and_bytes(shape, context_length, batch, density)
    f = sum(v[0] for k, v in fb.items() if k != "lm_head") * n_layers + fb["lm_head"][0]
    b = sum(v[1] for k, v in fb.items() if k != "lm_head") * n_layers + fb["lm_head"][1]
    return float(f), float(b)


@dataclass(frozen=True)
class Obs:
    gpu: str
    flops: float
    bytes: float
    n_ops: int
    measured: float


@dataclass(frozen=True)
class Bound:
    phi_c: float
    phi_m: float
    tau: float

    def terms(self, o: Obs) -> dict[str, float]:
        p, bw = PEAKS[o.gpu]
        return {"compute": o.flops / (self.phi_c * p), "memory": o.bytes / (self.phi_m * bw),
                "host": o.n_ops * self.tau}

    def at(self, o: Obs) -> float:
        return max(self.terms(o).values())


def fit(train: Sequence[Obs]) -> Bound:
    """The envelope over `train` (P1's sparse cells)."""
    if not train:
        raise ValueError("no training cell")
    for o in train:
        if o.measured <= 0 or o.n_ops < 1 or o.gpu not in PEAKS:
            raise ValueError(f"unusable training cell {o}")
    return Bound(phi_c=max(o.flops / (o.measured * PEAKS[o.gpu][0]) for o in train),
                 phi_m=max(o.bytes / (o.measured * PEAKS[o.gpu][1]) for o in train),
                 tau=min(o.measured / o.n_ops for o in train))


def check(b: Bound, cells: Mapping[str, Obs], close: float = CLOSE) -> dict:
    """name -> measured / bound, the binding term, and break / close / clear."""
    out = {}
    for name, o in cells.items():
        t = b.terms(o)
        lb = max(t.values())
        r = o.measured / lb
        out[name] = {"ratio": r, "binding": max(t, key=t.get), "bound_s": lb,
                     "status": "break" if r < 1 else ("close" if r < 1 + close else "clear")}
    return out
