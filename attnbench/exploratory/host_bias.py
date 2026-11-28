"""EXPLORATORY / POST HOC: how far the host prices overstate host time (the
researcher's paper-quality wave of 28 Nov 2026, item 10). Nothing here
scores, rescores or replaces a pre-registered result, no host price is
changed, and BUNGA is not refitted. No new measurement: every input is a
banked session file.

Two instruments price host cost in the banked sessions, and they are kept
apart here:

  - the *null pass* (`host_profile`, `host_profile_null_pass.json`): the
    prefill with every launching call replaced by one null kernel, under the
    tape's and the null device's Python modes. This is the null-kernel
    substitution adapted from TaxBreak;
  - the *committed price* (`host_profile.json`, pricing `bare_timer`): each
    call bare-timed alone (`bare_timer`), plus the per-layer idle-gap residual
    (`layer_residual`), plus a few positions priced another way (the
    isolated replay). This is what P2's and P4's committed predictions read.

So "null-kernel substitution overprices" is measured two ways: the null pass
against the bare timer per operation class (`class_sums`, both instruments on
the same calls), and the committed price against the measured untimed
prefill per cell (`lower_bound`). The second is a lower bound only: the host
issues every operation before the timer stops, so the measured time is at
least the true host time, and priced / measured understates priced / true.
"""

from __future__ import annotations

import bisect
from typing import Mapping, Optional, Sequence

import numpy as np

LABEL = "EXPLORATORY / POST HOC"

# Operation classes (plan default of this wave, flagged in the write-up).
CLASSES = {
    "attention": ("aten::_scaled_dot_product_flash_attention", "aten::_softmax", "aten::bmm",
                  "flash_attn::_block_sparse_attn_forward"),
    "gemm": ("aten::mm", "aten::addmm"),
    "copy": ("aten::_to_copy", "aten::copy_", "aten::clone", "aten::constant_pad_nd"),
    "elementwise": ("aten::mul", "aten::add", "aten::div", "aten::neg", "aten::pow",
                    "aten::rsqrt", "aten::silu", "aten::cos", "aten::sin", "aten::round"),
    "reduction": ("aten::mean", "aten::sum", "aten::cumsum", "aten::sort", "aten::all"),
    "mask_and_index": ("aten::eq", "aten::lt", "aten::gt", "aten::ne", "aten::where",
                       "aten::masked_fill", "aten::masked_scatter", "aten::bitwise_and",
                       "aten::bitwise_not", "aten::index", "aten::index_put_", "aten::scatter_",
                       "aten::flip", "aten::tril", "aten::triu", "aten::cat"),
    "fill_and_create": ("aten::fill_", "aten::zeros", "aten::zeros_like", "aten::ones",
                        "aten::eye", "aten::arange", "aten::scalar_tensor"),
}
_OF = {op: c for c, ops in CLASSES.items() for op in ops}


def op_class(operator: Optional[str]) -> str:
    """The class of one priced call. A call with no aten operator is a Python
    entry point (a Triton kernel called from the arm)."""
    if operator is None:
        return "entry"
    return _OF.get(operator, "other")


def host_parts(profile: Mapping) -> dict:
    """One committed `host_profile.json` split by route, in seconds: `bare`
    (each call's bare time), `residual` (the per-layer idle gaps added back),
    `other` (positions priced another way) and `total`."""
    host, res = profile["host_s"], profile.get("residual_s", {})
    route = profile.get("price_route", {})
    total = float(sum(host.values()))
    residual = float(sum(res.values()))
    bare = float(sum(h - res.get(k, 0.0) for k, h in host.items()
                     if str(route.get(k, "")).startswith("bare")))
    return {"total": total, "bare": bare, "residual": residual,
            "other": total - bare - residual}


def null_total(profile: Mapping) -> float:
    """Total host seconds of one null-pass profile."""
    if profile.get("pricing") not in (None, "null_pass"):
        raise ValueError(f"not a null-pass profile: {profile.get('pricing')}")
    return float(sum(profile["host_s"].values()))


def position_classes(bare_prices: Mapping, priced: Sequence[int]) -> dict:
    """Position -> operation class. A call prices `positions` host positions
    starting at `position`, the next ones in `priced` (the profile's own
    positions: a synchronisation inside a call has no host price, so a call's
    positions need not be contiguous). Positions no call prices (the
    embedding, by the isolated replay) are left out."""
    keys = sorted(int(k) for k in priced)
    out = {}
    for c in sorted(bare_prices["calls"], key=lambda c: c["position"]):
        start, n = int(c["position"]), int(c["positions"])
        i = bisect.bisect_left(keys, start)
        if i + n > len(keys):
            raise ValueError(f"call at {start} prices positions the profile has not got")
        for k in keys[i:i + n]:
            if k in out:
                raise ValueError(f"position {k} priced by two calls")
            out[k] = op_class(c["operator"])
    return out


def class_sums(bare_prices: Mapping, null_profile: Mapping, committed: Mapping) -> dict:
    """Per operation class over one cell: the null pass's seconds and the
    committed price's seconds at the same positions, and the calls.

    Like for like: a null-pass position holds the call's gap (the arm's Python
    since the previous call) and its launch; the committed position holds the
    bare price plus its share of the layer's idle-gap residual, which is what
    adds that Python back. The residual is shared out in proportion to the
    null pass's gaps, so the class split of the residual follows the null
    pass by construction; the bare part does not."""
    cls = position_classes(bare_prices, committed["host_s"])
    out: dict = {}
    for k, c in cls.items():
        s = out.setdefault(c, {"null_s": 0.0, "committed_s": 0.0, "positions": 0})
        s["null_s"] += float(null_profile["host_s"][str(k)])
        s["committed_s"] += float(committed["host_s"][str(k)])
        s["positions"] += 1
    return out


def ratio_of_sums(pairs: Sequence[tuple[float, float]], *, n_boot: int, seed: int,
                  level: float) -> dict:
    """sum(num) / sum(den) over clusters (cells or layers), with a percentile
    bootstrap interval that resamples whole clusters."""
    a = np.asarray(pairs, dtype=float)
    if a.ndim != 2 or len(a) == 0 or a[:, 1].sum() <= 0:
        raise ValueError("need at least one cluster with a positive denominator")
    point = a[:, 0].sum() / a[:, 1].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    num, den = a[idx, 0].sum(axis=1), a[idx, 1].sum(axis=1)
    boots = num[den > 0] / den[den > 0]
    q = (1 - level) / 2
    return {"ratio": float(point), "lo": float(np.quantile(boots, q)),
            "hi": float(np.quantile(boots, 1 - q)), "clusters": int(len(a))}


def lower_bound(host_s: float, measured_s: float, lo_s: float, hi_s: float) -> dict:
    """priced host / measured prefill, a lower bound on priced / true host
    time, with the interval the measured median's interval gives it."""
    if min(measured_s, lo_s, hi_s) <= 0 or lo_s > hi_s:
        raise ValueError("measured times must be positive with lo <= hi")
    return {"ratio": host_s / measured_s, "lo": host_s / hi_s, "hi": host_s / lo_s}


def layer_spans(trace: Mapping, prefix: str = "attnbench::layer.") -> dict:
    """Host seconds of each layer annotation in a Kineto trace, by layer."""
    out = {}
    for e in trace["traceEvents"]:
        n = e.get("name", "")
        if e.get("cat") == "user_annotation" and n.startswith(prefix):
            layer = int(n[len(prefix):])
            if layer in out:
                raise ValueError(f"layer {layer} annotated twice")
            out[layer] = float(e["dur"]) * 1e-6
    return out


def null_by_layer(profile: Mapping, sequence: Mapping) -> dict:
    """Null-pass host seconds summed per layer (positions index `ops`)."""
    out: dict = {}
    for k, h in profile["host_s"].items():
        layer = sequence["ops"][int(k)]["layer"]
        if layer is not None:
            out[layer] = out.get(layer, 0.0) + float(h)
    return out
