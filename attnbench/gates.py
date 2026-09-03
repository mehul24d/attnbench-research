"""Stage 0 (capability matrix) and Stage 1 (correctness gate).

Both run before any timing. Stage 0 tells us which cells exist; Stage 1 tells us
which of those cells produce trustworthy output. A backend that runs fast and
wrong is the most dangerous thing in a benchmark, so nothing enters Stage 2
without passing here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Optional

import torch

from .config import AttnConfig
from .backends.base import AttentionBackend, UnsupportedConfig

# Loosely calibrated to bf16 accumulation error. Tighten per-dtype once we have
# baseline numbers from the reference implementations.
TOL = {
    "bfloat16": dict(atol=2e-2, rtol=2e-2),
    "float16": dict(atol=5e-3, rtol=5e-3),
    "float32": dict(atol=1e-4, rtol=1e-4),
}


@dataclass
class ProbeResult:
    backend: str
    config_key: str
    claimed: bool
    claim_reason: str
    actual: str              # supported | unsupported | oom | error
    detail: str = ""
    claim_mismatch: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def probe(backend: AttentionBackend, cfg: AttnConfig,
          mask=None) -> ProbeResult:
    """Determine empirically whether this cell runs.

    The claim/actual split is the point. Documentation for these kernels is
    wrong often enough that disagreement between the two is a reportable finding
    rather than a bug in our harness.
    """
    claimed, reason = backend.claims_support(cfg)

    try:
        backend.run_once(cfg, mask=mask)
        torch.cuda.synchronize()
        actual, detail = "supported", ""
    except UnsupportedConfig as e:
        actual, detail = "unsupported", str(e)[:300]
    except torch.cuda.OutOfMemoryError as e:
        actual, detail = "oom", str(e)[:300]
    except RuntimeError as e:
        msg = str(e)
        actual = "oom" if "out of memory" in msg.lower() else "error"
        detail = msg[:300]
    except Exception as e:
        actual, detail = "error", f"{type(e).__name__}: {e}"[:300]
    finally:
        torch.cuda.empty_cache()

    return ProbeResult(
        backend=backend.name,
        config_key=cfg.key(),
        claimed=claimed,
        claim_reason=reason,
        actual=actual,
        detail=detail,
        claim_mismatch=(claimed != (actual == "supported")),
    )


@dataclass
class CorrectnessResult:
    backend: str
    config_key: str
    passed: bool
    max_abs_err: Optional[float] = None
    max_rel_err: Optional[float] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def check_correctness(backend: AttentionBackend, cfg: AttnConfig,
                      mask=None, seed: int = 0) -> CorrectnessResult:
    """Compare against a float64 naive reference on identical inputs.

    The error magnitudes are recorded even on pass, because numerical fidelity
    is an independent axis of the study: a low-precision kernel that drifts at
    long sequence lengths is losing accuracy in a way task metrics will not
    always surface.
    """
    from .backends.impls import NaiveAttention

    ref_backend = NaiveAttention()
    q, k, v = backend.make_inputs(cfg, seed=seed)

    try:
        with torch.no_grad():
            expected = ref_backend.reference(q.detach(), k.detach(),
                                             v.detach(), cfg)
            got = backend.forward(q.detach(), k.detach(), v.detach(),
                                  cfg, mask=mask)
    except UnsupportedConfig as e:
        return CorrectnessResult(backend.name, cfg.key(), False,
                                 detail=f"unsupported: {e}"[:300])
    except Exception as e:
        return CorrectnessResult(backend.name, cfg.key(), False,
                                 detail=f"{type(e).__name__}: {e}"[:300])

    got = got.double()
    abs_err = (got - expected).abs()
    rel_err = abs_err / expected.abs().clamp_min(1e-8)

    # A quantized backend's error profile comes from cfg.quant_scheme, not
    # cfg.dtype -- dtype is the storage/output precision (e.g. bfloat16),
    # unrelated to what a kernel like SageAttention actually computed
    # internally (int8 QK, block-scaled). Keying only on dtype would silently
    # grade a quantized backend against a tolerance that has nothing to do
    # with its error profile.
    tol_key = cfg.quant_scheme if cfg.quant_scheme is not None else cfg.dtype
    if tol_key not in TOL:
        kind = "quant_scheme" if cfg.quant_scheme is not None else "dtype"
        raise KeyError(
            f"no correctness tolerance defined for {kind}={tol_key!r}; "
            f"add one to TOL rather than silently reusing bfloat16's"
        )
    tol = TOL[tol_key]
    # Elementwise combined bound (torch.allclose convention), not two
    # independent whole-tensor maxima OR'd together: a single element that is
    # simultaneously large in absolute error and, at some *other* element,
    # small in relative error must not pass just because each metric looked
    # fine somewhere in the tensor.
    passed = bool((abs_err <= tol["atol"] + tol["rtol"] * expected.abs()).all())

    return CorrectnessResult(
        backend=backend.name,
        config_key=cfg.key(),
        passed=passed,
        max_abs_err=float(abs_err.max()),
        max_rel_err=float(rel_err.max()),
    )


def probe_grid(backends: Iterable[AttentionBackend],
               configs: Iterable[AttnConfig]) -> list[dict]:
    """Run the full Stage 0 matrix. Returns rows ready for a dataframe."""
    configs = list(configs)
    rows = []
    for b in backends:
        for cfg in configs:
            r = probe(b, cfg)
            rows.append({**r.to_dict(), **cfg.to_dict()})
    return rows
