"""Stage 0 (capability matrix) and Stage 1 (correctness gate).

Both run before any timing. Stage 0 tells us which cells exist; Stage 1 tells us
which of those cells produce trustworthy output. A backend that runs fast and
wrong is the most dangerous thing in a benchmark, so nothing enters Stage 2
without passing here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
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
    """One correctness verdict.

    `check_kind` records WHAT a pass certifies, because one column named
    `passed` now carries four different meanings and conflating them is how
    an inapplicable check gets read as a real one:

      "exact"        -- dense backends: agreement with a float64 naive
                        softmax reference within the dtype's tolerance.
      "masked_exact" -- sparse backends: the same comparison, but against the
                        naive reference given the SAME block-sparse mask.
      "cross_backend"-- lengths where the float64 oracle cannot exist (it
                        needs 64 GiB at 16384, 256 GiB at 32768): agreement
                        among THREE independent implementations at float32
                        tolerance. Weaker than an oracle, but two kernels
                        sharing a bug is plausible where three from different
                        authors is much less so.
      "structural"   -- linear backends: finite/shape/dtype/determinism and
                        causality. NOT numerical agreement, because a linear
                        attention kernel does not compute softmax attention
                        and never will.

    See docs/limitations.md. A reader treating a "structural" pass as
    evidence of numerical equivalence would be badly misled, which is why
    the kind travels with the row rather than living in a docstring -- and
    why it has NO DEFAULT. A default would let a cross_backend or structural
    pass be silently constructed as "exact" and read that way forever after.
    """

    backend: str
    config_key: str
    passed: bool
    check_kind: str            # REQUIRED -- deliberately no default.
    max_abs_err: Optional[float] = None
    max_rel_err: Optional[float] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def check_correctness(backend: AttentionBackend, cfg: AttnConfig,
                      mask=None, seed: int = 0,
                      device: str = "cuda") -> CorrectnessResult:
    """Compare against a float64 naive reference on identical inputs.

    The error magnitudes are recorded even on pass, because numerical fidelity
    is an independent axis of the study: a low-precision kernel that drifts at
    long sequence lengths is losing accuracy in a way task metrics will not
    always surface.
    """
    from .backends.impls import NaiveAttention

    ref_backend = NaiveAttention()
    q, k, v = backend.make_inputs(cfg, device=device, seed=seed)

    try:
        with torch.no_grad():
            expected = ref_backend.reference(q.detach(), k.detach(),
                                             v.detach(), cfg, mask=mask)
            got = backend.forward(q.detach(), k.detach(), v.detach(),
                                  cfg, mask=mask)
    except UnsupportedConfig as e:
        return CorrectnessResult(backend.name, cfg.key(), False, "exact",
                                 detail=f"unsupported: {e}"[:300])
    except Exception as e:
        return CorrectnessResult(backend.name, cfg.key(), False, "exact",
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
        check_kind="exact",
        max_abs_err=float(abs_err.max()),
        max_rel_err=float(rel_err.max()),
    )


def check_structural(backend: AttentionBackend, cfg: AttnConfig,
                     mask=None, seed: int = 0,
                     device: str = "cuda") -> CorrectnessResult:
    """Correctness for a backend that does NOT compute softmax attention.

    Gated Linear Attention is not an approximation of softmax attention; it is
    a different function. Grading it against an exact oracle produced
    `max_abs_err=1.42e+01, failed=6/6` -- not a failure, just confirmation
    that two different computations differ, which was known in advance.

    A reference GLA implementation was considered and rejected: it would
    either be the same `fla` code path under test (circular -- certifies
    nothing) or a reimplementation (a fresh source of bugs, with no way to
    tell which of the two was wrong when they disagreed).

    So this checks properties a real bug would violate:

      1. finite      -- no NaN/Inf anywhere in the output
      2. shape/dtype -- matches the query's, as every backend must
      3. determinism -- two calls on identical inputs agree bitwise
      4. CAUSALITY   -- the load-bearing one. Perturbing token t must not
                        change any output before t. A linear-attention kernel
                        that leaked future information would be badly broken
                        in a way no throughput measurement would reveal, and
                        the corruption would reach Stage 3 as plausible
                        accuracy numbers.

    Passing this does NOT certify numerical agreement with attention, and
    `check_kind="structural"` records that on the row.
    """
    q, k, v = backend.make_inputs(cfg, device=device, seed=seed)
    q, k, v = q.detach(), k.detach(), v.detach()

    def fail(detail: str) -> CorrectnessResult:
        return CorrectnessResult(backend.name, cfg.key(), False,
                                 detail=detail[:300], check_kind="structural")

    try:
        with torch.no_grad():
            out = backend.forward(q, k, v, cfg, mask=mask)
            again = backend.forward(q, k, v, cfg, mask=mask)
    except UnsupportedConfig as e:
        return fail(f"unsupported: {e}")
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")

    if not torch.isfinite(out).all():
        n = int((~torch.isfinite(out)).sum())
        return fail(f"output has {n} non-finite values")
    if out.shape != q.shape:
        return fail(f"shape {tuple(out.shape)} != query {tuple(q.shape)}")
    if out.dtype != q.dtype:
        return fail(f"dtype {out.dtype} != query {q.dtype}")
    if not torch.equal(out, again):
        d = (out.double() - again.double()).abs().max().item()
        return fail(f"non-deterministic: two identical calls differ by {d:.3e}")

    # Causality: perturb the second half of the keys/values and require every
    # output position before the split to be unchanged.
    split = cfg.seq_len // 2
    k2, v2 = k.clone(), v.clone()
    k2[:, :, split:, :] += 1.0
    v2[:, :, split:, :] += 1.0
    try:
        with torch.no_grad():
            perturbed = backend.forward(q, k2, v2, cfg, mask=mask)
    except Exception as e:
        return fail(f"causality probe raised {type(e).__name__}: {e}")

    leak = (out[:, :, :split, :].double()
            - perturbed[:, :, :split, :].double()).abs().max().item()
    if leak > 0.0:
        return fail(f"CAUSALITY VIOLATED: perturbing tokens >= {split} changed "
                    f"earlier outputs by up to {leak:.3e}")

    return CorrectnessResult(backend.name, cfg.key(), True, max_abs_err=0.0,
                             max_rel_err=0.0, check_kind="structural",
                             detail="structural only: finite/shape/dtype/"
                                    "determinism/causality, NOT numerical "
                                    "agreement with softmax attention")


# Above this length the float64 naive oracle cannot be allocated: it needs
# 64 GiB at 16384 and 256 GiB at 32768 for Stage 2's geometry (batch 1, 32
# heads), against a 23 GiB card. 4096 costs 4 GiB and fits.
EXACT_ORACLE_MAX_SEQ_LEN = 4096

# Independent implementations required to agree before a cross-backend pass is
# recorded. THREE, not two: two kernels sharing a bug is plausible (a common
# upstream, a shared CUTLASS path); three from different authors much less so.
MIN_CROSS_BACKEND_AGREEING = 3


def check_cross_backend(backend: AttentionBackend, cfg: AttnConfig,
                        references: list[AttentionBackend], mask=None,
                        seed: int = 0, device: str = "cuda",
                        ) -> CorrectnessResult:
    """Agreement among independent implementations, where no oracle can exist.

    Weaker evidence than a float64 reference, and used only where that
    reference is impossible to allocate. The error is recorded per row (and
    every row carries its seq_len), so cross-backend divergence AS A FUNCTION
    OF LENGTH is readable as a result -- numerical fidelity at long context is
    one of the study's target gaps, not merely a gate to clear.

    A single disagreeing backend is a FINDING, not a skip: it means one of
    three implementations is wrong at this shape, and which one is a question
    worth answering rather than routing around.
    """
    tol = TOL[cfg.quant_scheme if cfg.quant_scheme is not None else "float32"]

    def fail(detail: str) -> CorrectnessResult:
        return CorrectnessResult(backend.name, cfg.key(), False,
                                 "cross_backend", detail=detail[:300])

    usable = [r for r in references if r.name != backend.name]
    if len(usable) + 1 < MIN_CROSS_BACKEND_AGREEING:
        return fail(f"only {len(usable) + 1} independent implementations "
                    f"available, need {MIN_CROSS_BACKEND_AGREEING}; a "
                    f"two-way agreement is not evidence enough to stand in "
                    f"for an oracle")

    q, k, v = backend.make_inputs(cfg, device=device, seed=seed)
    q, k, v = q.detach(), k.detach(), v.detach()

    try:
        with torch.no_grad():
            got = backend.forward(q, k, v, cfg, mask=mask).float()
    except UnsupportedConfig as e:
        return fail(f"unsupported: {e}")
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")

    errors: dict[str, float] = {}
    for ref in usable:
        try:
            with torch.no_grad():
                other = ref.forward(q, k, v, cfg, mask=mask).float()
        except UnsupportedConfig:
            continue          # this reference cannot run this shape
        except Exception as e:
            return fail(f"reference {ref.name} raised {type(e).__name__}: {e}")
        errors[ref.name] = float((got - other).abs().max())

    if len(errors) + 1 < MIN_CROSS_BACKEND_AGREEING:
        return fail(f"only {len(errors) + 1} implementations could run this "
                    f"shape, need {MIN_CROSS_BACKEND_AGREEING}")

    worst = max(errors.values())
    disagreeing = {n: e for n, e in errors.items() if e > tol["atol"]}
    if disagreeing:
        return CorrectnessResult(
            backend.name, cfg.key(), False, "cross_backend",
            max_abs_err=worst,
            detail=f"disagreement beyond atol={tol['atol']}: {disagreeing}. "
                   f"One of these implementations is wrong at this shape -- "
                   f"a finding, not a cell to skip.")

    return CorrectnessResult(
        backend.name, cfg.key(), True, "cross_backend", max_abs_err=worst,
        detail=f"agrees with {len(errors)} independent implementations "
               f"({', '.join(sorted(errors))}) within atol={tol['atol']}; "
               f"NOT verified against a float64 oracle, which cannot be "
               f"allocated at seq_len={cfg.seq_len}")


def check_for_family(backend: AttentionBackend, cfg: AttnConfig,
                     mask=None, seed: int = 0,
                     device: str = "cuda",
                     references: Optional[list[AttentionBackend]] = None,
                     ) -> CorrectnessResult:
    """Dispatch to the check that is meaningful for this backend's family.

    Applying the exact comparison to every family is what produced Stage 2's
    near-miss: `gla` (linear) failed by construction and `block_sparse`
    (sparse) was never probed at all, so both would have been rejected by the
    pass table and the sweep would have measured only dense backends -- a
    clean-looking result missing the study's entire subject matter.
    """
    family = backend.capability.family
    if family == "linear":
        return check_structural(backend, cfg, mask=mask, seed=seed, device=device)

    if cfg.seq_len > EXACT_ORACLE_MAX_SEQ_LEN:
        # No float64 oracle can exist here (64 GiB at 16384, 256 at 32768).
        # Fall back to agreement among independent implementations rather than
        # either skipping the length -- which would delete the study's
        # subject matter -- or inheriting a short-length pass, which would
        # assume exactly what Stage 1 exists to measure: that numerical error
        # does not accumulate with sequence length.
        if not references:
            return CorrectnessResult(
                backend.name, cfg.key(), False, "cross_backend",
                detail=f"seq_len={cfg.seq_len} exceeds the float64 oracle's "
                       f"reach ({EXACT_ORACLE_MAX_SEQ_LEN}) and no reference "
                       f"backends were supplied to compare against")
        if cfg.mask == "block_sparse" and mask is None and cfg.mask_source:
            from .masks import mask_for
            mask = mask_for(cfg)
        return check_cross_backend(backend, cfg, references, mask=mask,
                                   seed=seed, device=device)

    if cfg.mask == "block_sparse":
        # A sparse backend compared with NO mask is comparing two different
        # computations, so the probe must supply one. It was omitted before,
        # which is why block_sparse had zero correctness rows and would have
        # been rejected wholesale by the Stage 2 pass table.
        #
        # The comparison itself is already validated: on 2026-09-03 BSA agreed
        # with the naive oracle to 8.09e-03 (bf16 tol 2e-2) given a real mask,
        # while the pre-causality-fix mask disagreed by 3.70. So this is
        # wiring a proven check, not inventing one.
        if mask is None:
            from .masks import mask_for
            if cfg.mask_source is None:
                return CorrectnessResult(
                    backend.name, cfg.key(), False, "masked_exact",
                    detail="block_sparse config has mask_source=None, so no "
                           "mask can be generated. Stage 2 uses "
                           "mask_source='random'; the probe config must match "
                           "or the comparison is against a different mask "
                           "than the sweep will time.")
            mask = mask_for(cfg)
        # to_dense_bool/to_flex_block_mask move the mask to the right device
        result = check_correctness(backend, cfg, mask=mask, seed=seed, device=device)
        return replace(result, check_kind="masked_exact")

    return check_correctness(backend, cfg, mask=mask, seed=seed, device=device)


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
