"""Stage 0 (capability matrix) and Stage 1 (correctness gate).

Both run before any timing. Stage 0 tells us which cells exist; Stage 1 tells us
which of those cells produce trustworthy output. A backend that runs fast and
wrong is the most dangerous thing in a benchmark, so nothing enters Stage 2
without passing here.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, asdict, replace
from typing import Iterable, Optional

import torch

from . import compile_guard
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
    from .backends.base import FAULT_REASON_PREFIX

    claimed, reason = backend.claims_support(cfg)

    # A kernel that faults the device is NOT launched, and the cell is recorded
    # as the finding it is rather than as a gap. `sdpa_cudnn` at seq_len=16384
    # on sm_89 reads unmapped memory: Xid 31, MMU Fault ENGINE GRAPHICS, which
    # kills the whole process. Nine other backends completed all 84 configs in
    # that band and cuDNN wrote zero.
    #
    # This is deliberately its own status. "unsupported" would say the kernel
    # declines the config, which is false and much milder than the truth, and
    # it would sit in the same column as fifty ordinary declines where nobody
    # would ever look at it again.
    if not claimed and reason.startswith(FAULT_REASON_PREFIX):
        return ProbeResult(
            backend=backend.name,
            config_key=cfg.key(),
            claimed=False,
            claim_reason=reason,
            actual="illegal_memory_access",
            detail=reason[len(FAULT_REASON_PREFIX):][:300],
            # Not a mismatch: claim and behaviour agree. The backend says it
            # faults here and it does.
            claim_mismatch=False,
        )

    # A block_sparse config with no mask cannot run: every backend that
    # implements it raises "block_sparse requires an explicit mask". Probing
    # without one marked all 540 block_sparse cells "unsupported", so they
    # never reached the correctness gate, so they had no Stage 1 pass, so
    # Stage 2 would have rejected every one of them -- the sparse arm of the
    # study, absent, with "unsupported" as the only trace.
    if mask is None and cfg.mask == "block_sparse" and cfg.mask_source:
        from .masks import mask_for
        mask = mask_for(cfg)

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
      "cross_backend_pair"
                     -- the same comparison, but only TWO implementations
                        survived the shape: the third could not run or could
                        not be compared here. Weaker again, deliberately
                        labelled so, and never silently merged into
                        "cross_backend". See MIN_CROSS_BACKEND_AGREEING.
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
    # The memory budget that produced this verdict, in bytes.
    #
    # Budgets became a FRACTION of device memory when the study gained a
    # second card, so "declined: memory infeasible" no longer means one fixed
    # number -- it means 12.1 GiB on an L4 and 44.0 GiB on an A100. A reader
    # who has to infer the budget from the GPU model is reconstructing the
    # standard from context, and will get it wrong the first time a third card
    # appears.
    #
    # Same principle as `check_kind`: the verdict and the standard that
    # produced it travel together, on the row.
    memory_budget_bytes: Optional[int] = None

    # The denominator floor `max_rel_err` was computed over, or None where no
    # relative-error comparison was made (structural checks, declined configs).
    #
    # A ratio is only as precise as its denominator. Elements whose |expected|
    # is below this floor are excluded from max_rel_err entirely, so the
    # number reports the largest relative error among the elements where
    # relative error decides anything -- see check_correctness. Recorded on the
    # row for the same reason as `memory_budget_bytes`: a reader must not have
    # to reconstruct the standard a verdict was measured against.
    #
    # `max_rel_err=None` with a floor set means NOTHING cleared the floor. That
    # is not agreement, and must never be read as 0.0.
    rel_err_floor: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def resolvable_rel_err(abs_err, expected, floor: float):
    """Largest relative error among elements whose denominator resolves it.

    Returns `(max_rel_err_or_None, n_resolvable, n_total)`.

    Elements with `|expected| < floor` are excluded, not clamped. Clamping
    reports a ratio against a denominator that was never measured, which is
    the same fiction one magnitude smaller; excluding says how many elements
    the answer rests on, which is what `n_resolvable` is for.

    `None` when nothing clears the floor -- absence of a measurement. Never
    0.0, which reads as perfect agreement and is the strongest possible claim
    to make from no data.
    """
    denom = expected.abs()
    resolvable = denom >= floor
    n_resolvable = int(resolvable.sum())
    n_total = int(denom.numel())
    if not n_resolvable:
        return None, 0, n_total
    return float((abs_err[resolvable] / denom[resolvable]).max()), n_resolvable, n_total


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

    # The ORACLE is asked whether it supports this config, for the same reason
    # the cross-backend references are: an implementation that declines a
    # config and is run anyway computes a different function and answers with
    # confidence. Naive happens to claim every config this study uses today,
    # so this changes nothing now -- which is exactly why it was missing.
    # Correctness by luck of one backend's breadth is not correctness by
    # design, and the next oracle, dtype or mask kind would inherit the gap.
    ref_claimed, ref_why = ref_backend.claims_support(cfg)
    if not ref_claimed:
        return CorrectnessResult(
            backend.name, cfg.key(), False, "exact",
            detail=f"the float64 oracle declines this config ({ref_why}), so "
                   f"there is nothing to compare against. Not a verdict on "
                   f"{backend.name}: the reference, not the backend, is "
                   f"missing here."[:300])

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

    if got.shape != expected.shape:
        return CorrectnessResult(
            backend.name, cfg.key(), False, "exact",
            detail=f"backend returned shape {tuple(got.shape)}, expected "
                   f"{tuple(expected.shape)}")
    if not torch.isfinite(got).all():
        return CorrectnessResult(
            backend.name, cfg.key(), False, "exact",
            detail="backend returned non-finite output")
    if not torch.isfinite(expected).all():
        return CorrectnessResult(
            backend.name, cfg.key(), False, "exact",
            detail="reference returned non-finite output")

    got = got.double()
    abs_err = (got - expected).abs()

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

    # Relative error is only as precise as its denominator, and the previous
    # `abs_err / expected.abs().clamp_min(1e-8)` had no floor -- 1e-8 is a
    # divide-by-zero guard, which is a different thing. On a block-sparse
    # config most of `expected` is at or near zero by construction (the mask
    # zeroes it), so the ratio was dividing a real bf16 absolute error by a
    # number that carries no information. The A100 rows show the result:
    # max_rel_err reached 3.3e+04 while max_abs_err stayed at 1.3e-02.
    #
    # The floor is `atol`, and not by analogy. Below atol the pass/fail bound
    # `atol + rtol*|expected|` is dominated by atol, so relative error governs
    # nothing there; above it, relative error is what decides. Reporting the
    # maximum over exactly the elements where the metric has a decision behind
    # it is what makes the number mean something.
    #
    # Sub-floor elements are EXCLUDED, not clamped. Clamping would report a
    # ratio computed against a denominator that was never measured, which is
    # the same fiction one magnitude smaller. If nothing clears the floor the
    # answer is None -- absence of a measurement, never 0.0, which would read
    # as perfect agreement.
    rel_floor = float(tol["atol"])
    max_rel, n_resolvable, n_total = resolvable_rel_err(abs_err, expected,
                                                        rel_floor)

    max_abs = float(abs_err.max())
    n_violating = int((abs_err > tol["atol"] + tol["rtol"] * expected.abs()).sum())

    # `detail` was empty on both outcomes, so a FAILING row carried a number
    # and no account of it -- a reader had the max error and not the tolerance
    # it was judged against, the count of offending elements, or where in the
    # tensor they were. That is the same shape as a flag nobody reads: the
    # field exists, so an auditor concludes the reason is recorded.
    if passed:
        detail = (f"{n_resolvable}/{n_total} elements above the "
                  f"rel_err floor {rel_floor:g}")
    else:
        worst = int(abs_err.argmax())
        detail = (f"{n_violating}/{n_total} elements exceed "
                  f"atol={tol['atol']:g} + rtol={tol['rtol']:g}*|expected|; "
                  f"worst at flat index {worst} with |err|={max_abs:.3e} "
                  f"against |expected|={float(expected.abs().flatten()[worst]):.3e}; "
                  f"{n_resolvable}/{n_total} elements above the rel_err floor "
                  f"{rel_floor:g}")

    return CorrectnessResult(
        backend=backend.name,
        config_key=cfg.key(),
        passed=passed,
        check_kind="exact",
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        rel_err_floor=rel_floor,
        detail=detail[:300],
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

    # Both error columns are None, not 0.0. A structural check makes NO
    # numerical comparison -- there is no oracle to differ from -- and 0.0 in
    # an error column reads as perfect agreement, which is the strongest
    # possible claim and the exact opposite of what this verdict certifies.
    # GLA contributes 30 such rows to the A100 dataset; under the old value a
    # plot of max_abs_err by backend would have shown GLA as the most accurate
    # kernel in the study.
    return CorrectnessResult(backend.name, cfg.key(), True, max_abs_err=None,
                             max_rel_err=None, check_kind="structural",
                             detail="structural only: finite/shape/dtype/"
                                    "determinism/causality, NOT numerical "
                                    "agreement with softmax attention; error "
                                    "columns are null because no comparison "
                                    "was made")


# Memory the float64 oracle is allowed to use for its score matrix. The
# oracle materialises (batch, heads, S, S) in float64, so feasibility is a
# function of the WHOLE config, not of seq_len alone -- an early version
# thresholded on seq_len<=4096 assuming batch=1 and promptly OOM'd on the
# batch-16 configs, where 4096 costs 68 GB rather than 4.
#
# 8 GiB leaves room on a 23 GiB card for weights, the backend under test, and
# fragmentation. An OOM here is not a correctness failure but it is recorded
# as one, so predicting it is better than discovering it.
# Budgets are a FRACTION of the device, not an absolute number of bytes.
#
# Both constants below were sized by hand for a 22.03 GiB L4. Carrying them to
# an 80 GB A100 would make the same commit mean different things on different
# cards -- verification strength conditioned on which machine happened to run
# it, which is the failure the commit-pinning machinery exists to prevent. So
# the fraction is the constant and the byte count is derived.
#
# The fractions reproduce the hand-picked L4 values almost exactly:
#
#     0.37 x 22.03 GiB =  8.15 GiB   (was 8)
#     0.55 x 22.03 GiB = 12.12 GiB   (was 12)
#
# Slightly above rather than below, deliberately: nothing that fit before may
# stop fitting now. And nothing NEW is admitted either, because oracle and
# cross-backend costs across this grid are widely spaced powers of two -- the
# nearest values to the oracle's 8 GiB boundary are 4 and 16 GiB, so a 2%
# change in the threshold cannot move a single cell. Asserted in
# tests/test_device_aware_budgets.py rather than left as arithmetic in a
# comment.
#
# On an 80 GiB A100 the same fractions give 29.6 and 44.0 GiB, which restores
# the float64 oracle at three config classes and admits cross-backend at
# 32768/batch 16 (21.5 GiB) -- see docs/a100_session_plan.md.
ORACLE_BUDGET_FRACTION = 0.37
CROSS_BACKEND_BUDGET_FRACTION = 0.55

# Used when no CUDA device is visible: the card every existing result in this
# project was measured on. A CPU test run must produce the same verdicts as the
# L4 did, or the suite stops describing the hardware the data came from.
FALLBACK_DEVICE_BYTES = 23_660_000_000      # NVIDIA L4, as torch reports it

# Retained as the L4-equivalent values so existing references keep meaning what
# they meant. New code should call the budget functions, which are device-aware.
EXACT_ORACLE_BUDGET_BYTES = int(FALLBACK_DEVICE_BYTES * ORACLE_BUDGET_FRACTION)
CROSS_BACKEND_BUDGET_BYTES = int(FALLBACK_DEVICE_BYTES * CROSS_BACKEND_BUDGET_FRACTION)


def device_memory_bytes(device: str = "cuda") -> int:
    """Total memory of the device a check would run on."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return FALLBACK_DEVICE_BYTES
    return int(torch.cuda.get_device_properties(device).total_memory)


def exact_oracle_budget_bytes(device: str = "cuda") -> int:
    return int(device_memory_bytes(device) * ORACLE_BUDGET_FRACTION)


def cross_backend_budget_bytes(device: str = "cuda") -> int:
    return int(device_memory_bytes(device) * CROSS_BACKEND_BUDGET_FRACTION)


def exact_oracle_fits(cfg: AttnConfig,
                      budget_bytes: Optional[int] = None,
                      device: str = "cuda") -> bool:
    """Whether a float64 naive reference can be allocated for this config."""
    if budget_bytes is None:
        budget_bytes = exact_oracle_budget_bytes(device)
    score_bytes = cfg.batch * cfg.n_heads_q * cfg.seq_len * cfg.seq_len * 8
    return score_bytes <= budget_bytes


def oracle_bytes(cfg: AttnConfig) -> int:
    return cfg.batch * cfg.n_heads_q * cfg.seq_len * cfg.seq_len * 8


# Memory a cross-backend check is allowed to use, on the same principle as
# EXACT_ORACLE_BUDGET_BYTES and for the same reason: a check whose cost is
# computable from the config should be DECLINED IN ADVANCE with a recorded
# reason, not attempted and discovered by an OOM. `exact_oracle_fits` began
# life as a seq_len cutoff, was wrong by 16x at batch 16, and became a memory
# bound; this is that lesson applied to the other check that allocates.
#
# 12 GiB against a ~22 GiB card. The remaining ~10 GiB is not slack: it covers
# each kernel's own workspace beyond the tensors modelled below, the hoisted
# mask caches, inductor's compiled-kernel state, and allocator fragmentation
# -- and fragmentation is the reason a check can fail with several GiB
# nominally free, which is exactly how Stage 1 died three times in a row.
CROSS_BACKEND_BUDGET_BYTES = 12 * 2**30


def _dtype_bytes(dtype: str) -> int:
    return torch.empty(0, dtype=getattr(torch, dtype)).element_size()


def cross_backend_bytes(cfg: AttnConfig) -> int:
    """Peak resident tensor bytes for one cross-backend check.

    Modelled term by term rather than as a single fudge factor, because a
    wrong-by-16x guess is what this function exists to prevent:

      inputs      q + k + v, held for the whole loop (every reference needs
                  them, so none can be freed early).
      outputs     the tested output plus ONE reference output. Not one per
                  reference: the loop frees each before running the next, so
                  the peak does not grow with the number of references -- and
                  therefore comparing against fewer of them saves no memory,
                  which is why the pair fallback below is triggered by what
                  actually happens rather than predicted here.
      expansion   GQA KV expansion inside a reference's forward, which
                  materialises k and v at the query head count. At 32:8 that
                  is 1.5x a query-shaped tensor, transient but concurrent
                  with everything above, and far too large to bury in a
                  headroom factor.
      compare     `_max_abs_diff` casts one batch slice of each side to
                  float32 and takes a difference: three slice-sized float32
                  tensors. Coarse at small batch, where one slice is a large
                  fraction of the whole.
    """
    itemsize = _dtype_bytes(cfg.dtype)
    per_q = cfg.batch * cfg.n_heads_q * cfg.seq_len * cfg.head_dim * itemsize
    per_kv = cfg.batch * cfg.n_heads_kv * cfg.seq_len * cfg.head_dim * itemsize

    inputs = per_q + 2 * per_kv
    outputs = 2 * per_q
    expansion = 2 * (per_q - per_kv)
    compare = 3 * (per_q // cfg.batch) * 4 // itemsize
    return int(inputs + outputs + expansion + compare)


def cross_backend_fits(cfg: AttnConfig,
                       budget_bytes: Optional[int] = None,
                       device: str = "cuda") -> bool:
    """Whether a cross-backend check can be attempted at all for this config.

    False is a RESULT -- "cross-backend verification is infeasible at this
    shape on this hardware" is a real statement about what this study can
    verify, and it belongs in the results table. It is not the same as a
    crash, and it is not the same as a backend being wrong.
    """
    if budget_bytes is None:
        budget_bytes = cross_backend_budget_bytes(device)
    return cross_backend_bytes(cfg) <= budget_bytes


# Independent implementations required to agree before a cross-backend pass is
# recorded. THREE, not two: two kernels sharing a bug is plausible (a common
# upstream, a shared CUTLASS path); three from different authors much less so.
MIN_CROSS_BACKEND_AGREEING = 3

# ...but three is not always ACHIEVABLE. Where references were offered and
# then lost to the shape -- OOM running, OOM comparing, or an outright
# decline -- insisting on three would delete the 8192 and 16384 bands from
# the study to protect a standard that the hardware, not the backend, made
# unreachable. So two implementations agreeing is recorded as a pass under a
# DIFFERENT check_kind, carrying its own weakness on the row.
#
# The distinction that keeps this honest: attempted-and-lost, versus
# never-offered. A caller who simply supplies too few references still gets a
# hard failure below, because nothing was tried and no hardware limit was
# reached -- that is a study-design gap, and silently downgrading it would
# hide the one case where the standard is genuinely within reach.
MIN_CROSS_BACKEND_PAIR = 2
CHECK_KIND_PAIR = "cross_backend_pair"

# Cross-backend comparisons allow twice the exact check's tolerance: both
# sides are approximate, so each contributes its own error from truth.
CROSS_BACKEND_TOL_FACTOR = 2.0


# A REFERENCE hitting a hardware limit is one fewer opinion; a reference
# hitting a bug is a hard failure. Both arrive as the same exception type
# (torch._inductor's InductorError wraps everything), so the two are separated
# on the MESSAGE, deliberately and narrowly.
#
# Matching on the type instead -- `except InductorError: skip` -- would be a
# hatch that swallows real compilation bugs in a reference implementation and
# silently downgraded the evidence for every cell that reference touched. Only
# these two signatures are known to be device limits, both already recorded in
# docs/limitations.md, and anything else stays a hard failure:
#
#   1. Triton shared memory. flex block-sparse at head_dim=128 needs 114688 B
#      per block; sm_89 has 101376 B. "Required: 114688 Hardware limit:101376".
#   2. flex's BlockMask block size against inductor's default tile. With
#      max_autotune off there is exactly one candidate config, so a
#      divisibility mismatch raises rather than falling back to another tile.
#
# Both were found on 2026-09-04, where they cost 66 correctness cells -- 42 of
# block_sparse's 78, the sparse arm, at exactly the lengths this study is
# about -- because flex-as-reference could not lower and the whole check was
# recorded as the backend under test having failed.
_DEVICE_LIMIT_SIGNATURES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("out of resource", "Hardware limit"),
     "Triton shared-memory limit"),
    (("must be divisible by BLOCK_M and BLOCK_N",),
     "BlockMask block size vs inductor's default tile"),
)


def device_limit_reason(exc: BaseException) -> Optional[str]:
    """A short label if `exc` is a known device limit, else None.

    None means "not recognised", and the caller must treat that as a real
    failure. Erring toward None is the safe direction: an unrecognised error
    fails a cell loudly, where a wrongly-recognised one quietly weakens the
    evidence behind a pass.
    """
    msg = str(exc)
    for needles, label in _DEVICE_LIMIT_SIGNATURES:
        if all(n in msg for n in needles):
            return label
    return None


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor, chunk: int = 1) -> float:
    """max |a - b| in float32, without materialising a full difference tensor.

    `(a - b).abs().max()` allocates two more tensors the size of the inputs. At
    batch 16 / seq 8192 / 32 heads / head_dim 128 that is ~1 GB apiece, on top
    of the two outputs already resident -- which is exactly how Stage 1 died at
    the 8192 band on 2026-09-04, *after* both forwards had succeeded.

    Reducing over the batch dimension one slice at a time bounds the extra
    allocation to one slice, and casting per slice keeps the comparison in
    float32 (where the tolerance reasoning lives) without ever holding a
    float32 copy of the whole output.

    The result is identical to the unchunked computation: max is associative
    over a partition, so chunking changes peak memory and nothing else --
    asserted in tests/test_correctness_families.py.
    """
    worst = 0.0
    for i in range(0, a.shape[0], chunk):
        d = (a[i:i + chunk].float() - b[i:i + chunk].float()).abs().max()
        worst = max(worst, float(d))
        del d
    return worst


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
    # Tolerance comes from the COMPUTE dtype, not from the dtype the
    # comparison happens to be cast to. An earlier version used float32
    # (atol=1e-4) because the tensors are compared as float32, and every
    # bf16 config failed at 0.004-0.023 -- ordinary bf16 kernel variation.
    #
    # And it is deliberately LOOSER than the exact check by
    # CROSS_BACKEND_TOL_FACTOR. The exact check compares one kernel against
    # float64 truth, so its error budget is one kernel's epsilon. Here BOTH
    # sides are approximate: if each is within epsilon of truth, they can
    # differ from each other by up to 2*epsilon without either being wrong.
    # Holding cross-backend to the oracle's tolerance would report ordinary
    # rounding as disagreement.
    base = TOL[cfg.quant_scheme if cfg.quant_scheme is not None else cfg.dtype]
    tol = {k: v * CROSS_BACKEND_TOL_FACTOR for k, v in base.items()}

    budget = cross_backend_budget_bytes(device)

    def fail(detail: str) -> CorrectnessResult:
        return CorrectnessResult(backend.name, cfg.key(), False,
                                 "cross_backend", detail=detail[:300],
                                 memory_budget_bytes=budget)

    usable = [r for r in references if r.name != backend.name]
    if len(usable) + 1 < MIN_CROSS_BACKEND_AGREEING:
        return fail(f"only {len(usable) + 1} independent implementations "
                    f"available, need {MIN_CROSS_BACKEND_AGREEING}; a "
                    f"two-way agreement is not evidence enough to stand in "
                    f"for an oracle")

    # Declined before a single byte is allocated. The alternative -- find out
    # by OOMing -- costs the whole run, because an OOM mid-check leaves the
    # allocator in whatever state it died in and takes the process with it if
    # it lands somewhere unguarded. Three consecutive Stage 1 deaths at the
    # 8192 band were each at a different allocation site inside this check.
    if device.startswith("cuda") and not cross_backend_fits(cfg, device=device):
        need = cross_backend_bytes(cfg) / 2**30
        return fail(
            f"CROSS-BACKEND VERIFICATION INFEASIBLE at this shape: needs "
            f"{need:.1f} GiB of concurrent tensors (batch={cfg.batch}, "
            f"heads={cfg.n_heads_q}:{cfg.n_heads_kv}, seq_len={cfg.seq_len}) "
            f"against a {budget / 2**30:.1f} GiB budget "
            f"({CROSS_BACKEND_BUDGET_FRACTION:.0%} of a "
            f"{device_memory_bytes(device) / 2**30:.1f} GiB device), "
            f"and the float64 oracle would need {oracle_bytes(cfg) / 2**30:.1f} "
            f"GiB. Not a verdict on this backend: nothing was run. This cell "
            f"cannot be verified on this hardware by any means available here.")

    q, k, v = backend.make_inputs(cfg, device=device, seed=seed)
    q, k, v = q.detach(), k.detach(), v.detach()

    try:
        with torch.no_grad():
            # Deliberately NOT .float() here. A float32 copy of the output at
            # batch 16 / seq 8192 / 32 heads / head_dim 128 is 2.1 GB, held for
            # the whole loop while each reference allocates its own. The
            # comparison still happens in float32 -- `_max_abs_diff` casts one
            # slice at a time, which is where the tolerance reasoning needs it.
            got = backend.forward(q, k, v, cfg, mask=mask)
    except UnsupportedConfig as e:
        return fail(f"unsupported: {e}")
    except Exception as e:
        return fail(f"{type(e).__name__}: {e}")

    if got.shape != q.shape:
        return fail(f"backend returned shape {tuple(got.shape)}, expected "
                    f"{tuple(q.shape)}")
    if not torch.isfinite(got).all():
        return fail("backend returned non-finite output")

    errors: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for ref in usable:
        # RESPECT THE CLAIM, exactly as the Stage 1 loop does for the backend
        # under test. A dense backend handed a block_sparse config ignores the
        # mask argument entirely and computes plain causal attention, which
        # "succeeds" -- so it produces a confident, wrong opinion about a
        # different function.
        #
        # This is instance 1's shape (an oracle computing something other than
        # the kernel) arriving through the reference path. The guard was added
        # to run_probe.py's choice of BACKEND on 2026-09-03, after it produced
        # max_abs_err=4.81 and 114/126 failures, and not to the choice of
        # REFERENCE -- so on 2026-09-04 the identical failure reappeared as 66
        # block_sparse/naive cells disagreeing by ~4.9 with five dense
        # references that were identical to six decimal places. Five
        # independent kernels agreeing exactly is not five opinions; it is one
        # computation, and it was not the one under test.
        claimed, why = ref.claims_support(cfg)
        if not claimed:
            skipped[ref.name] = f"declines this config: {why}"[:80]
            continue
        try:
            with torch.no_grad():
                other = ref.forward(q, k, v, cfg, mask=mask)
        except UnsupportedConfig as e:
            skipped[ref.name] = f"unsupported: {e}"[:80]
            continue
        except torch.cuda.OutOfMemoryError:
            # A reference that cannot fit this shape is one fewer opinion,
            # not a verdict on the backend under test. Treated like
            # "unsupported" -- if too few remain, that is reported honestly
            # below rather than being blamed on the backend.
            skipped[ref.name] = "OOM at this shape"
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            # A reference that cannot LOWER on this device is in the same
            # position as one that cannot fit: it has no opinion here. It is
            # not evidence about the backend under test, and recording it as a
            # failure of that backend is a misattribution, not a conservative
            # choice -- it deleted 42 of block_sparse's 78 cells on 2026-09-04
            # because flex could not compile a mask it never needed to.
            limit = device_limit_reason(e)
            if limit is None:
                return fail(f"reference {ref.name} raised {type(e).__name__}: {e}")
            skipped[ref.name] = f"device limit ({limit})"
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            continue

        if other.shape != got.shape:
            return fail(f"reference {ref.name} returned shape "
                        f"{tuple(other.shape)}, expected {tuple(got.shape)}")
        if not torch.isfinite(other).all():
            return fail(f"reference {ref.name} returned non-finite output")

        try:
            errors[ref.name] = _max_abs_diff(got, other)
        except torch.cuda.OutOfMemoryError:
            # The COMPARISON can OOM even when both forwards fit -- it was the
            # full-tensor `(got - other).abs()` that killed Stage 1 at the 8192
            # band on 2026-09-04, asking for 1024 MiB with 559 MiB free, after
            # both outputs had been computed successfully.
            #
            # The existing policy covered a reference that could not RUN; this
            # is a reference we could not FINISH COMPARING, and it is the same
            # thing for the same reason -- one fewer opinion, never a verdict
            # on the backend under test.
            skipped[ref.name] = "OOM comparing at this shape"
            torch.cuda.empty_cache()
        # Free before the next reference. Three dense outputs at batch 16 /
        # seq 4096 held simultaneously is what made flex OOM in 27 of 29
        # cross-backend configs on the first run.
        del other
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if len(errors) + 1 < MIN_CROSS_BACKEND_PAIR:
        return fail(f"only {len(errors) + 1} implementations could run this "
                    f"shape ({', '.join(sorted(errors)) or 'none'}), need "
                    f"{MIN_CROSS_BACKEND_PAIR} even for a pair. "
                    f"Skipped: {skipped}")

    # Three references were offered; fewer survived the shape. Recorded as a
    # pass under the weaker kind rather than discarded -- see
    # MIN_CROSS_BACKEND_PAIR for why attempted-and-lost is treated differently
    # from never-offered.
    kind = ("cross_backend"
            if len(errors) + 1 >= MIN_CROSS_BACKEND_AGREEING
            else CHECK_KIND_PAIR)

    worst = max(errors.values())
    disagreeing = {n: e for n, e in errors.items() if e > tol["atol"]}
    if disagreeing:
        return CorrectnessResult(
            backend.name, cfg.key(), False, kind,
            max_abs_err=worst, memory_budget_bytes=budget,
            detail=f"disagreement beyond atol={tol['atol']}: {disagreeing}. "
                   f"One of these implementations is wrong at this shape -- "
                   f"a finding, not a cell to skip.")

    if kind == CHECK_KIND_PAIR:
        note = (f" WEAKER EVIDENCE: only {len(errors) + 1} implementations "
                f"survived this shape, below the {MIN_CROSS_BACKEND_AGREEING} "
                f"this study normally requires, so a shared bug between two "
                f"kernels would not be caught here. Lost: {skipped}.")
    elif skipped:
        # Reported even when the check passed at full strength. Otherwise a
        # row reading "agrees with 2 implementations" is indistinguishable
        # between "two were offered" and "three were offered and one hit a
        # hardware limit" -- and the second is a fact about this card that the
        # study is partly about.
        note = f" One reference did not run here: {skipped}."
    else:
        note = ""

    return CorrectnessResult(
        backend.name, cfg.key(), True, kind, max_abs_err=worst,
        memory_budget_bytes=budget,
        detail=f"agrees with {len(errors)} independent implementations "
               f"({', '.join(sorted(errors))}) within atol={tol['atol']}; "
               f"NOT verified against a float64 oracle, which would need "
               f"{oracle_bytes(cfg) / 2**30:.1f} GiB for this config.{note}")


def check_for_family(backend: AttentionBackend, cfg: AttnConfig,
                     mask=None, seed: int = 0,
                     device: str = "cuda",
                     references: Optional[list[AttentionBackend]] = None,
                     ) -> CorrectnessResult:
    """`_dispatch_for_family`, but a pass obtained via an eager fallback is void.

    Stage 1 exists to license Stage 2. A pass earned by code Stage 2 will not
    run is worse than no pass at all: it is a precondition that reports
    satisfied while certifying a different implementation. That is exactly what
    happened to `flex` block-sparse on 2026-09-03 -- 72/72 passes with real
    numerical agreement, from the eager path dynamo silently switched to after
    the probe exhausted the recompile limit, for 72 cells the compiled kernel
    cannot lower on an L4 at all.

    So the fallback voids the verdict rather than annotating it. `passed` goes
    to False and the reason is written into `detail`; `check_kind` is
    preserved, because what was attempted is still the honest label for what
    kind of check this row was.

    This is also where the correctness path releases memory between checks.
    `timing.measure` has had that `finally` since it was written and this
    function never did, which is the whole of why Stage 1 died at the 8192
    band three times running, at three different allocation sites, each
    "fixed" in turn while the actual defect sat here.

    The mechanism is not leaked tensors -- every check's tensors are freed by
    refcount when its frame dies. It is that freeing them returns them to
    PyTorch's caching allocator, which keeps the blocks RESERVED and hands
    them back only to allocations that fit. Successive checks at different
    shapes fragment that reserve until a 1 GiB contiguous request fails with
    the card nominally almost empty -- the last crash reported 47 MiB free
    against 22 GiB total, which is that state exactly. `empty_cache()` is what
    returns the blocks to the driver; nothing else does.
    """
    try:
        with compile_guard.guard() as cg:
            result = _dispatch_for_family(backend, cfg, mask=mask, seed=seed,
                                          device=device, references=references)
        if cg.fell_back:
            return replace(result, passed=False,
                           detail=f"VOID ({cg.detail}) | {result.detail}"[:400])
        return result
    finally:
        # gc.collect() first: a reference cycle (an exception traceback
        # holding a frame that holds a tensor) survives refcounting, and
        # empty_cache() cannot release a block that is still referenced.
        # Order matters, and it is the order timing.measure uses.
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _dispatch_for_family(backend: AttentionBackend, cfg: AttnConfig,
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

    if not exact_oracle_fits(cfg, device=device):
        # No float64 oracle can exist here (64 GiB at 16384, 256 at 32768).
        # Fall back to agreement among independent implementations rather than
        # either skipping the length -- which would delete the study's
        # subject matter -- or inheriting a short-length pass, which would
        # assume exactly what Stage 1 exists to measure: that numerical error
        # does not accumulate with sequence length.
        if not references:
            return CorrectnessResult(
                backend.name, cfg.key(), False, "cross_backend",
                detail=f"float64 oracle would need "
                       f"{oracle_bytes(cfg) / 2**30:.1f} GiB for this config "
                       f"(batch={cfg.batch}, heads={cfg.n_heads_q}, "
                       f"seq_len={cfg.seq_len}) and no reference backends "
                       f"were supplied to compare against")
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
        return replace(result, check_kind="masked_exact",
                       memory_budget_bytes=exact_oracle_budget_bytes(device))

    return replace(
        check_correctness(backend, cfg, mask=mask, seed=seed, device=device),
        memory_budget_bytes=exact_oracle_budget_bytes(device))


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
