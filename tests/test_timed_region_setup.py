"""What is inside the timed region: setup that repeats on every call.

**The gap this closes.** Every other test in this suite asserts correctness --
outputs, masks, shapes, dtypes, determinism, causality. None asserted anything
about *cost*. For a study whose entire output is timing measurements that is a
structural hole, not an oversight, and it is why `FlexAttentionBackend` called
`create_block_mask` on every forward for as long as the backend existed: the
suite was green throughout, because the answers were right.

It cost a 10x error in the reported number. Segment 1 measured flex at 4.22
useful TFLOPS at seq_len=1024/batch=1 against FA2's 47.9 on the same card;
with mask construction hoisted out of the hot path the same cell reads 40.72.
"FlexAttention is an order of magnitude slower than FlashAttention-2" would
have been a clean, plausible, entirely artifactual finding.

**The general property.** `timing.measure` allocates q/k/v once and then calls
`timed_call` repeatedly, so anything a backend rebuilds per call that depends
only on `cfg` is a tax no deployment pays -- a real one builds a mask once and
reuses it across calls and layers. So: for each backend, is this work done
once or per call?

**How that is detected, mechanically.**

  1. *Factory ops sized by seq_len.* `torch.ones/zeros/full/arange/...` take no
     input tensor, so whatever they produce cannot depend on the VALUES in
     q/k/v -- it is a function of cfg alone, and therefore hoistable by
     construction. This is what catches a causal mask built with
     `torch.ones(S, S).triu(1)`.
  2. *Named view constructors.* `BlockSparseMask.to_*` and flex's
     `create_block_mask` convert a shared mask into a backend-specific
     representation. Those take a tensor argument, so (1) cannot see them.

Backends whose kernels cannot run here (fa2, gla, block_sparse, sage,
xformers need CUDA or an extension module) are reported as NOT_EXERCISED
rather than silently passing -- an unrunnable backend must not look like a
clean one. They ARE exercised by the same test on the instance, where the
suite is a per-session precondition. A backend that raises AFTER doing setup
work is still judged, since the counters have already seen what they need.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Optional

import pytest
import torch
from torch.overrides import TorchFunctionMode

from attnbench.accuracy.grid_configs import backend_instance
from attnbench.backends import all_backends
from attnbench.config import AttnConfig
from attnbench import masks as M

SEQ_LEN = 128
CALLS = 3

_FACTORIES = {"ones", "zeros", "full", "arange", "eye", "empty", "linspace",
              "ones_like", "zeros_like", "full_like", "empty_like"}

# Backends that legitimately rebuild per call, with the reason. Empty on
# purpose: an entry here is a decision to leave a tax inside a timed region,
# and it should have to be argued for in writing. `test_no_stale_allowlist`
# fails if an entry stops being necessary, so this cannot rot into a
# permanent exemption list.
ALLOWLIST: dict[tuple[str, str], str] = {}


class _CountSeqLenFactories(TorchFunctionMode):
    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len
        self.n: collections.Counter = collections.Counter()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = getattr(func, "__name__", "")
        if (name in _FACTORIES and isinstance(out, torch.Tensor)
                and self.seq_len in tuple(out.shape)):
            self.n[name] += 1
        return out


@dataclass
class Observation:
    backend: str
    mask: str
    per_call: Optional[list[int]]     # None when the backend never ran
    detail: str = ""

    @property
    def exercised(self) -> bool:
        return self.per_call is not None

    @property
    def repeats_setup(self) -> bool:
        # Call 1 legitimately builds. Anything on a later call is a rebuild.
        return bool(self.per_call) and sum(self.per_call[1:]) > 0


def _watch_view_constructors(monkeypatch, counter: collections.Counter):
    """Count mask->backend-representation conversions, which the factory mode
    cannot see because they take a tensor argument."""
    for name in ("to_dense_bool", "to_flex_block_mask",
                 "to_block_sparse_attn_mask", "to_xformers_bias"):
        original = getattr(M.BlockSparseMask, name, None)
        if original is None:
            continue

        def make(orig, nm):
            def wrapper(self, *a, **kw):
                counter[nm] += 1
                return orig(self, *a, **kw)
            return wrapper

        monkeypatch.setattr(M.BlockSparseMask, name, make(original, name))

    try:
        import torch.nn.attention.flex_attention as FA
    except Exception:
        return
    orig_cbm = FA.create_block_mask

    def counting_cbm(*a, **kw):
        counter["create_block_mask"] += 1
        return orig_cbm(*a, **kw)

    monkeypatch.setattr(FA, "create_block_mask", counting_cbm)


def _cfg_for(cls, mask: str) -> tuple[AttnConfig, int, str]:
    cap = cls.capability
    dtype = "bfloat16" if "bfloat16" in cap.dtypes else cap.dtypes[0]
    head_dim = 64 if 64 in cap.head_dims else cap.head_dims[0]
    extra = (dict(block_size=64, sparsity=0.5, mask_source="random")
             if mask == "block_sparse" else {})
    cfg = AttnConfig(seq_len=SEQ_LEN, batch=1, n_heads_q=4, n_heads_kv=4,
                     head_dim=head_dim, dtype=dtype, mask=mask,
                     pass_kind="fwd", regime="prefill", **extra)
    return cfg, head_dim, dtype


# SDPA is ONE registered class with a kernel pinned per INSTANCE, so
# enumerating classes observes whichever kernel the constructor defaults to.
# That default is `efficient` -- a kernel this study never measures, and the
# one variant that raises UnsupportedConfig on CPU and on an L4 alike ("No
# viable backend for scaled_dot_product_attention was found").
#
# The cost of that was three sessions of `sdpa` reading NOT_EXERCISED, a
# withdrawn coverage claim in limitations.md, and an audit item (S14) asking
# whether the dense reference arm's timed region was checkable anywhere --
# while `sdpa_math` and `sdpa_flash`, the two the study actually uses, ran
# fine on the workstation the whole time. Both are clean.
#
# A backend whose identity includes a kernel must be enumerated by instance.
# Any class not listed here is enumerated as itself.
INSTANCES: dict[str, tuple[str, ...]] = {
    "sdpa": ("sdpa_math", "sdpa_flash", "sdpa_efficient"),
}


def _observe(cls, mask: str, monkeypatch, instance: Optional[str] = None) -> Observation:
    name = instance or cls.capability.name
    cfg, head_dim, dtype = _cfg_for(cls, mask)
    counter: collections.Counter = collections.Counter()
    _watch_view_constructors(monkeypatch, counter)

    try:
        backend = backend_instance(instance) if instance else cls()
        m = M.mask_for(cfg) if mask == "block_sparse" else None
        counter.clear()                       # mask_for is not the backend's work
    except Exception as e:
        return Observation(name, mask, None, f"construction failed: {type(e).__name__}")

    dt = getattr(torch, dtype)
    q = torch.zeros(1, 4, SEQ_LEN, head_dim, dtype=dt)
    k, v = q.clone(), q.clone()

    per_call, errors = [], []
    for _ in range(CALLS):
        before = sum(counter.values())
        with _CountSeqLenFactories(SEQ_LEN) as fm:
            try:
                backend.forward(q, k, v, cfg, mask=m)
            except Exception as e:
                errors.append(type(e).__name__)
        per_call.append(sum(fm.n.values()) + sum(counter.values()) - before)

    if errors and sum(per_call) == 0:
        # It raised before touching anything watched -- an import error or an
        # early reject. Nothing was observed, so nothing may be concluded.
        return Observation(name, mask, None, f"not runnable here: {errors[0]}")
    return Observation(name, mask, per_call,
                       f"raised {errors[0]} after setup" if errors else "")


def _all_observations(monkeypatch) -> list[Observation]:
    out = []
    for registered, cls in all_backends().items():
        cap = cls.capability
        for mask in ("causal", "block_sparse"):
            if mask == "causal" and not cap.supports_causal:
                continue
            if mask == "block_sparse" and not cap.supports_block_sparse:
                continue
            for instance in INSTANCES.get(registered, (None,)):
                out.append(_observe(cls, mask, monkeypatch, instance))
    return out


def test_no_backend_rebuilds_setup_inside_the_timed_region(monkeypatch):
    offenders = []
    for obs in _all_observations(monkeypatch):
        if not obs.exercised or not obs.repeats_setup:
            continue
        if (obs.backend, obs.mask) in ALLOWLIST:
            continue
        offenders.append(
            f"{obs.backend}/{obs.mask}: setup ops per call {obs.per_call} -- "
            f"work that depends only on cfg is being redone every call, and "
            f"Stage 2 times every call")
    assert not offenders, (
        "hoistable setup inside the timed region:\n  " + "\n  ".join(offenders))


def test_something_was_actually_exercised(monkeypatch):
    """The vacuous-check guard. Most backends need CUDA or an extension, so on
    a workstation this test could quietly observe nothing at all and pass."""
    exercised = [o for o in _all_observations(monkeypatch) if o.exercised]
    assert exercised, "no backend ran: this test proved nothing"
    names = {o.backend for o in exercised}
    assert "naive" in names, (
        f"naive must always be exercisable -- it is pure torch. Got {names}")


def test_every_registered_backend_is_considered(monkeypatch):
    """A new backend must appear in the report rather than escape it."""
    considered = {o.backend for o in _all_observations(monkeypatch)}
    # Expanded the same way _all_observations expands it, so a class listed in
    # INSTANCES must have every one of its instances examined -- adding a
    # kernel variant there without observing it is the hole this test closes.
    registered: set[str] = set()
    for registered_name, cls in all_backends().items():
        registered |= set(INSTANCES.get(registered_name, (cls.capability.name,)))
    missing = registered - considered
    assert not missing, (
        f"registered but never examined: {sorted(missing)}. Give it a causal "
        f"or block_sparse capability, or state why it is exempt.")


def test_no_stale_allowlist(monkeypatch):
    """An exemption that is no longer needed must be deleted, not left to rot
    into a permanent hole."""
    obs = {(o.backend, o.mask): o for o in _all_observations(monkeypatch)}
    for key in ALLOWLIST:
        o = obs.get(key)
        if o is None or not o.exercised:
            continue          # cannot judge here; the instance run will
        assert o.repeats_setup, (
            f"{key} is allowlisted but no longer rebuilds per call -- "
            f"remove the entry")


def test_the_detector_catches_a_deliberately_reintroduced_rebuild(monkeypatch):
    """A check nobody has watched fail is a guess. This reintroduces exactly
    the flex defect -- rebuilding the mask on every call -- and requires the
    detector to notice."""
    from attnbench.backends.impls import FlexAttentionBackend

    def rebuild_every_time(self, cfg, mask, device):
        from torch.nn.attention.flex_attention import create_block_mask

        def mod(b, h, qi, ki):
            return qi >= ki
        return create_block_mask(mod, B=None, H=None, Q_LEN=cfg.seq_len,
                                 KV_LEN=cfg.seq_len, device=device)

    monkeypatch.setattr(FlexAttentionBackend, "_block_mask_for",
                        rebuild_every_time)
    obs = _observe(FlexAttentionBackend, "causal", monkeypatch)
    assert obs.exercised, "the reintroduced defect must still be observable"
    assert obs.repeats_setup, (
        f"detector missed a per-call mask rebuild: {obs.per_call}")


# --- the other half of hoisting: it must not accumulate ---------------------
#
# Added after the fix above OOMed the Stage 1 probe on 2026-09-04. Hoisting
# setup out of the timed region is only correct if what is kept is BOUNDED.
# NaiveAttention's cached mask is (S, S) bool -- 268 MB at seq_len 16384, 1.07
# GB at 32768 -- and the probe reuses one backend instance across 504 configs,
# so an unbounded dict reached 21.69 GiB and killed the run after 24 minutes.
#
# "Cache it" and "cache all of them forever" are one keystroke apart, and only
# the second one shows up as an OOM half an hour into a billed session.

@pytest.mark.parametrize("backend_name", ["naive", "flex"])
def test_hoisted_setup_is_bounded_not_accumulated(backend_name):
    cls = all_backends()[backend_name]
    backend = cls()
    mask_kind = "causal" if backend_name == "naive" else "causal"

    def cached_tensor_bytes(obj) -> int:
        """Every tensor reachable from the instance, INCLUDING inside
        containers. The first version of this walked only top-level attributes
        and therefore passed against the exact bug it was written for -- the
        original cache was a `dict`, and a dict is not a Tensor. Watched it
        fail to fail, which is the only reason it is written this way now."""
        seen: set[int] = set()

        def walk(value) -> int:
            if id(value) in seen:
                return 0
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(walk(v) for v in value.values())
            if isinstance(value, (list, tuple, set)):
                return sum(walk(v) for v in value)
            if hasattr(value, "__dict__"):        # e.g. a flex BlockMask
                return sum(walk(v) for v in vars(value).values())
            return 0

        return sum(walk(v) for v in obj.__dict__.values())

    seen_sizes = []
    for seq_len in (64, 128, 256, 512):
        cfg = AttnConfig(seq_len=seq_len, batch=1, n_heads_q=4, n_heads_kv=4,
                         head_dim=64, dtype="float32", mask=mask_kind,
                         pass_kind="fwd", regime="prefill")
        q = torch.zeros(1, 4, seq_len, 64)
        try:
            backend.forward(q, q.clone(), q.clone(), cfg)
        except Exception:
            pass                      # flex cannot lower on CPU; setup still ran
        seen_sizes.append(cached_tensor_bytes(backend))

    # Four distinct configs of growing size. If every one were retained the
    # total would grow monotonically; bounded means it tracks the CURRENT
    # config only, so it never exceeds the largest single mask.
    largest_single = 512 * 512          # bool, one element per byte
    assert max(seen_sizes) <= largest_single, (
        f"{backend_name} retained {max(seen_sizes)} bytes across 4 configs; "
        f"one mask is at most {largest_single}. The cache is accumulating -- "
        f"this is what OOMed the probe at 21.69 GiB.")

    keys = [k for k in backend.__dict__ if k.endswith("_key")]
    assert len(keys) <= 1, f"expected a single-entry cache, found keys {keys}"


# ---------------------------------------------------------------------------
# Coverage, asserted rather than assumed
#
# `limitations.md` claimed for months that the CUDA backends were "covered by
# the same test on the instance, where the suite is a per-session
# precondition". Both were untrue: the only two session logs that record the
# suite running on a GPU show this file's main test FAILING, no log records it
# passing, and the session that produced results/stage5/phases.parquet has no
# suite record at all. The claim was withdrawn 2026-09-20.
#
# A prose claim about coverage is exactly the thing that rots. These assert it.
# ---------------------------------------------------------------------------

# The arms of every end-to-end comparison in this study, named as the study
# names them. `sdpa` is not on this list because it is not a thing the study
# measures -- `sdpa_math` and `sdpa_flash` are, and they are different kernels
# of one class (see INSTANCES above).
HEADLINE_BACKENDS = ("block_sparse", "sdpa_math", "sdpa_flash")

# Which of them cannot run without a GPU. Everything else must be observed
# HERE, on whatever machine runs the suite -- a skip for those would be the
# "pass if unexercised" shape this file exists to remove.
NEEDS_CUDA = {"block_sparse"}


def test_coverage_on_this_machine_is_reported_not_assumed(monkeypatch, capsys):
    """Always prints the exercised/not-exercised split, so a reader of CI
    output can see what the green tick covers. Never fails: on a workstation
    the gap is expected and is the reason NOT_EXERCISED exists."""
    obs = _all_observations(monkeypatch)
    exercised = sorted({o.backend for o in obs if o.exercised})
    missing = sorted({o.backend for o in obs if not o.exercised})
    with capsys.disabled():
        print(f"\n  timed-region coverage: {len(exercised)}/{len({o.backend for o in obs})} "
              f"backends exercised here")
        print(f"    exercised:     {', '.join(exercised) or 'none'}")
        print(f"    NOT exercised: {', '.join(missing) or 'none'}")
    assert obs, "no backends were even considered"


@pytest.mark.parametrize("backend", HEADLINE_BACKENDS)
def test_the_headline_backends_timed_region_is_checked_where_it_can_be(
        monkeypatch, backend):
    """On a CUDA machine, the backends the study's conclusions rest on must
    actually be observed -- not reported NOT_EXERCISED and waved through.

    Skipped on CPU, which is honest: the check cannot run there. It is
    deliberately NOT written as "pass if unexercised", because that is the
    shape of the claim this replaces.
    """
    if backend in NEEDS_CUDA and not torch.cuda.is_available():
        pytest.skip(f"{backend} needs CUDA; coverage for it is unverified here "
                    f"-- see docs/limitations.md, the withdrawn timed-region "
                    f"coverage claim")
    obs = [o for o in _all_observations(monkeypatch) if o.backend == backend]
    if not obs:
        pytest.skip(f"{backend} is not registered in this build")
    assert any(o.exercised for o in obs), (
        f"{backend} reported NOT_EXERCISED on this machine: "
        f"{[(o.mask, o.detail) for o in obs]}.\n\n"
        f"This backend is one arm of every end-to-end comparison in the study. "
        f"If its timed region cannot be observed here it is observed nowhere, "
        f"and no result depending on it has a checked timed region. Fix the "
        f"import/config that stops it running rather than accepting the skip.")
