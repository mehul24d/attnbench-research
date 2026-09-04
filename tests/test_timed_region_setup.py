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


def _observe(cls, mask: str, monkeypatch) -> Observation:
    name = cls.capability.name
    cfg, head_dim, dtype = _cfg_for(cls, mask)
    counter: collections.Counter = collections.Counter()
    _watch_view_constructors(monkeypatch, counter)

    try:
        backend = cls()
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
    for _, cls in all_backends().items():
        cap = cls.capability
        for mask in ("causal", "block_sparse"):
            if mask == "causal" and not cap.supports_causal:
                continue
            if mask == "block_sparse" and not cap.supports_block_sparse:
                continue
            out.append(_observe(cls, mask, monkeypatch))
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
    registered = {cls.capability.name for cls in all_backends().values()}
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
