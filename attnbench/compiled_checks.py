"""What the ground truth's probe shows for an eager arm, shown for a compiled
arm whose estimator runs inside its compiled function (researcher's ruling 1
of 7 Nov 2026, after the exploratory session).

The probe (`ground_truth.ScoringProbe`) wraps the scoring functions in
Python and logs each call. Inside a compiled region Dynamo traces the
wrapper: in the exploratory session's `compiled_default` group it guarded
compiled XAttention's `forward_and_mask` on the length of the probe's log,
so every call recompiled until the recompile limit (256). A Python probe
cannot observe per-call work inside a compiled region, so it is kept out of
it (`CompiledBackend.compiled_region_targets`), and two untimed checks, run
after the arm's timed blocks and its recompile gate, stand in for the score
events a timed block of an eager arm must log:

  - **The Kineto twin check** (`kineto_twin_check`). The estimator's kernels
    per layer: the device work launched inside each layer's compiled call
    (`CompiledBackend.annotate`, a host annotation outside the compiled
    function) and outside the block-sparse kernel interface
    (`block_sparse.annotated`), read from a Kineto trace
    (`estimator_structure`). The compiled path's recorded structure is a
    plain forward of the compiled arm, recorded so; its twin is the same
    forward in the untimed twin pass (each forward a new CUDA-graph step,
    `CompiledBackend.twin_pass`). Every layer must run at least one estimator
    kernel in both, and the same kernels in both. With the recompile gate
    (no compile after the warm-up), the timed calls ran the compiled code
    these two forwards ran, so the estimator was computed in every layer of
    every timed call, as the probe would have shown.
  - **The two-prompt mask dependence check** (`mask_dependence_check`). The
    compiled arm's per-layer masks on the cell's prompt and on a second
    prompt of the same length, and the eager arm's on the same two
    (`EagerMasks`). The dependence is the share of causal blocks whose
    selection differs between the two prompts. The compiled masks must
    depend on the prompt (a share above zero: not computed once and reused),
    the eager masks too (else the two prompts test nothing), and the
    compiled share must be the eager share within +-0.01
    (`MASK_DEPENDENCE_TOLERANCE`, the compiled-versus-eager tolerance of
    `densities`, a T4.1 lock value).

`run_checks` is the session's: both checks for one arm, given `time_arms`'
untimed `run`. Not run on a GPU: whether a CUDA graph's kernels appear under
the host annotation of the call that launched it is the first compiled
session's to show.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import torch

from .densities import DENSITY_FLAG_TOLERANCE

MASK_DEPENDENCE_TOLERANCE = DENSITY_FLAG_TOLERANCE


class CompiledCheckError(ValueError):
    """Inputs the checks cannot compare."""


# --- the Kineto twin check ----------------------------------------------------------------------

def estimator_structure(trace) -> dict[int, tuple[str, ...]]:
    """Per layer, the names of the device activities launched inside that
    layer's `COMPILED_CALL` annotation and outside every `KERNEL_INTERFACE`
    one, in device order. `trace` is a Kineto trace (a path or its dict)."""
    from .backends.block_sparse import KERNEL_INTERFACE
    from .backends.compiled import COMPILED_CALL
    from .components import op_sequence
    ev = op_sequence._events(op_sequence.load_trace(trace))
    calls, kernel = [], []
    for e in ev:
        if e.cat != "user_annotation":
            continue
        if e.name.startswith(COMPILED_CALL):
            calls.append((e.ts, e.end, int(e.name[len(COMPILED_CALL):]), e.pid, e.tid))
        elif e.name == KERNEL_INTERFACE:
            kernel.append((e.ts, e.end))
    eps = op_sequence._EPS
    layer_of: dict = {}
    for e in ev:
        if e.cat not in op_sequence._HOST_CATS:
            continue
        corr = op_sequence._int_arg(e.args, "correlation")
        if corr is None:
            continue
        if any(a - eps <= e.ts and e.end <= b + eps for a, b in kernel):
            continue
        for a, b, layer, pid, tid in calls:
            if (e.pid, e.tid) == (pid, tid) and a - eps <= e.ts and e.end <= b + eps:
                layer_of[corr] = layer
                break
    acts = sorted((e.ts, layer_of[c], e.name) for e in ev if e.cat in op_sequence._DEVICE_CATS
                  for c in [op_sequence._int_arg(e.args, "correlation")] if c in layer_of)
    out: dict = {layer: () for _, _, layer, _, _ in calls}
    for _, layer, name in acts:
        out[layer] = out[layer] + (name,)
    return out


def kineto_twin_check(recorded: Mapping[int, Sequence[str]], twin: Mapping[int, Sequence[str]],
                      n_layers: int) -> dict:
    """Every layer `0..n_layers-1` runs at least one estimator kernel in the
    recorded compiled path and in its twin, the same kernels in both; a layer
    outside the model is a problem too."""
    problems = []
    for l in range(n_layers):
        r, t = tuple(recorded.get(l, ())), tuple(twin.get(l, ()))
        if not r:
            problems.append(f"layer {l}: the recorded compiled path ran no estimator kernel")
        if not t:
            problems.append(f"layer {l}: the twin ran no estimator kernel")
        if r and t and sorted(r) != sorted(t):
            problems.append(f"layer {l}: the twin ran {len(t)} estimator kernels "
                            f"({sorted(set(t))[:3]}), the recorded path {len(r)} "
                            f"({sorted(set(r))[:3]})")
    for l in sorted({l for l in (*recorded, *twin) if not 0 <= l < n_layers}):
        problems.append(f"layer {l}: outside the model's {n_layers} layers")
    return {"passed": not problems, "layers": n_layers,
            "kernels_per_layer": {str(l): len(twin.get(l, ())) for l in range(n_layers)},
            "problems": problems}


# --- the two-prompt mask dependence check --------------------------------------------------------

def _causal_disagreement(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> float:
    """The share of causal blocks (query block >= key block), over every layer
    and head, whose selection differs between two per-layer mask lists."""
    if len(xs) != len(ys):
        raise CompiledCheckError(f"{len(xs)} and {len(ys)} layers of masks")
    differ = total = 0
    for x, y in zip(xs, ys):
        if x.shape != y.shape:
            raise CompiledCheckError(f"masks of shape {tuple(x.shape)} and {tuple(y.shape)}")
        nq, nk = x.shape[-2], x.shape[-1]
        causal = torch.ones(nq, nk, dtype=torch.bool, device=x.device).tril(nk - nq)
        d = (x.bool() != y.bool()) & causal
        differ += int(d.sum())
        total += int(causal.sum()) * (x.numel() // (nq * nk))
    if not total:
        raise CompiledCheckError("no causal block to compare")
    return differ / total


def mask_dependence_check(compiled_a, compiled_b, eager_a, eager_b, *,
                          tolerance: float = MASK_DEPENDENCE_TOLERANCE) -> dict:
    """The compiled masks change with the prompt (`compiled_dependence` > 0),
    the eager masks too, and by the same share within `tolerance`
    (`difference`, compiled minus eager). Each path's masks against the
    other's on the same prompt are reported (`compiled_vs_eager`)."""
    dc = _causal_disagreement(compiled_a, compiled_b)
    de = _causal_disagreement(eager_a, eager_b)
    vs = [_causal_disagreement(compiled_a, eager_a), _causal_disagreement(compiled_b, eager_b)]
    diff = dc - de
    problems = []
    if dc == 0:
        problems.append("the compiled masks are the same on both prompts: they do not depend "
                        "on the prompt")
    if de == 0:
        problems.append("the eager masks are the same on both prompts: the two prompts test "
                        "no dependence")
    if abs(diff) > tolerance + 1e-12:
        problems.append(f"the compiled masks change with the prompt by {dc:.4f}, the eager "
                        f"by {de:.4f}: {diff:+.4f}, outside +-{tolerance:g}")
    return {"passed": not problems, "compiled_dependence": dc, "eager_dependence": de,
            "difference": diff, "tolerance": tolerance, "compiled_vs_eager": vs,
            "layers": len(compiled_a), "problems": problems}


class EagerMasks:
    """The eager arm with each forward's block mask kept (`masks`): its
    forward runs the arm's `forward_and_mask`. Everything else is the arm's."""

    def __init__(self, arm):
        self.__dict__.update(arm=arm, masks=[])

    def forward(self, q, k, v, cfg, mask=None):
        out, m = self.arm.forward_and_mask(q, k, v, cfg, mask=mask)
        self.masks.append(m.clone())
        return out

    def __getattr__(self, name):
        return getattr(self.__dict__["arm"], name)


# --- the session's checks ---------------------------------------------------------------------------

def _kineto(fn: Callable[[], None]) -> dict:
    """One call of `fn` under the profiler, CPU and CUDA; the trace's dict."""
    import json
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=acts) as prof:
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "twin_check.json"
        prof.export_chrome_trace(str(path))
        return json.loads(path.read_text())


def run_checks(arm, run: Callable[..., None], *, second_ids, n_layers: int,
               profile: Optional[Callable[[Callable[[], None]], object]] = None) -> dict:
    """Both checks for one compiled arm, untimed, after its timed blocks.
    `run(backend=None, input_ids=None)` is one forward (`time_arms`'), on
    the cell's prompt unless `input_ids` names the second; `profile(fn)`
    returns the Kineto trace of one call of `fn` (`_kineto`)."""
    from .backends import block_sparse
    be = arm.backend
    profile = profile or _kineto
    with be.annotate(), block_sparse.annotated():
        recorded = estimator_structure(profile(lambda: run()))
        with be.twin_pass():
            twin = estimator_structure(profile(lambda: run()))
    masks = {}
    for which, ids in (("a", None), ("b", second_ids)):
        with be.twin_pass(), be.keep_masks() as kept:
            run(input_ids=ids)
        eager = EagerMasks(be.arm)
        run(backend=eager, input_ids=ids)
        masks[which] = (list(kept), eager.masks)
    return {"kineto_twin": kineto_twin_check(recorded, twin, n_layers),
            "mask_dependence": mask_dependence_check(masks["a"][0], masks["b"][0],
                                                     masks["a"][1], masks["b"][1])}
