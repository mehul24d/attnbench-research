"""Compiled variants of an arm under `torch.compile` (T2.21; plan Section 5.5).

`CompiledBackend(arm, setting)` wraps an arm's layer forward in
`torch.compile(mode="reduce-overhead")`, which uses CUDA graphs where the
graph allows. The authors' code is wrapped, never edited (rule 8). Two
settings, as the plan names them:

  - "default": every Dynamo setting at its default;
  - "capture": the capture options for scalar outputs and data-dependent
    shapes (`capture_scalar_outputs`, `capture_dynamic_output_shape_ops`),
    applied while each call traces.

Each variant is its own code path (K8), named `<arm>@compile-<setting>`, and
is never pooled with the eager arm.

**One setting per process.** Dynamo caches compiled code per Python code
object and does not guard on its own settings. Measured on the CPU with
PyTorch 2.14.1 (25 Oct 2026): a function compiled under the default
settings, then compiled again under `capture_scalar_outputs=True` in the same
process, reused the first compile, `.item()` break and all; in a fresh
process the capture setting gave one graph. A second variant in one process
would therefore silently be the first. So this module refuses a second
setting in a process until `reset_compiled_state()` has cleared Dynamo's
caches, and the harness interleaves compiled variants across processes, not
within one.

**Compile time and recompiles stay out of timed blocks.** Every call counts
the graphs Dynamo compiled during it; a call that compiled anything is a
compile event, with its wall time. `timed_block()` wraps a timed block and
reports any graph compiled inside it, which flags the block (Section 5.6).
`compile_guard` raises the recompile limit (the XAttention arm's per-layer
threshold is read from `layer_idx`, so each layer is its own guard set) and
refuses results after Dynamo has fallen back to eager.

**The graph-break report** (`graph_break_report`) runs
`torch._dynamo.explain` on the arm's forward for each layer and records the
graph count, the breaks and where each break happened. It runs wherever the
arm runs. XAttention's estimator moves its inputs to "cuda" itself (upstream
`xattn_estimate`), so its report needs a CUDA device; dense SDPA (math) and
the block-masked kernel run on a CPU.

**Mask agreement** (`MaskAgreement`) compares the compiled estimator's mask
with the eager one in every layer of every recorded prompt. A variant whose
masks differ on any prompt is reported as a different arm, with its share of
agreeing layers.

Not here: the mean-pool arm builds its mask in `accuracy/model.py`
(`SwappedAttention.forward`), outside any backend, so wrapping a backend
does not compile its estimator; and the vertical-slash arm is not built
(T1.7).
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import torch

from .. import compile_guard
from ..config import AttnConfig
from .impls import _expand_kv

MODE = "reduce-overhead"
SETTINGS: dict[str, dict[str, bool]] = {
    "default": {},
    "capture": {"capture_scalar_outputs": True,
                "capture_dynamic_output_shape_ops": True},
}


class CompiledVariantError(RuntimeError):
    """A compiled variant that would not be the variant it claims to be."""


_active_setting: Optional[str] = None


def reset_compiled_state() -> None:
    """Clear Dynamo's caches, so another setting can compile from scratch."""
    global _active_setting
    torch._dynamo.reset()
    _active_setting = None


def _claim(setting: str) -> None:
    global _active_setting
    if _active_setting is not None and _active_setting != setting:
        raise CompiledVariantError(
            f"this process already compiled with the {_active_setting!r} setting; "
            f"a {setting!r} variant would reuse that compiled code. Run it in "
            f"another process, or call reset_compiled_state() first")
    _active_setting = setting


def graphs_compiled() -> int:
    """Graphs Dynamo has compiled in this process so far."""
    from torch._dynamo.utils import counters
    return int(counters["stats"]["unique_graphs"])


def _compile(fn: Callable, mode: Optional[str], dynamo_backend: str) -> Callable:
    """`torch.compile`; a mode applies to inductor only (the CPU tests use
    Dynamo's "eager" backend, which traces the same graphs, to stay fast)."""
    kwargs = {"backend": dynamo_backend}
    if mode is not None:
        if dynamo_backend != "inductor":
            raise ValueError(f"mode {mode!r} needs the inductor backend")
        kwargs["mode"] = mode
    return torch.compile(fn, **kwargs)


def variant_name(arm: str, setting: str) -> str:
    return f"{arm}@compile-{setting}"


@dataclass(frozen=True)
class CompileEvent:
    layer: Optional[int]
    graphs: int
    seconds: float


@dataclass
class TimedBlock:
    graphs_before: int
    graphs_inside: int = 0

    @property
    def flagged(self) -> bool:
        return self.graphs_inside > 0


class CompiledBackend:
    """An arm's layer forward under `torch.compile` (module docstring)."""

    def __init__(self, arm, setting: str, *, mode: Optional[str] = MODE,
                 dynamo_backend: str = "inductor"):
        if setting not in SETTINGS:
            raise ValueError(f"setting {setting!r}; expected one of {sorted(SETTINGS)}")
        if isinstance(arm, CompiledBackend):
            raise TypeError("an arm is compiled once")
        _claim(setting)
        compile_guard.configure()
        self.arm, self.setting, self.mode = arm, setting, mode
        self.name = variant_name(arm.capability.name, setting)
        self.layer_idx: Optional[int] = None
        self.events: list[CompileEvent] = []
        self._forward = _compile(arm.forward, mode, dynamo_backend)

    # The model wrapper's hooks, passed to the eager arm.
    def set_layer(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        if hasattr(self.arm, "set_layer"):
            self.arm.set_layer(layer_idx)

    def reset_density(self) -> None:
        self.arm.reset_density()

    @property
    def last_layer_density(self):
        return self.arm.last_layer_density

    def _call(self, fn: Callable, *args, **kwargs):
        before = graphs_compiled()
        start = time.perf_counter()
        with torch._dynamo.config.patch(**SETTINGS[self.setting]), \
                compile_guard.guard() as record:
            out = fn(*args, **kwargs)
        seconds = time.perf_counter() - start
        compile_guard.raise_if_fallen_back(record, self.name)
        made = graphs_compiled() - before
        if made:
            self.events.append(CompileEvent(self.layer_idx, made, seconds))
        return out

    def forward(self, q, k, v, cfg: AttnConfig, mask=None):
        return self._call(self._forward, q, k, v, cfg, mask=mask)

    @property
    def compile_seconds(self) -> float:
        """Wall time of every call that compiled, reported apart from timing."""
        return sum(e.seconds for e in self.events)

    @property
    def recompiles(self) -> int:
        """Compile events after the first."""
        return max(0, len(self.events) - 1)

    @contextlib.contextmanager
    def timed_block(self) -> Iterator[TimedBlock]:
        """Wrap a timed block; `flagged` if any graph compiled inside it."""
        block = TimedBlock(graphs_compiled())
        yield block
        block.graphs_inside = graphs_compiled() - block.graphs_before


# --- the graph-break report --------------------------------------------------

@dataclass(frozen=True)
class GraphBreak:
    reason: str
    where: tuple[str, ...]      # "file.py:line in function", innermost last


@dataclass(frozen=True)
class LayerBreaks:
    layer: int
    graphs: int
    breaks: tuple[GraphBreak, ...]


def _where(frames) -> tuple[str, ...]:
    return tuple(f"{Path(f.filename).name}:{f.lineno} in {f.name}" for f in frames or ())


def graph_break_report(arm, layers: Sequence[tuple[int, tuple, AttnConfig]],
                       setting: str) -> tuple[LayerBreaks, ...]:
    """Per layer, `torch._dynamo.explain` of the arm's forward under the
    setting. `layers` holds (layer index, (q, k, v[, mask]), cfg).

    Refused while a compiled variant is active in the process, since it
    clears Dynamo's caches before each layer. It claims no setting. The final
    reset covers PyTorch versions whose `explain` leaves its compiled code
    cached (2.14.1 does not)."""
    if setting not in SETTINGS:
        raise ValueError(f"setting {setting!r}; expected one of {sorted(SETTINGS)}")
    if _active_setting is not None:
        raise CompiledVariantError(
            f"a {_active_setting!r} compiled variant is active in this process; "
            f"run the report in a process of its own")
    out = []
    try:
        for layer, inputs, cfg in layers:
            if hasattr(arm, "set_layer"):
                arm.set_layer(layer)
            torch._dynamo.reset()
            with torch._dynamo.config.patch(**SETTINGS[setting]):
                e = torch._dynamo.explain(arm.forward)(*inputs[:3], cfg,
                                                       mask=inputs[3] if len(inputs) > 3 else None)
            out.append(LayerBreaks(layer, int(e.graph_count), tuple(
                GraphBreak(str(r.reason).strip().splitlines()[0], _where(r.user_stack))
                for r in e.break_reasons)))
    finally:
        torch._dynamo.reset()
    return tuple(out)


def same_report(a: Sequence[LayerBreaks], b: Sequence[LayerBreaks]) -> bool:
    """Whether two variants split identically in every layer. The plan
    times only one of two compiled variants that T2.21 shows identical."""
    return [(x.layer, x.graphs, x.breaks) for x in a] == \
        [(y.layer, y.graphs, y.breaks) for y in b]


# --- mask agreement with the eager arm ----------------------------------------

@dataclass(frozen=True)
class MaskRecord:
    prompt: int
    layer: Optional[int]
    equal: bool


class MaskAgreement:
    """The compiled estimator's mask against the eager one, per layer.

    For arms with an `estimate` (both XAttention arms), the estimate is
    compiled under the same setting as the variant. Use it as the model's
    prefill backend over the recorded prompts, calling `next_prompt()`
    before each; it returns the eager output, so every layer sees the eager
    arm's hidden states. Untimed.
    """

    def __init__(self, arm, setting: str, *, mode: Optional[str] = MODE,
                 dynamo_backend: str = "inductor"):
        if not hasattr(arm, "estimate"):
            raise TypeError(f"{arm.capability.name} has no estimator of its own "
                            f"to compare")
        self.compiled = CompiledBackend(arm, setting, mode=mode, dynamo_backend=dynamo_backend)
        self._estimate = _compile(arm.estimate, mode, dynamo_backend)
        self.arm = arm
        self.prompt = -1
        self.records: list[MaskRecord] = []

    def next_prompt(self) -> None:
        self.prompt += 1

    def set_layer(self, layer_idx: int) -> None:
        self.compiled.set_layer(layer_idx)

    def reset_density(self) -> None:
        self.arm.reset_density()

    @property
    def last_layer_density(self):
        return self.arm.last_layer_density

    def forward(self, q, k, v, cfg: AttnConfig, mask=None):
        if self.prompt < 0:
            raise RuntimeError("call next_prompt() before each recorded prompt")
        k_rep, _ = _expand_kv(k, v, cfg)
        eager = self.arm.estimate(q, k_rep)
        compiled = self.compiled._call(self._estimate, q, k_rep)
        self.records.append(MaskRecord(self.prompt, self.compiled.layer_idx,
                                       torch.equal(eager, compiled)))
        return self.arm.forward(q, k, v, cfg, mask=mask)

    @property
    def agreement_share(self) -> float:
        if not self.records:
            raise RuntimeError("no layers recorded")
        return sum(r.equal for r in self.records) / len(self.records)

    @property
    def same_arm(self) -> bool:
        """False if the masks differ on any prompt: then the variant is a
        different arm (Section 5.5)."""
        return bool(self.records) and all(r.equal for r in self.records)

    def differing_prompts(self) -> tuple[int, ...]:
        return tuple(sorted({r.prompt for r in self.records if not r.equal}))
