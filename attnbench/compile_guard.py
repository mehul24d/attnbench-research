"""Refuse results produced by torch.compile's silent eager fallback.

On 2026-09-03 the Stage 1 correctness gate certified `flex` block-sparse
72/72, with genuine bf16-magnitude errors (0.013-0.021), and Stage 2 then
failed the same 72 cells at *verified-identical* geometry twelve minutes
later on the same machine. Neither result was wrong. They were measuring
two different implementations under one backend name.

The mechanism, from `stage1.log`:

    W0903 18:41:46 torch/_dynamo/convert_frame.py:1358] [0/8]
        torch._dynamo hit config.recompile_limit (8)
        function: 'flex_attention'
        last reason: 0/7: tensor 'key' requires_grad mismatch
    torch/nn/attention/flex_attention.py:1687: UserWarning:
        flex_attention called without torch.compile() - this will be slow

Dynamo recompiles per distinct guard set. The probe sweeps hundreds of
configs through one process, so `flex_attention` blew past the default limit
of 8 -- driven here partly by `requires_grad` differing between `fwd` and
`fwd_bwd` cells, not only by shape. Past the limit dynamo stops compiling and
**runs the function eagerly**, which for `flex_attention` means a
score-materialising fallback that handles any block size on any card. So the
gate exercised code the sweep would never run, and passed.

This is the nastiest shape of silent failure the project has hit, because the
fallback is *designed* to preserve correctness. It converts a performance
property into a semantic swap, and it announces itself only in a warning that
Python's default `once` dedup prints a single time no matter how many calls
are affected -- one line in `stage1.log` covering 72 result rows.

Two defences, because either alone is insufficient:

  **Prevention** -- `configure()` raises the recompile limit far above the
  number of distinct shapes any segment produces, so ordinary variety never
  trips the fallback.

  **Detection** -- `guard()` fails closed if it happens anyway. Prevention by
  a tuned constant is exactly the kind of thing a future grid quietly
  outgrows, and the failure it prevents does not announce itself.

Detection is deliberately **sticky at process scope**. The dynamo warning
fires *once*, when the limit is crossed; every later call runs eager in
silence. Per-call detection alone would therefore flag the one cell that
happened to cross the threshold and clear the hundreds after it -- the
precise inversion of what is true. Once this process has been observed
falling back, every subsequent guarded result from it is refused.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch

# Far above the distinct-guard-set count any one segment produces (segment 1's
# band produced 6 for flex), and well below anything that would exhaust memory
# with cached variants. The point is that the limit is never the operative
# constraint; `guard()` is what actually enforces the invariant.
RECOMPILE_LIMIT = 256

# Substrings identifying a dynamo give-up in a log record. Both spellings are
# listed because the config was renamed (`cache_size_limit` -> `recompile_limit`)
# between torch versions this project has run on: 2.9.1 on the L4, 2.13 locally.
_DYNAMO_MARKERS = (
    "recompile_limit",
    "cache_size_limit",
    "falling back to eager",
)

# torch's own admission, emitted per call from flex_attention's entry point.
_WARNING_MARKERS = (
    "called without torch.compile",
)

_LOGGER_NAME = "torch._dynamo"


class CompileFallbackError(RuntimeError):
    """A gated or measured call ran uncompiled.

    Raised rather than warned. A warning here would join the one torch already
    emits, which was present in `stage1.log` and read by nobody.
    """


@dataclass
class FallbackRecord:
    """What was observed during a guarded region, plus process history."""

    reasons: list[str] = field(default_factory=list)

    @property
    def fell_back(self) -> bool:
        return bool(self.reasons)

    @property
    def detail(self) -> str:
        return "compile fallback: " + "; ".join(dict.fromkeys(self.reasons))[:400]


# Process-scoped, and deliberately never cleared except by an explicit
# `reset_process_state()` in tests: see the module docstring on stickiness.
_PROCESS_REASONS: list[str] = []


def reset_process_state() -> None:
    """Clear the sticky flag. For tests only -- never call this in a run."""
    _PROCESS_REASONS.clear()


def process_has_fallen_back() -> bool:
    return bool(_PROCESS_REASONS)


class _DynamoLogCatcher(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.WARNING)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        if any(m in msg for m in _DYNAMO_MARKERS):
            self._sink.append(msg.strip()[:200])


_global_handler: Optional[logging.Handler] = None


def _install_global_catcher() -> None:
    """Record a dynamo give-up anywhere in the process, guarded region or not.

    `guard()` alone would miss a crossing that happens outside one -- during
    Stage 0's `probe()`, say, or any warmup a future caller forgets to wrap.
    That gap matters more than it looks, because torch emits the companion
    "called without torch.compile" warning through its own `_warn_once` set,
    which `simplefilter("always")` cannot defeat: if the FIRST occurrence lands
    outside a guard, every later call is silent and the process flag would
    never be set at all.

    Attaching to the logger instead makes the root event -- dynamo announcing
    it has stopped compiling -- impossible to miss regardless of where it
    happens.
    """
    global _global_handler
    if _global_handler is not None:
        return
    logger = logging.getLogger(_LOGGER_NAME)
    _global_handler = _DynamoLogCatcher(_PROCESS_REASONS)
    logger.addHandler(_global_handler)
    if logger.level > logging.WARNING:
        logger.setLevel(logging.WARNING)


def configure(limit: int = RECOMPILE_LIMIT) -> None:
    """Raise the recompile ceiling. Call once per process, before any measuring.

    Idempotent and safe on any torch version: the config attribute was renamed
    across the versions this project runs on, and a missing attribute is not
    worth taking down a measurement run for -- `guard()` still catches the
    condition this prevents.
    """
    _install_global_catcher()
    cfg = torch._dynamo.config
    for attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, limit)
    # accumulated_* bounds the total across all guard sets; leaving it at the
    # default would let it become the binding limit instead.
    for attr in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, max(getattr(cfg, attr), limit * 8))


@contextlib.contextmanager
def guard() -> Iterator[FallbackRecord]:
    """Observe a region for compile fallback. Does not raise; the caller decides.

    `timing.measure` turns a hit into a failed cell and `gates` into a failed
    check, rather than raising through the sweep -- one contaminated cell is
    data about that cell, not a reason to lose the session's other results.

    Warnings are captured with `simplefilter("always")` to defeat the default
    `once` dedup (which is why 72 affected rows produced a single log line),
    then **re-emitted** on exit so nothing that would have reached the session
    log is swallowed by the act of checking for it.
    """
    record = FallbackRecord(reasons=list(_PROCESS_REASONS))

    fresh: list[str] = []          # what this region newly observed
    logger = logging.getLogger(_LOGGER_NAME)
    handler = _DynamoLogCatcher(fresh)
    logger.addHandler(handler)
    restore_level = None
    if logger.level > logging.WARNING:
        restore_level, logger.level = logger.level, logging.WARNING

    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught_list:
            warnings.simplefilter("always")
            caught = caught_list        # bound before yield, so a raising
            yield record                # body still surrenders its warnings
    finally:
        logger.removeHandler(handler)
        if restore_level is not None:
            logger.level = restore_level

        for w in caught:
            text = str(w.message)
            if any(m in text for m in _WARNING_MARKERS):
                fresh.append(text.strip()[:200])
            else:
                # Not ours: put it back so the session log is unchanged by the
                # fact that a guard was watching.
                warnings.warn_explicit(w.message, w.category, w.filename,
                                       w.lineno)

        for r in fresh:
            if r not in _PROCESS_REASONS:
                _PROCESS_REASONS.append(r)
            if r not in record.reasons:
                record.reasons.append(r)


def raise_if_fallen_back(record: FallbackRecord, context: str = "") -> None:
    """Convert a record into an exception, for callers that want fail-fast."""
    if record.fell_back:
        raise CompileFallbackError(f"{context}: {record.detail}" if context
                                   else record.detail)
