"""A tape of every host-device interaction in one prefill, so the prefill
can be captured as a CUDA graph (T2.4).

A CUDA graph cannot contain a synchronisation, and the arms synchronise:
an `assert` or `bool()` on a GPU tensor, `.item()`, sizing from `nonzero()`,
a copy to the host (Section 5.1). They also build small device tensors from
Python values (`torch.tensor([0, s], device=...)` in both block-sparse call
sites), which is a host-to-device copy that waits. The device replay
(Section 5.1: "the replay contains no host, link or synchronisation cost")
needs the arm's device work without any of these.

So the replay runs in two passes:

  1. `TapeRecorder`, in an untimed eager pass on the recorded prompt, lets
     every call through and writes down what each interaction returned: the
     value read, the size a data-dependent operation produced, the host data
     a device tensor was built from.
  2. `TapeReplayer`, active while the graph is captured, serves those
     answers in order instead of waiting. Device tensors built from host data
     are staged on the device before capture (`SyncTape.stage`), so the
     graph holds no host-to-device copy. Sizing from `nonzero()` and boolean
     indexing become `torch.nonzero_static` with the recorded size, which
     runs on the device without reading back. An explicit synchronise becomes
     a no-op.

Every replayed interaction is checked against the tape: same kind, same
function, same input shape and dtype. Anything else raises `TapeMismatch`,
because it means the replayed prefill took another path than the recorded
one, and its device costs would belong to a different sequence.

**What it intercepts.** `TorchFunctionMode` sees the reads `Tensor.__bool__`
(which an `assert` calls), `__int__`, `__float__`, `__index__`, `item`,
`tolist`, `torch.equal` and `torch.allclose`, and `cpu`, `to`, `copy_`,
`nonzero`, `torch.where(cond)`, indexing, `torch.tensor` and
`torch.as_tensor` (each checked on PyTorch 2.8 and 2.14 on CPU);
`torch.cuda.synchronize` is patched. A boolean index that is not a single
bool tensor, and the operations in `UNHANDLED`, which size their output from
the data (`masked_select`, `unique`, `unique_consecutive`, `bincount`,
`repeat_interleave` with tensor repeats and no `output_size`), are not
served: the recorder writes them down as "unhandled", and the replay refuses
at that point rather than letting the capture fail somewhere less clear.
A wait the mode cannot see (`torch.cuda.Event.synchronize`, a read inside a
C++ extension) is not on the tape; the CUDA-graph capture itself refuses a
synchronisation, so such a path fails at capture rather than being timed.

Replacements change the kernels: `nonzero_static` is not `nonzero`, and a
host-to-device `copy_` into an existing device tensor becomes a
device-to-device copy from the staged tensor. `SyncTape.substitutions`
counts them so the replay-versus-trace check can name them.

`device_type` is a parameter so the bookkeeping is tested on CPU tensors.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from torch.overrides import TorchFunctionMode

SCHEMA = "attnbench.sync_tape/1"
READS = ("__bool__", "__int__", "__float__", "__index__", "item", "tolist",
         "equal", "allclose")
UNHANDLED = ("masked_select", "unique", "unique_consecutive", "bincount")
SUBSTITUTED_KINDS = ("nonzero", "nonzero_tuple", "bool_index", "bool_setitem",
                     "to_device_inplace")


class TapeMismatch(RuntimeError):
    """The replayed prefill departed from the recorded one."""


@dataclass
class TapeEntry:
    kind: str
    name: str
    sig: tuple
    value: Any = None
    note: str = ""


def _on(t, device_type: str) -> bool:
    return isinstance(t, torch.Tensor) and t.device.type == device_type


def _as_device(d) -> Optional[torch.device]:
    if isinstance(d, torch.device):
        return d
    if isinstance(d, str):
        return torch.device(d)
    if isinstance(d, int) and not isinstance(d, bool):
        return torch.device("cuda", d)
    return None


def _to_target(args, kwargs) -> Optional[torch.device]:
    """The device `Tensor.to(...)` moves to, or None if it keeps the device."""
    if "device" in kwargs:
        return _as_device(kwargs["device"])
    for a in args[1:]:
        if isinstance(a, torch.Tensor):
            return a.device
        d = _as_device(a)
        if d is not None:
            return d
    return None


def _sig(t) -> tuple:
    return (tuple(t.shape), str(t.dtype)) if isinstance(t, torch.Tensor) else (type(t).__name__,)


def _single_bool_index(index, device_type) -> Optional[torch.Tensor]:
    if isinstance(index, tuple) and len(index) == 1:
        index = index[0]
    if _on(index, device_type) and index.dtype in (torch.bool, torch.uint8):
        return index
    return None


def _has_bool_index(index, device_type) -> bool:
    parts = index if isinstance(index, tuple) else (index,)
    return any(_on(p, device_type) and p.dtype in (torch.bool, torch.uint8) for p in parts)


def classify(func, args, kwargs, device_type: str) -> Optional[str]:
    """The kind of interaction a call is, or None if it is ordinary device
    work the graph can hold."""
    name = getattr(func, "__name__", "")
    self = args[0] if args else None
    if name in READS and _on(self, device_type):
        return "read"
    if name in UNHANDLED and _on(self, device_type):
        return "unhandled"
    if name == "repeat_interleave" and "output_size" not in kwargs and \
            (len(args) == 1 and _on(self, device_type) or
             len(args) >= 2 and _on(args[1], device_type)):
        return "unhandled"
    if name == "cpu" and _on(self, device_type):
        return "to_host"
    if name == "to" and isinstance(self, torch.Tensor):
        target = _to_target(args, kwargs)
        if target is not None:
            if _on(self, device_type) and target.type != device_type:
                return "to_host"
            if not _on(self, device_type) and target.type == device_type:
                return "to_device"
        return None
    if name == "copy_" and len(args) >= 2:
        dst, src = args[0], args[1]
        if _on(src, device_type) and not _on(dst, device_type):
            return "to_host_inplace"
        if _on(dst, device_type) and not _on(src, device_type):
            return "to_device_inplace"
        return None
    if name == "nonzero" and _on(self, device_type):
        return "nonzero_tuple" if kwargs.get("as_tuple") else "nonzero"
    if name == "where" and len(args) == 1 and not kwargs and _on(self, device_type):
        return "nonzero_tuple"
    if name in ("__getitem__", "__setitem__") and len(args) >= 2 and \
            _has_bool_index(args[1], device_type):
        if _single_bool_index(args[1], device_type) is None:
            return "unhandled"
        return "bool_index" if name == "__getitem__" else "bool_setitem"
    if name in ("tensor", "as_tensor"):
        d = _as_device(kwargs.get("device"))
        if d is not None and d.type == device_type and not _on(self, device_type):
            return "to_device"
    return None


@dataclass
class SyncTape:
    device_type: str
    entries: list[TapeEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def substitutions(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.entries:
            if e.kind in SUBSTITUTED_KINDS:
                out[e.kind] = out.get(e.kind, 0) + 1
        return out

    @property
    def unhandled(self) -> list[int]:
        return [i for i, e in enumerate(self.entries) if e.kind == "unhandled"]

    def stage(self, device) -> dict[int, torch.Tensor]:
        """Device copies of every host value a device tensor is built from,
        made now, so the graph captured afterwards holds no such copy."""
        return {i: e.value.to(device) for i, e in enumerate(self.entries)
                if e.kind in ("to_device", "to_device_inplace")}

    def save(self, path) -> Path:
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} exists; a tape is never overwritten")
        torch.save({"schema": SCHEMA, "device_type": self.device_type,
                    "entries": [{"kind": e.kind, "name": e.name, "sig": list(e.sig),
                                 "value": e.value, "note": e.note} for e in self.entries]},
                   path)
        return path

    @classmethod
    def load(cls, path) -> "SyncTape":
        d = torch.load(path, weights_only=True)
        if d.get("schema") != SCHEMA:
            raise TapeMismatch(f"schema {d.get('schema')!r}, expected {SCHEMA!r}")
        return cls(d["device_type"], [TapeEntry(e["kind"], e["name"], _tuplify(e["sig"]),
                                                e["value"], e["note"]) for e in d["entries"]])


def _tuplify(x):
    return tuple(_tuplify(v) for v in x) if isinstance(x, (list, tuple)) else x


def _input_sig(kind, func, args) -> tuple:
    if kind in ("to_host_inplace", "to_device_inplace"):
        return _sig(args[0]) + _sig(args[1])
    if kind in ("bool_index", "bool_setitem") or \
            kind == "unhandled" and getattr(func, "__name__", "") in ("__getitem__", "__setitem__"):
        idx = args[1]
        return _sig(args[0]) + (tuple(_sig(p) for p in (idx if isinstance(idx, tuple) else (idx,))),)
    return _sig(args[0]) if args else ()


class _Patched:
    """torch.cuda.synchronize, replaced for the life of a mode."""

    def __init__(self, on_call):
        self.on_call = on_call
        self.original = torch.cuda.synchronize

    def __enter__(self):
        torch.cuda.synchronize = self.on_call
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize = self.original


class TapeRecorder(TorchFunctionMode):
    """Pass every call through and write down every interaction."""

    def __init__(self, device_type: str = "cuda"):
        super().__init__()
        self.tape = SyncTape(device_type)
        self._sync = _Patched(self._synchronize)

    def _synchronize(self, *args, **kwargs):
        self.tape.entries.append(TapeEntry("synchronize", "synchronize", ()))
        if self.tape.device_type == "cuda":
            return self._sync.original(*args, **kwargs)

    def __enter__(self):
        self._sync.__enter__()
        return super().__enter__()

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self._sync.__exit__(*exc)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        dt = self.tape.device_type
        kind = classify(func, args, kwargs, dt)
        out = func(*args, **kwargs)
        if kind is None:
            return out
        name = getattr(func, "__name__", str(func))
        sig = _input_sig(kind, func, args)
        if kind == "read":
            value = out
        elif kind == "to_host":
            value = out.clone()
        elif kind == "to_host_inplace":
            value = args[0].clone()
        elif kind == "to_device":
            value = out.detach().cpu().clone()
        elif kind == "to_device_inplace":
            value = args[1].detach().clone()
        elif kind in ("nonzero", "nonzero_tuple"):
            value = int((out[0] if isinstance(out, tuple) else out).shape[0])
        elif kind == "bool_index":
            value = int(out.shape[0])
        elif kind == "bool_setitem":
            value = int(_single_bool_index(args[1], dt).sum())
        else:   # unhandled
            value = None
        note = "" if kind != "unhandled" else (
            f"{name} sizes its output from the data: the replay cannot serve it"
            if name in UNHANDLED or name == "repeat_interleave" else
            f"{name} with a boolean index that is not one bool tensor: the replay "
            f"cannot serve it")
        self.tape.entries.append(TapeEntry(kind, name, sig, value, note))
        return out


class TapeReplayer(TorchFunctionMode):
    """Serve the tape's answers in order; refuse any departure from it."""

    def __init__(self, tape: SyncTape, staged: Optional[dict[int, torch.Tensor]] = None):
        super().__init__()
        self.tape = tape
        self.staged = staged if staged is not None else {}
        self.pos = 0
        self._sync = _Patched(self._synchronize)

    def _next(self, kind: str, name: str, sig: tuple) -> tuple[int, TapeEntry]:
        if self.pos >= len(self.tape.entries):
            raise TapeMismatch(f"interaction {self.pos} ({kind} {name}) is past the end "
                               f"of the tape ({len(self.tape.entries)} entries)")
        i, e = self.pos, self.tape.entries[self.pos]
        if e.kind == "unhandled":
            raise TapeMismatch(f"interaction {i}: {e.note}")
        if (e.kind, e.name, e.sig) != (kind, name, sig):
            raise TapeMismatch(f"interaction {i}: replayed {kind} {name} {sig}, "
                               f"recorded {e.kind} {e.name} {e.sig}")
        self.pos += 1
        return i, e

    def _synchronize(self, *args, **kwargs):
        self._next("synchronize", "synchronize", ())

    def __enter__(self):
        self._sync.__enter__()
        return super().__enter__()

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self._sync.__exit__(*exc)

    def finish(self) -> None:
        if self.pos != len(self.tape.entries):
            raise TapeMismatch(f"the replay used {self.pos} of {len(self.tape.entries)} "
                               f"recorded interactions; it took a shorter path")

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        dt = self.tape.device_type
        kind = classify(func, args, kwargs, dt)
        if kind is None:
            return func(*args, **kwargs)
        name = getattr(func, "__name__", str(func))
        i, e = self._next(kind, name, _input_sig(kind, func, args))
        if kind == "read":
            return copy.deepcopy(e.value)
        if kind == "to_host":
            return e.value.clone()
        if kind == "to_host_inplace":
            return args[0].copy_(e.value)
        if kind == "to_device":
            if i not in self.staged:
                raise TapeMismatch(f"interaction {i}: no staged device tensor; call "
                                   f"SyncTape.stage before capture")
            return self.staged[i]
        if kind == "to_device_inplace":
            if i not in self.staged:
                raise TapeMismatch(f"interaction {i}: no staged device tensor")
            return args[0].copy_(self.staged[i])
        if kind in ("nonzero", "nonzero_tuple"):
            idx = torch.nonzero_static(args[0], size=e.value)
            return tuple(idx.unbind(1)) if kind == "nonzero_tuple" else idx
        mask = _single_bool_index(args[1], dt)
        idx = tuple(torch.nonzero_static(mask, size=e.value).unbind(1))
        if kind == "bool_index":
            return args[0][idx]
        args[0][idx] = args[2]       # bool_setitem
        return None
