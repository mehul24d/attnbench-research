"""The estimator's selection: masks captured at the kernel and banked (T2.4).

Plan Section 5.1 counts "the estimator's selection (its masks on each
recorded prompt)" as part of an arm's recorded artifact, and the device
replay (Section 5.3 step 5) runs "with each prompt's recorded mask". This
module takes the masks where they are used, at the block-sparse kernel, and
writes them to disk.

**Where the masks are taken.** Every sparse arm reaches one kernel,
`block_sparse_attn.block_sparse_attn_func`, and all three call sites
(`kernels.bsa_prefill`, `backends/xattention.py`, `backends/block_sparse.py`)
import it from its module at call time. `MaskCapture` replaces that module
attribute for the duration of a block, so it sees every call without editing
any call site (the XAttention call is the authors' and is not edited,
rule 8). The mask banked is `base_blockmask`, the tensor the kernel was
handed, one per call, which is one per layer per prefill.

**No synchronisation inside the region.** During a recorded run the capture
keeps a reference to each mask and copies nothing, so it adds no device work
and no wait to the trace it sits in. The masks are moved to the host only by
`bank`, after the region. A reference is safe only if nothing writes the mask
in place afterwards. An untraced pass can afford a copy, so
`MaskCapture(copy=True)` clones each mask as the kernel receives it, and
`recorder.record_arm_prefill` compares the traced run's references bitwise
with those copies: a buffer reused across layers shows its last contents in
every reference and fails the comparison.

**Substitution.** At replay, `MaskCapture(substitute=...)` hands the kernel
the banked mask for each call instead of the one the estimator produced, so
the sparse kernel runs on exactly the recorded selection while the
estimator's own kernels still run. The estimator's live masks are kept too,
for the agreement check.

Only `head_mask_type` all ones is accepted, the one value this study passes
(`kernels.py`): each head then runs its own mask.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from .op_sequence import CodePath

KERNEL_MODULE = "block_sparse_attn"
KERNEL_NAME = "block_sparse_attn_func"
# block_sparse_attn_func(q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
# streaming_info, base_blockmask, ...), as kernels.py records it.
_HEAD_MASK_TYPE_ARG = 5
_BASE_BLOCKMASK_ARG = 7
SCHEMA = "attnbench.mask_bank/1"


class MaskBankError(ValueError):
    """Masks that cannot be banked, loaded or substituted as asked."""


def _arg(args, kwargs, name, pos):
    if name in kwargs:
        return kwargs[name]
    if len(args) > pos:
        return args[pos]
    raise MaskBankError(f"{KERNEL_NAME} called without {name}")


def _with_arg(args, kwargs, name, pos, value):
    if name in kwargs:
        return args, {**kwargs, name: value}
    args = list(args)
    args[pos] = value
    return tuple(args), kwargs


@dataclass
class CapturedCall:
    mask: torch.Tensor            # what the kernel was handed
    live: torch.Tensor            # what the estimator produced (same as mask
                                  # unless substituted)
    causal: bool
    head_mask_type: torch.Tensor


class MaskCapture:
    """Record (and optionally substitute) the mask of every kernel call.

    `substitute` is one bool tensor per expected call, on the kernel's
    device; call i gets substitute[i]. A call beyond the substitutes, or a
    substitute whose shape differs from the live mask, is refused. `copy`
    clones each mask at the call instead of keeping a reference; it adds a
    device copy per call, so it is for untraced passes only. `optional`
    makes a missing kernel module a capture of nothing (a dense arm on a
    machine without the library), instead of an ImportError. `into` is one
    buffer per expected call: call i's live mask is copied into into[i] (an
    operation of the call, so a CUDA graph writes it at a replay, not at the
    capture), and that buffer is what the call keeps as its live mask.
    """

    def __init__(self, module=None, substitute: Optional[Sequence[torch.Tensor]] = None,
                 copy: bool = False, optional: bool = False,
                 into: Optional[Sequence[torch.Tensor]] = None):
        self._module = module
        self._copy = copy
        self.into = list(into) if into is not None else None
        self._optional = optional
        self._original = None
        self.substitute = list(substitute) if substitute is not None else None
        self.calls: list[CapturedCall] = []

    def __enter__(self) -> "MaskCapture":
        mod = self._module or sys.modules.get(KERNEL_MODULE)
        if mod is None:
            import importlib
            try:
                mod = importlib.import_module(KERNEL_MODULE)
            except ImportError:
                if not self._optional:
                    raise
                return self
        self._module = mod
        self._original = getattr(mod, KERNEL_NAME)
        original = self._original

        def capturing(*args, **kwargs):
            live = _arg(args, kwargs, "base_blockmask", _BASE_BLOCKMASK_ARG)
            hmt = _arg(args, kwargs, "head_mask_type", _HEAD_MASK_TYPE_ARG)
            mask = live
            if self.substitute is not None:
                i = len(self.calls)
                if i >= len(self.substitute):
                    raise MaskBankError(
                        f"kernel call {i} has no banked mask ({len(self.substitute)} "
                        f"banked); the replay ran more sparse layers than the recording")
                mask = self.substitute[i]
                if mask.shape != live.shape or mask.dtype != live.dtype or \
                        mask.device != live.device:
                    raise MaskBankError(
                        f"call {i}: banked mask {tuple(mask.shape)} {mask.dtype} on "
                        f"{mask.device}, live mask {tuple(live.shape)} {live.dtype} "
                        f"on {live.device}")
                args, kwargs = _with_arg(args, kwargs, "base_blockmask",
                                         _BASE_BLOCKMASK_ARG, mask)
            kept, kept_live = (mask.clone(), live.clone()) if self._copy else (mask, live)
            if self.into is not None:
                i = len(self.calls)
                if i >= len(self.into) or self.into[i].shape != live.shape:
                    raise MaskBankError(f"kernel call {i} has no live-mask buffer of shape "
                                        f"{tuple(live.shape)}")
                self.into[i].copy_(live)
                kept_live = self.into[i]
            self.calls.append(CapturedCall(kept, kept_live,
                                           bool(kwargs.get("is_causal", False)), hmt))
            return original(*args, **kwargs)

        setattr(mod, KERNEL_NAME, capturing)
        return self

    def __exit__(self, *exc) -> None:
        if self._module is not None:
            setattr(self._module, KERNEL_NAME, self._original)

    def masks(self) -> list[torch.Tensor]:
        return [c.mask for c in self.calls]

    def live_masks(self) -> list[torch.Tensor]:
        return [c.live for c in self.calls]


# --- the bank on disk ----------------------------------------------------------

def _valid(nq: int, nk: int, causal: bool) -> np.ndarray:
    v = np.ones((nq, nk), dtype=bool)
    return np.tril(v, k=nk - nq) if causal else v


def density(mask: np.ndarray, causal: bool) -> float:
    """Kept blocks over valid blocks times leading dims (Section 5.5's
    convention, as `backends.xattention.density_tensor` has it)."""
    nq, nk = mask.shape[-2:]
    valid = _valid(nq, nk, causal)
    lead = mask.size // (nq * nk)
    return float((mask & valid).sum()) / float(valid.sum() * lead)


@dataclass(frozen=True)
class BankedMask:
    shape: tuple[int, ...]
    causal: bool
    density: float
    sha256: str
    packed: np.ndarray = field(compare=False)   # np.packbits of the flattened
                                                # mask; sha256 is its identity

    def array(self) -> np.ndarray:
        n = int(np.prod(self.shape))
        return np.unpackbits(self.packed, count=n).astype(bool).reshape(self.shape)

    def tensor(self, device="cpu") -> torch.Tensor:
        return torch.from_numpy(self.array()).to(device).contiguous()


@dataclass(frozen=True)
class MaskRecord:
    code_path: CodePath
    context_length: int
    batch: int
    prompt_id: str
    masks: tuple[BankedMask, ...]

    @property
    def sha256(self) -> str:
        return hashlib.sha256("".join(m.sha256 for m in self.masks).encode()).hexdigest()

    def tensors(self, device="cpu") -> list[torch.Tensor]:
        return [m.tensor(device) for m in self.masks]


def _bank_one(t: torch.Tensor, causal: bool) -> BankedMask:
    if t.dtype != torch.bool:
        raise MaskBankError(f"masks are bool; got {t.dtype}")
    a = t.detach().cpu().numpy()
    packed = np.packbits(a.reshape(-1))
    return BankedMask(tuple(a.shape), causal, density(a, causal),
                      hashlib.sha256(packed.tobytes()).hexdigest(), packed)


def from_capture(capture: MaskCapture, *, code_path: CodePath, context_length: int,
                 batch: int, prompt_id: str) -> MaskRecord:
    """Move every captured mask to the host (after the region) and check the
    kernel arguments the bank depends on."""
    if not capture.calls:
        raise MaskBankError("no block-sparse kernel call was captured")
    for i, c in enumerate(capture.calls):
        if bool((c.head_mask_type != 1).any()):
            raise MaskBankError(f"call {i}: head_mask_type is not all ones")
    return MaskRecord(code_path, int(context_length), int(batch), str(prompt_id),
                      tuple(_bank_one(c.mask, c.causal) for c in capture.calls))


def _code_path_dict(cp: CodePath) -> dict:
    return {"arm": cp.arm, "gpu": cp.gpu, "implementation": cp.implementation,
            "libraries": [list(x) for x in cp.libraries],
            "python_optimize": cp.python_optimize}


def calls_sha256(calls) -> str:
    """sha256 over the manifest's call entries (shape, causal, density, mask
    hash), canonical JSON: the masks' own hashes do not cover the density and
    causal fields (audit C4, finding 2)."""
    blob = json.dumps(list(calls), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def save(record: MaskRecord, path) -> tuple[Path, Path]:
    """`<path>.npz` (packed masks) and `<path>.json` (manifest); both paths."""
    path = Path(path)
    npz, manifest = path.with_suffix(".npz"), path.with_suffix(".json")
    if npz.exists() or manifest.exists():
        raise MaskBankError(f"{path} is already banked; masks are never overwritten")
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"call_{i}": m.packed for i, m in enumerate(record.masks)})
    npz.write_bytes(buf.getvalue())
    calls = [{"shape": list(m.shape), "causal": m.causal, "density": m.density,
              "sha256": m.sha256} for m in record.masks]
    manifest.write_text(json.dumps({
        "schema": SCHEMA, "code_path": _code_path_dict(record.code_path),
        "context_length": record.context_length, "batch": record.batch,
        "prompt_id": record.prompt_id, "sha256": record.sha256,
        "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "calls": calls, "calls_sha256": calls_sha256(calls)}, indent=1))
    return npz, manifest


def load(path) -> MaskRecord:
    """Read a bank back, checking every sha256 it records, the hash over the
    call entries, and each call's density against its own mask."""
    path = Path(path)
    npz, manifest = path.with_suffix(".npz"), path.with_suffix(".json")
    meta = json.loads(manifest.read_text())
    if meta.get("schema") != SCHEMA:
        raise MaskBankError(f"schema {meta.get('schema')!r}, expected {SCHEMA!r}")
    if hashlib.sha256(npz.read_bytes()).hexdigest() != meta["npz_sha256"]:
        raise MaskBankError(f"{npz} does not match its manifest")
    if "calls_sha256" not in meta:
        raise MaskBankError(f"{manifest} has no calls_sha256, so its density and causal "
                            f"fields are covered by no hash")
    if calls_sha256(meta["calls"]) != meta["calls_sha256"]:
        raise MaskBankError(f"{manifest}: the call entries do not match their calls_sha256")
    arrays = np.load(npz)
    masks = []
    for i, c in enumerate(meta["calls"]):
        packed = arrays[f"call_{i}"]
        if hashlib.sha256(packed.tobytes()).hexdigest() != c["sha256"]:
            raise MaskBankError(f"call {i} does not match its recorded sha256")
        m = BankedMask(tuple(c["shape"]), bool(c["causal"]), float(c["density"]),
                       c["sha256"], packed)
        n = int(np.prod(m.shape))
        if len(m.shape) < 2 or (n + 7) // 8 != packed.size:
            raise MaskBankError(f"call {i}: shape {list(m.shape)} does not fit its "
                                f"{packed.size} packed bytes")
        got = density(m.array(), m.causal)
        if got != m.density:
            raise MaskBankError(f"call {i}: the manifest's density {m.density} does not "
                                f"follow from its mask ({got}, causal={m.causal})")
        masks.append(m)
    cp = meta["code_path"]
    record = MaskRecord(
        CodePath(arm=cp["arm"], gpu=cp["gpu"], implementation=cp["implementation"],
                 libraries=tuple(tuple(x) for x in cp["libraries"]),
                 python_optimize=int(cp["python_optimize"])),
        int(meta["context_length"]), int(meta["batch"]), meta["prompt_id"], tuple(masks))
    if record.sha256 != meta["sha256"]:
        raise MaskBankError(f"{manifest}: the combined sha256 does not match")
    return record


def agreement(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> dict:
    """Per-call bitwise agreement of two mask lists (shape-checked). Refused
    over no masks: an agreement of nothing is not an agreement (ruling,
    1 Nov 2026)."""
    if not a or not b:
        raise MaskBankError(f"{len(a)} masks against {len(b)}: the agreement would compare "
                            f"nothing")
    if len(a) != len(b):
        raise MaskBankError(f"{len(a)} masks against {len(b)}")
    per_call = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape:
            raise MaskBankError(f"call {i}: shapes {tuple(x.shape)} and {tuple(y.shape)}")
        x, y = x.detach().cpu(), y.detach().cpu()
        per_call.append(float((x == y).float().mean()))
    return {"identical": all(v == 1.0 for v in per_call), "per_call_agreement": per_call}
