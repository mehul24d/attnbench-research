"""Recording one arm's prefill: trace, sequence, banked masks and tape (T2.4).

`record_arm_prefill` is the recorder the replay and the structural probe
read from. One call, for one arm, prompt and cell, writes to a new
directory:

  - `trace.json`: the Kineto trace of one prefill, through T2.3's
    `trace_recording.record_prefill_trace` (its contract and annotations);
  - `sequence.json`: the operation sequence parsed from it, with no
    durations (`op_sequence.to_record`);
  - `masks.npz` and `masks.json`: the estimator's selection, the mask the
    block-sparse kernel was handed at every call **of the traced run**
    (`mask_bank`), so the banked masks belong to exactly the recorded
    sequence;
  - `tape.pt`: every host-device interaction of the prefill (`sync_tape`),
    from a second, untraced pass, which the CUDA-graph replay needs;
  - `recording.json`: what was recorded, with the sha256 of each file.

The second pass also captures the masks again, this time as copies taken
at each kernel call. The traced run keeps references, not copies, so as to
add nothing to the trace, and the two must be bitwise equal: if they are
not, either the arm is not deterministic on this prompt or a mask was
written in place after the kernel used it (a buffer reused across layers),
and either way the references would not describe the sequence. The recorder
refuses.

`record_masks` is the untraced half alone: masks (copied at each call) and
tape at a cell, with no trace. It is the mask-recording pass of Section 5.1.

A dense arm reaches no block-sparse kernel; it banks no masks and says so.
Where the kernel library is not installed no arm can reach it, so the
captures are optional and a dense arm records without it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

from . import mask_bank, op_sequence
from .op_sequence import CodePath, OpSequence
from .sync_tape import SyncTape, TapeRecorder
from .trace_recording import record_prefill_trace

SCHEMA = "attnbench.recording/1"


class RecordingError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fresh_dir(out_dir) -> Path:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise RecordingError(f"{out} is not empty; a recording is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass(frozen=True)
class Recording:
    directory: Path
    sequence: Optional[OpSequence]
    masks: Optional[mask_bank.MaskRecord]
    tape: SyncTape


def _untraced_pass(run_prefill, device_type: str, kernel_module=None):
    with TapeRecorder(device_type) as rec, \
            mask_bank.MaskCapture(module=kernel_module, copy=True, optional=True) as cap:
        run_prefill()
    if device_type == "cuda":
        torch.cuda.synchronize()
    return rec.tape, cap


def _bank(cap, out, *, code_path, context_length, batch, prompt_id):
    if not cap.calls:
        return None, None
    record = mask_bank.from_capture(cap, code_path=code_path, context_length=context_length,
                                    batch=batch, prompt_id=prompt_id)
    npz, manifest = mask_bank.save(record, out / "masks")
    return record, (npz, manifest)


def _write_manifest(out: Path, *, code_path, context_length, batch, prompt_id,
                    files: Sequence[Path], tape: SyncTape, masks, n_ops) -> None:
    (out / "recording.json").write_text(json.dumps({
        "schema": SCHEMA, "code_path": mask_bank._code_path_dict(code_path),
        "context_length": int(context_length), "batch": int(batch),
        "prompt_id": str(prompt_id),
        "files": {p.name: _sha(p) for p in files},
        "sequence_ops": n_ops,
        "mask_calls": len(masks.masks) if masks else 0,
        "masks_sha256": masks.sha256 if masks else None,
        "tape_entries": len(tape), "tape_substitutions": tape.substitutions,
        "tape_unhandled": tape.unhandled}, indent=1))


def record_masks(run_prefill: Callable[[], None], out_dir, *, code_path: CodePath,
                 context_length: int, batch: int, prompt_id: str,
                 device_type: str = "cuda", kernel_module=None) -> Recording:
    """The untraced mask-recording pass: masks and tape at one cell."""
    out = _fresh_dir(out_dir)
    tape, cap = _untraced_pass(run_prefill, device_type, kernel_module)
    masks, mfiles = _bank(cap, out, code_path=code_path, context_length=context_length,
                          batch=batch, prompt_id=prompt_id)
    tape_path = tape.save(out / "tape.pt")
    _write_manifest(out, code_path=code_path, context_length=context_length, batch=batch,
                    prompt_id=prompt_id, files=[tape_path, *(mfiles or ())],
                    tape=tape, masks=masks, n_ops=None)
    return Recording(out, None, masks, tape)


def record_arm_prefill(run_prefill: Callable[[], None], out_dir, *, code_path: CodePath,
                       context_length: int, batch: int, prompt_id: str,
                       layers: Sequence[torch.nn.Module] = (), warmup: int = 1,
                       cuda: Optional[bool] = None, kernel_module=None,
                       parse=op_sequence.parse_trace,
                       record_trace=record_prefill_trace) -> Recording:
    """Trace, sequence, masks of the traced run, and tape (module docstring).

    `parse` and `record_trace` are T2.3's functions; they are parameters so
    the orchestration is tested on CPU.
    """
    cuda = torch.cuda.is_available() if cuda is None else cuda
    out = _fresh_dir(out_dir)
    traced = mask_bank.MaskCapture(module=kernel_module, optional=True)
    n = {"calls": 0}

    def run():
        n["calls"] += 1
        if n["calls"] == warmup + 1:      # the profiled call: keep only its masks
            traced.calls.clear()
        run_prefill()

    with traced:
        trace_path = Path(record_trace(run, out / "trace.json", layers=layers,
                                       warmup=warmup, cuda=cuda))
    if n["calls"] != warmup + 1:
        raise RecordingError(f"the recorder ran the prefill {n['calls']} times, "
                             f"expected {warmup + 1}")
    seq = parse(trace_path, code_path, context_length=context_length, batch=batch)
    seq_path = out / "sequence.json"
    record = op_sequence.to_record(seq)
    seq_path.write_text(json.dumps({**record, "sha256": op_sequence.record_sha256(record)},
                                   indent=1))

    tape, again = _untraced_pass(run_prefill, "cuda" if cuda else "cpu", kernel_module)
    if len(traced.calls) != len(again.calls):
        raise RecordingError(f"the traced run made {len(traced.calls)} kernel calls and "
                             f"the untraced pass {len(again.calls)}")
    if traced.calls:
        same = mask_bank.agreement(traced.masks(), again.masks())
        if not same["identical"]:
            raise RecordingError(
                "the masks of the traced run and the untraced pass differ "
                f"(per-call agreement {same['per_call_agreement']}); the bank would "
                f"not describe the recorded sequence")
    masks, mfiles = _bank(traced, out, code_path=code_path, context_length=context_length,
                          batch=batch, prompt_id=prompt_id)
    tape_path = tape.save(out / "tape.pt")
    _write_manifest(out, code_path=code_path, context_length=context_length, batch=batch,
                    prompt_id=prompt_id, files=[trace_path, seq_path, tape_path, *(mfiles or ())],
                    tape=tape, masks=masks, n_ops=len(seq))
    return Recording(out, seq, masks, tape)


def load_recording(directory) -> Recording:
    """Read a recording back, checking every file against its sha256."""
    d = Path(directory)
    meta = json.loads((d / "recording.json").read_text())
    if meta.get("schema") != SCHEMA:
        raise RecordingError(f"schema {meta.get('schema')!r}")
    for name, sha in meta["files"].items():
        if _sha(d / name) != sha:
            raise RecordingError(f"{d / name} does not match recording.json")
    seq = None
    if (d / "sequence.json").exists():
        rec = json.loads((d / "sequence.json").read_text())
        sha = rec.pop("sha256")
        if op_sequence.record_sha256(rec) != sha:
            raise RecordingError("sequence.json does not match its own sha256")
        seq = op_sequence.from_record(rec)
    masks = mask_bank.load(d / "masks") if (d / "masks.json").exists() else None
    return Recording(d, seq, masks, SyncTape.load(d / "tape.pt"))
