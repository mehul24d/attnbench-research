"""T2.4: one arm's recording, trace, sequence, banked masks and tape
(`components.recorder`).

T2.3's real `record_prefill_trace` runs here on CPU (`cuda=False`), so the
warmup count and the profiled call are its own; `parse_trace` needs device
events a CPU trace does not have, so a stand-in parser returns a fixed
sequence. The kernel is a stand-in module (as in the mask-bank tests).

The arm is a toy whose masks can be made to depend on which call of the
prefill this is, so the tests can tell the warmup's masks from the profiled
call's, and the profiled call's from the untraced pass's.
"""

from __future__ import annotations

import json
import types

import pytest
import torch

from attnbench.components import op_sequence as S
from attnbench.components import recorder as R
from attnbench.components.op_sequence import CodePath, OpSequence, SeqKind, SeqOp
from attnbench.components.recorder import RecordingError

CP = CodePath(arm="xattn", gpu="NVIDIA L4", implementation="xattn:torch",
              libraries=(("torch", "2.9.1"),))
SEQ = OpSequence(code_path=CP, context_length=4096, batch=1, streams=(7,), ops=(
    SeqOp(SeqKind.KERNEL, layer=0, api="cudaLaunchKernel", aten_op="aten::mm",
          kernels=("sgemm",), stream=7),
    SeqOp(SeqKind.SYNC, layer=0, api="cudaStreamSynchronize", aten_op="aten::item",
          mechanism="item", payload_bytes=4),
    SeqOp(SeqKind.KERNEL, layer=1, api="cudaLaunchKernel", kernels=("bsa_fwd",), stream=7)))


def fake_kernel():
    mod = types.SimpleNamespace()
    mod.block_sparse_attn_func = lambda q, k, v, cq, ck, hmt, si, mask, *a, **kw: q
    return mod


def mask(seed, shape=(1, 2, 4, 4)):
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed)) > 0.5


class Arm:
    """`layers` kernel calls per prefill. `seed(call)` picks each call's masks."""

    def __init__(self, layers=2, seed=lambda call: 0, reuse_buffer=False, calls_at=None):
        self.mod, self.layers, self.seed = fake_kernel(), layers, seed
        self.reuse, self.calls_at = reuse_buffer, calls_at
        self.call = 0

    def __call__(self):
        self.call += 1
        x = torch.arange(4.0)
        assert bool((x >= 0).all())                 # one read for the tape
        n = self.layers if self.calls_at is None else self.calls_at(self.call)
        buf = torch.zeros(1, 2, 4, 4, dtype=torch.bool)
        for layer in range(n):
            m = mask(100 * self.seed(self.call) + layer)
            if self.reuse:
                buf.copy_(m)
                m = buf
            q = torch.zeros(2)
            self.mod.block_sparse_attn_func(q, q, q, None, None,
                                            torch.ones(2, dtype=torch.int32), None, m,
                                            is_causal=True)


def parse(path, code_path, *, context_length, batch):
    assert json.loads(path.read_text())["traceEvents"]    # a real trace was written
    assert (code_path, context_length, batch) == (CP, 4096, 1)
    return SEQ


def rec(arm, d, **kw):
    return R.record_arm_prefill(arm, d, code_path=CP, context_length=4096, batch=1,
                                prompt_id="p0", cuda=False, kernel_module=arm.mod,
                                parse=parse, **kw)


def test_a_sparse_arm_is_recorded_whole(tmp_path):
    arm = Arm(seed=lambda call: 0 if call == 1 else 1)    # the warmup's masks differ
    r = rec(arm, tmp_path / "r")
    assert arm.call == 3                                    # warmup, profiled, untraced
    names = sorted(p.name for p in (tmp_path / "r").iterdir())
    assert names == ["masks.json", "masks.npz", "recording.json", "sequence.json",
                     "tape.pt", "trace.json"]
    assert [m.sha256 for m in r.masks.masks] == \
        [S_sha(mask(100 + i)) for i in range(2)]           # the profiled call's
    assert [(e.kind, e.name) for e in r.tape.entries] == [("read", "__bool__")]
    meta = json.loads((tmp_path / "r" / "recording.json").read_text())
    assert set(meta["files"]) == {"trace.json", "sequence.json", "tape.pt",
                                  "masks.npz", "masks.json"}
    assert (meta["sequence_ops"], meta["mask_calls"], meta["tape_entries"]) == (3, 2, 1)
    assert meta["masks_sha256"] == r.masks.sha256 and meta["prompt_id"] == "p0"


def S_sha(m):
    import hashlib
    import numpy as np
    return hashlib.sha256(np.packbits(m.numpy().reshape(-1)).tobytes()).hexdigest()


def test_a_recording_reads_back_with_every_file_checked(tmp_path):
    r = rec(Arm(), tmp_path / "r")
    back = R.load_recording(tmp_path / "r")
    assert back.sequence == SEQ
    assert back.masks == r.masks
    assert [(e.kind, e.value) for e in back.tape.entries] == [("read", True)]
    seq_json = json.loads((tmp_path / "r" / "sequence.json").read_text())
    assert "dur" not in json.dumps(seq_json)               # no durations banked


def test_a_tampered_file_is_refused(tmp_path):
    rec(Arm(), tmp_path / "r")
    t = tmp_path / "r" / "tape.pt"
    t.write_bytes(t.read_bytes() + b"x")
    with pytest.raises(RecordingError, match="tape.pt does not match"):
        R.load_recording(tmp_path / "r")


def test_a_sequence_edited_with_its_manifest_is_still_refused(tmp_path):
    rec(Arm(), tmp_path / "r")
    d = tmp_path / "r"
    s = json.loads((d / "sequence.json").read_text())
    s["ops"][1]["mechanism"] = "bool"
    (d / "sequence.json").write_text(json.dumps(s))
    meta = json.loads((d / "recording.json").read_text())
    meta["files"]["sequence.json"] = R._sha(d / "sequence.json")
    (d / "recording.json").write_text(json.dumps(meta))
    with pytest.raises(RecordingError, match="its own sha256"):
        R.load_recording(d)


def test_an_arm_whose_masks_change_between_passes_is_refused(tmp_path):
    with pytest.raises(RecordingError, match="masks of the traced run and the untraced"):
        rec(Arm(seed=lambda call: call), tmp_path / "r")


def test_a_mask_buffer_reused_across_layers_is_refused(tmp_path):
    # Both passes are deterministic; only the traced run's references show the
    # buffer's last contents in every call.
    with pytest.raises(RecordingError, match="masks of the traced run and the untraced"):
        rec(Arm(reuse_buffer=True), tmp_path / "r")


def test_a_different_number_of_kernel_calls_is_refused(tmp_path):
    with pytest.raises(RecordingError, match="2 kernel calls and the untraced pass 3"):
        rec(Arm(calls_at=lambda call: 2 if call <= 2 else 3), tmp_path / "r")


def test_a_dense_arm_banks_no_masks_and_says_so(tmp_path):
    r = rec(Arm(layers=0), tmp_path / "r")
    assert r.masks is None
    assert not (tmp_path / "r" / "masks.npz").exists()
    meta = json.loads((tmp_path / "r" / "recording.json").read_text())
    assert (meta["mask_calls"], meta["masks_sha256"]) == (0, None)
    assert R.load_recording(tmp_path / "r").masks is None


def test_a_dense_arm_records_without_the_kernel_library(tmp_path):
    import importlib.util
    if importlib.util.find_spec("block_sparse_attn") is not None:
        pytest.skip("block_sparse_attn is installed here")
    x = torch.arange(4.0)

    def dense():
        assert bool((x >= 0).all())
    r = R.record_arm_prefill(dense, tmp_path / "r", code_path=CP, context_length=4096,
                             batch=1, prompt_id="p0", cuda=False, parse=parse)
    assert r.masks is None and len(r.tape) == 1


def test_a_recording_is_never_overwritten(tmp_path):
    rec(Arm(), tmp_path / "r")
    with pytest.raises(RecordingError, match="never overwritten"):
        rec(Arm(), tmp_path / "r")


def test_a_trace_recorder_that_runs_the_prefill_the_wrong_number_of_times_is_refused(tmp_path):
    def twice_traced(run, path, *, layers, warmup, cuda):
        for _ in range(warmup + 2):
            run()
        path.write_text(json.dumps({"traceEvents": [{}]}))
        return path
    with pytest.raises(RecordingError, match="3 times, expected 2"):
        rec(Arm(), tmp_path / "r", record_trace=twice_traced)


def test_warmup_is_passed_through(tmp_path):
    arm = Arm(seed=lambda call: 0 if call <= 3 else 1)
    with pytest.raises(RecordingError, match="differ"):
        rec(arm, tmp_path / "a", warmup=2)          # profiled call 3, untraced call 4
    arm = Arm(seed=lambda call: 0 if call <= 2 else 1)
    r = rec(arm, tmp_path / "c", warmup=2)          # calls 3 and 4 agree
    assert arm.call == 4 and r.masks.masks[0].sha256 == S_sha(mask(100))


def test_record_masks_copies_each_mask_at_its_call(tmp_path):
    arm = Arm(reuse_buffer=True)
    r = R.record_masks(arm, tmp_path / "m", code_path=CP, context_length=4096, batch=1,
                       prompt_id="p0", device_type="cpu", kernel_module=arm.mod)
    assert [m.sha256 for m in r.masks.masks] == [S_sha(mask(i)) for i in range(2)]
    assert r.sequence is None
    assert sorted(p.name for p in (tmp_path / "m").iterdir()) == \
        ["masks.json", "masks.npz", "recording.json", "tape.pt"]
    back = R.load_recording(tmp_path / "m")
    assert back.sequence is None and back.masks == r.masks
    assert json.loads((tmp_path / "m" / "recording.json").read_text())["sequence_ops"] is None
