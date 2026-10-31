"""T2.4: the tape of host-device interactions (`components.sync_tape`).

On CPU with `device_type="cpu"`: the CPU stands in for the device, so every
interaction the mode classifies is one the CUDA path would see. What a CPU
cannot show is a capture refusing a wait the mode missed; that is the GPU
session's.

The test of record for "the replay serves the tape" is replaying on
different data of the same shapes: every value read back, every size a
data-dependent operation produced, must be the recorded one, not the live
one.
"""

from __future__ import annotations

import pytest
import torch

from attnbench.components import sync_tape as T
from attnbench.components.sync_tape import SyncTape, TapeMismatch, TapeRecorder, TapeReplayer


def prefill(x):
    """One of every interaction the tape serves, in a fixed order."""
    assert bool((x > -100).all())
    out = {"n": (x > 2).sum().item(), "head": x[:2].tolist(), "first": int(x[0]),
           "second": float(x[1]), "eq": torch.equal(x, x)}
    out["nz"] = torch.nonzero(x > 2)
    (out["where"],) = torch.where(x > 3)
    out["sel"] = x[x > 1]
    y = x.clone()
    y[y > 4] = -1.0
    out["set"] = y
    out["built"] = torch.tensor([0, 5], device="cpu")
    out["host"] = x.cpu()
    torch.cuda.synchronize()
    return out


def record(fn, *a):
    with TapeRecorder("cpu") as rec:
        out = fn(*a)
    return rec.tape, out


def replay(tape, fn, *a, staged=None):
    staged = tape.stage("cpu") if staged is None else staged
    rep = TapeReplayer(tape, staged)
    with rep:
        out = fn(*a)
    rep.finish()
    return out


X = torch.arange(6.0)


def test_the_recorder_writes_down_every_interaction_in_order():
    tape, _ = record(prefill, X)
    assert [e.kind for e in tape.entries] == [
        "read", "read", "read", "read", "read", "read",
        "nonzero", "nonzero_tuple", "bool_index", "bool_setitem",
        "to_device", "to_host", "synchronize"]
    assert [e.name for e in tape.entries[:6]] == [
        "__bool__", "item", "tolist", "__int__", "__float__", "equal"]
    assert [e.value for e in tape.entries[:9]] == [True, 3, [0.0, 1.0], 0, 1.0, True, 3, 2, 4]
    assert tape.entries[9].value == 1
    assert tape.substitutions == {"nonzero": 1, "nonzero_tuple": 1, "bool_index": 1,
                                  "bool_setitem": 1}
    assert tape.unhandled == []


def test_the_replay_serves_recorded_values_not_live_ones():
    tape, rec = record(prefill, X)
    other = torch.full((6,), 10.0)        # same shapes; every live answer differs
    out = replay(tape, prefill, other)
    for k in ("n", "head", "first", "second", "eq"):
        assert out[k] == rec[k], k
    # sizes come from the tape: 3, 2 and 4 rows, where the live data has 6
    assert out["nz"].shape == rec["nz"].shape == (3, 1)
    assert out["where"].shape == rec["where"].shape == (2,)
    assert out["sel"].shape == rec["sel"].shape == (4,)
    assert int((out["set"] == -1).sum()) == 1          # one element written, not six
    # the staged tensor and the recorded host copy, not new ones
    assert torch.equal(out["built"], torch.tensor([0, 5]))
    assert torch.equal(out["host"], X)


def test_a_replayed_setitem_writes_the_recorded_number_of_elements():
    tape, rec = record(prefill, X)
    out = replay(tape, prefill, X.clone())
    assert torch.equal(out["set"], rec["set"])


def test_a_replayed_list_is_a_copy():
    tape, _ = record(prefill, X)
    out = replay(tape, prefill, X)
    out["head"].append("mutated")
    assert replay(tape, prefill, X)["head"] == [0.0, 1.0]


def test_a_different_shape_is_a_mismatch_at_the_first_shaped_interaction():
    # the first five reads are of 0-d tensors or a fixed slice; equal sees x
    tape, _ = record(prefill, X)
    with pytest.raises(TapeMismatch, match=r"interaction 5: replayed read equal \(\(7,\)"):
        replay(tape, prefill, torch.arange(7.0))


def test_a_different_path_is_a_mismatch():
    def other(x):
        _ = (x > 2).sum().item()          # skips the assert the tape starts with
    tape, _ = record(prefill, X)
    with pytest.raises(TapeMismatch, match="recorded read __bool__"):
        replay(tape, other, X)


def test_a_shorter_path_fails_finish_and_a_longer_one_runs_past_the_end():
    def short(x):
        assert bool((x > -100).all())

    def long(x):
        assert bool((x > -100).all())
        assert bool((x > -100).all())
    tape, _ = record(short, X)
    with pytest.raises(TapeMismatch, match="past the end"):
        replay(tape, long, X)
    tape, _ = record(prefill, X)
    with pytest.raises(TapeMismatch, match="used 1 of 13"):
        replay(tape, short, X)


@pytest.mark.parametrize("fn", [
    lambda x: torch.masked_select(x, x > 1),
    lambda x: x.unique(),
    lambda x: torch.unique_consecutive(x),
    lambda x: torch.bincount(x.long()),
    lambda x: torch.repeat_interleave(x, x.long()),
    lambda x: torch.repeat_interleave(x.long()),
    lambda x: x.reshape(2, 3)[x.reshape(2, 3).sum(1) > 0, 1:],
], ids=["masked_select", "unique", "unique_consecutive", "bincount",
        "repeat_interleave_tensor", "repeat_interleave_self", "bool_index_with_slice"])
def test_reads_the_replay_cannot_serve_are_written_down_and_refused(fn):
    tape, _ = record(fn, X)
    assert [e.kind for e in tape.entries] == ["unhandled"]
    assert "cannot serve it" in tape.entries[0].note
    with pytest.raises(TapeMismatch, match="cannot serve it"):
        replay(tape, fn, X)


@pytest.mark.parametrize("fn", [
    lambda x: x.repeat_interleave(2),
    lambda x: torch.repeat_interleave(x, x.long(), output_size=15),
    lambda x: x.sum() + x.max(),
    lambda x: x.any(),
], ids=["repeat_interleave_int", "repeat_interleave_sized", "reductions", "any"])
def test_ordinary_device_work_is_not_on_the_tape(fn):
    tape, _ = record(fn, X)
    assert tape.entries == []


def test_the_index_protocol_is_a_read():
    tape, _ = record(lambda x: [10, 20, 30][x[1].long()], X)
    assert [(e.kind, e.name, e.value) for e in tape.entries] == [("read", "__index__", 1)]


def test_copies_across_the_device_boundary_are_classified_by_direction():
    cpu, meta = torch.zeros(3), torch.empty(3, device="meta")
    c = lambda f, *a, **k: T.classify(f, a, k, "cpu")
    assert c(torch.Tensor.copy_, cpu, meta) == "to_device_inplace"
    assert T.classify(torch.Tensor.copy_, (cpu, meta), {}, "meta") == "to_host_inplace"
    assert c(torch.Tensor.copy_, cpu, cpu) is None
    assert c(torch.Tensor.to, cpu, "meta") == "to_host"
    assert c(torch.Tensor.to, cpu, device="meta") == "to_host"
    assert c(torch.Tensor.to, cpu, meta) == "to_host"
    assert c(torch.Tensor.to, cpu, torch.float16) is None
    assert T.classify(torch.Tensor.to, (cpu, "meta"), {}, "meta") == "to_device"
    assert c(torch.tensor, [1, 2], device="cpu") == "to_device"
    assert c(torch.tensor, [1, 2]) is None
    assert T._as_device(0) == torch.device("cuda", 0) and T._as_device(True) is None


def test_staging_is_required_for_tensors_built_from_host_data():
    tape, _ = record(lambda: torch.tensor([0, 5], device="cpu"))
    staged = tape.stage("cpu")
    assert list(staged) == [0] and torch.equal(staged[0], torch.tensor([0, 5]))
    with pytest.raises(TapeMismatch, match="no staged device tensor"):
        replay(tape, lambda: torch.tensor([0, 5], device="cpu"), staged={})


def test_synchronize_is_patched_for_the_life_of_the_mode_only():
    original = torch.cuda.synchronize
    with pytest.raises(RuntimeError):
        with TapeRecorder("cpu"):
            assert torch.cuda.synchronize is not original
            raise RuntimeError("boom")
    assert torch.cuda.synchronize is original
    tape, _ = record(lambda: torch.cuda.synchronize())
    rep = TapeReplayer(tape)
    with rep:
        torch.cuda.synchronize()          # served, so a CPU-only build does not raise
    rep.finish()
    assert torch.cuda.synchronize is original


def test_the_tape_round_trips_and_is_never_overwritten(tmp_path):
    tape, _ = record(prefill, X)
    path = tape.save(tmp_path / "tape.pt")
    back = SyncTape.load(path)
    assert back.device_type == "cpu" and len(back) == len(tape)
    for a, b in zip(tape.entries, back.entries):
        assert (a.kind, a.name, a.sig, a.note) == (b.kind, b.name, b.sig, b.note)
        assert torch.equal(a.value, b.value) if isinstance(a.value, torch.Tensor) \
            else a.value == b.value
    out = replay(back, prefill, torch.full((6,), 10.0))
    assert out["n"] == 3
    with pytest.raises(FileExistsError):
        tape.save(path)


def test_a_tape_of_another_schema_is_refused(tmp_path):
    torch.save({"schema": "attnbench.sync_tape/0", "device_type": "cpu", "entries": []},
               tmp_path / "t.pt")
    with pytest.raises(TapeMismatch, match="schema"):
        SyncTape.load(tmp_path / "t.pt")


def test_a_replayed_scalar_setitem_builds_its_value_on_the_device():
    """Shakedown 2's replay: `nonzero_idx[nonzero_val == 0] = -1` inside the
    block-sparse kernel's wrapper reached `index_put_` with the -1 as a host
    tensor, whose copy to the device is not permitted during graph capture.
    The replay builds the value with a fill on the device instead, and still
    writes the recorded number of elements."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Ops(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.seen = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.seen.append(str(func))
            return func(*args, **(kwargs or {}))

    def setter(x):
        y = x.clone()
        y[y > 4] = -1.0
        return y

    tape, rec = record(setter, X)
    rep = TapeReplayer(tape, tape.stage("cpu"))
    with rep, Ops() as ops:
        out = setter(X.clone())
    rep.finish()
    assert torch.equal(out, rec)
    put = next(i for i, f in enumerate(ops.seen) if f.startswith("aten.index_put"))
    assert any(f.startswith("aten.full") for f in ops.seen[:put])
    assert not any(f.startswith(("aten.lift_fresh", "aten._to_copy", "aten.scalar_tensor"))
                   for f in ops.seen[:put])
