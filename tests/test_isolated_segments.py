"""The isolated replay in blocks of layers (researcher's ruling, 2 Nov 2026,
after T3.1's fourth session: `oracle_d0.5_L32768` was killed at the host
memory ceiling inside one profiler session).

Three things are tested. The stitch: segments' compact streams become the
stream of the whole replay, and anything that is not one timeline of one
thread is refused. The roll: on a CPU, a toy prefill replayed in blocks has
the operators of the replay in one session, none cut at a boundary, one
region, and leaves nothing running or hooked. And the equivalence on the
fourth session's banked streams: a banked monolithic stream cut at gaps
between top-level calls, given a region per piece and stitched, is read by
`host_profile` into the same launches, call chains, flushes and host costs as
the original. What a CUDA profiler's memory does is the GPU session's."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from attnbench.components import host_profile as hp  # noqa: E402
from attnbench.components import replay as RP  # noqa: E402
from attnbench.components import trace_stream as ts  # noqa: E402
from attnbench.components.sync_tape import TapeRecorder  # noqa: E402

REGION = RP.REPLAY_REGION
BANKED = (ROOT / "results" / "t31_p1_s4" / "results" / "t31_p1_20261102T1208Z"
          / "05_host_components")


def write_part(path, events, *, schema=ts.SCHEMA):
    with open(path, "w") as f:
        f.write(json.dumps({"schema": schema, "source": "p", "cats": [], "fields": list(ts.FIELDS)}) + "\n")
        for e in events:
            f.write(json.dumps(list(e)) + "\n")
    return Path(path)


def part_events(t0, n, *, base=0, thread=(1, 2), region=True, span=None):
    """One segment: a region and n operators one microsecond apart, from t0."""
    ev = [(base + 1 + k, "cpu_op", f"aten::op{k}", *thread, t0 + 1 + 2 * k, 1.0) for k in range(n)]
    if region:
        ev.insert(0, (base, "user_annotation", REGION, *thread, t0, span or 2 * n + 1))
    return ev


# --- the stitch --------------------------------------------------------------------------------

def test_the_stitched_stream_is_the_segments_in_order_under_one_region(tmp_path):
    a = write_part(tmp_path / "a.jsonl", part_events(100.0, 3))
    b = write_part(tmp_path / "b.jsonl", part_events(200.0, 2))
    c = write_part(tmp_path / "c.jsonl", part_events(300.0, 4))
    out = ts.stitch_compact([a, b, c], tmp_path / "s.events.jsonl", region=REGION)
    ev = ts.load_compact(tmp_path / "s.events.jsonl")
    regions = [e for e in ev if e.name == REGION]
    assert len(regions) == 1 and (regions[0].ts, regions[0].end) == (100.0, 300.0 + 9.0)
    assert len(ev) == 1 + 3 + 2 + 4 == out["kept"] and out["segments"] == 3
    ops = [e for e in ev if e.name != REGION]
    assert [e.ts for e in ops] == sorted(e.ts for e in ops)              # in order
    assert len({e.idx for e in ev}) == len(ev)                            # indices stay unique
    assert [e.name for e in ops] == [f"aten::op{k}" for n in (3, 2, 4) for k in range(n)]
    assert ts.is_compact(tmp_path / "s.events.jsonl") and json.loads(
        (tmp_path / "s.events.jsonl").open().readline())["segments"] == 3


def test_one_segment_stitches_to_itself(tmp_path):
    a = write_part(tmp_path / "a.jsonl", part_events(5.0, 3))
    ts.stitch_compact([a], tmp_path / "s.events.jsonl", region=REGION)
    assert [(e.name, e.ts, e.end) for e in ts.load_compact(tmp_path / "s.events.jsonl")] == [
        (e.name, e.ts, e.end) for e in ts.load_compact(a)]


@pytest.mark.parametrize("what", ["no region", "two regions", "other thread", "overlap",
                                  "other schema", "none"])
def test_a_set_of_segments_that_is_not_one_replay_is_refused(tmp_path, what):
    a = write_part(tmp_path / "a.jsonl", part_events(100.0, 3))
    b_ev = part_events(200.0, 2)
    schema = ts.SCHEMA
    if what == "no region":
        b_ev = part_events(200.0, 2, region=False)
    elif what == "two regions":
        b_ev = b_ev + [(99, "user_annotation", REGION, 1, 2, 250.0, 1.0)]
    elif what == "other thread":
        b_ev = part_events(200.0, 2, thread=(1, 3))
    elif what == "overlap":
        b_ev = part_events(104.0, 2)               # starts inside a's region (100 to 107)
    elif what == "other schema":
        schema = "something/else"
    b = write_part(tmp_path / "b.jsonl", b_ev, schema=schema)
    parts = [] if what == "none" else [a, b]
    with pytest.raises(ts.TraceStreamError):
        ts.stitch_compact(parts, tmp_path / "s.events.jsonl", region=REGION)
    assert not (tmp_path / "s.events.jsonl").exists() and not (tmp_path / "s.events.jsonl.part").exists()


# --- the roll, on a CPU --------------------------------------------------------------------------

class Toy:
    """A prefill of `n` layers, each a linear and a relu, on the CPU."""

    def __init__(self, n):
        torch.manual_seed(0)
        self.layers = [torch.nn.Linear(4, 4) for _ in range(n)]
        self.x = torch.ones(2, 4)

    def __call__(self):
        x = self.x
        for layer in self.layers:
            x = torch.relu(layer(x))


def _replay(tmp_path, n, per=None, name="iso"):
    toy = Toy(n)
    with TapeRecorder("cpu") as rec:
        toy()
    kw = dict(device="cpu", flush=lambda: None, sync=lambda: None)
    if per is None:
        path = RP.isolated_replay(toy, rec.tape, tmp_path / f"{name}.json", substitute=False, **kw)
        info = ts.write_compact(path)
        return toy, Path(info["path"]), None
    stream, counts = RP.isolated_replay_compact(toy, rec.tape, tmp_path / f"{name}.json",
                                                layers=toy.layers, segment_layers=per, **kw)
    return toy, stream, counts


def _names(path, *, skip_region=True):
    return [(e.cat, e.name) for e in sorted(ts.load_compact(path), key=lambda e: e.ts)
            if not (skip_region and e.name == REGION)]


def test_a_replay_in_blocks_has_the_operators_of_the_whole_replay_under_one_region(tmp_path):
    _, whole, _ = _replay(tmp_path, 6)
    toy, rolled, counts = _replay(tmp_path, 6, per=2, name="seg")
    assert counts["segments"] == 3 and counts["segment_layers"] == 2
    ev = ts.load_compact(rolled)
    assert sum(e.name == REGION for e in ev) == 1
    # every operator the whole replay has, none cut at a boundary or added by a roll
    assert _names(rolled) == _names(whole)
    assert sum(e.name == "aten::linear" for e in ev) == 6
    assert sum(e.name == RP.FLUSH for e in ev) == sum(e.name == RP.FLUSH for e in ts.load_compact(whole))
    assert len({e.idx for e in ev}) == len(ev)
    reg = next(e for e in ev if e.name == REGION)
    assert all(reg.ts - 1e-3 <= e.ts and e.end <= reg.end + 1e-3
               for e in ev if e.name in ("aten::linear", RP.FLUSH))
    # one timeline: the blocks followed one another
    lin = sorted(e.ts for e in ev if e.name == "aten::linear")
    assert lin == sorted(lin) and len(lin) == 6


def test_a_roll_is_made_before_each_block_of_layers_and_nothing_stays_hooked(tmp_path):
    toy = Toy(7)
    rolls = []
    real = RP._Rolled.roll

    def spy(self, layer):
        rolls.append(len(self.parts))
        return real(self, layer)
    with TapeRecorder("cpu") as rec:
        toy()
    RP._Rolled.roll = spy
    try:
        _, counts = RP.isolated_replay_compact(toy, rec.tape, tmp_path / "x.json", device="cpu",
                                               layers=toy.layers, segment_layers=3,
                                               flush=lambda: None, sync=lambda: None)
    finally:
        RP._Rolled.roll = real
    assert rolls == [0, 1] and counts["segments"] == 3                  # before layers 3 and 6
    assert all(not l._forward_pre_hooks for l in toy.layers)
    assert not list(tmp_path.glob("*.seg*")) and not list(tmp_path.glob("*.part"))
    assert [p.name for p in tmp_path.iterdir()] == ["x.events.jsonl"]    # the stream, nothing else


def test_a_failed_replay_leaves_no_profiler_hook_or_segment(tmp_path):
    toy = Toy(6)
    with TapeRecorder("cpu") as rec:
        toy()
    # fail after the second block has opened
    def failing():
        x = toy.x
        for j, layer in enumerate(toy.layers):
            x = torch.relu(layer(x))
            if j == 3:
                raise RuntimeError("prefill failed")
    with pytest.raises(RuntimeError, match="prefill failed"):
        RP.isolated_replay_compact(failing, rec.tape, tmp_path / "x.json", device="cpu",
                                   layers=toy.layers, segment_layers=2, flush=lambda: None,
                                   sync=lambda: None)
    assert all(not l._forward_pre_hooks for l in toy.layers)
    assert not list(tmp_path.iterdir())
    # no profiler left running: another can start and a replay run
    _, stream, _ = _replay(tmp_path, 4, per=2, name="again")
    assert stream.exists()


def test_a_replay_in_blocks_holds_the_bank_as_the_whole_replay_does(tmp_path):
    import test_components_replay as tcr
    arm = tcr.Arm(torch.arange(4.0), layers=3)
    tape, bank = tcr.recorded(arm)
    ids = [torch.nn.Identity() for _ in range(3)]

    def prefill():                     # the blocks' layers run, so each roll is at its boundary
        for layer in ids:
            layer(torch.zeros(1))
        arm()
    kw = dict(device="cpu", masks=bank, flush=lambda: None, sync=lambda: None, kernel_module=arm.mod,
              layers=ids, segment_layers=1)
    RP.isolated_replay_compact(prefill, tape, tmp_path / "a.json", **kw)
    assert all(torch.equal(s, b) for s, b in zip(arm.mod.seen, bank.tensors()))
    arm.layers = 2
    with pytest.raises(RP.ReplayError, match="made 2 kernel calls; 3 masks are banked"):
        RP.isolated_replay_compact(prefill, tape, tmp_path / "b.json", **kw)
    assert not [p for p in tmp_path.iterdir() if p.name != "a.events.jsonl"]


@pytest.mark.parametrize("per", [0, 6, 7])
def test_fewer_than_two_blocks_is_not_a_segmented_replay(tmp_path, per):
    toy = Toy(6)
    with TapeRecorder("cpu") as rec:
        toy()
    with pytest.raises(RP.ReplayError, match="not segmented"):
        RP.isolated_replay_compact(toy, rec.tape, tmp_path / "x.json", device="cpu",
                                   layers=toy.layers, segment_layers=per, flush=lambda: None,
                                   sync=lambda: None)


# --- the wiring ------------------------------------------------------------------------------------

def _host_cell(monkeypatch, tmp_path, *, layers, segment_layers):
    import t31_steps as st
    import test_t31_wiring as tw
    from attnbench.components import replay
    tw._components(monkeypatch, tw.FakeHarness())
    seen = {"compact": [], "whole": []}

    def compact(run, tape, path, **kw):
        seen["compact"].append(kw["segment_layers"])
        return Path("iso_stream"), {"stream": "iso_stream", "segments": 3}
    monkeypatch.setattr(replay, "isolated_replay_compact", compact)
    monkeypatch.setattr(replay, "isolated_replay", lambda *a, **k: seen["whole"].append(1) or "iso_trace")
    a = SimpleNamespace(run_prefill=lambda: None, layers=layers)
    rec = SimpleNamespace(sequence=tw.SEQ, tape=None, masks=None)
    tmp_path.mkdir(exist_ok=True)
    st.host_profile_cell(a, rec, tmp_path, reps=1, tolerance={"aggregate": 0.1},
                         segment_layers=segment_layers)
    return seen


def test_the_host_cell_replays_in_blocks_when_it_has_more_layers_than_a_block(tmp_path, monkeypatch):
    seen = _host_cell(monkeypatch, tmp_path / "a", layers=[object()] * 28, segment_layers=2)
    assert seen == {"compact": [2], "whole": []}
    seen = _host_cell(monkeypatch, tmp_path / "b", layers=[object()] * 2, segment_layers=2)
    assert seen == {"compact": [], "whole": [1]}                   # one block: as before
    seen = _host_cell(monkeypatch, tmp_path / "c", layers=[object()] * 28, segment_layers=0)
    assert seen == {"compact": [], "whole": [1]}                   # switched off


def test_the_host_child_passes_the_sessions_block_size(tmp_path, monkeypatch):
    import t31_steps as st
    import test_t31_wiring as tw
    seen = []

    def spy(*a, **k):
        seen.append(k["segment_layers"])
        raise RuntimeError("stop here")
    tw._components(monkeypatch, tw.FakeHarness())
    monkeypatch.setattr(st, "host_profile_cell", spy)
    for cfg_extra, want in (({}, st.ISOLATED_SEGMENT_LAYERS), ({"isolated_segment_layers": 5}, 5),
                            ({"isolated_segment_layers": 0}, 0)):
        session = tw._session(tmp_path / f"s{want}")
        args = SimpleNamespace(dry_run=False, session=str(session),
                               out=str(st.step_dir(session, "host_components")))
        with pytest.raises(st.StepFailed, match="stop here"):
            st.step_host_components(args, dict(tw._grid(), **cfg_extra))
        assert seen[-1] == want
    assert st.ISOLATED_SEGMENT_LAYERS == 2


# --- equivalence on the fourth session's banked streams ----------------------------------------------

def _top_level_gaps(events):
    """Times between top-level calls (no operator, flush or launch spans them):
    where a roll could have been made."""
    spans = sorted((e.ts, e.end) for e in events
                   if e.cat in ("cpu_op", "user_annotation") and e.name != REGION)
    gaps, hi = [], float("-inf")
    for t0, t1 in spans:
        if t0 > hi + 1e-6 and hi > float("-inf"):
            gaps.append((hi + t0) / 2)
        hi = max(hi, t1)
    return gaps


def _cut(src, k, tmp_path):
    """The banked stream cut into k pieces at evenly chosen gaps, each given a
    region of its own, as a profiler session per piece would have."""
    ev = ts.load_compact(src)
    reg = next(e for e in ev if e.name == REGION)
    gaps = _top_level_gaps([e for e in ev if e.name != REGION])
    cuts = [gaps[len(gaps) * j // k] for j in range(1, k)]
    pieces = [[] for _ in range(k)]
    for e in ev:
        if e.name == REGION:
            continue
        pieces[sum(e.ts >= c for c in cuts)].append(e)
    paths = []
    for n, p in enumerate(pieces):
        lo, hi = min(e.ts for e in p), max(e.end for e in p)
        rows = [(1_000_000_000 + n, "user_annotation", REGION, reg.pid, reg.tid, lo - 1.0, hi - lo + 2.0)]
        rows += [(e.idx, e.cat, e.name, e.pid, e.tid, e.ts, e.end - e.ts) for e in p]
        paths.append(write_part(tmp_path / f"piece{n}.jsonl", rows))
    return paths


def _reading(path, seq):
    iso = hp.read_isolated(path, seq)
    launches = {i: (e.ts, e.end, tuple((c.name, c.ts, c.end) for c in iso.calls.get(e.idx, ())))
                for i, e in iso.launch.items()}
    return launches, len(iso.flushes), iso.flushes.within(float("-inf"), float("inf"))


@pytest.mark.skipif(not BANKED.exists(), reason="session 4's pull is not in this tree")
@pytest.mark.parametrize("cell", ["oracle_d0.25_L2048_b1", "xattention_dcal_L2048_b1",
                                  "oracle_d0.5_L4096_b1"])
@pytest.mark.parametrize("k", [2, 7])
def test_a_banked_stream_cut_and_stitched_reads_as_the_original(tmp_path, cell, k):
    from attnbench.components import recorder
    cd = BANKED / cell
    seq = recorder.load_recording(cd / "recording").sequence
    src = cd / "isolated_host.events.jsonl"
    if not src.exists():
        pytest.skip("the event log is not in this pull")
    stitched = tmp_path / "stitched.events.jsonl"
    ts.stitch_compact(_cut(src, k, tmp_path), stitched, region=REGION)
    assert _reading(stitched, seq) == _reading(src, seq)
    pos = hp.isolated_positions(seq, hp.ISOLATED_OUTSIDE_LAYERS)
    assert hp.isolated_host_s(stitched, seq, pos) == hp.isolated_host_s(src, seq, pos)
    assert len(pos) > 0 or cell.startswith("xattention")
    assert len(ts.load_compact(stitched)) == len(ts.load_compact(src))      # k regions became one


# --- matching by segment (researcher's ruling 2 of 7 Nov 2026) ---------------------------------------
#
# The exploratory session's oracle cells left 16 positions unpriced (L8075 d0.5: the
# `masked_fill_` of layers 25 and 26 and the `nonzero` of layers 26 and 27; L16370 and
# L32700 d0.25: layer 26's `masked_fill_` and layer 27's `nonzero`). The layers are one
# structure repeated, so a match of the whole stream reports a layer's worth of nodes
# lost anywhere at the last layers, and a stretch can run across a cut. Matched within
# its own segment, a position is compared only with the nodes of the profiler session
# that ran it, and what is lost is reported where it was lost.

from attnbench.components.op_sequence import CodePath, OpSequence, SeqKind, SeqOp  # noqa: E402
from attnbench.components.replay import Node  # noqa: E402

_CP = CodePath(arm="oracle", gpu="NVIDIA L4", implementation="oracle:bsa")
_LAYER = ("q", "nz_a", "nz_b", "mf", "o")


def _layered(n=6, *, between=None):
    """`embed`, n layers of `_LAYER` (one kernel per position), `norm`; with
    `between`, an unlayered position after that layer."""
    ops = [SeqOp(SeqKind.KERNEL, aten_op="aten::embedding", kernels=("embed",))]
    for l in range(n):
        ops += [SeqOp(SeqKind.KERNEL, layer=l, aten_op=f"aten::{k}", kernels=(k,)) for k in _LAYER]
        if between == l:
            ops.append(SeqOp(SeqKind.KERNEL, aten_op="aten::add", kernels=("between",)))
    ops.append(SeqOp(SeqKind.KERNEL, aten_op="aten::norm", kernels=("norm",)))
    return OpSequence(code_path=_CP, context_length=2048, batch=1, streams=(7,), ops=tuple(ops))


SEGMENTS = [{"layers": [0, 1], "correlations": [0, 99]},
            {"layers": [2, 3], "correlations": [100, 199]},
            {"layers": [4, 5], "correlations": [200, 299]}]


def _seg_of_layer(layer, segments=SEGMENTS):
    return next(k for k, s in enumerate(segments) if s["layers"][0] <= layer <= s["layers"][1])


def _stream(seq, *, keep=lambda p: True, segments=SEGMENTS):
    """The isolated original's nodes for `seq`, each with a correlation in the
    range of the segment that ran its position; `keep` drops positions."""
    out, seg, used = [], 0, {}
    for p, op in enumerate(seq.ops):
        if op.layer is not None:
            seg = _seg_of_layer(op.layer, segments)
        c = segments[seg]["correlations"][0] + used.get(seg, 0)
        used[seg] = used.get(seg, 0) + 1
        if keep(p):
            out.append(Node("kernel", op.kernels[0], 1e-6, c, None))
    return out


def _positions(seq, layer, names=_LAYER):
    return [p for p, op in enumerate(seq.ops) if op.layer == layer and op.kernels[0] in names]


def _in_segment(seq, k, positions):
    lo, hi = SEGMENTS[k]["layers"]
    return all(seq.ops[p].layer is not None and lo <= seq.ops[p].layer <= hi for p in positions)


def test_a_layer_lost_from_the_stream_is_reported_in_the_segment_that_lost_it():
    seq = _layered()
    lost = set(_positions(seq, 2))
    nodes = _stream(seq, keep=lambda p: p not in lost)
    whole = RP.match(seq, nodes)
    # the defect: the whole stream's match puts the lost layer at the end
    assert {seq.ops[p].layer for p in whole.unmatched_positions} == {5}
    m = RP.match_segmented(seq, nodes, SEGMENTS)
    # within a segment its layers are still one structure repeated, so the loss is
    # placed in its segment (layers 2 and 3), not in its layer
    assert len(m.unmatched_positions) == len(lost) and _in_segment(seq, 1, m.unmatched_positions)
    assert [len(s["unmatched"]) for s in m.by_segment] == [0, 5, 0]
    assert m.by_segment[1]["unmatched"] == list(m.unmatched_positions)
    assert [s["layers"] for s in m.by_segment] == [[0, 1], [2, 3], [4, 5]]
    assert [s["positions"] for s in m.by_segment] == [11, 10, 11]   # embed, norm at the ends
    assert [s["nodes"] for s in m.by_segment] == [11, 5, 11]
    assert m.device_s.keys() == {p for p in range(len(seq.ops)) if p not in m.unmatched_positions}


def test_no_stretch_is_matched_across_a_cut():
    seq = _layered()
    # the last two nodes before the cut at layer 2 and the first two after it, lost
    lost = set(_positions(seq, 1, ("mf", "o")) + _positions(seq, 2, ("q", "nz_a")))
    m = RP.match_segmented(seq, _stream(seq, keep=lambda p: p not in lost), SEGMENTS)
    assert set(m.unmatched_positions) == lost
    assert m.by_segment[0]["unmatched"] == _positions(seq, 1, ("mf", "o"))
    assert m.by_segment[1]["unmatched"] == _positions(seq, 2, ("q", "nz_a"))
    assert m.by_segment[2]["unmatched"] == []


def test_a_complete_stream_matches_by_segment_as_it_does_whole():
    seq = _layered()
    nodes = _stream(seq)
    whole, seg = RP.match(seq, nodes), RP.match_segmented(seq, nodes, SEGMENTS)
    assert seg.unmatched_positions == whole.unmatched_positions == ()
    assert seg.device_s == whole.device_s and seg.extra_nodes == whole.extra_nodes == ()
    assert (seg.n_expected, seg.n_nodes) == (whole.n_expected, whole.n_nodes)


def test_an_unlayered_position_goes_with_the_segment_that_ran_it():
    # an operator after layer 1's forward runs before the roll at layer 2's
    seq = _layered(between=1)
    where = RP.position_segments(seq, SEGMENTS)
    first = {k: min(p for p, s in where.items() if s == k) for k in range(3)}
    assert where[0] == 0 and where[len(seq.ops) - 1] == 2                  # embed, norm
    b = next(p for p, op in enumerate(seq.ops) if op.kernels[0] == "between")
    assert where[b] == 0 and first[1] == b + 1
    assert RP.match_segmented(seq, _stream(seq), SEGMENTS).unmatched_positions == ()


def test_a_node_outside_every_segment_is_refused():
    seq = _layered()
    nodes = _stream(seq)
    stray = Node("kernel", "q", 1e-6, 5000, None)
    with pytest.raises(RP.ReplayError, match="no segment holds"):
        RP.match_segmented(seq, nodes + [stray], SEGMENTS)
    with pytest.raises(RP.ReplayError, match="no segment holds"):
        RP.match_segmented(seq, nodes + [Node("kernel", "q", 1e-6, None, None)], SEGMENTS)


@pytest.mark.parametrize("bad", ["gap", "overlap", "short", "unordered"])
def test_segments_that_do_not_cover_the_layers_in_order_are_refused(bad):
    seq = _layered()
    segs = [dict(s) for s in SEGMENTS]
    if bad == "gap":
        segs[1] = {"layers": [3, 3], "correlations": [100, 199]}
    elif bad == "overlap":
        segs[1] = {"layers": [1, 3], "correlations": [100, 199]}
    elif bad == "short":
        segs = segs[:2]
    else:
        segs[1], segs[2] = segs[2], segs[1]
    with pytest.raises(RP.ReplayError, match="segments"):
        RP.match_segmented(seq, _stream(seq), segs)


def test_the_segments_come_from_the_replays_counts():
    counts = {"segments": 3, "segment_layer_ranges": [[0, 1], [2, 3], [4, 5]],
              "segment_correlations": [[0, 99], [100, 199], [200, 299]]}
    assert RP.segments_from_counts(counts) == SEGMENTS
    assert RP.segments_from_counts({"stream": "x"}) is None                # not segmented
    with pytest.raises(RP.ReplayError, match="segments"):
        RP.segments_from_counts(dict(counts, segment_correlations=[[0, 99]]))


# --- the cuts are made at layer boundaries only, and each segment's bounds are kept -----------------

def test_each_segment_is_a_block_of_whole_layers(tmp_path):
    _, _, counts = _replay(tmp_path, 6, per=2)
    assert counts["segment_layer_ranges"] == [[0, 1], [2, 3], [4, 5]]
    assert len(counts["segment_correlations"]) == 3
    _, _, counts = _replay(tmp_path, 7, per=3, name="seven")
    assert counts["segment_layer_ranges"] == [[0, 2], [3, 5], [6, 6]]


@pytest.mark.parametrize("how", ["a layer twice", "a layer skipped"])
def test_a_roll_that_is_not_at_its_layer_boundary_is_refused(tmp_path, how):
    toy = Toy(6)
    with TapeRecorder("cpu") as rec:
        toy()
    order = [0, 1, 2, 2, 3, 4, 5] if how == "a layer twice" else [0, 1, 2, 3, 5]

    def prefill():
        x = toy.x
        for j in order:
            x = torch.relu(toy.layers[j](x))
    with pytest.raises(RP.ReplayError, match="layer boundar"):
        RP.isolated_replay_compact(prefill, rec.tape, tmp_path / "x.json", device="cpu",
                                   layers=toy.layers, segment_layers=2, flush=lambda: None,
                                   sync=lambda: None)
    assert all(not l._forward_pre_hooks for l in toy.layers)
    assert not list(tmp_path.iterdir())


def test_the_stitch_keeps_each_segments_correlation_range(tmp_path):
    import test_components_replay as tcr
    parts = []
    for k in range(3):
        t = tcr.Trace()
        base = 10_000 * k
        t.dev_t = base
        t.x("user_annotation", RP.REPLAY_REGION, base, 1000)
        for i in range(4 + k):
            t.launch(base + 10 + 10 * i, f"s{k}k{i}")
        f = tmp_path / f"seg{k}.json"
        f.write_text(json.dumps(t.dict()))
        parts.append(ts.write_compact(f, cats=ts.REPLAY_CATS, fields=ts.DEVICE_FIELDS)["path"])
    out = ts.stitch_compact(parts, tmp_path / ("all" + ts.SUFFIX), region=REGION)
    # each session's correlations start at 1: 1..4, 1..5, 1..6, offset by the ones before
    assert out["correlations"] == [[1, 4], [6, 10], [12, 17]]
    head = json.loads(Path(out["path"]).open().readline())
    assert head["segment_correlations"] == out["correlations"]
    got = RP.replay_nodes(out["path"])
    for k, (lo, hi) in enumerate(out["correlations"]):
        assert [n.name for n in got if lo <= n.correlation <= hi] == [f"s{k}k{i}" for i in range(4 + k)]
    host = [write_part(tmp_path / f"h{k}.jsonl", part_events(100.0 * (k + 1), 2)) for k in range(2)]
    assert ts.stitch_compact(host, tmp_path / ("h" + ts.SUFFIX), region=REGION)["correlations"] == [None, None]


def test_a_launch_with_no_device_activity_is_counted():
    import test_components_replay as tcr
    t = tcr.Trace()
    t.x("user_annotation", RP.REPLAY_REGION, 10, 1000)
    t.launch(20, "a")
    lost = t.launch(30)                                   # launched, no activity in the trace
    t.x("cuda_runtime", "cudaStreamSynchronize", 40, 1.0, correlation=t.corr)   # no activity, not a launch
    t.corr += 1
    t.x("user_annotation", RP.FLUSH, 49, 5)
    t.launch(50)                                          # inside a flush: not the replay's
    t.launch(60, "b")
    assert RP.launches_without_activity(t.dict(), exclude=RP.FLUSH) == [lost]


def test_the_segment_report_names_each_segments_losses():
    seq = _layered()
    lost = set(_positions(seq, 2))
    m = RP.match_segmented(seq, _stream(seq, keep=lambda p: p not in lost), SEGMENTS)
    rep = RP.segment_report(m, SEGMENTS, [150, 151, 250])
    assert [r["launches_without_activity"] for r in rep] == [0, 2, 1]
    assert [len(r["unmatched"]) for r in rep] == [0, 5, 0]
    assert [r["layers"] for r in rep] == [[0, 1], [2, 3], [4, 5]]
    assert RP.segment_report(m, SEGMENTS, [999])[0]["launches_without_activity"] == 0


# --- the step: the isolated original matched by segment, its report in the step's result -----------

def test_the_step_matches_a_segmented_original_by_segment_and_a_whole_one_whole(monkeypatch):
    import t31_steps as st
    seq = _layered()
    lost = set(_positions(seq, 2))
    nodes = _stream(seq, keep=lambda p: p not in lost)
    monkeypatch.setattr(RP, "replay_nodes", lambda path, **k: nodes)
    counts = {"stream": "iso", "segments": 3, "segment_layer_ranges": [[0, 1], [2, 3], [4, 5]],
              "segment_correlations": [[0, 99], [100, 199], [200, 299]]}
    monkeypatch.setattr(RP, "launches_without_activity", lambda path, **k: [150])
    m, report = st.match_original(seq, "iso", counts)
    assert len(m.unmatched_positions) == 5 and _in_segment(seq, 1, m.unmatched_positions)
    assert [len(r["unmatched"]) for r in report] == [0, 5, 0]
    assert [r["launches_without_activity"] for r in report] == [0, 1, 0]
    m, report = st.match_original(seq, "iso", {"stream": "iso"})          # one session
    assert {seq.ops[p].layer for p in m.unmatched_positions} == {5} and report is None


def test_the_steps_result_names_only_the_segments_with_a_loss():
    import t31_steps as st
    entries = [
        {"arm": "oracle", "nominal_density": 0.5, "budget": 8192, "batch": 1,
         "isolated_segments": [{"segment": 0, "layers": [0, 1], "positions": 11, "nodes": 11,
                                "unmatched": [], "launches_without_activity": 0},
                               {"segment": 1, "layers": [2, 3], "positions": 10, "nodes": 5,
                                "unmatched": [6, 7], "launches_without_activity": 3}]},
        {"arm": "oracle", "nominal_density": 0.25, "budget": 8192, "batch": 1,
         "isolated_segments": [{"segment": 0, "layers": [0, 1], "positions": 11, "nodes": 11,
                                "unmatched": [], "launches_without_activity": 0}]},
        {"arm": "dense", "nominal_density": 1, "budget": 8192, "batch": 1},
    ]
    assert st.segment_findings(entries) == {
        "oracle_d0.5_L8192_b1": {"segments": 2, "with_loss": [
            {"segment": 1, "layers": [2, 3], "positions": 10, "nodes": 5, "unmatched": [6, 7],
             "launches_without_activity": 3}]},
        "oracle_d0.25_L8192_b1": {"segments": 1, "with_loss": []}}
