"""The researcher's rulings of 1 Nov 2026 after Shakedown 6.

Once-only host pricing (`host_profile.align_calls`), on Shakedown 6's own
sequence: XAttention at 2048 tokens on the L4, recorded on the instance
during the diagnosis (`results/shakedown6_l4_20261101/results/s6_diag/
s6_diag_fixed/xattention_None_2048/recording/sequence.json`, sha256
11f03cbf...1350). Its null pass was refused there: "the recording's call 65
is to aten::where in layer 0 (position 77, 2 operation(s)), and the pass's
launching call 65 was to aten::scalar_tensor in layer 0". The committed
fixture is that sequence up to the first operation of layer 1 (187 of 4948
operations, unchanged; `tests/fixtures/s6_xattention_2048_layer0.json.gz`);
with `results/` present the whole sequence is tested too.

The pass's calls are the dispatcher's view of the recording: one per
recorded call, at the operator the pass meets it at, and before each
`where` whose recording nests a `scalar_tensor`, that `scalar_tensor`, as
the refusal showed. The conservation test: every pass call is used exactly
once, and the positions' costs sum to the pass's calls plus one floor per
further kernel; the bare prices sum to the bare calls' prices.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from attnbench import cpu_boundary as cb
from attnbench.components import bare_timer as bt
from attnbench.components import host_profile as hp
from attnbench.components import op_sequence as S

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "s6_xattention_2048_layer0.json.gz"
BANKED = (HERE.parent / "results" / "shakedown6_l4_20261101" / "results" / "s6_diag" /
          "s6_diag_fixed" / "xattention_None_2048" / "recording" / "sequence.json")
WHERE_SCALAR = ("aten::where", "aten::scalar_tensor", "aten::fill_")


def _fixture() -> S.OpSequence:
    return S.from_record(json.loads(gzip.decompress(FIXTURE.read_bytes())))


def _banked() -> S.OpSequence:
    return S.from_record(json.loads(BANKED.read_text()))


def _dispatcher_view(seq: S.OpSequence):
    """(isolated positions, the pass's launching calls as (operator, layer))."""
    iso = hp.isolated_positions(seq)
    visible = lambda ch: next((n for n in ch if not cb.composite_implicit(n)), ch[0])
    names = {visible(hp.chain_of(o)) for i, o in enumerate(seq.ops)
             if hp._is_issue(o) and i not in iso}
    met = hp.met_at(seq, names | {"aten::scalar_tensor"}, skip=iso)
    issued = []
    for name, layer, pos in hp.recorded_calls(seq, met, skip=iso):
        issued += [("aten::scalar_tensor", layer) for p in pos
                   if seq.ops[p].chain == WHERE_SCALAR]
        issued.append((name, layer))
    return iso, issued


def _log(issued) -> hp.PassLog:
    log = hp.PassLog()
    for k, (name, layer) in enumerate(issued):
        log.issues.append(hp.Issue(name=name, layer=layer, gap_s=(k + 1) * 1e-6,
                                   launch_s=3e-6))
        log.launches.append(3e-6)
    log.isolated.append(5e-6)            # the embedding's gather, kept out
    return log


def _conserves(seq: S.OpSequence) -> None:
    iso, issued = _dispatcher_view(seq)
    log = _log(issued)
    host, source, _ = hp.align(seq, log, isolated=iso)
    met = hp.met_at(seq, (e.name for e in log.issues), skip=iso)
    groups = hp.align_calls(seq, met, issued, skip=iso)
    # every pass call used exactly once, in order
    assert [i for i, _, _ in groups] == list(range(len(issued)))
    # every priced position in exactly one share
    shares = [p for _, _, share in groups for p in share]
    assert sorted(shares) == sorted(host) and len(shares) == len(set(shares))
    floor = 3e-6
    extra = len(shares) - len(issued)
    assert sum(host.values()) == pytest.approx(
        sum(e.gap_s + e.launch_s for e in log.issues) + floor * extra, rel=1e-12)

    # the bare prices, aligned the same way, sum to the bare calls' prices
    null = hp.HostProfile(host_s={**host, **{i: 5e-6 for i in iso}}, reps_s={},
                          launch_floor_s=floor, reps=1, sequence_sha256="x",
                          cost_source={**{i: s for i, s in source.items() if s != hp.NULL_PASS},
                                       **{i: hp.ISOLATED_REPLAY for i in iso}})
    bp = bt.BarePass(floor_s=1e-6, reps=bt.MIN_REPS)
    bp.calls = [bt.BareCall(n, layer, direct_s=(k + 2) * 1e-6, chain_s=None)
                for k, (n, layer) in enumerate(issued)]
    profile, record = bt.bare_profile(seq, bp, null, aggregate_tol=0.10)
    priced = sum(v for i, v in profile.host_s.items() if i not in iso)
    assert priced == pytest.approx(sum(c.direct_s for c in bp.calls), rel=1e-12)
    assert record["calls_priced"] == len(bp.calls)


def test_the_fixture_holds_shakedown_6s_refusal():
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    met = hp.met_at(seq, (n for n, _ in issued), skip=iso)
    calls = hp.recorded_calls(seq, met, skip=iso)
    assert calls[65] == ("aten::where", 0, (77, 78))
    assert seq.ops[77].chain == WHERE_SCALAR
    assert issued[65] == ("aten::scalar_tensor", 0) and issued[66] == ("aten::where", 0)
    assert sum(1 for o in seq.ops if o.chain == WHERE_SCALAR) == 5


def test_each_pass_call_is_priced_once_inside_the_call_that_nests_it():
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    met = hp.met_at(seq, (n for n, _ in issued), skip=iso)
    groups = {i: (n, share) for i, n, share in hp.align_calls(seq, met, issued, skip=iso)}
    assert groups[65] == (65, (77,))       # scalar_tensor: its fill_
    assert groups[66] == (65, (78,))       # where: its own kernel
    log = _log(issued)
    host, source, _ = hp.align(seq, log, isolated=iso)
    assert host[77] == pytest.approx(66e-6 + 3e-6) and host[78] == pytest.approx(67e-6 + 3e-6)
    assert source[77] == hp.CHAIN and source[78] == hp.NULL_PASS


def test_conservation_on_the_shakedown_6_fixture():
    _conserves(_fixture())


@pytest.mark.skipif(not BANKED.exists(), reason=f"needs the banked sequence {BANKED}")
def test_conservation_on_the_whole_shakedown_6_sequence():
    seq = _banked()
    assert len(seq.ops) == 4948
    assert sum(1 for o in seq.ops if o.chain == WHERE_SCALAR) == 140
    _conserves(seq)


def test_a_nested_call_with_no_call_of_its_own_after_it_is_still_refused():
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    del issued[66]                         # the where after the scalar_tensor
    with pytest.raises(hp.ProfileMismatch, match="call 65 is to aten::where"):
        hp.align(seq, _log(issued), isolated=iso)


def test_a_call_the_recording_does_not_nest_there_is_still_refused():
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    issued.insert(65, ("aten::sum", 0))    # sum is not in where's chains
    with pytest.raises(hp.ProfileMismatch, match="call 65 is to aten::where"):
        hp.align(seq, _log(issued), isolated=iso)


def test_no_more_nested_calls_than_the_call_has_further_kernels():
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    issued.insert(65, ("aten::scalar_tensor", 0))   # two before a two-kernel call
    with pytest.raises(hp.ProfileMismatch):
        hp.align(seq, _log(issued), isolated=iso)


def test_the_stand_in_gives_scalar_tensor_its_launch():
    # On the L4 every scalar_tensor (140 per XAttention prefill) was under a
    # where, with a fill_ child: a launch. On a CPU it is a leaf.
    assert cb._may_launch("aten::scalar_tensor")
