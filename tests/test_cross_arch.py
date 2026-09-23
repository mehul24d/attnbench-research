"""The cross-architecture results path, exercised end to end on CPU with
two synthetic hosts.

This mechanism carries the study's entire hardware-conditional claim
("backend X wins on an L4 and loses on an H100") and had never been run
before these tests. Everything here is synthetic parquet -- no GPU, no
model -- because the thing under test is the bookkeeping, and the
bookkeeping is where an invalid comparison would be formed silently.

The invariant being defended throughout: **a latency ratio is only ever
taken between two rows from the same physical machine.** Architectures are
compared between ratios, never between raw latencies.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from attnbench.analysis.cross_arch import (
    CrossArchError, compare_across_architectures, hosts_by_architecture,
    load_segments, speedup_within_host)
from attnbench.sweep import HostMismatchError, check_host_continuity

L4, H100 = "NVIDIA L4", "NVIDIA H100 80GB HBM3"


# A plausible 40-hex commit. Real rows carry one from provenance.capture();
# fixtures must too, or every test here would exercise the allow_unverified
# path rather than the one production uses.
COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40


def _row(host, gpu_name, backend, config_key, latency, ok=True,
         git_commit=COMMIT):
    return dict(host=host, gpu_name=gpu_name, backend=backend,
                config_key=config_key, latency_ms_p50=latency, ok=ok,
                git_commit=git_commit)


def _write(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    p = tmp_path / name
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def _two_machine_sweep(tmp_path):
    """The legitimate workflow: each machine writes its own file, and each
    re-measures the dense baseline locally.

    Numbers are chosen so the H100 is faster in absolute terms for BOTH
    backends -- the trap this path exists to avoid. Compared raw, block_sparse
    on the H100 would look faster than dense on the L4 and mean nothing.
    Compared as within-host ratios, block_sparse is 2.0x on the L4 and 0.8x
    on the H100: a genuine flip.
    """
    l4 = _write(tmp_path, "l4.parquet", [
        _row("host-a", L4, "sdpa_math", "cfg1", 100.0),
        _row("host-a", L4, "block_sparse", "cfg1", 50.0),
    ])
    h100 = _write(tmp_path, "h100.parquet", [
        _row("host-b", H100, "sdpa_math", "cfg1", 20.0),
        _row("host-b", H100, "block_sparse", "cfg1", 25.0),
    ])
    return l4, h100


# ---------------------------------------------------------------------------
# collection-time guard: check_host_continuity
# ---------------------------------------------------------------------------

def test_resuming_into_another_hosts_file_is_refused(tmp_path):
    """The collection-time half of the invariant. Without this, a resumed
    sweep appends the new machine's rows next to the old machine's in one
    file, and load_done_keys (which keys on host) makes them look like
    ordinary un-run cells."""
    p = _write(tmp_path, "sweep.parquet", [_row("host-a", L4, "sdpa_math", "cfg1", 100.0)])
    with pytest.raises(HostMismatchError, match="host-a"):
        check_host_continuity(p, "host-b")


def test_resuming_on_the_same_host_is_allowed(tmp_path):
    """The ordinary case -- a process restart on the same machine -- must
    not be blocked, or the guard would make resume useless."""
    p = _write(tmp_path, "sweep.parquet", [_row("host-a", L4, "sdpa_math", "cfg1", 100.0)])
    check_host_continuity(p, "host-a")  # must not raise


def test_two_architecture_workflow_does_not_trip_the_guard(tmp_path):
    """The legitimate two-machine workflow writes SEPARATE files, so the
    continuity guard never fires. If this failed, the guard would be
    blocking the very workflow it exists to make safe."""
    l4, h100 = _two_machine_sweep(tmp_path)
    check_host_continuity(l4, "host-a")
    check_host_continuity(h100, "host-b")


def test_a_fresh_file_on_a_new_host_is_allowed(tmp_path):
    """Pointing --out at a new directory is the prescribed recovery. A
    non-existent checkpoint must be permitted for any host."""
    check_host_continuity(tmp_path / "does_not_exist.parquet", "host-b")


# ---------------------------------------------------------------------------
# analysis-time join
# ---------------------------------------------------------------------------

def test_segments_are_joined_only_at_analysis_time(tmp_path):
    l4, h100 = _two_machine_sweep(tmp_path)
    df = load_segments([l4, h100])
    assert len(df) == 4
    assert set(df["host"]) == {"host-a", "host-b"}
    assert set(df["gpu_name"]) == {L4, H100}


def test_segment_missing_provenance_columns_is_refused(tmp_path):
    """A file without host/gpu_name cannot be kept on the right side of a
    same-machine comparison, so it is refused at load rather than
    concatenated into something that looks fine."""
    bad = _write(tmp_path, "bad.parquet",
                 [{"backend": "sdpa_math", "config_key": "cfg1",
                   "latency_ms_p50": 100.0}])
    with pytest.raises(CrossArchError, match="missing"):
        load_segments([bad])


def test_hosts_by_architecture_surfaces_multiple_machines(tmp_path):
    a = _write(tmp_path, "a.parquet", [_row("host-a", L4, "sdpa_math", "cfg1", 100.0)])
    b = _write(tmp_path, "b.parquet", [_row("host-c", L4, "sdpa_math", "cfg1", 110.0)])
    mapping = hosts_by_architecture(load_segments([a, b]))
    assert mapping[L4] == ["host-a", "host-c"]


# ---------------------------------------------------------------------------
# the invariant: ratios are formed within a host, never across
# ---------------------------------------------------------------------------

def test_speedups_are_computed_against_the_same_machines_baseline(tmp_path):
    l4, h100 = _two_machine_sweep(tmp_path)
    speedups = speedup_within_host(load_segments([l4, h100]),
                                   baseline_backend="sdpa_math")
    by_host = {s.host: s for s in speedups}
    # L4: 100/50 = 2.0 against ITS OWN baseline, not the H100's 20ms
    assert by_host["host-a"].speedup == pytest.approx(2.0)
    assert by_host["host-a"].baseline_latency_ms == 100.0
    # H100: 20/25 = 0.8
    assert by_host["host-b"].speedup == pytest.approx(0.8)
    assert by_host["host-b"].baseline_latency_ms == 20.0


def test_a_machine_missing_the_baseline_raises_rather_than_borrowing_one(tmp_path):
    """The failure this module exists to prevent. Falling back to another
    machine's baseline yields a plausible-looking number that is a
    comparison between two GPUs wearing a backend's name."""
    l4 = _write(tmp_path, "l4.parquet", [
        _row("host-a", L4, "sdpa_math", "cfg1", 100.0),
        _row("host-a", L4, "block_sparse", "cfg1", 50.0),
    ])
    h100 = _write(tmp_path, "h100.parquet", [
        _row("host-b", H100, "block_sparse", "cfg1", 25.0),   # no baseline!
    ])
    with pytest.raises(CrossArchError, match="not measured on host"):
        speedup_within_host(load_segments([l4, h100]), baseline_backend="sdpa_math")


def test_raw_cross_host_latency_comparison_is_never_formed(tmp_path):
    """Every emitted ratio's baseline must belong to its own host."""
    l4, h100 = _two_machine_sweep(tmp_path)
    df = load_segments([l4, h100])
    baselines = {(r["host"], r["config_key"]): r["latency_ms_p50"]
                 for _, r in df[df["backend"] == "sdpa_math"].iterrows()}
    for s in speedup_within_host(df, baseline_backend="sdpa_math"):
        assert s.baseline_latency_ms == baselines[(s.host, s.config_key)]


def test_a_per_cell_baseline_gap_is_skipped_not_raised(tmp_path):
    """A baseline that OOM'd at one shape is a legitimate gap, unlike a
    whole machine missing the baseline -- the two must not be conflated."""
    p = _write(tmp_path, "l4.parquet", [
        _row("host-a", L4, "sdpa_math", "cfg1", 100.0),
        _row("host-a", L4, "block_sparse", "cfg1", 50.0),
        _row("host-a", L4, "block_sparse", "cfg_oom", 50.0),  # no baseline here
    ])
    speedups = speedup_within_host(load_segments([p]), baseline_backend="sdpa_math")
    assert {s.config_key for s in speedups} == {"cfg1"}


def test_failed_rows_are_excluded_from_speedups(tmp_path):
    p = _write(tmp_path, "l4.parquet", [
        _row("host-a", L4, "sdpa_math", "cfg1", 100.0),
        _row("host-a", L4, "block_sparse", "cfg1", 1.0, ok=False),
    ])
    assert speedup_within_host(load_segments([p]), baseline_backend="sdpa_math") == []


# ---------------------------------------------------------------------------
# the hardware-conditional claim itself
# ---------------------------------------------------------------------------

def test_architecture_comparison_detects_a_flip(tmp_path):
    """The study's headline shape: faster than dense on one architecture,
    slower on another. Note this is invisible in raw latency -- block_sparse
    on the H100 (25ms) beats dense on the L4 (100ms) by 4x, which means
    nothing."""
    l4, h100 = _two_machine_sweep(tmp_path)
    comparisons = compare_across_architectures(
        speedup_within_host(load_segments([l4, h100]), baseline_backend="sdpa_math"))
    assert len(comparisons) == 1
    c = comparisons[0]
    assert c.architectures == sorted([L4, H100])
    assert c.speedup_by_architecture[L4] == pytest.approx(2.0)
    assert c.speedup_by_architecture[H100] == pytest.approx(0.8)
    assert c.flips()


def test_a_backend_measured_on_one_architecture_only_is_excluded(tmp_path):
    """No cross-architecture claim exists for it; emitting a one-entry
    comparison invites reading it as one."""
    p = _write(tmp_path, "l4.parquet", [
        _row("host-a", L4, "sdpa_math", "cfg1", 100.0),
        _row("host-a", L4, "block_sparse", "cfg1", 50.0),
    ])
    assert compare_across_architectures(
        speedup_within_host(load_segments([p]), baseline_backend="sdpa_math")) == []


def test_consistent_winner_is_not_reported_as_a_flip(tmp_path):
    l4 = _write(tmp_path, "l4.parquet", [
        _row("host-a", L4, "sdpa_math", "cfg1", 100.0),
        _row("host-a", L4, "block_sparse", "cfg1", 50.0),
    ])
    h100 = _write(tmp_path, "h100.parquet", [
        _row("host-b", H100, "sdpa_math", "cfg1", 20.0),
        _row("host-b", H100, "block_sparse", "cfg1", 10.0),
    ])
    c = compare_across_architectures(
        speedup_within_host(load_segments([l4, h100]), baseline_backend="sdpa_math"))[0]
    assert not c.flips()


# ---------------------------------------------------------------------------
# preemption: the same case, arriving by accident
# ---------------------------------------------------------------------------

def test_preemption_recovery_is_a_new_file_per_segment(tmp_path):
    """A preempted Spot instance resuming elsewhere produces exactly the
    cross-host case. The prescribed recovery is a new file per segment,
    joined here -- and the guard blocks the wrong alternative (resuming into
    the original file)."""
    seg1 = _write(tmp_path, "seg1.parquet", [
        _row("host-a", L4, "sdpa_math", "cfg1", 100.0),
        _row("host-a", L4, "block_sparse", "cfg1", 50.0),
    ])
    # ... preemption ... resumes on a different machine, same GPU model
    seg2 = _write(tmp_path, "seg2.parquet", [
        _row("host-c", L4, "sdpa_math", "cfg2", 200.0),
        _row("host-c", L4, "block_sparse", "cfg2", 80.0),
    ])

    # the wrong recovery is blocked
    with pytest.raises(HostMismatchError):
        check_host_continuity(seg1, "host-c")

    # the right one works, and each segment keeps its own baseline
    speedups = speedup_within_host(load_segments([seg1, seg2]),
                                   baseline_backend="sdpa_math")
    by_config = {s.config_key: s for s in speedups}
    assert by_config["cfg1"].speedup == pytest.approx(2.0)
    assert by_config["cfg2"].speedup == pytest.approx(2.5)


def test_many_segments_per_architecture_are_supported(tmp_path):
    """Nothing downstream may assume one file per architecture: repeated
    preemption produces many segments, and a re-rental produces several
    hosts for one gpu_name."""
    paths = []
    for i, host in enumerate(["host-a", "host-c", "host-d"]):
        paths.append(_write(tmp_path, f"seg{i}.parquet", [
            _row(host, L4, "sdpa_math", f"cfg{i}", 100.0),
            _row(host, L4, "block_sparse", f"cfg{i}", 50.0),
        ]))
    df = load_segments(paths)
    assert hosts_by_architecture(df)[L4] == ["host-a", "host-c", "host-d"]
    assert len(speedup_within_host(df, baseline_backend="sdpa_math")) == 3


def test_repeated_measurements_of_one_architecture_average_ratios_not_latencies(tmp_path):
    """Averaging ratios across machines is valid because each is already
    normalised by its own machine's baseline. Averaging latencies would not
    be -- here the two L4 hosts have very different absolute speeds but the
    same 2.0x ratio, and the average must be 2.0, not something pulled by
    the slower machine."""
    fast = _write(tmp_path, "fast.parquet", [
        _row("host-a", L4, "sdpa_math", "cfg1", 100.0),
        _row("host-a", L4, "block_sparse", "cfg1", 50.0),
    ])
    slow = _write(tmp_path, "slow.parquet", [
        _row("host-c", L4, "sdpa_math", "cfg1", 400.0),
        _row("host-c", L4, "block_sparse", "cfg1", 200.0),
    ])
    h100 = _write(tmp_path, "h100.parquet", [
        _row("host-b", H100, "sdpa_math", "cfg1", 20.0),
        _row("host-b", H100, "block_sparse", "cfg1", 25.0),
    ])
    c = compare_across_architectures(
        speedup_within_host(load_segments([fast, slow, h100]),
                            baseline_backend="sdpa_math"))[0]
    assert c.speedup_by_architecture[L4] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Segment integrity. The threats here are self-inflicted, not adversarial:
# every one of these has a real 2026-09-03 incident behind it.
# ---------------------------------------------------------------------------

def test_segments_from_different_commits_are_refused(tmp_path):
    """A code change between segments makes them different experiments.

    The concrete case: the `to_dense_bool` sub-block causality fix changed
    what a causal block-sparse mask means. Segments either side of it are not
    comparable, and averaging them yields a number belonging to neither.
    """
    a = _write(tmp_path, "a.parquet", [_row("host-a", L4, "sdpa_math", "c1", 100.0)])
    b = _write(tmp_path, "b.parquet", [_row("host-b", H100, "sdpa_math", "c1", 20.0,
                                            git_commit=OTHER_COMMIT)])
    with pytest.raises(CrossArchError, match="different commits"):
        load_segments([a, b])

    joined = load_segments([a, b], allow_mixed_commits=True)
    assert len(joined) == 2, "the escape hatch must still work when asked for"


def test_a_segment_without_a_usable_commit_is_refused(tmp_path):
    """`git rev-parse HEAD` echoes the literal string 'HEAD' to stdout when a
    repository has no commits, so provenance recorded "HEAD" as if it were a
    SHA. Files written that way must not be read as verified.
    """
    bad = _write(tmp_path, "bad.parquet",
                 [_row("host-a", L4, "sdpa_math", "c1", 100.0, git_commit="HEAD")])
    with pytest.raises(CrossArchError, match="no usable git commit"):
        load_segments([bad])

    assert len(load_segments([bad], allow_unverified=True)) == 1


def test_a_segment_missing_the_commit_column_is_refused(tmp_path):
    rows = [_row("host-a", L4, "sdpa_math", "c1", 100.0)]
    rows[0].pop("git_commit")
    p = _write(tmp_path, "nocol.parquet", rows)
    with pytest.raises(CrossArchError, match="no usable git commit"):
        load_segments([p])


def test_duplicate_cells_are_refused_rather_than_silently_deduplicated(tmp_path):
    """Two measurements of one (config_key, backend, host) is a question.

    Last-wins would hide whether this was a deliberate re-measure or the same
    checkpoint joined twice -- and those want opposite treatment.
    """
    a = _write(tmp_path, "a.parquet", [_row("host-a", L4, "sdpa_math", "c1", 100.0)])
    b = _write(tmp_path, "b.parquet", [_row("host-a", L4, "sdpa_math", "c1", 103.0)])
    with pytest.raises(CrossArchError, match="duplicate"):
        load_segments([a, b])


def test_the_same_cell_on_different_hosts_is_not_a_duplicate(tmp_path):
    """The legitimate two-architecture workflow must not trip the duplicate
    check -- the same config on two machines is the entire point."""
    l4, h100 = _two_machine_sweep(tmp_path)
    assert len(load_segments([l4, h100])) == 4


def test_segment_digest_is_order_independent_but_content_sensitive(tmp_path):
    from attnbench.analysis.cross_arch import segment_digest

    rows = [_row("host-a", L4, "sdpa_math", "c1", 100.0),
            _row("host-a", L4, "block_sparse", "c1", 50.0)]
    d1 = segment_digest(pd.DataFrame(rows))
    d2 = segment_digest(pd.DataFrame(list(reversed(rows))))
    assert d1 == d2, "row order is incidental and must not change the digest"

    changed = [dict(rows[0], latency_ms_p50=101.0), rows[1]]
    assert segment_digest(pd.DataFrame(changed)) != d1


def test_segment_digest_handles_a_cell_that_produced_no_latency():
    """A NaN latency is a real banked row, not a malformed one.

    Every other fixture in this file hands `segment_digest` concrete floats,
    which is why the suite stayed green while
    `scripts/run_cross_arch_analysis.py` could not run at all: under pandas 3
    `astype(str)` leaves missing values missing, `"|".join` gets a float, and
    the load raises `TypeError` before any analysis happens. The real tree has
    72 such rows in segment 1 alone (unsupported configs and OOMs), so the
    path this exercises is the ordinary one, not an edge case.
    """
    from attnbench.analysis.cross_arch import segment_digest

    rows = [_row("host-a", L4, "sdpa_math", "c1", 100.0),
            _row("host-a", L4, "block_sparse", "c1", float("nan"), ok=False)]

    # Anti-vacuity: if a future edit makes this fixture all-concrete, the test
    # goes on passing while covering nothing. Assert the NaN is really here.
    assert pd.DataFrame(rows)["latency_ms_p50"].isna().sum() == 1

    d = segment_digest(pd.DataFrame(rows))
    assert len(d) == 16

    # Order-independent, as for any other segment.
    assert segment_digest(pd.DataFrame(list(reversed(rows)))) == d

    # And still content-sensitive: the failed cell is part of what was
    # measured, so dropping it must change the digest.
    assert segment_digest(pd.DataFrame(rows[:1])) != d


def test_load_segments_admits_a_segment_containing_a_failed_cell(tmp_path):
    """The production entry point, at the point it actually broke.

    `load_segments` calls `segment_digest` on the raw frame BEFORE any
    `ok`/status filtering, so one unsupported cell anywhere in a sweep was
    enough to take the whole cross-architecture analysis down.
    """
    rows = [_row("host-a", L4, "sdpa_math", "c1", 100.0),
            _row("host-a", L4, "block_sparse", "c1", float("nan"), ok=False),
            _row("host-b", H100, "sdpa_math", "c1", 50.0),
            _row("host-b", H100, "block_sparse", "c1", 25.0)]
    path = _write(tmp_path, "seg_with_failed_cell.parquet", rows)

    loaded = load_segments([path])
    assert len(loaded) == 4
    assert loaded["latency_ms_p50"].isna().sum() == 1
    assert loaded["segment_digest"].nunique() == 1


def test_digest_is_attached_to_every_loaded_row(tmp_path):
    l4, h100 = _two_machine_sweep(tmp_path)
    joined = load_segments([l4, h100])
    assert "segment_digest" in joined.columns
    assert joined.groupby("segment")["segment_digest"].nunique().eq(1).all()
    assert joined["segment_digest"].nunique() == 2


# --- flip materiality --------------------------------------------------------
#
# Added 2026-09-05 with a flat 5% bar, corrected 2026-09-06. The flat bar was
# wrong in the dangerous direction: it certified four flips this dataset's own
# instrument cannot resolve. See RATIO_RESOLUTION_BANDS.

def test_flip_margin_takes_the_weaker_side():
    """A flip is only as strong as its half nearest parity. 0.999 against
    1.44 is not a reversal -- it is one card being fast and the other being
    exactly average, and `flips()` alone cannot tell those apart."""
    from attnbench.analysis.cross_arch import ArchitectureComparison

    wide = ArchitectureComparison("gla", "k", {"L4": 0.823, "A100": 1.443})
    assert wide.flip_margin == pytest.approx(0.177)
    assert wide.flips()

    nominal = ArchitectureComparison("flex", "k", {"L4": 1.001, "A100": 0.837})
    assert nominal.flip_margin == pytest.approx(0.001)
    assert nominal.flips()
    assert not nominal.flips_materially()


def test_materiality_is_measured_from_the_latency_not_fixed():
    """The same flip margin is material at 40 ms and not at 2 ms, because the
    instrument resolves differently there. A fixed bar cannot express that."""
    from attnbench.analysis.cross_arch import ArchitectureComparison

    margin = dict(backend="gla", config_key="k",
                  speedup_by_architecture={"L4": 0.823, "A100": 1.443})

    slow = ArchitectureComparison(
        **margin, min_latency_by_architecture={"L4": 42.0, "A100": 33.0})
    assert slow.resolution == pytest.approx(0.133)
    assert slow.flips_materially(), "0.177 clears 0.133 at long latencies"

    fast = ArchitectureComparison(
        **margin, min_latency_by_architecture={"L4": 9.5, "A100": 2.05})
    assert fast.resolution == pytest.approx(0.272)
    assert not fast.flips_materially(), (
        "0.177 does not clear 0.272 -- two L4 hosts in this dataset disagree "
        "with each other by 27.2% at sub-3 ms latencies")


def test_the_shortest_latency_on_either_side_sets_the_resolution():
    """A comparison is as imprecise as its least precise half, and the flip
    has to survive both halves. Taking the mean, or the A100's own latency,
    would let one long-running side buy precision for a short-running one."""
    from attnbench.analysis.cross_arch import ArchitectureComparison

    c = ArchitectureComparison("gla", "k", {"L4": 0.8, "A100": 1.4},
                               min_latency_by_architecture={"L4": 90.0,
                                                            "A100": 2.0})
    assert c.min_latency_ms == pytest.approx(2.0)
    assert c.resolution == pytest.approx(0.272)


def test_an_unknown_latency_buys_the_widest_band_not_the_narrowest():
    """A missing measurement must not buy a claim more precision than a
    measured one -- the same direction rule as `_both_locked` on None."""
    from attnbench.analysis.cross_arch import (ArchitectureComparison,
                                                ratio_resolution)

    assert ratio_resolution(None) == pytest.approx(0.272)
    c = ArchitectureComparison("gla", "k", {"L4": 0.823, "A100": 1.443})
    assert c.min_latency_ms is None
    assert c.resolution == pytest.approx(0.272)
    assert not c.flips_materially()


def test_the_bands_come_from_measured_spread_not_a_round_number():
    """Both values are maxima observed in this study's own cross-host data.
    A round 0.05 or 0.10 here would be a guess wearing a measurement's
    clothes."""
    from attnbench.analysis.cross_arch import RATIO_RESOLUTION_BANDS

    assert RATIO_RESOLUTION_BANDS == ((5.0, 0.272), (float("inf"), 0.133))
    for _, r in RATIO_RESOLUTION_BANDS:
        assert r * 100 % 1 != 0, "a round percentage is not a measurement"


def test_a_nonmaterial_flip_explains_why_it_is_not_reportable():
    from attnbench.analysis.cross_arch import ArchitectureComparison

    c = ArchitectureComparison("fa2", "k", {"L4": 0.921, "A100": 1.090},
                               min_latency_by_architecture={"L4": 2.3,
                                                            "A100": 0.83})
    text = c.resolution_caveat()
    assert "0.272" in text and "Not reportable" in text
    assert ArchitectureComparison("fa2", "k", {"L4": 1.2, "A100": 1.3}
                                  ).resolution_caveat() == ""


def test_an_explicit_margin_still_overrides_at_the_call_site():
    from attnbench.analysis.cross_arch import ArchitectureComparison

    c = ArchitectureComparison("gla", "k", {"L4": 0.823, "A100": 1.443},
                               min_latency_by_architecture={"A100": 2.0})
    assert not c.flips_materially()
    assert c.flips_materially(margin=0.05)


def test_a_one_sided_comparison_has_no_flip_margin():
    from attnbench.analysis.cross_arch import ArchitectureComparison

    assert ArchitectureComparison("fa2", "k", {"L4": 2.0}).flip_margin == 0.0
    assert not ArchitectureComparison("fa2", "k", {"L4": 2.0}).flips_materially()


# --- coverage_gaps: the exclusion stays, the silence does not ---------------

def _sp(backend, cfg, gpu, host="h"):
    from attnbench.analysis.cross_arch import Speedup
    return Speedup(host=f"{host}-{gpu}", gpu_name=gpu, backend=backend,
                   config_key=cfg, latency_ms=10.0, baseline_latency_ms=10.0,
                   clocks_locked=False)


def test_coverage_gaps_names_what_the_comparison_drops():
    from attnbench.analysis.cross_arch import coverage_gaps
    sp = [_sp("fa2", "c1", "L4"), _sp("fa2", "c1", "A100"),   # compared
          _sp("fa2", "c2", "L4"),                             # dropped
          _sp("flex", "c1", "L4"), _sp("flex", "c1", "A100")] # fully covered
    assert len(compare_across_architectures(sp)) == 2
    gaps = coverage_gaps(sp)
    assert [g.backend for g in gaps] == ["fa2"]
    assert (gaps[0].pairs_total, gaps[0].pairs_compared, gaps[0].pairs_dropped) == (2, 1, 1)
    assert not gaps[0].wholly_absent


def test_a_backend_missing_from_one_architecture_is_named_not_vanished():
    """The case the old code handled silently: every pair of a backend
    dropped, and nothing in the output mentioned that backend at all."""
    from attnbench.analysis.cross_arch import coverage_gaps
    sp = [_sp("flex", "c1", "L4"), _sp("flex", "c1", "A100"),
          _sp("sage", "c1", "L4"), _sp("sage", "c2", "L4")]
    assert {c.backend for c in compare_across_architectures(sp)} == {"flex"}
    (g,) = coverage_gaps(sp)
    assert g.backend == "sage" and g.absent_from == ("A100",)
    assert g.pairs_compared == 0 and "ALL 2 pairs dropped" in g.describe()


def test_full_coverage_reports_no_gaps():
    from attnbench.analysis.cross_arch import coverage_gaps
    sp = [_sp("fa2", "c1", "L4"), _sp("fa2", "c1", "A100")]
    assert coverage_gaps(sp) == []
