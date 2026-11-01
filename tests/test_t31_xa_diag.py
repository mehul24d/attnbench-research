"""The XAttention crash diagnostic (`scripts/t31_xa_diag.py`; researcher's
ruling, 2 Nov 2026), on a CPU: the case plans, the session's order, deadline
and sanitizer choice with a fake runner, the unprofiled recording's call and
synchronise pattern, and case E's hook guard. The cases themselves run on
the L4; behind the CPU boundary they are the card's dry run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import t31_xa_diag as xd  # noqa: E402

CFG = {"lengths": [2048, 4096, 8192, 16384, 32768]}
XA_PROBES = (20480, 28672, 32768)


def test_c_is_step_4s_order():
    """T3.1 step 4: each arm's probes, then its grid lengths, arm by arm."""
    plan = xd.case_plan("C", CFG)
    arms = [a for a, _, _ in plan]
    assert list(dict.fromkeys(arms)) == list(xd.C_ARMS)
    xa = [(k, L) for a, k, L in plan if a == "xattention"]
    assert xa == [("probe", L) for L in XA_PROBES] + [("full", L) for L in CFG["lengths"]]
    dense = [(k, L) for a, k, L in plan if a == "dense"]
    assert dense == [("probe", 2048), ("probe", 4096)] + [("full", L) for L in CFG["lengths"]]
    # the crash point is the first XAttention full recording, after 3 arms
    first_full = next(i for i, r in enumerate(plan) if r == ("xattention", "full", 2048))
    assert plan[first_full - 1] == ("xattention", "probe", 32768)


def test_b_a_and_e():
    assert xd.case_plan("B", CFG) == xd.case_plan("C", CFG)[21:29]
    assert xd.case_plan("A", CFG) == [("xattention", "full", 2048)] * xd.A_REPEATS
    e = xd.case_plan("E", CFG)
    assert "xattention_noassert" not in {a for a, _, _ in e}
    assert e == [r for r in xd.case_plan("C", CFG) if r[0] != "xattention_noassert"]


class FakeRunner:
    """Writes each case's case.json with the status asked for; records the
    commands, environments and limits."""

    def __init__(self, session: Path, statuses: dict, seconds: dict | None = None,
                 clock=None):
        self.session, self.statuses, self.calls = session, statuses, []
        self.seconds, self.clock = seconds or {}, clock

    def __call__(self, cmd, log, *, env, timeout):
        sanitizer = cmd[0] == "compute-sanitizer"
        case = cmd[cmd.index("case") + 1]
        self.calls.append(("S" + case if sanitizer else case, dict(env), timeout, cmd))
        if self.clock is not None:
            self.clock[0] += self.seconds.get(case, 0)
        if not sanitizer:
            out = self.session / case
            out.mkdir(parents=True, exist_ok=True)
            (out / "case.json").write_text(json.dumps({"status": self.statuses[case]}))
        return (99 if sanitizer else 0), False


def _session(tmp_path, statuses, *, anyway_a=False, deadline_s=3600, seconds=None,
             monkeypatch=None):
    clock = [0.0]
    if monkeypatch is not None:
        monkeypatch.setattr(xd.time, "monotonic", lambda: clock[0])
    r = FakeRunner(tmp_path, statuses, seconds, clock)
    s = xd.run_session(tmp_path, deadline_s=deadline_s, cpu_boundary=False, sync=True,
                       anyway_a=anyway_a, runner=r, sanitizer="compute-sanitizer")
    return s, r


ALL_CLEAN = {c: xd.CLEAN for c in xd.CASES}


def test_order_env_and_fresh_processes(tmp_path):
    s, r = _session(tmp_path, ALL_CLEAN)
    assert [c for c, *_ in r.calls] == ["C", "B", "A", "D", "E"]
    envs = {c: e for c, e, *_ in r.calls}
    assert envs["C"] == {"CUDA_LAUNCH_BLOCKING": "1"}
    assert all(envs[c] == {} for c in "BADE")
    for c, _, _, cmd in r.calls:      # each its own process, synced as a phase
        assert cmd[:3] == ["bash", str(ROOT / "scripts" / "run_phase.sh"), f"t31xa_{c}"]
    assert s["sanitizer"]["status"] == "not run: neither A nor B reproduced"


@pytest.mark.parametrize("repro, target", [("A", "A"), ("B", "B"), ("AB", "A")])
def test_the_sanitizer_takes_the_smallest_and_runs_after_a(tmp_path, repro, target):
    statuses = dict(ALL_CLEAN, **{c: xd.REPRODUCED for c in repro})
    s, r = _session(tmp_path, statuses)
    order = [c for c, *_ in r.calls]
    assert order == ["C", "B", "A", "S" + target, "D", "E"]
    _, env, limit, cmd = r.calls[3]
    assert env == {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
    assert limit <= xd.SANITIZER_LIMIT_S
    assert cmd[:3] == ["compute-sanitizer", "--tool", "memcheck"]
    assert "run_phase.sh" not in " ".join(cmd)
    assert s["sanitizer"]["status"] == "memcheck reported errors"


def test_only_c_reproducing_skips_the_sanitizer_unless_asked(tmp_path):
    statuses = dict(ALL_CLEAN, C=xd.REPRODUCED, D=xd.REPRODUCED, E=xd.REPRODUCED)
    s, r = _session(tmp_path / "x", statuses)
    assert not any(c.startswith("S") for c, *_ in r.calls)
    s, r = _session(tmp_path / "y", statuses, anyway_a=True)
    assert [c for c, *_ in r.calls] == ["C", "B", "A", "SA", "D", "E"]


def test_the_deadline_cuts_later_cases_and_bounds_each(tmp_path, monkeypatch):
    s, r = _session(tmp_path, ALL_CLEAN, deadline_s=1500,
                    seconds={"C": 900, "B": 300, "A": 200}, monkeypatch=monkeypatch)
    limits = {c: t for c, _, t, _ in r.calls}
    assert limits["C"] == 1500 and limits["B"] == 600 and limits["A"] == 300
    assert [c for c, *_ in r.calls] == ["C", "B", "A"]     # 100 s left: under 2 min
    assert s["cases"]["D"]["status"] == xd.CUT and s["cases"]["E"]["status"] == xd.CUT


def test_a_case_with_no_result_is_reported(tmp_path):
    def runner(cmd, log, *, env, timeout):
        return -9, True
    s = xd.run_session(tmp_path, deadline_s=3600, cpu_boundary=False, sync=False,
                       anyway_a=False, runner=runner, sanitizer="x")
    assert s["cases"]["C"]["status"] == xd.CUT
    assert json.loads((tmp_path / "summary.json").read_text())["cases"]["C"]["exit"] == -9


def test_the_unprofiled_recording_matches_the_recorders_prefills(monkeypatch, tmp_path):
    """D: warm-up, the profiler's warm-up step and the recorded prefill
    under the traced capture, synchronised as `record_prefill_trace` does,
    then the untraced pass."""
    from attnbench.components import mask_bank, recorder
    log = []

    class Capture:
        def __init__(self, **kw):
            self.calls = ["stale"]

        def __enter__(self):
            log.append("enter")
            return self

        def __exit__(self, *a):
            log.append("exit")

    monkeypatch.setattr(mask_bank, "MaskCapture", Capture)
    monkeypatch.setattr(xd, "_sync", lambda cuda: log.append("sync"))
    monkeypatch.setattr(recorder, "_untraced_pass",
                        lambda run, dev, *a: log.append(f"untraced {dev}"))
    arm = SimpleNamespace(run_prefill=lambda: log.append("prefill"))
    xd.record_unprofiled(arm, tmp_path / "d", cuda=True)
    assert log == ["enter", "prefill", "sync", "prefill", "sync", "prefill", "sync",
                   "exit", "untraced cuda"]


def test_e_holds_the_hook_off(monkeypatch):
    from attnbench import assert_strip
    monkeypatch.setattr(assert_strip, "install", assert_strip.install)
    assert_strip.uninstall()
    xd._hold_hook_off()
    with pytest.raises(RuntimeError, match="holds the assert-strip hook off"):
        assert_strip.install()
    assert not xd.hook_installed()


def test_e_refuses_a_hook_already_installed_outside_the_boundary(monkeypatch):
    from attnbench import assert_strip
    monkeypatch.setattr(assert_strip, "install", assert_strip.install)
    assert_strip.install()
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            xd._hold_hook_off()
    finally:
        assert_strip.uninstall()


def test_the_layer_watch_names_the_last_layer_entered(monkeypatch):
    from attnbench.backends.xattention import XAttentionBackend
    monkeypatch.setattr(XAttentionBackend, "forward_and_mask",
                        lambda self, q, k, v, cfg, mask=None: "out")
    ev = xd.Events(Path("/dev/null"))
    ev.arm = "xattention"
    xd._watch_layers(ev)
    be = XAttentionBackend.__new__(XAttentionBackend)
    be.layer_idx = 7
    q = SimpleNamespace(shape=(1, 28, 2048, 128))
    assert be.forward_and_mask(q, None, None, None) == "out"
    assert ev.last_layer == ("xattention", 7, 2048)
