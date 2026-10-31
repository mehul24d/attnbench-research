"""The P1 session runner (T3.1): the order of the prediction commit and the
first clocked block, the runner's stop and resume rules, the real-run
refusals, and a CPU dry run end to end.

The hold is checked against the analysis's own refusal
(`commit.load_committed`) with a settable store clock, so "the runner
waited long enough" means "the analysis accepts the file", not a number
restated here. GPU wiring (`dense_select`, `link`) is checked with
`autospec` fakes, which refuse any call the real functions would refuse.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t31_session as sess  # noqa: E402
import t31_steps as st  # noqa: E402
from attnbench.predict import commit as C  # noqa: E402
from attnbench.predict import session_order as so  # noqa: E402

UTC = timezone.utc
T0 = datetime(2026, 11, 7, 3, 0, 0, tzinfo=UTC)
SKEW, GUARD = so.REQUIRED_SKEW, so.PLANNED_GUARD
ORDER = ("preflight", "link", "dense_select", "probe", "host_components",
         "device_components", "predict", "hold", "ground_truth", "verify", "postflight")


# --- the hold against the analysis's refusal -------------------------------------

class ClockedStore:
    """An in-memory store whose creation times come from a settable clock."""

    def __init__(self):
        self.objs, self.times, self.now = {}, {}, T0

    def put(self, local, remote):
        if remote not in self.objs:
            self.objs[remote] = Path(local).read_bytes()
            self.times[remote] = self.now

    def get(self, remote):
        return self.objs[remote]

    def created(self, remote):
        return self.times[remote]


def _synced(tmp_path):
    cfg = dict(st.DRY_CONFIG)
    session = tmp_path / "s"
    for step in ("link", "device_components"):
        st.step_dir(session, step).mkdir(parents=True)
    args = SimpleNamespace(out=str(st.step_dir(session, "link")), session=str(session),
                           dry_run=True, config=None)
    with mock.patch.object(st, "_t227"):
        st.step_link(args, cfg)
    args.out = str(st.step_dir(session, "device_components"))
    st.step_device_components(args, cfg)
    record = st.build_prediction(cfg, session, dry_run=True, now=T0 - timedelta(seconds=1))
    committed = C.write_prediction(record, tmp_path)
    store = ClockedStore()
    receipt = C.sync_prediction(committed, store, "mem://p")
    return store, receipt


def test_the_hold_release_is_accepted_by_the_analysis(tmp_path):
    store, receipt = _synced(tmp_path)
    start = so.release_at(receipt.synced_at, clock_skew=SKEW, guard=GUARD)
    cp = C.load_committed(store, receipt.remote, ground_truth_started_at=start,
                          clock_skew=SKEW, banked_at={})
    assert cp.sha256 == receipt.sha256
    assert (start - receipt.synced_at) >= timedelta(seconds=C.PLANNED_SYNC_WAIT_S)


def test_a_block_at_exactly_sync_plus_skew_is_late_so_the_guard_is_needed(tmp_path):
    store, receipt = _synced(tmp_path)
    with pytest.raises(C.LatePredictionError):
        C.load_committed(store, receipt.remote, clock_skew=SKEW, banked_at={},
                         ground_truth_started_at=receipt.synced_at + SKEW)
    with pytest.raises(so.HoldError):
        so.ready_to_clock(receipt.synced_at, receipt.synced_at + SKEW,
                          clock_skew=SKEW, guard=GUARD)


def test_ready_to_clock_refuses_until_the_release_and_not_after():
    rel = so.release_at(T0, clock_skew=SKEW, guard=GUARD)
    with pytest.raises(so.HoldError):
        so.ready_to_clock(T0, rel - timedelta(microseconds=1), clock_skew=SKEW, guard=GUARD)
    so.ready_to_clock(T0, rel, clock_skew=SKEW, guard=GUARD)


def test_hold_waits_on_the_instance_clock_and_returns_at_the_release():
    t = [T0]
    slept = []

    def sleep(s):
        slept.append(s)
        t[0] += timedelta(seconds=s)

    got = so.hold(T0, clock_skew=SKEW, guard=GUARD, now=lambda: t[0], sleep=sleep)
    assert got >= T0 + SKEW + GUARD and sum(slept) >= 65
    with pytest.raises(so.HoldError):      # a receipt far ahead is not this session's
        so.hold(T0 + timedelta(hours=1), clock_skew=SKEW, guard=GUARD, now=lambda: T0,
                sleep=sleep)


def test_a_real_session_holds_at_least_sixty_seconds():
    with pytest.raises(so.HoldError):
        so.check_session_skew(timedelta(seconds=59), GUARD, dry_run=False)
    so.check_session_skew(timedelta(seconds=2), GUARD, dry_run=True)
    with pytest.raises(so.HoldError):
        so.check_session_skew(SKEW, timedelta(seconds=-1), dry_run=False)


def test_the_session_start_is_the_earliest_stamp_of_either_kind():
    s = so.session_start([T0.timestamp() + 5, T0.timestamp() + 2],
                         [T0.timestamp() + 3, T0.timestamp() + 9])
    assert s == T0 + timedelta(seconds=2)


def test_ground_truth_refuses_to_clock_before_the_release(tmp_path):
    session = tmp_path / "s"
    pdir = st.step_dir(session, "predict")
    pdir.mkdir(parents=True)
    now = datetime.now(UTC)
    (pdir / "receipt.json").write_text(json.dumps(
        {"remote": "local://x", "remote_sidecar": "local://x.sha256", "sha256": "0" * 64,
         "synced_at": now.isoformat(), "local": "x.json"}))
    args = SimpleNamespace(config=None, dry_run=True, out=str(tmp_path / "g"),
                           session=str(session), group="eager")
    with pytest.raises(so.HoldError):
        st.cmd_ground_truth_one(args)
    assert not (tmp_path / "g" / "stamps.json").exists()


# --- the runner ---------------------------------------------------------------------

def fake_runner(outcomes: dict, ran: list):
    def run(cmd, log):
        step = cmd[cmd.index("--out") - 1]
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        log.write_text(f"ran {step}\n")
        status, rc = outcomes.get(step, ("passed", 0))
        if status is not None:
            (out / "result.json").write_text(json.dumps(
                {"step": step, "status": status, "reason": f"{status} by the test"}))
        ran.append(step)
        return rc
    return run


def _run(tmp_path, outcomes=None, start=ORDER[0], dry_run=False):
    ran = []
    rc = sess.run_session(tmp_path, start=start, extra=[], dry_run=dry_run, sync=False,
                          runner=fake_runner(outcomes or {}, ran))
    return rc, ran


def test_the_steps_are_in_the_plan_order_with_predict_before_any_clock():
    assert st.STEPS == ORDER and tuple(sess.CHECKLIST) == ORDER
    assert ORDER.index("predict") < ORDER.index("hold") < ORDER.index(sess.FIRST_CLOCKED)


def test_every_step_runs_with_a_log_each_and_the_session_passed(tmp_path):
    rc, ran = _run(tmp_path)
    assert rc == 0 and ran == list(ORDER)
    for i, s in enumerate(ORDER, 1):
        assert (tmp_path / f"{i:02d}_{s}" / "step.log").read_text() == f"ran {s}\n"
    cmd = sess.step_command("predict", tmp_path, [], dry_run=False, sync=True)
    assert cmd[:3] == ["bash", str(ROOT / "scripts" / "run_phase.sh"), "t31_07_predict"]


@pytest.mark.parametrize("outcome", [("failed", 1), ("skipped", 3), (None, 1), ("passed", 1)])
def test_it_stops_at_the_first_failure(tmp_path, outcome):
    rc, ran = _run(tmp_path, {"hold": outcome})
    assert rc == 1 and ran == list(ORDER[:ORDER.index("hold") + 1])


def test_resume_needs_every_earlier_step_passed(tmp_path):
    _run(tmp_path, {"link": ("failed", 1)})
    rc, ran = _run(tmp_path, start="predict")
    assert rc == 2 and ran == []
    rc, ran = _run(tmp_path, start="link")
    assert rc == 0 and ran == list(ORDER[1:])
    assert len(list(tmp_path.glob("02_link.failed.*"))) == 1


def test_no_new_prediction_once_ground_truth_has_started(tmp_path):
    _run(tmp_path, {"ground_truth": ("failed", 1)})
    for start in ("predict", "device_components", "preflight"):
        rc, ran = _run(tmp_path, start=start)
        assert rc == 2 and ran == []
    rc, ran = _run(tmp_path, start="ground_truth")
    assert rc == 0 and ran == list(ORDER[ORDER.index("ground_truth"):])


def test_a_ground_truth_crash_without_a_result_still_counts_as_started(tmp_path):
    _run(tmp_path, {"ground_truth": (None, -9)})
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert [a["status"] for a in m["attempts"] if a["step"] == "ground_truth"] == \
        ["started", "failed"]
    assert _run(tmp_path, start="hold")[0] == 0     # hold only waits; it writes nothing
    assert _run(tmp_path, start="predict")[0] == 2


def test_a_dry_session_cannot_be_continued_as_a_real_one(tmp_path):
    _run(tmp_path, dry_run=True)
    assert _run(tmp_path)[0] == 2


# --- real-run refusals and wiring ----------------------------------------------------

def test_the_repository_config_leaves_the_lock_values_null_in_the_file(tmp_path, monkeypatch):
    """The file keeps every undecided value null; a real run fills the working
    ones from `docs/lock_value_table.md` (test_t31_wiring) and refuses if a
    required one is still missing after that."""
    cfg = json.loads(st.DEFAULT_CONFIG.read_text())
    for k in ("densities", "reps", "resolution_floor_ms", "max_rerun_rounds"):
        assert cfg[k] is None
    assert "clock_skew_s" not in cfg and "hold_guard_s" not in cfg and cfg["card"] == "L4"
    assert cfg["lengths"] == [2048, 4096, 8192, 16384, 32768]
    assert cfg["batches"] == {"16384": [1, 4]}
    t = tmp_path / "table.md"
    t.write_text(st.WORKING_TABLE.read_text().replace('"reps": 10', '"reps": null'))
    monkey = st.WORKING_TABLE
    try:
        st.WORKING_TABLE = t
        with pytest.raises(st.StepFailed, match="reps"):
            st.load_config(None, dry_run=False)
        # The runner refuses on the config itself, on a clean tree too.
        started = []
        monkeypatch.setattr(sess, "tree_clean", lambda: True)
        monkeypatch.setattr(sess, "run_session", lambda *a, **k: started.append(a) or 0)
        assert sess.main(["--session", str(tmp_path / "never")]) == 2
        assert started == [] and not (tmp_path / "never").exists()
    finally:
        st.WORKING_TABLE = monkey


def test_a_real_session_syncs_to_the_results_bucket_and_banks_no_bucket_name(
        tmp_path, monkeypatch):
    monkeypatch.delenv("ATTNBENCH_BUCKET", raising=False)
    monkeypatch.setenv("ATTNBENCH_METADATA_URL", "http://127.0.0.1:9/none")
    with pytest.raises(st.StepFailed, match="no results bucket"):
        st.make_store(tmp_path, dry_run=False)
    monkeypatch.setenv("ATTNBENCH_BUCKET", "gs://b/")
    store, base = st.make_store(tmp_path, dry_run=False)
    assert isinstance(store, C.GsutilStore) and base == "gs://b"
    assert st.prediction_dir(json.loads(st.DEFAULT_CONFIG.read_text()),
                             tmp_path / "t31_p1_X") == "predictions/P1/t31_p1_X"
    with pytest.raises(st.StepFailed):
        st.prediction_dir({"prediction_prefix": ""}, tmp_path)


def test_the_dry_receipt_holds_bucket_relative_paths(tmp_path):
    assert sess.main(["--dry-run", "--session", str(tmp_path / "dry")]) == 0
    r = json.loads((tmp_path / "dry" / "07_predict" / "receipt.json").read_text())
    assert r["remote"].startswith("predictions/P1-dryrun/dry/prediction_")
    assert r["remote_sidecar"] == r["remote"] + ".sha256"


def test_fake_components_are_refused_in_a_real_prediction(tmp_path):
    session = tmp_path / "s"
    for step in ("link", "device_components"):
        st.step_dir(session, step).mkdir(parents=True)
    args = SimpleNamespace(out=str(st.step_dir(session, "link")), session=str(session),
                           dry_run=True, config=None)
    with mock.patch.object(st, "_t227"):
        st.step_link(args, st.DRY_CONFIG)
    args.out = str(st.step_dir(session, "device_components"))
    st.step_device_components(args, st.DRY_CONFIG)
    with pytest.raises(st.StepFailed, match="fake components"):
        st.build_prediction(st.DRY_CONFIG, session, dry_run=False)


def test_the_prediction_covers_exactly_the_grid(tmp_path):
    session = tmp_path / "s"
    for step in ("link", "device_components"):
        st.step_dir(session, step).mkdir(parents=True)
    args = SimpleNamespace(out=str(st.step_dir(session, "link")), session=str(session),
                           dry_run=True, config=None)
    with mock.patch.object(st, "_t227"):
        st.step_link(args, st.DRY_CONFIG)
    args.out = str(st.step_dir(session, "device_components"))
    st.step_device_components(args, st.DRY_CONFIG)
    bigger = {**st.DRY_CONFIG, "lengths": [64, 128, 256]}
    with pytest.raises(st.StepFailed, match="missing"):
        st.build_prediction(bigger, session, dry_run=True)


def test_dense_select_wiring_on_a_real_run(tmp_path):
    from attnbench import dense_selection as ds
    cfg = {**st.DRY_CONFIG, "card": "L4", "lengths": [2048, 16384],
           "batches": {"16384": [1, 4]}, "dense_rounds": 5, "resolution_floor_ms": 0.5}
    harness = mock.MagicMock()
    harness._cfg.return_value = SimpleNamespace(seq_len=0, batch=1)
    choice = SimpleNamespace(chosen="sdpa_flash", complete=True)
    with mock.patch("attnbench.components.shakedown_arms.Harness", return_value=harness), \
            mock.patch("dataclasses.replace", side_effect=lambda c, **kw: SimpleNamespace(**{**vars(c), **kw})), \
            mock.patch.object(ds, "registry_candidates", autospec=True, return_value={}), \
            mock.patch.object(ds, "select_dense_baseline", autospec=True,
                              return_value=choice) as sel, \
            mock.patch.object(ds, "write_choice", autospec=True,
                              return_value=tmp_path / "c.json"):
        found = st.step_dense_select(SimpleNamespace(dry_run=False, out=str(tmp_path)), cfg)
    assert sorted(found) == ["L4_16384_1", "L4_16384_4", "L4_2048_1"]
    kw = sel.call_args.kwargs
    assert kw == {"rounds": 5, "resolution_floor_ms": 0.5}
    assert [c.args[1].batch for c in sel.call_args_list] == [1, 1, 4]


def test_link_wiring_on_a_real_run(tmp_path):
    from attnbench.components import launch_queue, link
    from attnbench.predict.recursion import MechanismCost
    from attnbench.predict.training import HardwareFeatures
    args = SimpleNamespace(dry_run=False, out=str(tmp_path), session=str(tmp_path), config=None)
    frame = mock.MagicMock()
    frame.iloc.__getitem__.return_value.to_dict.return_value = {}
    with mock.patch.object(st, "_t227"), mock.patch("pandas.read_parquet", return_value=frame), \
            mock.patch.object(link, "hardware_features", autospec=True,
                              return_value=HardwareFeatures(1.0, 2.0)), \
            mock.patch.object(link, "mechanism_costs", autospec=True,
                              return_value={"item": MechanismCost(1e-5, 1e9, 5e8)}) as mc, \
            mock.patch.object(launch_queue, "queue_depth", autospec=True, return_value=7) as qd:
        found = st.step_link(args, {**st.DRY_CONFIG, "lengths": [2048, 4096]})
    assert found == {"queue_depth": 7, "fake": False}
    assert qd.call_args.args[1] == "sleep0"
    assert [c.args[1] for c in mc.call_args_list] == [2048, 4096]
    inputs = json.loads((tmp_path / "recursion_inputs.json").read_text())
    assert inputs["by_length"]["4096"]["item"]["loaded_bandwidth"] == 5e8


# --- the CPU dry run, end to end -------------------------------------------------------

def test_the_cpu_dry_run_passes_and_verify_accepts_the_prediction(tmp_path):
    rc = sess.main(["--dry-run", "--session", str(tmp_path / "dry")])
    assert rc == 0
    v = json.loads((tmp_path / "dry" / "10_verify" / "result.json").read_text())["findings"]
    h = json.loads((tmp_path / "dry" / "08_hold" / "result.json").read_text())["findings"]
    assert v["margin_s"] > st.DRY_CONFIG["clock_skew_s"]
    assert h["released_at"] >= h["synced_at"]
    stamps = json.loads((tmp_path / "dry" / "09_ground_truth" / "eager" / "stamps.json")
                        .read_text())
    assert stamps["clock_not_before"] <= min(stamps["first_clocked_block_started_at"])


# --- the rehearsal (Shakedown 2) -------------------------------------------------------

def test_a_rehearsal_stops_after_hold_and_never_clocks(tmp_path):
    ran = []
    rc = sess.run_session(tmp_path, start=ORDER[0], extra=[], dry_run=False, sync=False,
                          runner=fake_runner({}, ran), rehearsal=True)
    assert rc == 0 and ran == list(ORDER[:ORDER.index("hold") + 1])
    assert sess.FIRST_CLOCKED not in ran
    assert sess.run_session(tmp_path, start="ground_truth", extra=[], dry_run=False, sync=False,
                            runner=fake_runner({}, []), rehearsal=True) == 2


def test_a_rehearsal_cannot_be_continued_as_a_session_or_the_reverse(tmp_path):
    sess.run_session(tmp_path / "r", start=ORDER[0], extra=[], dry_run=False, sync=False,
                     runner=fake_runner({}, []), rehearsal=True)
    assert sess.run_session(tmp_path / "r", start="predict", extra=[], dry_run=False,
                            sync=False, runner=fake_runner({}, [])) == 2
    _run(tmp_path / "s", {"hold": ("failed", 1)})
    assert sess.run_session(tmp_path / "s", start="hold", extra=[], dry_run=False, sync=False,
                            runner=fake_runner({}, []), rehearsal=True) == 2


def test_the_rehearsal_config_has_its_own_prefix_and_is_written_once(tmp_path):
    p = sess.rehearsal_config(tmp_path, None)
    cfg = json.loads(p.read_text())
    assert cfg["rehearsal"] is True and cfg["prediction_prefix"] == sess.REHEARSAL_PREFIX
    assert cfg["pairing"] == "P1"                 # the working values are P1's
    assert cfg["lengths"] == [2048, 16384] and cfg["batches"] == {"16384": [1, 4]}
    assert st.prediction_dir(cfg, tmp_path).startswith("predictions/P1-rehearsal/")
    p.write_text(json.dumps({**cfg, "marker": 1}))
    assert json.loads(sess.rehearsal_config(tmp_path, None).read_text())["marker"] == 1


@pytest.mark.parametrize("rehearsal,prefix", [(True, "predictions/P1"),
                                               (None, "predictions/P1-rehearsal")])
def test_a_prediction_goes_under_the_rehearsal_prefix_only_from_a_rehearsal(
        tmp_path, rehearsal, prefix):
    with pytest.raises(st.StepFailed, match="rehearsal"):
        st.prediction_dir({"rehearsal": rehearsal, "prediction_prefix": prefix}, tmp_path)
