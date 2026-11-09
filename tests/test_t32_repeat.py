"""A T3.2 repeat of the P1 session (researcher's ruling 4 of 7 Nov 2026):
T3.2a is the eager group only, real data, its code identity frozen before
launch, with no fix within the session.

`t31_session.py --repeat a --groups eager` writes the session's
`repeat_config.json` once (the session config with the repeat's label and
prediction prefix, every other ground-truth group set to run nothing, and
`oracle_d0.5_L32768` not measured, the drop fallback in force), and freezes
the code identity: the commit of the session's first start is recorded, and a
run of the session at any other commit is refused."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import t31_session as TS  # noqa: E402
import t31_steps as st  # noqa: E402


def test_the_repeat_config_runs_the_named_groups_only_and_is_written_once(tmp_path):
    s = tmp_path / "s"
    path = TS.repeat_config(s, "a", ["eager"])
    cfg = json.loads(path.read_text())
    assert path == s / "repeat_config.json"
    assert cfg["repeat"] == "T3.2a" and cfg["prediction_prefix"] == "predictions/P1-T3.2a"
    assert cfg["gt_groups"] == {"eager": None, "compiled_default": [], "compiled_capture": []}
    stuck = [e for e in cfg["not_measured"] if (e["arm"], e["density"], e["length"]) ==
             ("oracle", 0.5, 32768)]
    assert len(stuck) == 1 and "drop fallback" in stuck[0]["reason"]
    assert not cfg.get("debug") and not cfg.get("rehearsal")
    assert TS.repeat_config(s, "a", ["eager"]) == path               # a resume reuses it
    with pytest.raises(TS.RepeatError, match="T3.2a with groups"):
        TS.repeat_config(s, "a", ["eager", "compiled_default"])
    with pytest.raises(TS.RepeatError, match="T3.2a"):
        TS.repeat_config(s, "b", ["eager"])
    with pytest.raises(TS.RepeatError, match="no group 'compiled_fast'"):
        TS.repeat_config(tmp_path / "t", "a", ["compiled_fast"])


def test_the_working_values_fill_the_run_group_and_leave_the_others_empty(tmp_path):
    cfg = st.load_config(TS.repeat_config(tmp_path / "s", "a", ["eager"]), dry_run=False)
    assert cfg["gt_groups"]["eager"] == ["dense", "oracle", "meanpool", "xattention",
                                         "xattention_noassert"]
    assert cfg["gt_groups"]["compiled_default"] == [] == cfg["gt_groups"]["compiled_capture"]
    assert ("oracle", 0.5, 32768, 1) not in st.cells(cfg)


def test_a_group_with_no_arm_is_not_run(tmp_path, monkeypatch):
    ran = []

    def run(cmd):
        group = cmd[cmd.index("--group") + 1]
        ran.append(group)
        d = Path(cmd[cmd.index("--out") + 1])
        d.mkdir(parents=True, exist_ok=True)
        (d / "stamps.json").write_text(json.dumps({"group": group}))
        return 0
    monkeypatch.setattr(st, "_run", run)
    cfg = {"gt_groups": {"eager": ["dense", "oracle"], "compiled_default": [],
                         "compiled_capture": []}}
    args = SimpleNamespace(out=str(tmp_path / "gt"), session=str(tmp_path), dry_run=True,
                           config=None)
    found = st.step_ground_truth(args, cfg)
    assert ran == ["eager"] and found["eager"] == {"group": "eager"}
    assert found["compiled_default"] == found["compiled_capture"] == {"not_run": st.GROUP_NOT_RUN}


@pytest.mark.parametrize("complete", [True, False])
def test_a_cell_short_of_clean_rounds_makes_ground_truth_incomplete(tmp_path, monkeypatch, complete):
    def run(cmd):
        d = Path(cmd[cmd.index("--out") + 1])
        d.mkdir(parents=True, exist_ok=True)
        (d / "stamps.json").write_text(json.dumps({"group": "eager"}))
        (d / "gt_L2048_b1.json").write_text(json.dumps({"complete": True, "clean_rounds": 10,
                                                        "rounds_run": 10}))
        (d / "gt_L16384_b4.json").write_text(json.dumps(
            {"complete": complete, "clean_rounds": 10 if complete else 2, "rounds_run": 10}))
        return 0
    monkeypatch.setattr(st, "_run", run)
    cfg = {"gt_groups": {"eager": ["dense", "oracle"]}}
    args = SimpleNamespace(out=str(tmp_path / "gt"), session=str(tmp_path), dry_run=True,
                           config=None)
    if complete:
        assert st.step_ground_truth(args, cfg)["eager"] == {"group": "eager"}
        return
    with pytest.raises(st.StepIncomplete) as e:
        st.step_ground_truth(args, cfg)
    assert e.value.findings["incomplete"] == [{"group": "eager", "file": "gt_L16384_b4.json",
                                               "clean_rounds": 2, "rounds_run": 10}]
    assert "2 clean of 10" in str(e.value)


def test_run_step_writes_incomplete_and_exits_4(tmp_path, monkeypatch):
    def short(args, cfg):
        raise st.StepIncomplete("1 ground-truth cell(s) with too few clean rounds", {"x": 1})
    monkeypatch.setitem(st.STEP_FUNCS, "ground_truth", short)
    monkeypatch.setattr(st, "load_config", lambda *a, **k: {})
    args = SimpleNamespace(out=str(tmp_path / "09"), dry_run=True, cpu_boundary=False, config=None)
    assert st.run_step("ground_truth", args) == 4
    res = json.loads((tmp_path / "09" / "result.json").read_text())
    assert res["status"] == "incomplete" and res["findings"] == {"x": 1}


def test_the_code_identity_is_frozen_at_the_first_start(tmp_path):
    s = tmp_path / "s"
    assert TS.freeze_code(s, "T3.2a", head="aaa") is None
    assert json.loads((s / "manifest.json").read_text())["frozen_commit"] == "aaa"
    assert TS.freeze_code(s, "T3.2a", head="aaa") is None             # a resume at the same commit
    why = TS.freeze_code(s, "T3.2a", head="bbb")
    assert why and "frozen at aaa" in why and "no fix within the session" in why
    assert json.loads((s / "manifest.json").read_text())["frozen_commit"] == "aaa"


def test_a_repeat_runs_at_its_frozen_commit_and_refuses_any_other(tmp_path, monkeypatch):
    monkeypatch.setenv(TS.EXIT_WRAPPER_ENV, "armed")
    seen = []
    monkeypatch.setattr(TS, "tree_clean", lambda: True)
    monkeypatch.setattr(TS, "run_session", lambda session, **kw: seen.append((session, kw)) or 0)
    heads = iter(["c1", "c1", "c2"])
    monkeypatch.setattr(TS, "git_head", lambda: next(heads))
    s = tmp_path / "s"
    argv = ["--repeat", "a", "--groups", "eager", "--session", str(s)]
    assert TS.main(argv) == 0
    assert seen[0][1]["extra"] == ["--config", str(s / "repeat_config.json")]
    assert TS.main(argv + ["--from", "link"]) == 0                    # a resume, same commit
    assert TS.main(argv + ["--from", "link"]) == 2 and len(seen) == 2  # another commit: refused


@pytest.mark.parametrize("extra,why", [
    (["--debug"], "not a debug session"),
    (["--rehearsal"], "not a rehearsal"),
    ([], "--groups"),
])
def test_a_repeat_is_neither_debug_nor_rehearsal_and_names_its_groups(tmp_path, extra, why, capsys):
    argv = ["--repeat", "a", "--session", str(tmp_path / "s"), *extra]
    if why != "--groups":
        argv += ["--groups", "eager"]
    assert TS.main(argv) == 2
    assert why in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--repeat", "c", "--groups", "eager"],
    ["--rehearsal"],
    [],
])
def test_a_session_on_the_card_refuses_to_start_outside_the_exit_wrapper(tmp_path, argv,
                                                                         monkeypatch, capsys):
    """Ruling B4 of 8 Nov 2026: the sync and the halt are the session's own
    last act, never a watcher's; a dry run and the CPU boundary need neither."""
    monkeypatch.delenv(TS.EXIT_WRAPPER_ENV, raising=False)
    seen = []
    monkeypatch.setattr(TS, "run_session", lambda session, **kw: seen.append(kw) or 0)
    monkeypatch.setattr(TS, "tree_clean", lambda: True)
    monkeypatch.setattr(TS, "git_head", lambda: "c1")
    s = tmp_path / "s"
    assert TS.main([*argv, "--session", str(s)]) == 2 and seen == []
    assert "scripts/session_exit.sh" in capsys.readouterr().err
    assert not (s / "manifest.json").exists()           # nothing frozen by a refused start
    monkeypatch.setenv(TS.EXIT_WRAPPER_ENV, "armed")
    assert TS.main([*argv, "--session", str(s)]) == 0 and len(seen) == 1


def test_t32c_is_a_repeat_of_its_own():
    import json
    from pathlib import Path as P
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg = json.loads(TS.repeat_config(P(d), "c", ["eager"]).read_text())
    assert cfg["repeat"] == "T3.2c" and cfg["prediction_prefix"] == "predictions/P1-T3.2c"
    assert [g for g, arms in cfg["gt_groups"].items() if arms != []] == ["eager"]
