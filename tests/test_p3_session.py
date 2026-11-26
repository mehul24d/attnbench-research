"""P3's session (docs/launch_cards/aws_p3_g6.md): its config, as the pinned
step code loads it, and the runner's blocks (`t31_session.py --until`).

The config: pairing P3, P1's 35 cells one for one, every T4.1 item from the
lock as for P2 and P4 (nothing filled from P1's working table), the eager
group only. Neither it nor the new P3 scripts enter the timed-path pin, and
the prediction committed on 22 Nov 2026 is the file it was.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import t31_session as S  # noqa: E402
import t31_steps as T  # noqa: E402
import timed_path  # noqa: E402

P3 = ROOT / "configs" / "t31" / "p3_session.json"
P1 = ROOT / "configs" / "t31" / "p1_session.json"


def test_the_p3_config_loads_with_p1s_35_cells_and_the_locks_values():
    cfg = T.load_config(P3, dry_run=False)
    assert cfg["pairing"] == "P3" and cfg["prediction_prefix"] == "predictions/P3"
    assert "working_values" not in cfg                  # nothing from P1's working table
    assert sorted(cfg["lock_values"]["filled"]) == sorted(T.LOCK_SOURCED)
    lock = json.loads((ROOT / "docs" / "lock_values.json").read_text())
    for k in T.PAIRING_KEYS:                            # the lock's P2 and P4 values, copied
        assert cfg[k] == lock["pairing_values"]["P2"][k] == lock["pairing_values"]["P4"][k]
    assert list(cfg["gt_groups"]) == ["eager"]
    p1 = T.load_config(P1, dry_run=False)
    assert T.cells(cfg) == T.cells(p1) and len(T.cells(cfg)) == 35
    assert cfg["not_measured"] == json.loads(P1.read_text())["not_measured"]


def test_the_pin_and_the_committed_prediction_are_untouched():
    closure = timed_path.pin_closure("HEAD")
    for name in ("scripts/aws_p3_gsutil.py", "scripts/t31_session.py", "configs/t31/p3_session.json"):
        assert name not in closure
    assert "configs/t31/p3_session.json" not in timed_path.SESSION_CONFIGS
    assert timed_path.pin_hash("HEAD")["sha256"].startswith("0e0fb012")
    pred = ROOT / "docs" / "p3_prediction" / "prediction_P3_exploratory_from_P1_20261122T211249520458Z.json"
    assert hashlib.sha256(pred.read_bytes()).hexdigest() == \
        "72b907cf4605336bba91ef3f04f8503945e49dabef207d4e3c4fc7b26183e1b2"


def _runner(seen):
    def run(cmd, log):
        step = cmd[cmd.index("--out") - 1]
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps({"status": "passed", "reason": ""}))
        seen.append(step)
        return 0
    return run


def test_until_ends_a_block_and_from_resumes_it(tmp_path):
    seen = []
    kw = dict(extra=[], dry_run=False, sync=False, runner=_runner(seen))
    assert S.run_session(tmp_path, start="preflight", until="probe", **kw) == 0
    assert seen == ["preflight", "link", "dense_select", "probe"]
    assert S.run_session(tmp_path, start="host_components", until="host_components", **kw) == 0
    assert seen[4:] == ["host_components"]
    assert S.run_session(tmp_path, start="device_components", **kw) == 0
    assert seen[5:] == list(T.STEPS[5:])


def test_until_before_from_is_refused(tmp_path):
    seen = []
    assert S.run_session(tmp_path, start="probe", until="link", extra=[], dry_run=False,
                         sync=False, runner=_runner(seen)) == 2 and seen == []


def test_a_block_cannot_skip_an_unpassed_step(tmp_path):
    seen = []
    assert S.run_session(tmp_path, start="host_components", until="host_components", extra=[],
                         dry_run=False, sync=False, runner=_runner(seen)) == 2 and seen == []


@pytest.mark.parametrize("flag", [["--rehearsal"], ["--debug"], ["--cpu-boundary"],
                                  ["--repeat", "d", "--groups", "eager"]])
def test_until_is_for_a_session_in_blocks_only(flag, capsys, monkeypatch):
    monkeypatch.setenv(S.EXIT_WRAPPER_ENV, "armed")
    assert S.main([*flag, "--until", "probe", "--session", "/nonexistent/x"]) == 2
    assert "REFUSING" in capsys.readouterr().err
