"""scripts/p6_backend_check.py, block 1's abort gate on P6 (the researcher's task
of 26 Nov 2026 after P3's push, item 2c). On this Mac: the wheel hash read from
pip's record, a check that cannot run named as a failure, the GPU check
stopping the rest, and a missing K2 candidate named. The real run is the
p5.4xlarge's."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p6_backend_check as B  # noqa: E402

H = B.FA3_WHEEL_SHA256


def test_the_wheel_hash_is_read_from_pips_record_in_each_form():
    assert B.wheel_sha256(json.dumps({"url": "file:///x.whl", "archive_info": {"hashes": {"sha256": H}}})) == H
    assert B.wheel_sha256(json.dumps({"url": "file:///x.whl", "archive_info": {"hash": f"sha256={H}"}})) == H
    assert B.wheel_sha256(json.dumps({"url": f"file:///x.whl#sha256={H}", "archive_info": {}})) == H
    assert B.wheel_sha256(json.dumps({"url": "file:///x.whl", "archive_info": {}})) is None
    assert B.wheel_sha256(None) is None


def test_a_gpu_that_is_not_an_h100_stops_the_checks_and_fails():
    seen = []

    def gpu(rep, problems):
        problems.append("the GPU is 'NVIDIA L4', not an H100")

    def later(rep, problems):
        seen.append("ran")
    rep, problems = B.run((("gpu", gpu), ("candidates", later)))
    assert rep["passed"] is False and problems == ["the GPU is 'NVIDIA L4', not an H100"] and not seen
    assert rep["priced_dense_kernel"] == "sdpa_flash"


def test_a_check_that_cannot_run_is_a_named_failure():
    def boom(rep, problems):
        raise ImportError("no module named flash_attn_interface")
    rep, problems = B.run((("gpu", lambda r, p: None), ("distributions", boom)))
    assert rep["passed"] is False
    assert problems == ["distributions check could not run: ImportError: no module named flash_attn_interface"]


def test_a_k2_candidate_not_supplied_is_named(monkeypatch):
    from attnbench import dense_selection as ds
    planned = ds.planned_candidates("H100")
    assert ds.FA3 in planned
    monkeypatch.setattr(ds, "registry_candidates", lambda card: {n: (None if n == ds.FA3 else object())
                                                                 for n in planned})
    monkeypatch.setattr(B, "geometry", lambda: type("G", (), {"n_heads_q": 12, "n_heads_kv": 2,
                                                              "head_dim": 128, "dtype": "bfloat16"})())
    monkeypatch.setattr(ds, "screen", lambda cfg, planned, cands, device: (
        {n: ds.CandidateResult(n, ds.CORRECT) for n in planned if cands[n] is not None}, {}))
    rep, problems = {}, []
    B.check_candidates(rep, problems)
    assert problems[0] == f"K2 candidates not supplied on the H100: ['{ds.FA3}']"
    assert len(problems) == 2 and "'fa3': 'not screened'" in problems[1]
    assert rep["supplied"][ds.FA3] is False and rep["screen_config"]["n_heads_kv"] == 2


def test_the_dense_arm_prices_the_kernel_the_record_names():
    from attnbench.components.shakedown_arms import ARMS
    assert ARMS["dense"].backend == B.PRICED_DENSE


def _screened(monkeypatch, statuses):
    from attnbench import dense_selection as ds
    planned = ds.planned_candidates("H100")
    monkeypatch.setattr(ds, "registry_candidates", lambda card: {n: object() for n in planned})
    monkeypatch.setattr(B, "geometry", lambda: type("G", (), {"n_heads_q": 12, "n_heads_kv": 2,
                                                              "head_dim": 128, "dtype": "bfloat16"})())
    monkeypatch.setattr(ds, "screen", lambda cfg, planned, cands, device: (
        {n: ds.CandidateResult(n, statuses.get(n, ds.CORRECT), "x") for n in planned}, {}))
    rep, problems = {}, []
    B.check_candidates(rep, problems)
    return rep, problems


def test_sdpa_efficient_failing_as_on_p4_does_not_stop_the_session(monkeypatch):
    from attnbench import dense_selection as ds
    rep, problems = _screened(monkeypatch, {ds.SDPA_EFFICIENT: ds.INCORRECT})
    assert problems == [] and rep["screen"][ds.SDPA_EFFICIENT]["status"] == ds.INCORRECT


def test_fa3_or_the_priced_kernel_not_correct_stops_the_session(monkeypatch):
    from attnbench import dense_selection as ds
    for name in (ds.FA3, ds.SDPA_FLASH, ds.FA2):
        rep, problems = _screened(monkeypatch, {name: ds.DECLINED})
        assert len(problems) == 1 and name in problems[0] and "not correct" in problems[0], (name, problems)
