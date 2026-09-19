"""The dense canary is a gate: it fails on any difference and on vacuity."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from attnbench.accuracy.dense_canary import DenseCanaryFailed, check, enforce

ROOT = Path(__file__).resolve().parent.parent
BANKED_2048 = ROOT / "results/stage3_s1/accuracy.parquet"


def _rows(n=3, task="niah_single", backend="sdpa_flash", pred="x"):
    return pd.DataFrame([{"backend": backend, "sparsity": None, "task": task,
                          "example_id": f"{task}_2048_{i}", "predicted": f"{pred}{i}",
                          "expected": f"e{i}", "context_length": 2000 + i}
                         for i in range(n)])


def _run(new, banked, min_checked):
    res = check(new, banked, backend="sdpa_flash", min_checked=min_checked)
    enforce(res, min_checked=min_checked)
    return res


def test_identical_passes_and_counts():
    res = _run(_rows(), _rows(), 3)
    assert res.passed and res.n_checked == 3


def test_one_different_prediction_fails():
    new = _rows()
    new.loc[1, "predicted"] = "different"
    with pytest.raises(DenseCanaryFailed, match="1 of 3"):
        _run(new, _rows(), 3)


def test_same_id_different_example_fails():
    new = _rows()
    new.loc[0, "expected"] = "not the same needle"
    with pytest.raises(DenseCanaryFailed):
        _run(new, _rows(), 3)


def test_same_text_different_tokenized_length_fails():
    new = _rows()
    new.loc[2, "context_length"] = 1
    with pytest.raises(DenseCanaryFailed):
        _run(new, _rows(), 3)


def test_zero_overlap_fails_rather_than_passing_vacuously():
    new = _rows(task="vt")
    with pytest.raises(DenseCanaryFailed, match="compared 0"):
        _run(new, _rows(), 3)


def test_zero_new_dense_rows_fails():
    new = _rows(backend="block_sparse")        # no dense rows at all
    with pytest.raises(DenseCanaryFailed, match="compared 0"):
        _run(new, _rows(), 1)


def test_partial_overlap_fails():
    new = pd.concat([_rows(), _rows(task="vt")], ignore_index=True)
    with pytest.raises(DenseCanaryFailed):
        _run(new, pd.concat([_rows(), _rows(n=2, task="vt")]), 5)


def test_fewer_than_required_fails_even_if_all_match():
    with pytest.raises(DenseCanaryFailed, match="required 300"):
        _run(_rows(), _rows(), 300)


def test_min_checked_zero_is_refused():
    with pytest.raises(ValueError):
        check(_rows(), _rows(), backend="sdpa_flash", min_checked=0)


def test_sparse_rows_are_not_mistaken_for_dense():
    new = _rows()
    sparse = _rows(pred="other")
    sparse["backend"] = "sdpa_flash"
    sparse["sparsity"] = 0.5                   # same backend label, not the dense arm
    res = _run(pd.concat([new, sparse], ignore_index=True), _rows(), 3)
    assert res.n_new_dense == 3


def test_reference_with_conflicting_duplicates_is_refused():
    banked = pd.concat([_rows(), _rows(n=1, pred="y")], ignore_index=True)
    with pytest.raises(DenseCanaryFailed, match="conflicting"):
        check(_rows(), banked, backend="sdpa_flash", min_checked=3)


# --- against the real banked reference ---------------------------------------

needs_banked = pytest.mark.skipif(not BANKED_2048.exists(),
                                  reason="banked results/ not present")


def _script(tmp_path, new_path, min_checked):
    rec = tmp_path / "canary.json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/check_dense_canary.py"),
                        "--new", str(new_path), "--banked", str(BANKED_2048),
                        "--seq-len", "2048", "--min-checked", str(min_checked),
                        "--record", str(rec)], capture_output=True, text=True)
    return p.returncode, json.loads(rec.read_text())


def _first_100(df):
    d = df[(df.backend == "sdpa_flash") & df.sparsity.isna()]
    idx = d.example_id.str.rsplit("_", n=1).str[-1].astype(int)
    return d[idx < 100]


@needs_banked
def test_real_reference_reproduces_itself(tmp_path):
    new = tmp_path / "new.parquet"
    _first_100(pd.read_parquet(BANKED_2048)).to_parquet(new)
    rc, rec = _script(tmp_path, new, 300)
    assert rc == 0 and rec["verdict"] == "PASS" and rec["n_checked"] == 300


@needs_banked
def test_real_reference_one_flipped_prediction_fails_the_script(tmp_path):
    d = _first_100(pd.read_parquet(BANKED_2048)).reset_index(drop=True)
    d.loc[57, "predicted"] = d.loc[57, "predicted"] + " "
    new = tmp_path / "new.parquet"
    d.to_parquet(new)
    rc, rec = _script(tmp_path, new, 300)
    assert rc != 0 and rec["verdict"] == "FAIL" and rec["n_mismatches"] == 1
    assert rec["n_checked"] == 300


@needs_banked
def test_real_reference_wrong_band_checks_nothing_and_fails(tmp_path):
    d = _first_100(pd.read_parquet(BANKED_2048))
    new = tmp_path / "new.parquet"
    d.to_parquet(new)
    rec = tmp_path / "c.json"
    p = subprocess.run([sys.executable, str(ROOT / "scripts/check_dense_canary.py"),
                        "--new", str(new), "--banked", str(BANKED_2048),
                        "--seq-len", "4096", "--min-checked", "300",
                        "--record", str(rec)], capture_output=True, text=True)
    r = json.loads(rec.read_text())
    assert p.returncode != 0 and r["verdict"] == "FAIL"
