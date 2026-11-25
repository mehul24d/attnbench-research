"""scripts/aws_p3_gsutil.py under the real commit.GsutilStore, sync_prediction
and load_committed, with a thread standing in for the Mac's witness.

What the in-session `predict` and `verify` steps rely on: the creation time is
the Mac's (the witness's) plus one second, as for GCS; the read-back is the
Mac's returned copy; nothing is clobbered; a Mac copy that differs, or no
witness in time, fails the sync; only gs://p3-mac/ paths are accepted.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attnbench.predict import commit as C  # noqa: E402

SCRIPT = ROOT / "scripts" / "aws_p3_gsutil.py"
BASE = "gs://p3-mac/predictions/P3/p3_session"


@pytest.fixture
def gsutil(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    g = b / "gsutil"
    g.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{SCRIPT}" "$@"\n')
    g.chmod(0o755)
    spool = tmp_path / "spool"
    monkeypatch.setenv("ATTNBENCH_P3_STORE", str(spool))
    monkeypatch.setenv("ATTNBENCH_P3_STORE_POLL_S", "0.05")
    monkeypatch.setenv("ATTNBENCH_P3_STORE_WAIT_S", "10")
    return str(g), spool


def mac(spool: Path, *, at="2026-11-27T01:02:03Z", corrupt=False, stop=None):
    """The driver's witness_store: for each request, keep a copy, return it, witness it."""
    def loop():
        while not (stop and stop.is_set()):
            for r in spool.rglob("*.request"):
                o = r.with_name(r.name[: -len(".request")])
                w = o.with_name(o.name + ".witness")
                if w.exists():
                    continue
                data = o.read_bytes() + (b"x" if corrupt else b"")
                o.with_name(o.name + ".mac").write_bytes(data)
                w.write_text(json.dumps({"sha256": hashlib.sha256(data).hexdigest(),
                                         "received_utc": at}))
            time.sleep(0.05)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def committed(tmp_path) -> C.CommittedFile:
    data = b'{"pairing": "P3"}\n'
    p = tmp_path / "prediction_P3_x.json"
    p.write_bytes(data)
    side = tmp_path / "prediction_P3_x.json.sha256"
    side.write_text(hashlib.sha256(data).hexdigest() + "\n")
    return C.CommittedFile(p, side, hashlib.sha256(data).hexdigest())


def test_the_sync_reads_the_macs_time_and_the_macs_copy(tmp_path, gsutil):
    g, spool = gsutil
    stop = threading.Event()
    mac(spool, stop=stop)
    store = C.GsutilStore(g)
    r = C.sync_prediction(committed(tmp_path), store, BASE)
    stop.set()
    assert r.synced_at == dt.datetime(2026, 11, 27, 1, 2, 4, tzinfo=dt.timezone.utc)   # + 1 s
    assert r.remote == f"{BASE}/prediction_P3_x.json"
    assert store.exists(r.remote) and not store.exists(f"{BASE}/nothing.json")
    # the reads `verify` makes through load_committed: both objects and their creation times
    c = committed(tmp_path)
    assert store.get(r.remote) == c.path.read_bytes()
    assert store.get(r.remote_sidecar).decode() == c.sidecar.read_text()
    assert store.created(r.remote_sidecar) == r.synced_at
    # the bytes came back from the Mac's copy, not from the spooled object
    (spool / "p3-mac/predictions/P3/p3_session/prediction_P3_x.json.mac").write_bytes(b"mac\n")
    assert store.get(r.remote) == b"mac\n"


def test_an_object_already_there_is_not_clobbered(tmp_path, gsutil):
    g, spool = gsutil
    stop = threading.Event()
    mac(spool, stop=stop)
    store = C.GsutilStore(g)
    a = tmp_path / "a.json"; a.write_bytes(b"first\n")
    b = tmp_path / "b.json"; b.write_bytes(b"second\n")
    store.put(a, f"{BASE}/x.json")
    store.put(b, f"{BASE}/x.json")
    stop.set()
    assert store.get(f"{BASE}/x.json") == b"first\n"


def test_a_mac_copy_that_differs_fails_the_sync(tmp_path, gsutil):
    g, spool = gsutil
    stop = threading.Event()
    mac(spool, corrupt=True, stop=stop)
    with pytest.raises(C.PredictionCommitError, match="sha256"):
        C.sync_prediction(committed(tmp_path), C.GsutilStore(g), BASE)
    stop.set()


def test_no_witness_in_time_fails_the_sync(tmp_path, gsutil, monkeypatch):
    g, _ = gsutil
    monkeypatch.setenv("ATTNBENCH_P3_STORE_WAIT_S", "0.3")
    with pytest.raises(C.PredictionCommitError, match="did not witness"):
        C.sync_prediction(committed(tmp_path), C.GsutilStore(g), BASE)


@pytest.mark.parametrize("remote", ["gs://other-bucket/x.json", "gs://p3-mac/../x.json",
                                    "gs://p3-mac/x.json.witness", "s3://p3-mac/x.json"])
def test_only_the_p3_store_is_accepted(tmp_path, gsutil, remote):
    g, _ = gsutil
    a = tmp_path / "a.json"; a.write_bytes(b"x\n")
    with pytest.raises(C.PredictionCommitError):
        C.GsutilStore(g).put(a, remote)
