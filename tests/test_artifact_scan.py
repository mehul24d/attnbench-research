"""The artifact's identifier scan (paper-quality wave, 28 Nov 2026, item 15):
it must find literals and patterns in plain, gzipped and zipped files, allow
the artifact's own placeholders, and never print a matched value."""

import gzip
import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("artifact_scan", ROOT / "scripts/artifact_scan.py")
AS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AS)

SECRET = b"proj-secret-4242"


def _tree(tmp_path):
    (tmp_path / "a.json").write_bytes(b'{"x": "gs://attnbench-store/p", "h": "/home/user/x"}')
    (tmp_path / "b.json.gz").write_bytes(gzip.compress(b"see " + SECRET))
    with zipfile.ZipFile(tmp_path / "c.npz", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("m.txt", "arn:aws:iam::x " * 40)
    (tmp_path / "d.log").write_bytes(b"account 123456789012 and /Users/someone/x and a@b.org")
    return tmp_path


def test_finds_literals_inside_gzip(tmp_path):
    res = AS.scan(_tree(tmp_path), [SECRET])
    assert res["hits"]["literal 1"] == {"b.json.gz (gunzipped)": 1}


def test_finds_patterns_inside_zip_members_and_plain_files(tmp_path):
    h = AS.scan(_tree(tmp_path), [])["hits"]
    assert h["aws_arn"] == {"c.npz [m.txt]": 40}
    assert h["aws_12_digit"] == {"d.log": 1}
    assert h["home_dir"] == {"d.log": 1}
    assert h["email"] == {"d.log": 1}


def test_placeholders_are_allowed(tmp_path):
    (tmp_path / "a.json").write_bytes(b'{"x": "gs://attnbench-store/p", "h": "/home/user/x"}')
    h = AS.scan(tmp_path, [])["hits"]
    assert not any(h.values())


def test_report_never_prints_a_value(tmp_path):
    md = AS.report(AS.scan(_tree(tmp_path), [SECRET]), "t")
    for v in (SECRET.decode(), "123456789012", "someone", "a@b.org", "arn:aws"):
        assert v not in md
    assert "NOT CLEAN" in md


def test_digits_inside_longer_numbers_are_not_account_ids(tmp_path):
    (tmp_path / "t.json").write_bytes(b'{"ts": 1795694324191501, "f": 0.123456789012345}')
    assert not AS.scan(tmp_path, [])["hits"]["aws_12_digit"]
