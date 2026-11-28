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
# Real-looking values that are not on the scanner's allowlist, built at run
# time so this file itself never carries one.
ARN = "arn:" + "aws:iam::y "
ACCT = "2109876" + "54321"
HOME = "/Users/" + "alice"
MAIL = "x" + "@example.net"


def _tree(tmp_path):
    (tmp_path / "a.json").write_bytes(b'{"x": "gs://attnbench-store/p", "h": "/home/user/x"}')
    (tmp_path / "b.json.gz").write_bytes(gzip.compress(b"see " + SECRET))
    with zipfile.ZipFile(tmp_path / "c.npz", "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("m.txt", ARN * 40)
    (tmp_path / "d.log").write_bytes(f"account {ACCT} and {HOME}/x and {MAIL}".encode())
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
    for v in (SECRET.decode(), ACCT, "alice", MAIL, ARN.strip()):
        assert v not in md
    assert "NOT CLEAN" in md


def test_digits_inside_longer_numbers_are_not_account_ids(tmp_path):
    (tmp_path / "t.json").write_bytes(b'{"ts": 1795694324191501, "f": 0.123456789012345}')
    assert not AS.scan(tmp_path, [])["hits"]["aws_12_digit"]


def test_allowlisted_fakes_are_excused(tmp_path):
    """Each known fake, exactly as it stands in the repository's history."""
    fakes = (b"account 123456789012 and /Users/someone/x and a@b.org\n"
             b'"arn:aws:iam::x " * 40\n'
             b'spec = {"bucket": "gs://b", "x": 1}\n'
             b'out["bucket"] = f"gs://{STORE}"\n'
             b"msg\n\nCo-Authored-By: Claude Opus 5.5 <noreply" + b"@anthropic.com>\n")
    (tmp_path / "f.txt").write_bytes(fakes)
    assert not any(AS.scan(tmp_path, [])["hits"].values())


def test_real_looking_values_off_the_list_still_trip(tmp_path):
    """Near misses of each fake, one at a time: a value sharing a fake's
    prefix, or the trailer's address outside a trailer line, is not excused."""
    gs, at = "gs:" + "//", "@"
    cases = [
        ("aws_12_digit", f"account {ACCT}"),
        ("aws_arn", "arn:" + "aws:iam::" + ACCT + ":role/r"),
        ("aws_arn", "arn:" + "aws:iam::xy "),
        ("email", f"see ba{at}b.org"),
        ("email", f"see a{at}b.org.uk"),
        ("email", f"Co-Authored-By: Claude Opus 5.5 <noreply{at}anthropic.com> inline"),
        ("email", f"x noreply{at}anthropic.com"),
        ("email", f"see Co-Authored-By: Claude Opus 5.5 <noreply{at}anthropic.com>\n"),
        ("store_url", f'"{gs}b-real-bucket"'),
        ("store_url", f'"{gs}{{STORE}}x"'),
        ("home_dir", "/Users/" + "someone2/x"),
    ]
    for i, (k, text) in enumerate(cases):
        d = tmp_path / str(i)
        d.mkdir()
        (d / "v.txt").write_bytes(text.encode())
        assert AS.scan(d, [])["hits"][k] == {"v.txt": 1}, (i, k)


def test_literals_are_never_excused(tmp_path):
    (tmp_path / "f.txt").write_bytes(b"account 123456789012")
    assert AS.scan(tmp_path, [b"123456789012"])["hits"]["literal 1"] == {"f.txt": 1}
