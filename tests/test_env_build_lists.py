"""The AWS environment build's list and wheel checks (scripts/env_build_lists.py)."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("env_build_lists", ROOT / "scripts" / "env_build_lists.py")
eb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eb)

V6 = ROOT / "docs" / "image_libraries" / "attnbench-env-v6-20260917"
WHEEL_NAME = "flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl"
FOUND = {"python": "3.10.12", "torch": "2.9.1+cu129"}


def _tag(*t):
    try:
        from packaging.tags import Tag
    except ImportError:
        from pip._vendor.packaging.tags import Tag
    return Tag(*t)


def test_usersite_list_refuses_xformers_local_build_without_a_stand_in():
    with pytest.raises(eb.Refused, match="xformers"):
        eb.requirements((V6 / "pip_freeze_usersite_mehuldahiya.txt").read_text(), {})


def test_usersite_list_with_the_stand_in():
    pins, local = eb.requirements((V6 / "pip_freeze_usersite_mehuldahiya.txt").read_text(),
                                  {"xformers": "0.0.33"})
    assert "xformers==0.0.33" in pins
    assert not any("5d4b92a5" in p for p in pins)
    assert "torchaudio==2.11.0+cu129" in pins and "transformers==4.46.0" in pins
    names = {p.split("==")[0].lower() for p in pins}
    assert not names & {"flash_attn", "flash-attn", "block_sparse_attn", "attnbench"}
    assert len(local) == 3 and all(x.startswith("-e ") or " @ file://" in x for x in local)


def test_system_list_refuses_apt_packages_without_dpkg():
    with pytest.raises(eb.Refused, match="ubuntu"):
        eb.requirements((V6 / "pip_freeze_root.txt").read_text(), {})


def test_system_list_less_apts_packages():
    pins, out = eb.requirements((V6 / "pip_freeze_root.txt").read_text(), {},
                                (V6 / "dpkg_l.txt").read_text())
    names = {p.split("==")[0] for p in pins}
    assert "torch==2.9.1+cu129" in pins and "triton==3.5.1" in pins
    assert "packaging==26.3" in pins          # pip's, over apt's 21.3
    assert {"numpy", "sympy", "nvidia-cudnn-cu12", "torchvision", "lightning"} <= names
    apt = {x.split("==")[0] for x in out}
    assert {"distro-info", "python-apt", "ubuntu-pro-client", "PyYAML", "Jinja2", "attrs",
            "command-not-found", "PyGObject", "pip", "setuptools", "sos"} <= apt
    assert len(pins) + len(out) == 110 and len(out) == 69
    assert not any("ubuntu" in p or p.startswith("pip==") for p in pins)


def test_apt_needs_the_version_unless_aliased():
    dpkg = {"python3-packaging": "21.3-1", "python3-attr": "21.2.0-1ubuntu1"}
    assert not eb.is_apt("packaging", "26.3", dpkg)
    assert eb.is_apt("packaging", "21.3", dpkg)
    assert eb.is_apt("attrs", "99", dpkg) and not eb.is_apt("attrs", "99", {})


@pytest.mark.parametrize("text,rep,match", [
    ("a==1\na==2\n", {}, "twice"),
    ("a>=1\n", {}, "not a pinned"),
    ("a==1\n", {"b": "2"}, "not in the list"),
    ("a==1+local7\n", {}, "local build"),
])
def test_requirements_refusals(text, rep, match):
    with pytest.raises(eb.Refused, match=match):
        eb.requirements(text, rep)


def test_requirements_names_match_case_and_separators():
    pins, _ = eb.requirements("Foo_Bar==1+x9\n", {"foo-bar": "1"})
    assert pins == ["foo-bar==1"]


def _wheel(tmp_path):
    w = tmp_path / WHEEL_NAME
    w.write_bytes(b"not really a wheel")
    return w, hashlib.sha256(w.read_bytes()).hexdigest()


def test_fa3_wheel_passes_on_its_bytes_stack_and_tag(tmp_path):
    w, sha = _wheel(tmp_path)
    eb.fa3_wheel(w, sha, None, FOUND, {_tag("cp39", "abi3", "linux_x86_64")})


@pytest.mark.parametrize("change,match", [
    ("sha", "is not the recorded"),
    ("torch", "torch here is"),
    ("python", "python here is"),
    ("tags", "no tag"),
])
def test_fa3_wheel_refusals(tmp_path, change, match):
    w, sha = _wheel(tmp_path)
    found, tags = dict(FOUND), {_tag("cp39", "abi3", "linux_x86_64")}
    if change == "sha":
        sha = "0" * 64
    elif change in found:
        found[change] = "other"
    else:
        tags = {_tag("cp39", "abi3", "macosx_14_0_arm64")}
    with pytest.raises(eb.Refused, match=match):
        eb.fa3_wheel(w, sha, None, found, tags)


def test_fa3_wheel_with_a_manifest_uses_the_full_check(tmp_path):
    from attnbench.fa3_wheel import WheelCheckError
    w, sha = _wheel(tmp_path)
    m = tmp_path / "manifest.json"
    good = {"sha256": sha, "wheel": WHEEL_NAME, "stack": dict(FOUND)}
    m.write_text(json.dumps(good))
    eb.fa3_wheel(w, sha, m, FOUND, {_tag("cp39", "abi3", "linux_x86_64")})
    m.write_text(json.dumps(dict(good, sha256="1" * 64)))
    with pytest.raises(WheelCheckError, match="manifest"):
        eb.fa3_wheel(w, sha, m, FOUND, {_tag("cp39", "abi3", "linux_x86_64")})


def _p2():
    return [list(x) for x in json.loads((ROOT / eb.LIBRARY_REFERENCE).read_text())["libraries"]]


def test_libraries_equal_to_p2s_with_fa3_added():
    assert eb.library_problems(_p2() + [["flash_attn_3", "3.0.0b1"]]) == []


def test_libraries_another_version_or_a_missing_one_is_a_problem():
    got = [[n, "9.9" if n == "torch" else v] for n, v in _p2() if n != "xattn"]
    problems = eb.library_problems(got)
    assert len(problems) == 2
    assert any("torch" in p for p in problems) and any("xattn" in p for p in problems)


def test_diff_freeze():
    assert eb.diff_freeze("a==1\nb==2\n", "b==2\nc==3\n") == {"only_new": ["a==1"], "only_old": ["c==3"]}


def test_cli_refusal_exits_2(tmp_path, capsys):
    f = tmp_path / "l.txt"
    f.write_text("a==1+local\n")
    assert eb.main(["requirements", str(f)]) == 2
    assert "REFUSING" in capsys.readouterr().err


def test_cli_libraries_exit_code(tmp_path, capsys):
    f = tmp_path / "got.json"
    f.write_text(json.dumps(_p2()))
    assert eb.main(["libraries", "--got", str(f)]) == 0
    f.write_text(json.dumps(_p2()[1:]))
    capsys.readouterr()
    assert eb.main(["libraries", "--got", str(f)]) == 1
    assert json.loads(capsys.readouterr().out)["problems"]
