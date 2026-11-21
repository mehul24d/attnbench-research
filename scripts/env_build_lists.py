#!/usr/bin/env python3
"""The list and wheel checks of the AWS environment build
(scripts/aws_env_build_instance.sh; docs/launch_cards/aws_env_build_m7a.md).

    requirements LIST [--replace NAME==VERSION ...] [--dpkg DPKG_LIST]
        The pip requirements of one of image v6's `pip freeze` lists
        (docs/image_libraries/attnbench-env-v6-20260917/), one `name==version`
        per line. Lines for packages built from local sources (`-e ...`,
        `name @ file://...`) are left out and named on stderr: the build
        script builds them. A version with a local label other than a CUDA
        build tag (`+cu129`) cannot be fetched from an index, so it is
        refused unless --replace names its stand-in. With --dpkg (the system
        list, against v6's `dpkg -l`), a package apt installed is left out and
        named: it comes with the base image, and no index has most of them.
        A pin is apt's when v6's dpkg list has it installed as `python3-<name>`
        (or under its own name) at the same upstream version, or under the
        name in APT_ALIASES; `packaging==26.3` over apt's 21.3 stays a pin.

    fa3-wheel WHEEL --sha256 HEX [--manifest FILE]
        Before FA3's install, never importing it: the bytes are the recorded
        ones and a tag of the wheel is one this interpreter installs. With
        the build's manifest, attnbench.fa3_wheel.check_wheel in full (the
        manifest's sha256, name and build stack); without it, this Python and
        torch against the stack the wheel was built on (build_fa3_wheel.sh).

    libraries [--got FILE]
        This environment's libraries (t61_session.session_libraries) against
        P2's (configs/t64/p2_libraries.json) by t61_session.library_problems,
        FA3 the one allowed difference. Prints JSON; exit 1 on a difference.

    diff-freeze NEW OLD
        Lines only in NEW and only in OLD, for the build's record.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIBRARY_REFERENCE = "configs/t64/p2_libraries.json"
# The stack the FA3 wheel was built on (scripts/build_fa3_wheel.sh, EXPECT_TORCH
# and its Python 3.10.12), checked when the build's manifest is not at hand.
FA3_BUILT_PYTHON = "3.10.12"
FA3_BUILT_TORCH = "2.9.1+cu129"

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)$")
_CUDA_LABEL = re.compile(r"^cu\d+$")
# pip names whose Ubuntu 22.04 package has another name; their pip-side
# versions are often the tool's own (command-not-found 0.3), so the name decides.
APT_ALIASES = {
    "attrs": "python3-attr", "command-not-found": "python3-commandnotfound",
    "dbus-python": "python3-dbus", "jsonpointer": "python3-json-pointer",
    "lazr-restfulclient": "python3-lazr.restfulclient", "lazr-uri": "python3-lazr.uri",
    "pygobject": "python3-gi", "sos": "sosreport", "systemd-python": "python3-systemd",
    "ubuntu-pro-client": "ubuntu-pro-client", "unattended-upgrades": "unattended-upgrades",
    "zope-interface": "python3-zope.interface",
}


class Refused(Exception):
    pass


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _dpkg(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 3 and f[0] == "ii":
            out[f[1].split(":")[0].lower()] = f[2]
    return out


def _upstream(deb_version: str) -> str:
    return re.split(r"[-~]", deb_version.split(":", 1)[-1])[0].replace("+", "")


def is_apt(name: str, version: str, dpkg: dict) -> bool:
    k = _norm(name)
    alias = APT_ALIASES.get(k)
    if alias is not None:
        return alias in dpkg
    v = version.replace("+", "")
    for c in {k, f"python3-{k}", f"python3-{k.replace('-', '')}",
              f"python3-{re.sub(r'^(python|py)-?', '', k)}"}:
        if c in dpkg and (_upstream(dpkg[c]).startswith(v) or v.startswith(_upstream(dpkg[c]))):
            return True
    return False


def requirements(text: str, replace: dict, dpkg_text: str | None = None) -> tuple[list, list]:
    """(pins, left out: built here or apt's) from a `pip freeze` list."""
    pins, local, seen = [], [], set()
    rep = {_norm(k): (k, v) for k, v in replace.items()}
    dpkg = _dpkg(dpkg_text) if dpkg_text is not None else None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-e ") or " @ file://" in line:
            local.append(line)
            continue
        m = _PIN.match(line)
        if not m:
            raise Refused(f"not a pinned requirement: {line!r}")
        name, ver = m.groups()
        if _norm(name) in seen:
            raise Refused(f"{name} is listed twice")
        seen.add(_norm(name))
        if dpkg is not None and _norm(name) not in rep and is_apt(name, ver, dpkg):
            local.append(f"{line} (apt's)")
            continue
        if _norm(name) in rep:
            name, ver = rep.pop(_norm(name))
        elif "+" in ver and not _CUDA_LABEL.match(ver.split("+", 1)[1]):
            raise Refused(f"{name}=={ver} is a local build; no index has it "
                          f"(name its stand-in with --replace)")
        pins.append(f"{name}=={ver}")
    if rep:
        raise Refused(f"--replace names packages not in the list: {sorted(v[0] for v in rep.values())}")
    return pins, local


def fa3_wheel(wheel: Path, sha: str, manifest: Path | None, found: dict, tags=None) -> None:
    from attnbench import fa3_wheel as fw
    if manifest is not None:
        fw.check_wheel(wheel, json.loads(manifest.read_text()), sha, found, tags)
        return
    got = fw.sha256(wheel)
    if got != sha:
        raise Refused(f"wheel sha256 {got} is not the recorded {sha}")
    for key, built in (("python", FA3_BUILT_PYTHON), ("torch", FA3_BUILT_TORCH)):
        if found.get(key) != built:
            raise Refused(f"{key} here is {found.get(key)}, the wheel was built against {built}")
    tags = tags if tags is not None else fw._sys_tags()
    if not set(fw._parse_wheel_filename(wheel.name)[3]) & set(tags):
        raise Refused(f"no tag of {wheel.name} is supported by this interpreter")


def library_problems(got: list) -> list:
    sys.path.insert(0, str(ROOT / "scripts"))
    import t61_session as t
    return t.library_problems(SimpleNamespace(library_reference=LIBRARY_REFERENCE), got)


def session_libraries() -> list:
    sys.path.insert(0, str(ROOT / "scripts"))
    import t61_session as t
    return t.session_libraries()


def diff_freeze(new: str, old: str) -> dict:
    a = {x.strip() for x in new.splitlines() if x.strip()}
    b = {x.strip() for x in old.splitlines() if x.strip()}
    return {"only_new": sorted(a - b), "only_old": sorted(b - a)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("requirements")
    r.add_argument("list", type=Path)
    r.add_argument("--replace", action="append", default=[])
    r.add_argument("--dpkg", type=Path)
    f = sub.add_parser("fa3-wheel")
    f.add_argument("wheel", type=Path)
    f.add_argument("--sha256", required=True)
    f.add_argument("--manifest", type=Path)
    lb = sub.add_parser("libraries")
    lb.add_argument("--got", type=Path)
    d = sub.add_parser("diff-freeze")
    d.add_argument("new", type=Path)
    d.add_argument("old", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "requirements":
            rep = {}
            for x in args.replace:
                m = _PIN.match(x)
                if not m:
                    raise Refused(f"--replace takes NAME==VERSION, not {x!r}")
                rep[m.group(1)] = m.group(2)
            pins, local = requirements(args.list.read_text(), rep,
                                       args.dpkg.read_text() if args.dpkg else None)
            for line in local:
                print(f"left out, not from an index: {line}", file=sys.stderr)
            print("\n".join(pins))
        elif args.cmd == "fa3-wheel":
            if not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
                raise Refused("--sha256 is not 64 lowercase hex")
            import torch
            fa3_wheel(args.wheel, args.sha256, args.manifest,
                      {"python": platform.python_version(), "torch": torch.__version__})
            print(f"FA3 wheel {args.wheel.name}: sha256 and tags checked; not imported")
        elif args.cmd == "libraries":
            got = json.loads(args.got.read_text()) if args.got else session_libraries()
            problems = library_problems(got)
            print(json.dumps({"reference": LIBRARY_REFERENCE, "libraries": got,
                              "problems": problems}, indent=1))
            return 1 if problems else 0
        else:
            print(json.dumps(diff_freeze(args.new.read_text(), args.old.read_text()), indent=1))
    except Exception as e:  # noqa: BLE001  (every refusal is one line and exit 2)
        if isinstance(e, SystemExit):
            raise
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
