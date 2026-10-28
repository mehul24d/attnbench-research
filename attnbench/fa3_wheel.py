"""The FA3 wheel: stack check before the build, manifest after it, and the
checks on the H100 before and after it is installed.

Called by `scripts/build_fa3_wheel.sh` (on the CPU build VM) and
`scripts/fa3_import_check.sh` (on the H100). The wheel's own tag is
`cp39-abi3-linux_x86_64` (`hopper/setup.py` builds with py_limited_api), so
the tag says nothing about torch; a torch mismatch is an ABI break the tag
cannot catch. So the manifest records the Python and torch the wheel was
built against, and the H100 check compares its own against them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

SCHEMA = "attnbench.fa3_wheel/1"


class WheelCheckError(RuntimeError):
    """The wheel, or the instance it is meant for, is not the one recorded."""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.split(".")[:2])


def stack(nvcc_version: str) -> dict:
    import torch
    return {"python": "%d.%d.%d" % sys.version_info[:3], "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "nvcc": nvcc_version,
            "machine": platform.machine()}


def check_stack(found: dict, expect_python: str, expect_torch: str, min_nvcc: str) -> None:
    problems = []
    if not found["python"].startswith(expect_python + "."):
        problems.append(f"python {found['python']}, expected {expect_python}.x")
    if found["torch"] != expect_torch:
        problems.append(f"torch {found['torch']}, expected {expect_torch}")
    if not found["nvcc"] or _version_tuple(found["nvcc"]) < _version_tuple(min_nvcc):
        problems.append(f"nvcc {found['nvcc'] or 'not found'}, need {min_nvcc} or later")
    if problems:
        raise WheelCheckError("build stack does not match the H100 image: " + "; ".join(problems))


def write_manifest(wheel: Path, out: Path, **fields) -> dict:
    m = {"schema": SCHEMA, "wheel": wheel.name, "sha256": sha256(wheel), **fields}
    out.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    return m


def _parse_wheel_filename(name: str):
    try:
        from packaging.utils import parse_wheel_filename
    except ImportError:                                   # pip vendors it
        from pip._vendor.packaging.utils import parse_wheel_filename
    return parse_wheel_filename(name)


def _sys_tags():
    try:
        from packaging.tags import sys_tags
    except ImportError:
        from pip._vendor.packaging.tags import sys_tags
    return set(sys_tags())


def check_wheel(wheel: Path, manifest: dict, expected_sha256: str, found: dict,
                supported_tags=None) -> None:
    """Before install, on the H100: the bytes are the recorded ones, the
    instance's Python and torch are the ones the wheel was built against, and
    the wheel's tag is one this interpreter installs."""
    got = sha256(wheel)
    if got != expected_sha256:
        raise WheelCheckError(f"wheel sha256 {got} is not the recorded {expected_sha256}")
    if manifest.get("sha256") != expected_sha256:
        raise WheelCheckError("the manifest's sha256 is not the recorded one")
    if manifest.get("wheel") != wheel.name:
        raise WheelCheckError(f"manifest names {manifest.get('wheel')}, not {wheel.name}")
    built = manifest.get("stack", {})
    for key in ("python", "torch"):
        if built.get(key) != found.get(key):
            raise WheelCheckError(f"{key} here is {found.get(key)}, the wheel was built "
                                  f"against {built.get(key)}")
    tags = supported_tags if supported_tags is not None else _sys_tags()
    _, _, _, wheel_tags = _parse_wheel_filename(wheel.name)
    if not (set(wheel_tags) & set(tags)):
        raise WheelCheckError(f"no tag of {wheel.name} is supported by this interpreter")


def import_check() -> dict:
    """After install, on the H100: the module imports, the device is Hopper,
    and the backend reports itself available."""
    import torch
    import flash_attn_interface  # noqa: F401  (imports flash_attn_3._C)
    from attnbench.backends.fa3 import FlashAttention3
    if not torch.cuda.is_available():
        raise WheelCheckError("no CUDA device")
    cc = tuple(torch.cuda.get_device_capability())
    if cc != (9, 0):
        raise WheelCheckError(f"compute capability {cc}, the wheel is sm_90a only")
    if not FlashAttention3.is_available():
        raise WheelCheckError("the fa3 backend reports itself unavailable")
    return {"module": getattr(flash_attn_interface, "__file__", None),
            "device": torch.cuda.get_device_name(), "compute_capability": list(cc),
            "fa3_available": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stack")
    s.add_argument("--nvcc", required=True)
    s.add_argument("--expect-python", required=True)
    s.add_argument("--expect-torch", required=True)
    s.add_argument("--min-nvcc", required=True)
    s.add_argument("--out", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--wheel", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--fields", required=True, help="JSON object of extra fields")
    c = sub.add_parser("check")
    c.add_argument("--wheel", required=True)
    c.add_argument("--manifest", required=True)
    c.add_argument("--sha256", required=True)
    sub.add_parser("import")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "stack":
            found = stack(a.nvcc)
            Path(a.out).write_text(json.dumps(found, sort_keys=True) + "\n")
            print("stack: " + json.dumps(found, sort_keys=True), flush=True)
            check_stack(found, a.expect_python, a.expect_torch, a.min_nvcc)
        elif a.cmd == "manifest":
            got = write_manifest(Path(a.wheel), Path(a.out), **json.loads(a.fields))
            print(got["sha256"])
        elif a.cmd == "check":
            manifest = json.loads(Path(a.manifest).read_text())
            check_wheel(Path(a.wheel), manifest, a.sha256, stack(""))
            print("wheel check ok: " + Path(a.wheel).name, flush=True)
        elif a.cmd == "import":
            print("import check ok: " + json.dumps(import_check(), sort_keys=True), flush=True)
    except WheelCheckError as e:
        print(f"FATAL: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
