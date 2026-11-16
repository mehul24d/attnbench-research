"""A stand-in for Nsight Systems, for the H100 diagnostic's tests and its
pre-launch CPU pass (`H100_DIAG_NSYS` names a wrapper that runs this file).

    python3 tests/_nsys_fakes.py --version
    python3 tests/_nsys_fakes.py profile [options] -o <base> <child command...>
    python3 tests/_nsys_fakes.py export --type=sqlite --force-overwrite=true \\
        --output=<db> <base>.nsys-rep

`profile` runs the child (so the real `one` subcommand runs) and writes a
report naming how many prefills the child was asked for; `export` writes a
synthetic sqlite export with the tables `nsys_prefills` reads. The numbers
are `write_db`'s, known exactly, so a test can check every field.
FAKE_NSYS_NO_CHILD=1 skips the child; FAKE_NSYS_FAIL=profile|export makes
that step exit 1."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

PREFIX = "h100diag:prefill:"
MS = 1_000_000          # ns
PERIOD = 1_000 * MS     # one prefill per second
WALL = 100 * MS
LAUNCH_EVERY, LAUNCH_NS, N_LAUNCHES = 2 * MS, 5_000, 10
KERNELS = (("flash_fwd_kernel", 20 * MS), ("sm90_gemm_bf16", 30 * MS))


def expected(i: int = 0) -> dict:
    """What `nsys_prefills` must read for prefill i (seconds)."""
    last = (N_LAUNCHES - 1) * LAUNCH_EVERY + LAUNCH_NS
    blocked = 1 * MS        # the one mid-prefill synchronise, before the last launch
    busy = sum(d for _, d in KERNELS) + 2 * MS     # plus the memcpy and the memset
    return {"range": f"{PREFIX}{i}", "wall_s": WALL * 1e-9, "device_busy_s": busy * 1e-9,
            "idle_s": (WALL - busy) * 1e-9, "submit_s": last * 1e-9, "blocked_s": blocked * 1e-9,
            "host_s": (last - blocked) * 1e-9, "launches": N_LAUNCHES,
            "kernels_s": {n: d * 1e-9 for n, d in KERNELS}}


def write_db(path, prefills: int) -> None:
    """A synthetic export: per prefill, ten launches 2 ms apart, a 1 ms
    synchronise between the 3rd and 4th, two kernels back to back from 25 ms,
    a 1 ms memcpy, a 1 ms memset, and the closing synchronise; plus an unrelated NVTX range
    and a kernel outside every prefill, which must not count."""
    p = Path(path)
    p.unlink(missing_ok=True)
    con = sqlite3.connect(p)
    con.executescript("""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, shortName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER);
    """)
    names = ["cudaLaunchKernel_v7000", "cudaStreamSynchronize_v3020", "flash_fwd_kernel",
             "sm90_gemm_bf16", "stray_kernel"]
    con.executemany("INSERT INTO StringIds VALUES (?, ?)", list(enumerate(names)))
    sid = {n: i for i, n in enumerate(names)}
    for i in range(prefills):
        a = MS + i * PERIOD
        con.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)", (a, a + WALL, f"{PREFIX}{i}"))
        for j in range(N_LAUNCHES):
            t = a + j * LAUNCH_EVERY
            con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?)",
                        (t, t + LAUNCH_NS, sid["cudaLaunchKernel_v7000"]))
        t = a + 3 * LAUNCH_EVERY + LAUNCH_NS
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?)",
                    (t, t + MS, sid["cudaStreamSynchronize_v3020"]))
        t = a + 25 * MS
        for n, d in KERNELS:
            con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)", (t, t + d, sid[n]))
            t += d
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?)", (t + MS, t + 2 * MS))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (?, ?)", (t + 3 * MS, t + 4 * MS))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?)",
                    (t + 4 * MS, a + WALL, sid["cudaStreamSynchronize_v3020"]))
        con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?)",
                    (a + 500 * MS, a + 510 * MS, sid["stray_kernel"]))
    con.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)", (0, 10, "warmup"))
    con.commit()
    con.close()


def _profile(argv: list[str]) -> int:
    i = argv.index("-o")
    base, child = Path(argv[i + 1]), argv[i + 2:]
    if not os.environ.get("FAKE_NSYS_NO_CHILD"):
        r = subprocess.run(child)
        if r.returncode != 0:
            return r.returncode
    n = int(child[child.index("--prefills") + 1])
    base.with_suffix(".nsys-rep").write_text(json.dumps({"fake": True, "prefills": n}))
    return 0


def _export(argv: list[str]) -> int:
    out = next(a.split("=", 1)[1] for a in argv if a.startswith("--output="))
    rep = json.loads(Path(argv[-1]).read_text())
    write_db(out, rep["prefills"])
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--version"]:
        print("NVIDIA Nsight Systems version FAKE (tests/_nsys_fakes.py)")
        return 0
    if argv[:1] and os.environ.get("FAKE_NSYS_FAIL") == argv[0]:
        print(f"fake nsys: {argv[0]} failed", file=sys.stderr)
        return 1
    if argv[:1] == ["profile"]:
        return _profile(argv[1:])
    if argv[:1] == ["export"]:
        return _export(argv[1:])
    print(f"fake nsys: unknown {argv[:1]}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
