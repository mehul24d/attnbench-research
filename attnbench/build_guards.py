"""Pre-flight resource guards for source builds of CUDA kernels.

Exists because of a specific, expensive failure: on 2026-09-02 a flash-attn
build was launched with `MAX_JOBS=5` and `NVCC_THREADS=2` on a 31 GB machine.
That is ~10 concurrent `cicc` processes (nvcc's CUDA frontend) at ~5 GB each,
so ~50 GB demanded against ~28 GB usable. The OOM killer fired eight times
and took the build with it, ninety minutes in, after SSH had already become
unreachable -- so the failure was neither visible nor recoverable until the
instance was torn down.

The arithmetic that was missed both times it was set, including once with a
second reader agreeing:

    concurrent cicc  ~=  max_jobs * nvcc_threads

`--threads` spawns one frontend per gencode target, so the two settings
MULTIPLY. Raising `max_jobs` while leaving `nvcc_threads` at its default is
not a partial fix; it is the same bug at a different scale.

This module refuses. It does not warn. A warning printed into a build log on
a machine whose SSH is about to die is not a control -- nobody reads it in
time, and the cost of proceeding is the whole session.
"""

from __future__ import annotations

from dataclasses import dataclass

# Measured from the 2026-09-02 OOM log: killed `cicc` processes reported
# anon-rss between 4.5 GB and 5.0 GB. Using the top of the observed range,
# since underestimating it is what caused the failure this guard prevents.
CICC_GB_PER_PROCESS = 5.0

# Memory that is not available to the build even when "free": kernel, page
# cache pressure, sshd, the monitoring agent, and the python process driving
# the build itself. Keeping SSH alive is not a nicety -- losing shell access
# to a billed instance blocks verification, measurement, and recovery.
SYSTEM_RESERVE_GB = 3.0


class BuildResourceError(RuntimeError):
    """Raised when a requested build concurrency cannot fit in memory."""


@dataclass(frozen=True)
class MemoryPlan:
    max_jobs: int
    nvcc_threads: int
    free_gb: float

    @property
    def concurrent_frontends(self) -> int:
        """The number that actually matters. Not max_jobs."""
        return self.max_jobs * self.nvcc_threads

    @property
    def required_gb(self) -> float:
        return self.concurrent_frontends * CICC_GB_PER_PROCESS

    @property
    def usable_gb(self) -> float:
        return max(0.0, self.free_gb - SYSTEM_RESERVE_GB)

    @property
    def fits(self) -> bool:
        return self.required_gb <= self.usable_gb

    def max_jobs_that_fit(self) -> int:
        """Largest max_jobs that fits at this nvcc_threads. May be 0, which
        means even a single job does not fit and the machine is too small --
        reported honestly rather than clamped up to 1 to look survivable."""
        per_job_gb = self.nvcc_threads * CICC_GB_PER_PROCESS
        return int(self.usable_gb // per_job_gb)


def plan_build_memory(max_jobs: int, nvcc_threads: int, free_gb: float) -> MemoryPlan:
    if max_jobs < 1:
        raise ValueError(f"max_jobs must be >= 1, got {max_jobs}")
    if nvcc_threads < 1:
        raise ValueError(f"nvcc_threads must be >= 1, got {nvcc_threads}")
    if free_gb < 0:
        raise ValueError(f"free_gb must be >= 0, got {free_gb}")
    return MemoryPlan(max_jobs=max_jobs, nvcc_threads=nvcc_threads, free_gb=free_gb)


def check_build_memory(max_jobs: int, nvcc_threads: int, free_gb: float) -> MemoryPlan:
    """Return the plan if it fits; raise BuildResourceError if it does not.

    Raising rather than warning is the whole point -- see module docstring.
    """
    plan = plan_build_memory(max_jobs, nvcc_threads, free_gb)
    if not plan.fits:
        fitting = plan.max_jobs_that_fit()
        suggestion = (f"lower MAX_JOBS to {fitting}"
                      if fitting >= 1 else
                      "this machine is too small for even one build job")
        raise BuildResourceError(
            f"MAX_JOBS={max_jobs} x NVCC_THREADS={nvcc_threads} = "
            f"{plan.concurrent_frontends} concurrent cicc processes needing "
            f"~{plan.required_gb:.0f} GB, but only ~{plan.usable_gb:.0f} GB is "
            f"usable ({free_gb:.0f} GB free minus {SYSTEM_RESERVE_GB:.0f} GB "
            f"system reserve). {suggestion}. "
            f"NOTE: max_jobs and nvcc_threads MULTIPLY -- lowering only one of "
            f"them may not be enough."
        )
    return plan


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-jobs", type=int, required=True)
    ap.add_argument("--nvcc-threads", type=int, required=True)
    ap.add_argument("--free-gb", type=float, default=None,
                    help="override measured free memory (for testing the guard)")
    args = ap.parse_args(argv)

    if args.free_gb is not None:
        free_gb = args.free_gb
    else:
        import psutil
        free_gb = psutil.virtual_memory().available / 2**30

    try:
        plan = check_build_memory(args.max_jobs, args.nvcc_threads, free_gb)
    except (BuildResourceError, ValueError) as e:
        print(f"FATAL: {e}", flush=True)
        return 1
    print(f"memory check ok: {plan.concurrent_frontends} concurrent cicc, "
          f"~{plan.required_gb:.0f} GB needed, ~{plan.usable_gb:.0f} GB usable",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
