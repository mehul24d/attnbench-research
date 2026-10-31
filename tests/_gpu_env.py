"""The device and dtype for the GPU variants of the five GQA tests (T2.26).

The five tests named in `_sdpa_env.py` pin `device="cpu"`, so on a GPU
instance they check nothing about the GPU path. Their variants, in
`test_*_gpu.py`, take the device from `ATTNBENCH_TEST_DEVICE` (default
"cuda") and skip unless it is a CUDA device that is present. They are not
xfail-marked: CUDA flash SDPA accepts GQA shapes, so on a GPU they must
pass.

CUDA flash attention runs only in half precision, so the variants run the
toy model in `ATTNBENCH_TEST_DTYPE` (default "bfloat16", the study's dtype;
"float16" is also accepted). Where a CPU test compares two computations at
an fp32 tolerance, its variant uses `tolerance`, which scales the fp32 bound
up to that dtype's rounding. These bounds have not yet been checked on a GPU.
"""

from __future__ import annotations

import os

import pytest
import torch

DEVICE_ENV = "ATTNBENCH_TEST_DEVICE"
DTYPE_ENV = "ATTNBENCH_TEST_DTYPE"
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def device() -> str:
    return os.environ.get(DEVICE_ENV, "cuda")


def dtype_name() -> str:
    name = os.environ.get(DTYPE_ENV, "bfloat16")
    if name not in DTYPES:
        raise ValueError(f"{DTYPE_ENV}={name!r}; CUDA flash SDPA needs one of "
                         f"{sorted(DTYPES)}")
    return name


def dtype() -> torch.dtype:
    return DTYPES[dtype_name()]


def _usable() -> bool:
    try:
        d = torch.device(device())
    except RuntimeError:
        return False
    return d.type == "cuda" and torch.cuda.is_available() and \
        (d.index is None or d.index < torch.cuda.device_count())


needs_gpu = pytest.mark.skipif(
    not _usable(),
    reason=f"needs the CUDA device named by {DEVICE_ENV} (default 'cuda')")


def tolerance(fp32_bound: float) -> float:
    """The fp32 bound, or 8 units of rounding in the test dtype if larger."""
    return max(fp32_bound, 8 * torch.finfo(dtype()).eps)
