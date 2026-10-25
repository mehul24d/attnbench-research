#!/usr/bin/env python3
"""Positive control for the page-fault check (T2.7): a process that
certainly faults on unified memory.

Run it under the same `nsys profile` options as the traced pass it vouches
for (`scripts/check_trace.py nsys-available` prints them). It allocates
managed memory through the CUDA driver API, writes it from the CPU, reads
and writes it in a GPU kernel (GPU faults, pages migrate to the device),
then reads it on the CPU again (CPU faults). If the export of this run
shows no fault rows, fault collection was not on, and the traced pass's
export cannot support "no page faults".

The managed buffer is handed to PyTorch through `__cuda_array_interface__`,
so the GPU work is an ordinary PyTorch kernel. Needs a CUDA device.

Usage, on a CUDA instance from the repo root:
    nsys profile --trace=cuda,nvtx --cuda-um-cpu-page-faults=true \\
        --cuda-um-gpu-page-faults=true -o OUT_BASE python3 scripts/um_fault_control.py
"""

from __future__ import annotations

import ctypes
import sys

CU_MEM_ATTACH_GLOBAL = 1
N_FLOATS = 1 << 24          # 64 MiB: many pages each way


class _Managed:
    def __init__(self, nbytes: int, ptr: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes // 4,), "typestr": "<f4",
            "data": (ptr, False), "version": 3}


def main() -> int:
    import torch
    if not torch.cuda.is_available():
        print("needs a CUDA device", file=sys.stderr)
        return 2
    torch.zeros(1, device="cuda")                       # the primary context
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_uint64),
                                       ctypes.c_size_t, ctypes.c_uint]
    cuda.cuMemFree.argtypes = [ctypes.c_uint64]
    ptr = ctypes.c_uint64()
    nbytes = N_FLOATS * 4
    rc = cuda.cuMemAllocManaged(ctypes.byref(ptr), nbytes, CU_MEM_ATTACH_GLOBAL)
    if rc != 0:
        print(f"cuMemAllocManaged failed: CUresult {rc}", file=sys.stderr)
        return 3
    try:
        ctypes.memset(ptr.value, 0, nbytes)             # pages first touched on the CPU
        t = torch.as_tensor(_Managed(nbytes, ptr.value), device="cuda")
        t.add_(1.0)                                     # GPU faults
        torch.cuda.synchronize()
        first = ctypes.c_float.from_address(ptr.value).value   # CPU fault
        last = ctypes.c_float.from_address(ptr.value + nbytes - 4).value
        print(f"managed buffer {nbytes >> 20} MiB: first {first}, last {last} "
              f"(both 1.0 if the GPU wrote it)")
        return 0 if first == last == 1.0 else 4
    finally:
        cuda.cuMemFree(ptr.value)


if __name__ == "__main__":
    raise SystemExit(main())
