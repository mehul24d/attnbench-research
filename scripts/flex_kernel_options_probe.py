#!/usr/bin/env python3
"""Find kernel_options under which flex block-sparse compiles on THIS card.

Written to be the entire GPU-side investigation of the segment-1 anomaly, in
one scripted attempt. The standing rule is that a step failing twice means
stop and report rather than debug on billed hardware, so the ladder of
candidates is decided here, on free hardware, and the instance only executes
it.

WHAT IS ALREADY SETTLED, off-GPU, and is not what this probe is for:

  * Why Stage 1 passed while Stage 2 failed at identical geometry.
    torch._dynamo hit its recompile limit of 8 during the probe (`stage1.log`,
    18:41:46) and silently ran `flex_attention` EAGERLY from then on. The
    eager path materialises scores and handles any block size on any card, so
    the gate certified an implementation the sweep never runs. Reproduced on
    CPU in tests/test_compile_guard.py; guarded by attnbench/compile_guard.py.

  * Why the compiled path fails. Inductor picks ONE config when max_autotune
    is off -- BLOCK_M=BLOCK_N=128 at head_dim=128 -- and then:
      block_size=64  -> 64 % 128 != 0, raises ValueError (it raises rather
                        than skipping precisely because there is one candidate)
      block_size=128 -> 114688 B of shared memory against sm_89's 101376 B
    Neither depends on cache state or fragmentation.

WHAT IS NOT SETTLED and needs this card: whether 64x64 tiles actually compile
and produce correct output here. That is a fact about sm_89 shared memory and
this triton build, and cannot be established anywhere else.

Each candidate runs after `torch._dynamo.reset()` in a fresh compile state, so
one candidate's failure cannot poison the next -- and so that every row is the
cold-process behaviour the sweep sees, not the accumulated-state behaviour
that produced the anomaly in the first place.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch                                          # noqa: E402

from attnbench import compile_guard                   # noqa: E402
from attnbench.backends.impls import NaiveAttention    # noqa: E402
from attnbench.config import AttnConfig               # noqa: E402
from attnbench.masks import mask_for                  # noqa: E402

# Ordered by preference: the first that compiles AND agrees with the oracle
# wins. 64x64 is the expected answer (it satisfies divisibility for both 64 and
# 128, and roughly halves shared memory); the rest are the documented retreats
# from the error message's own hint, "Reducing block sizes or num_stages may
# help."
CANDIDATES = [
    ("default", None),
    ("64x64", {"BLOCK_M": 64, "BLOCK_N": 64}),
    ("64x64/s2", {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 2}),
    ("64x64/s1", {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}),
    ("32x32/s1", {"BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1}),
]

# Smallest cells that still reproduce the failure: head_dim=128 is what makes
# inductor's default tile 128 wide, and both block sizes failed differently.
CASES = [(1024, 64), (1024, 128)]

BF16_TOL = 2e-2      # the tolerance the correctness gate uses for bfloat16


def _cfg(seq_len: int, block_size: int) -> AttnConfig:
    return AttnConfig(seq_len=seq_len, batch=1, n_heads_q=32, n_heads_kv=32,
                      head_dim=128, dtype="bfloat16", mask="block_sparse",
                      block_size=block_size, sparsity=0.9,
                      mask_source="random")


def try_one(cfg: AttnConfig, opts) -> tuple[str, str]:
    """(verdict, detail). Never raises -- a failure is the result."""
    from torch.nn.attention.flex_attention import flex_attention

    torch._dynamo.reset()
    compile_guard.reset_process_state()
    torch.cuda.empty_cache()

    mask = mask_for(cfg)
    naive = NaiveAttention()
    q, k, v = naive.make_inputs(cfg, device="cuda", seed=0)

    try:
        compiled = torch.compile(flex_attention, dynamic=False)
        with compile_guard.guard() as cg:
            with torch.no_grad():
                got = compiled(q, k, v,
                               block_mask=mask.to_flex_block_mask(device="cuda"),
                               kernel_options=opts)
            torch.cuda.synchronize()
        if cg.fell_back:
            # Would have looked like a pass. This is the anomaly's mechanism,
            # and here it is a REFUSAL rather than a result.
            return "EAGER", cg.detail
    except Exception as e:
        first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        return "COMPILE_FAIL", first[:160]

    try:
        with torch.no_grad():
            expected = naive.reference(q, k, v, cfg, mask=mask)
        err = (got.double() - expected).abs().max().item()
    except Exception as e:
        return "ORACLE_FAIL", f"{type(e).__name__}: {e}"[:160]

    verdict = "OK" if err <= BF16_TOL else "WRONG"
    return verdict, f"max_abs_err={err:.6f} (tol {BF16_TOL})"


def main() -> int:
    compile_guard.configure()
    if not torch.cuda.is_available():
        print("no CUDA device -- this probe is about sm_89 shared memory and "
              "cannot tell you anything on CPU")
        return 2

    print(f"device : {torch.cuda.get_device_name()} "
          f"(cc {'.'.join(map(str, torch.cuda.get_device_capability()))})")
    print(f"torch  : {torch.__version__}")
    print(f"smem/blk: "
          f"{torch.cuda.get_device_properties(0).shared_memory_per_block} B\n")

    winners = {}
    for seq_len, block_size in CASES:
        cfg = _cfg(seq_len, block_size)
        print(f"=== seq_len={seq_len} block_size={block_size} "
              f"head_dim=128 bf16 sparsity=0.9")
        for label, opts in CANDIDATES:
            verdict, detail = try_one(cfg, opts)
            print(f"    {label:<12} {verdict:<13} {detail}")
            if verdict == "OK":
                winners[block_size] = label
                break            # first that works wins; no point continuing
        else:
            print(f"    -> NOTHING WORKED at block_size={block_size}")
        print()

    print("SUMMARY:", winners if winners else "no working configuration")
    if set(winners) == {64, 128}:
        print("Both block sizes are recoverable. flex stays in the grid.")
        return 0
    print("Flex block-sparse is NOT available on this card for at least one "
          "block size. Stage 2's sparse arm shrinks accordingly -- report "
          "before changing the grid.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
