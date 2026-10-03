#!/usr/bin/env python
"""Llama-3.1-8B oracle-pass cost for the estimator-frontier bracket (sec. 8.2,
lines I2c / I2d / I2e), from measured numbers (2026-10-03).

    python scripts/derive_llama_oracle_cost.py --log <s12_vec_e2e.log> --toklens <json>

It replaces the earlier derivation, which multiplied a FLOP-split Llama
prefill model by an assumed mu = 2.0 lower bound and an assumed 1.6x upper
factor.

Inputs:

- **Measured.** The Qwen2.5-1.5B oracle scoring pass on the A100: one cold
  call per band in `results/s12_a100_vec_endtoend/logs/s12_vec_e2e.log`
  ("scoring pass 1.5s / 2.7s / 10.0s" at 8192 / 16384 / 32768). That is
  this harness's fp32 materialised-softmax pass.
- **From configs, no assumption.** The scale to Llama. The pass is attention
  work, which scales by layers x query heads x head_dim (3.048), plus the
  rest of the forward, which scales by non-embedding linear parameters
  (5.327). The split between the two is unknown, so the bracket takes the
  whole pass at 3.048 (lower) and at 5.327 (upper). Any split lies between.
- **Token counts.** `--toklens` is a JSON list of `text.json` lengths. They
  are Qwen-tokenizer counts, because the Llama tokenizer is gated. G9 step 3
  re-counts them with Llama on the instance.

Between measured bands, the cost is interpolated as a power law. Past 32768
it is extrapolated with the 16K-32K exponent (lower) or 2 (upper). Below 8192
it is linear (upper) or quadratic (lower) from the 8192 point. The XA
estimate pass is bounded by one dense prefill, which is at most
1/min(measured mu) of the oracle pass.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

DENSE_PREFILL_S = {8192: 0.18987, 16384: 0.43592, 32768: 1.10181}  # s12_a100_vec_endtoend, sdpa_flash
A100_RATE = 284  # Rs/h, docs/spend_ledger.md
OVERHEAD_MIN = (3, 10)


def linear_params(hidden, intermediate, kv_heads, layers, head_dim=128):
    return layers * (2 * hidden * hidden + 2 * hidden * kv_heads * head_dim
                     + 3 * hidden * intermediate)


QWEN_15B = linear_params(1536, 8960, 2, 28)      # local config, rev 989aa79
LLAMA_8B = linear_params(4096, 14336, 8, 32)     # G9 config.json, rev 0e9e39f
R_ATTN = (32 * 32) / (28 * 12)
R_LIN = LLAMA_8B / QWEN_15B


def read_scoring(log: Path) -> dict[int, float]:
    out, band = {}, None
    for line in log.read_text().splitlines():
        m = re.match(r"===== band (\d+) =====", line)
        if m:
            band = int(m.group(1))
        m = re.search(r"scoring pass ([\d.]+)s", line)
        if m and band is not None:
            out[band] = float(m.group(1))
    assert set(out) == set(DENSE_PREFILL_S), out
    return out


def t15(n, M, upper):
    if n <= 8192:
        return M[8192] * (n / 8192) ** (1 if upper else 2)
    if n <= 16384:
        return M[8192] * (n / 8192) ** math.log2(M[16384] / M[8192])
    e = math.log2(M[32768] / M[16384])
    if n <= 32768:
        return M[16384] * (n / 16384) ** e
    return M[32768] * (n / 32768) ** (2.0 if upper else e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--toklens", type=Path, required=True)
    args = ap.parse_args()
    M = read_scoring(args.log)
    toks = json.loads(args.toklens.read_text())
    mu = {n: M[n] / DENSE_PREFILL_S[n] for n in M}
    xa = (1.0, 1 + 1 / min(mu.values()))
    print(f"measured A100 mu (1.5B, this harness's oracle): "
          + ", ".join(f"{n}: {v:.2f}" for n, v in mu.items()))
    print(f"scale to Llama: attention {R_ATTN:.3f}, linear {R_LIN:.3f} "
          f"({QWEN_15B / 1e9:.3f}B -> {LLAMA_8B / 1e9:.3f}B non-embedding)")
    for name, sel, ovh in (("I2c texts <= 32K", lambda n: n <= 32768, True),
                           ("I2d texts > 32K", lambda n: n > 32768, False),
                           ("I2e all texts", lambda n: True, False)):
        mins = []
        for k, r in enumerate((R_ATTN, R_LIN)):
            sec = sum(r * t15(n, M, k == 1) for n in toks if sel(n)) * xa[k]
            mins.append(sec / 60 + (OVERHEAD_MIN[k] if ovh else 0))
        print(f"{name}: {mins[0]:.0f}-{mins[1]:.0f} min, "
              f"Rs {mins[0] / 60 * A100_RATE:.0f}-{mins[1] / 60 * A100_RATE:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
