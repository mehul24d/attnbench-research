#!/usr/bin/env python3
"""The adviser's rulings dated 13 Nov 2026, item 1: on P2's A100 step 7
recording of `xattention_dcal_L2048_b1`, are XAttention's two Triton kernels
device positions that step 8's pricing and the prediction's operations carry?
P2 stopped in step 7, so the recording's own eager trace stands in for the
graph replay (its nodes in the region `attnbench::prefill`). Two break
tests follow: the Triton prices dropped must refuse, not price lower.
Findings in `docs/p2_triton_priced_check.md`.

    python3 scripts/p2_triton_priced_check.py <step 7 cell>/recording
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.components import op_sequence as ops   # noqa: E402
from attnbench.components import replay   # noqa: E402

TRITON = {"flat_group_gemm_fuse_reshape_kernel", "softmax_fuse_block_sum_kernel_causal"}


def load_sequence(rec: Path):
    """`sequence.json`, checked against its own sha256 (the recording's
    `tape.pt` is not pulled, so `recorder.load_recording` cannot be used)."""
    record = json.loads((rec / "sequence.json").read_text())
    sha = record.pop("sha256")
    if ops.record_sha256(record) != sha:
        raise SystemExit("sequence.json does not match its own sha256")
    return ops.from_record(record)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("recording", type=Path)
    args = p.parse_args(argv)
    seq = load_sequence(args.recording)
    tri = [i for i, op in enumerate(seq.ops) if set(op.kernels) & TRITON]
    per_layer = defaultdict(list)
    for i in tri:
        per_layer[seq.ops[i].layer].append(i)
    print("triton positions", len(tri), "layers", len(per_layer),
          "per layer", sorted({len(v) for v in per_layer.values()}),
          "all device", all(seq.ops[i].is_device for i in tri),
          "no_device_record", sum(seq.ops[i].no_device_record for i in tri))

    nodes = replay.replay_nodes(args.recording / "trace.json", region=ops.REGION)
    m = replay.match(seq, nodes)
    print("nodes", len(nodes), "matched", len(m.device_s), "unmatched",
          len(m.unmatched_positions), "triton unmatched",
          sorted(set(m.unmatched_positions) & set(tri)))
    pr = replay.price_positions(seq, m, m if m.unmatched_positions else None)
    print("unpriced", len(pr.unpriced), "triton priced", sum(i in pr.device_s for i in tri))
    tot = pr.priced_s
    tri_s = sum(pr.device_s[i] for i in tri)
    lay = sorted(sum(pr.device_s[i] for i in v) for v in per_layer.values())
    print(f"priced_s {tot * 1e3:.3f} ms; triton {tri_s * 1e3:.3f} ms = {tri_s / tot:.4%} of "
          f"priced; per layer median {lay[len(lay) // 2] * 1e6:.1f} us, min {lay[0] * 1e6:.1f}, "
          f"max {lay[-1] * 1e6:.1f}; mismatch_s {pr.mismatch_s} share {pr.mismatch_s / tot:.4f} "
          f"(eager stand-in: no isolated original)")

    host = {i: 0.0 for i, op in enumerate(seq.ops) if op.kind is not ops.SeqKind.SYNC}
    link = {i: 0.0 for i, op in enumerate(seq.ops) if op.kind is ops.SeqKind.COPY}
    oper = ops.to_operations(seq, host=host, device=pr.device_s, link=link)
    dev_total = sum(getattr(o, "device", 0.0) or 0.0 for o in oper)
    print(f"operations {len(oper)}; their device total {dev_total * 1e3:.3f} ms; "
          f"equal to priced {abs(dev_total - tot) < 1e-12}")

    dropped = {i: v for i, v in pr.device_s.items() if i not in tri}
    try:
        ops.to_operations(seq, host=host, device=dropped, link=link)
        print("BREAK 1 SURVIVED: to_operations took the device costs without the Triton positions")
    except ops.SequenceError as e:
        print("break 1 killed:", str(e)[:100])
    m2 = replay.match(seq, [n for n in nodes if n.name not in TRITON])
    p2 = replay.price_positions(seq, m2, m2)
    print("break 2 (Triton nodes absent from both replays):",
          "killed" if set(p2.unpriced) >= set(tri) else "SURVIVED", f"unpriced {len(p2.unpriced)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
