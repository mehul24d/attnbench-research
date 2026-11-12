"""Item 1 (adviser's rulings of 13 Nov 2026): on P2's A100 step 7 recording of
xattention_dcal_L2048_b1, are the two Triton kernels device positions that
step 8's pricing and the prediction's operations carry? The recording's own
eager trace stands in for the graph replay (nodes by region attnbench::prefill)."""
import json, sys
from collections import defaultdict
from attnbench.components import op_sequence as ops, replay

REC = sys.argv[1]
TRITON = {"flat_group_gemm_fuse_reshape_kernel", "softmax_fuse_block_sum_kernel_causal"}
rec = json.load(open(f"{REC}/sequence.json"))
sha = rec.pop("sha256")
assert ops.record_sha256(rec) == sha, "sequence.json sha256"
seq = ops.from_record(rec)
tri = [i for i, op in enumerate(seq.ops) if set(op.kernels) & TRITON]
per_layer = defaultdict(list)
for i in tri:
    per_layer[seq.ops[i].layer].append(i)
print("triton positions", len(tri), "layers", len(per_layer),
      "per layer", sorted({len(v) for v in per_layer.values()}),
      "all device", all(seq.ops[i].is_device for i in tri),
      "no_device_record", sum(seq.ops[i].no_device_record for i in tri))

nodes = replay.replay_nodes(f"{REC}/trace.json", region=ops.REGION)
m = replay.match(seq, nodes)
print("nodes", len(nodes), "matched", len(m.device_s), "unmatched", len(m.unmatched_positions),
      "triton unmatched", sorted(set(m.unmatched_positions) & set(tri)))
p = replay.price_positions(seq, m, m if m.unmatched_positions else None)
print("unpriced", len(p.unpriced), "triton priced", sum(i in p.device_s for i in tri))
tot = p.priced_s
tri_s = sum(p.device_s[i] for i in tri)
lay = {l: sum(p.device_s[i] for i in v) for l, v in per_layer.items()}
print(f"priced_s {tot*1e3:.3f} ms; triton {tri_s*1e3:.3f} ms = {tri_s/tot:.4%} of priced; "
      f"per layer median {sorted(lay.values())[len(lay)//2]*1e6:.1f} us, "
      f"min {min(lay.values())*1e6:.1f}, max {max(lay.values())*1e6:.1f}; "
      f"mismatch_s {p.mismatch_s} share {p.mismatch_s/tot:.4f} (eager stand-in: no isolated original)")

host = {i: 0.0 for i, op in enumerate(seq.ops) if op.kind is not ops.SeqKind.SYNC}
link = {i: 0.0 for i, op in enumerate(seq.ops) if op.kind is ops.SeqKind.COPY}
oper = ops.to_operations(seq, host=host, device=p.device_s, link=link)
dev_total = sum(getattr(o, "device", 0.0) or 0.0 for o in oper)
print(f"operations {len(oper)}; their device total {dev_total*1e3:.3f} ms; equal to priced {abs(dev_total-tot) < 1e-12}")

# break tests: the Triton prices dropped must refuse, not price lower
dropped = {i: v for i, v in p.device_s.items() if i not in tri}
try:
    ops.to_operations(seq, host=host, device=dropped, link=link)
    print("BREAK 1 SURVIVED: to_operations took the device costs without the Triton positions")
except ops.SequenceError as e:
    print("break 1 killed:", str(e)[:100])
nodes_wo = [n for n in nodes if n.name not in TRITON]
m2 = replay.match(seq, nodes_wo)
p2 = replay.price_positions(seq, m2, m2)
print("break 2 (Triton nodes absent from both replays):", "killed" if set(p2.unpriced) >= set(tri)
      else "SURVIVED", f"unpriced {len(p2.unpriced)}")
