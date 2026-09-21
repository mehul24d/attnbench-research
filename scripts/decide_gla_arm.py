#!/usr/bin/env python
"""Run the pre-registered GLA accuracy-arm experiment and print the verdict.

    python scripts/decide_gla_arm.py            # ~5 minutes on an L4

Runs 100 niah_single examples at seq_len 2048 twice on the same prompts, in
one process: once through the dense arm (the positive control) and once
through GLA with `gate_source="ungated"` -- g = 0, decay factor 1, nothing
ever forgotten. Then applies accuracy/gla_arm.evaluate.

The thresholds are fixed in docs/gla_arm_decision.md and were written before
this was ever run. This script does not interpret; it prints what the rule
says. Run it at the HEAD of the next session, before the segment's bands, so
its answer decides whether the segment includes a GLA arm at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from attnbench.accuracy import stopping                              # noqa: E402
from attnbench.accuracy.config import load_grid                      # noqa: E402
from attnbench.accuracy.generation import (                          # noqa: E402
    ModelGeometry, StopTokens, generate_one)
from attnbench.accuracy.gla_arm import evaluate                      # noqa: E402
from attnbench.accuracy.grid_configs import (                        # noqa: E402
    backend_instance, build_examples_by_task_length, gate_source_of)
from attnbench.accuracy.model import SwappableAttentionModel          # noqa: E402
from attnbench.accuracy import ruler                                 # noqa: E402
from attnbench.backends.linear import GatedLinearAttention            # noqa: E402
from attnbench.config import AttnConfig                              # noqa: E402
from attnbench import provenance                                     # noqa: E402

TASK = "niah_single"          # its answer cannot be emitted by luck (~1e-7)
SEQ_LEN = 2048                # the easiest band; if it fails here it fails everywhere


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    # s1b, not s1: the verdict this writes is the GLA arm decision, and the
    # one that exists is at results/stage3_s1b/gla_arm_verdict.json -- the
    # session ran it there and the default here pointed one directory over.
    # A default output path that nobody reads is harmless until someone runs
    # the script with it and then looks for the file where the last one is.
    ap.add_argument("--out", default="results/stage3_s1b/gla_arm_verdict.json",
                    help="where the verdict is written. The decision "
                         "changes what the remaining segments measure, "
                         "so it must outlive the instance that made it.")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    grid = load_grid(args.grid)
    tok = AutoTokenizer.from_pretrained(grid.model_primary)
    examples = build_examples_by_task_length(
        grid, seed=0, tasks=(TASK,), seq_lens={SEQ_LEN: args.n},
        count_tokens=lambda t: len(tok(t).input_ids))[(TASK, SEQ_LEN)]

    model = AutoModelForCausalLM.from_pretrained(
        grid.model_primary, torch_dtype=getattr(torch, args.dtype))
    model = model.to(args.device).eval()
    geom = ModelGeometry.from_config(model.config, args.dtype)
    template = geom.onto(AttnConfig(seq_len=SEQ_LEN, batch=1, n_heads_q=1,
                                    n_heads_kv=1, head_dim=128, mask="causal"),
                         seq_len=SEQ_LEN)
    wrapped = SwappableAttentionModel(model, template, model_id=grid.model_primary,
                                       finest_block_size=grid.finest_block_size)
    stop = StopTokens.from_tokenizer(tok, getattr(model, "generation_config", None))

    arms = {
        "dense (control)": backend_instance(grid.dense_backend),
        "gla (ungated)": GatedLinearAttention(gate_source="ungated"),
    }
    # Read back off the instance rather than restating the literal above.
    # The gate is the one fact that makes this measurement interpretable, and
    # a constructor argument that silently failed to take would leave the
    # verdict file asserting a configuration that did not run.
    gate = gate_source_of(arms["gla (ungated)"])
    assert gate == "ungated", gate
    out: dict[str, tuple[list[str], list[float]]] = {}
    try:
        for label, backend in arms.items():
            preds, scores = [], []
            for i, ex in enumerate(examples, 1):
                g = generate_one(wrapped, tok, cfg=template, backend=backend,
                                 example=ex, geometry=geom, stop_tokens=stop,
                                 score_cache_dir=grid.score_cache_dir,
                                 device=args.device)
                preds.append(g.text)
                scores.append(ruler.score(TASK, g.text, ex.answer))
                if i % 25 == 0:
                    print(f"  {label}: {i}/{len(examples)}", flush=True)
            out[label] = (preds, scores)
            print(f"{label:16} distinct={len(set(preds))}/{len(preds)} "
                  f"mean_score={sum(scores)/len(scores):.1f}", flush=True)
    finally:
        wrapped.unwrap()

    v = evaluate(*out["gla (ungated)"],
                 control_predictions=out["dense (control)"][0],
                 control_scores=out["dense (control)"][1])

    print()
    print("=" * 70)
    print(f"VERDICT: {v.verdict.upper()}"
          + (f"  (failed gate {v.failed_gate})" if v.failed_gate else ""))
    print(f"  distinct     {v.distinct_fraction:.2f}   (need >= 0.90)")
    print(f"  answer-shape {v.format_valid_fraction:.2f}   (need >= 0.50)")
    print(f"  mean score   {v.mean_score:.1f}    (need >= 20.0)")
    print()
    print(f"  {v.reason}")
    print("=" * 70)
    print("Rule pre-registered in docs/gla_arm_decision.md before this ran.")

    # The verdict decides what the remaining Stage 3 segments measure, so it
    # cannot live only in terminal scrollback on a machine that is about to
    # be deleted. Written next to the results, with the gate that produced it
    # and the commit that ran it.
    record = {
        "verdict": v.verdict, "failed_gate": v.failed_gate, "reason": v.reason,
        "gate_source": gate, "task": TASK, "seq_len": SEQ_LEN, "n": len(examples),
        "distinct_fraction": v.distinct_fraction,
        "format_valid_fraction": v.format_valid_fraction,
        "mean_score": v.mean_score,
        "control_backend": grid.dense_backend,
        "control_mean_score": sum(out["dense (control)"][1]) / len(examples),
        "rule": "docs/gla_arm_decision.md (pre-registered 2026-09-07)",
        **provenance.capture().to_dict(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"written to {out_path}")
    # 0 keeps the arm, 1 drops it, 2 means the control failed and there is no
    # verdict -- so a wrapper script can branch without parsing prose.
    return {"keep": 0, "drop": 1, "no_verdict": 2}[v.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
