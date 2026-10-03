#!/usr/bin/env python3
"""Calibrate XAttention's per-(layer, head) thresholds for this study's model.

XAttention's own RULER evaluation does not use a scalar threshold. Its
`scripts/run_ruler.sh` passes none, so `FastPrefillConfig` loads
`llama_fuse_8`: a 32x32 table of per-(layer, head) thresholds, profiled for
Llama-3.1-8B by `xattn/threshold/profile_threshold/profile_threshold.py`.
No such table exists for Qwen2.5. This script makes one with that same
profiler, so the calibrated phase (docs/t4_xattention_calibrated.md) runs
the method as its authors ran RULER, not a scalar approximation of it.

The profiler is the authors' code, imported from the pinned checkout and
called per layer through `attnbench.backends.xattention.ThresholdProfiler`.
Per text, layer and head it finds the threshold on XAttention's own
estimate that selects at least the fewest blocks covering 90% of the exact
attention mass. The table is the maximum over texts, as the official
script's `final_threshold` is.

Two sources, one per run, each its own arm:

  authors        the method's own profiling set (`text.json` at the pinned
                 commit, 156 multi-document QA prompts), with the Llama-3
                 chat markers stripped because they mean nothing to Qwen.
                 Disjoint from RULER. This calibration carries the claim.
  ruler_heldout  RULER examples of the pilot's tasks at its bands, from
                 another seed, checked to share no context with any test
                 example. In-distribution and at the test lengths.
                 Descriptive only.

Writes `<out>/thresholds.json` (the table, its sha256 and its provenance,
committed to configs/xattn_thresholds/ before any test row uses it) and
`<out>/per_text.parquet` (every text's table, for the record).

    python scripts/calibrate_xattn_thresholds.py --source authors \\
        --out results/xattn_calibration_authors_<date>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import numerics, provenance                              # noqa: E402
from attnbench.accuracy import t4_pilot                                 # noqa: E402

CHAT_MARKER = re.compile(r"<\|[^|>]*\|>")


def strip_chat_markers(text: str) -> str:
    """Remove Llama-3 special-token markers (`<|begin_of_text|>`,
    `<|start_header_id|>` ...). To Qwen's tokenizer they are not special,
    so left in they would be profiled as literal text."""
    return CHAT_MARKER.sub("", text)


def split_identities(examples_by_key: dict) -> list[dict]:
    """The four identities sec. 5 compares, per example: id, context sha256,
    QA question index (the number after the band in the id, which is the
    index `_render` asks), and NIAH answer values (task-qualified)."""
    out = []
    for (task, _band), exs in examples_by_key.items():
        for ex in exs:
            row = {"example_id": ex.example_id,
                   "context_sha256": hashlib.sha256(ex.context.encode()).hexdigest()}
            m = re.search(r"_(\d+)$", ex.example_id)
            if task.startswith("qa_") and m:
                # The firewall keys on the tuple's first element: the
                # task-qualified question index (qa_1 and qa_2 index different
                # datasets).
                row["qa_question"] = ((task, int(m.group(1))),)
            if task.startswith("niah"):
                row["niah_needles"] = [(task, a) for a in ex.answer]
            out.append(row)
    return out


def assert_disjoint(calibration: dict, test: dict) -> None:
    """Refuse a calibration set that shares any example with the test by any
    of the four identities (estimator-frontier pre-registration sec. 5; T4
    amendment A1, 2026-10-03). Until then this compared whole contexts only,
    and a seed-1 qa_1 example asking test question 3 with other distractors
    passed it."""
    from attnbench.analysis.frontier_prereg import SplitLeak, check_splits_disjoint
    try:
        check_splits_disjoint({"calibration": split_identities(calibration),
                               "test": split_identities(test)})
    except SplitLeak as e:
        raise SystemExit(f"REFUSING: the calibration set leaks into the test: {e}")


def excluded_over_length(lengths: list[int], limit: int) -> list[int]:
    """Indices of texts longer than the model's max_position_embeddings (A2).
    They are excluded, never truncated."""
    return [i for i, n in enumerate(lengths) if n > limit]


def descriptive_records(per_text, used: list[int]) -> dict:
    """A3's descriptive records, never used to build the table: the p90
    table, how many entries have max - p90 above the gap, and which text
    sets each entry's max."""
    import torch
    stack = torch.stack([per_text[i] for i in used])
    mx, arg = stack.max(dim=0)
    p90 = torch.quantile(stack.float(), 0.9, dim=0)
    gap = t4_pilot.XATTN_CALIBRATION_DESCRIPTIVE_GAP
    return {"p90": [[round(float(x), 8) for x in row] for row in p90.tolist()],
            "n_max_minus_p90_above": int(((mx - p90) > gap).sum()),
            "gap": gap,
            "argmax_text": [[used[int(i)] for i in row] for row in arg.tolist()]}


def g12_or_refuse(texts: list[str]) -> dict:
    """Gate G12 (estimator-frontier pre-registration sec. 6): the calibration
    texts are held to the split firewall before anything is profiled on
    them. A leak, or QA data the gate cannot read, stops the run."""
    from attnbench.accuracy import frontier_splits
    from attnbench.analysis.frontier_prereg import SplitLeak
    try:
        return frontier_splits.g12(texts)
    except SplitLeak as e:
        raise SystemExit(f"STOP (G12): {e}")
    except Exception as e:
        raise SystemExit(f"STOP (G12 could not run, so it has not passed): "
                         f"{type(e).__name__}: {e}")


def authors_selection(texts: list[str], raw_sha256: str, *, full: bool) -> tuple[list[int], dict]:
    """Which of the authors' texts a table is profiled on, and its G12 record
    (T4 amendment A8; estimator-frontier pre-registration sec. 5).

    `authors` (full=False) leaves out the QA texts, and G12 must then pass on
    the rest. `authors_full` keeps every text; G12 is run and its failure is
    written into the table, which is descriptive.

    Both refuse a text.json other than the one the exclusion and the needle
    comparison were worked out on."""
    from attnbench.accuracy import frontier_splits as fs
    if raw_sha256 != fs.TEXT_JSON_SHA256 or len(texts) != fs.TEXT_JSON_N_TEXTS:
        raise SystemExit("STOP (G12): this is not the text.json the gate was worked out on "
                         f"(sha256 {raw_sha256[:12]}, {len(texts)} texts).")
    qa = fs.qa_text_indices(texts)
    if len(qa) != fs.TEXT_JSON_N_QA_TEXTS:
        raise SystemExit(f"STOP (G12): {len(qa)} QA texts found, not {fs.TEXT_JSON_N_QA_TEXTS}.")
    if full:
        try:
            hits = fs.g12_hits(texts)
        except Exception as e:
            raise SystemExit(f"STOP (G12 could not run): {type(e).__name__}: {e}")
        by = {}
        for h in hits:
            c = by.setdefault(h["split"], {"questions": 0, "gold_documents": 0})
            c["questions"] += bool(h["question_in_texts"])
            c["gold_documents"] += bool(h["gold_document_in_texts"])
        return list(range(len(texts))), {"gate": "G12", "passed": False, "role": "descriptive",
                                         "held_out_material": by, "qa_texts_excluded": []}
    keep = [k for k in range(len(texts)) if k not in set(qa)]
    record = g12_or_refuse([texts[k] for k in keep])
    record.update(qa_texts_excluded=qa, needles=fs.NEEDLE_COMPARISON)
    return keep, record


def authors_texts(*, full: bool = False) -> tuple[list[str], str, list[int], dict]:
    """(every text, the file's sha256, the indices to profile, the G12
    record). Indices stay those of the file, so the table can name what it
    left out."""
    import xattn
    path = (Path(xattn.__file__).resolve().parent / "threshold" / "profile_threshold"
            / "text.json")
    raw = path.read_bytes()
    texts = [strip_chat_markers(t) for t in json.loads(raw)]
    sha = hashlib.sha256(raw).hexdigest()
    keep, record = authors_selection(texts, sha, full=full)
    return texts, sha, keep, record


def ruler_texts(grid, count_tokens) -> tuple[list[str], str]:
    from attnbench.accuracy.grid_configs import build_examples_by_task_length
    bands = {b: t4_pilot.XATTN_CALIBRATION_RULER_N for b in t4_pilot.PILOT_BANDS}
    cal = build_examples_by_task_length(
        grid, seed=t4_pilot.XATTN_CALIBRATION_SEED, count_tokens=count_tokens,
        tasks=t4_pilot.SPARSE_PILOT_TASKS, seq_lens=bands,
        index_offset=t4_pilot.XATTN_CALIBRATION_INDEX_OFFSET)
    test = build_examples_by_task_length(
        grid, seed=0, count_tokens=count_tokens, tasks=t4_pilot.SPARSE_PILOT_TASKS,
        seq_lens={b: max(t4_pilot.SPARSE_PILOT_N.values()) for b in t4_pilot.PILOT_BANDS})
    assert_disjoint(cal, test)
    texts = [ex.context for key in sorted(cal) for ex in cal[key]]
    digest = hashlib.sha256("\x00".join(texts).encode()).hexdigest()
    return texts, digest


def main() -> int:
    numerics.enforce_fp32_matmul()
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(t4_pilot.XATTN_CALIBRATIONS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    prov = provenance.capture()
    if prov.git_dirty:
        raise SystemExit("REFUSING: uncommitted changes. The table is committed "
                         "and cited by commit; it must come from one.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from attnbench.accuracy.config import load_grid
    from attnbench.accuracy.generation import ModelGeometry
    from attnbench.accuracy.model import SwappableAttentionModel
    from attnbench.backends.xattention import (XATTN_COMMIT, ThresholdProfiler,
                                               ThresholdTable, installed_checkout)
    from attnbench.config import AttnConfig

    # A4: the profiler's use_triton=True takes the Triton path only where the
    # device name contains "100"; the table is made on the A100.
    from attnbench.backends.xattention import triton_for_device
    card = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    if t4_pilot.XATTN_CALIBRATION_CARD not in card or not triton_for_device(card):
        raise SystemExit(f"REFUSING: calibration runs on the {t4_pilot.XATTN_CALIBRATION_CARD} "
                         f"(T4 amendment A4); this device is {card!r}.")

    head, dirty = installed_checkout()
    if (head, dirty) != (XATTN_COMMIT, False):
        raise SystemExit(f"REFUSING: x-attention is {head} (dirty={dirty}), not "
                         f"{XATTN_COMMIT}. Run scripts/install_xattention.sh.")

    grid = load_grid(args.grid)
    model_id = grid.model_primary
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if args.source in ("authors", "authors_full"):
        texts, source_sha, keep, g12_record = authors_texts(full=args.source == "authors_full")
    else:
        texts, source_sha = ruler_texts(
            grid, lambda text: len(tokenizer(text).input_ids))
        keep, g12_record = list(range(len(texts))), None

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, args.dtype)).to(args.device).eval()
    geometry = ModelGeometry.from_config(model.config, args.dtype)
    template = geometry.onto(AttnConfig(seq_len=1024, batch=1, n_heads_q=1, n_heads_kv=1,
                                        head_dim=128, mask="causal"), seq_len=1024)
    wrapped = SwappableAttentionModel(model, template, model_id=model_id,
                                      score_source=t4_pilot.XATTN_SCORE_SOURCE)
    profiler = ThresholdProfiler(n_layers=wrapped.n_layers, stride=t4_pilot.XATTN_STRIDE)
    # A2: texts over the model's position limit, read from its config at run
    # time, are excluded, never truncated, and recorded in the table.
    limit = int(model.config.max_position_embeddings)
    all_lengths = [len(tokenizer(t).input_ids) for t in texts]
    excluded = excluded_over_length(all_lengths, limit)
    used = [i for i in keep if i not in set(excluded)]
    print(f"{len(used)} of {len(texts)} texts within {limit} positions; "
          f"excluded {excluded}", flush=True)
    lengths = []
    try:
        for n, i in enumerate(used):
            text = texts[i]
            ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
            lengths.append(int(ids.shape[-1]))
            cfg = geometry.onto(template, seq_len=lengths[-1])
            profiler.start_text()
            wrapped.run_measured(ids, profiler, cfg=cfg, logits_to_keep=1)
            profiler.end_text()
            print(f"  text {n + 1}/{len(used)} (index {i}): {lengths[-1]} tokens", flush=True)
    finally:
        wrapped.unwrap()

    table = profiler.table()
    values = [[round(float(x), 8) for x in row] for row in table.tolist()]
    doc = dict(
        name=args.source,
        sha256=ThresholdTable.digest(values),
        model=model_id, stride=t4_pilot.XATTN_STRIDE, block_size=128,
        exact_mass_coverage=0.9,                 # fixed inside the official profiler
        aggregation=t4_pilot.XATTN_CALIBRATION_STATISTIC,
        procedure="calibrated by the authors' released profiler "
                  "(profile_threshold.py), their substitute for the paper's "
                  "unreleased DP method (issue #13); T4 amendment A5",
        xattn_path="triton",
        card=card,
        max_position_embeddings=limit,
        n_texts=len(used),
        n_texts_offered=len(texts),
        excluded=[dict(index=i, tokens=all_lengths[i]) for i in excluded],
        g12=g12_record,
        descriptive=descriptive_records(
            {i: t for i, t in zip(used, profiler.per_text)}, used),
        tokens=dict(min=min(lengths), median=statistics.median(lengths), max=max(lengths)),
        source_sha256=source_sha,
        source=("x-attention xattn/threshold/profile_threshold/text.json, chat markers stripped, "
                "its 24 QA texts left out (T4 amendment A8)" if args.source == "authors" else
                "x-attention xattn/threshold/profile_threshold/text.json, chat markers stripped, "
                "every text (descriptive; T4 amendment A8)" if args.source == "authors_full" else
                f"RULER {list(t4_pilot.SPARSE_PILOT_TASKS)} at {list(t4_pilot.PILOT_BANDS)}, "
                f"seed {t4_pilot.XATTN_CALIBRATION_SEED}, "
                f"{t4_pilot.XATTN_CALIBRATION_RULER_N} per (task, band), "
                f"disjoint from the seed-0 test examples"),
        xattn_commit=XATTN_COMMIT,
        flash_attn_stubbed=profiler.flash_attn_stubbed,
        git_commit=prov.git_commit, git_dirty=prov.git_dirty,
        thresholds=values,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "thresholds.json").write_text(json.dumps(doc, indent=1) + "\n")
    per_text = torch.stack(profiler.per_text)
    pd.DataFrame([dict(text=t, layer=l, head=h, threshold=float(per_text[t, l, h]))
                  for t in range(per_text.shape[0]) for l in range(per_text.shape[1])
                  for h in range(per_text.shape[2])]).to_parquet(out / "per_text.parquet")

    flat = [x for row in values for x in row]
    print(f"\n{args.source}: {len(texts)} texts, {doc['tokens']} tokens; "
          f"table {len(values)}x{len(values[0])}, sha256 {doc['sha256'][:12]}")
    print(f"thresholds: min {min(flat):.3f}  median {statistics.median(flat):.3f}  "
          f"max {max(flat):.3f}  zero {sum(x == 0 for x in flat)}  "
          f">=0.95 {sum(x >= 0.95 for x in flat)} of {len(flat)}")
    for layer, row in enumerate(values):
        print(f"  layer {layer:2d}: " + " ".join(f"{x:.2f}" for x in row))
    print(f"\nwritten to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
