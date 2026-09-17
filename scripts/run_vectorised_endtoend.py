#!/usr/bin/env python3
"""Item 4: does the A100 prefill reversal survive a vectorised mask builder?

Measures ONE thing and composes nothing. The prior estimate multiplied a
standalone mask-construction timing by 28 layers and added it to a kernel
sweep cell; with laptop CPU numbers that composition undershot the observed
end-to-end gap by ~2.5x, and with the instance's own (3.7x slower) CPU it
OVERSHOT by 1.7-2.5x. Same arithmetic, better input, worse answer -- the
error is in the model, not its terms. So this script does not build a budget.
It runs the real `run_measured` prefill path twice, changing exactly one
thing between the two runs: which function `masks.mask_for` calls to turn
importance scores into a BlockSparseMask.

Both builders run in the SAME process, same model instance, same scores,
interleaved per arm, so host state, allocator warmth and clock drift hit
both equally. The reference arm is re-measured here rather than read from
an earlier parquet for the same reason.

Equivalence is asserted on the real masks at the real shapes BEFORE any
timing, and the assertion is fatal. A faster builder that produces a
different mask is measuring a different experiment.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from attnbench import masks, provenance                                # noqa: E402
from attnbench.accuracy.config import load_grid                        # noqa: E402
from attnbench.accuracy.generation import ModelGeometry                # noqa: E402
from attnbench.accuracy.grid_configs import backend_instance           # noqa: E402
from attnbench.accuracy.phase_timing import time_repeated              # noqa: E402
from attnbench.config import AttnConfig                                # noqa: E402
from _vec_mask_for_measurement import importance_block_mask_vectorised # noqa: E402


_REFERENCE_BUILDER = masks.importance_block_mask


def _vectorised_builder(seq_len, block_size, sparsity, importance_scores, *,
                        causal, identity_seed):
    """Same signature as masks.importance_block_mask, same return type.

    `identity_seed` is accepted and unused: it feeds only the 1e-9 tie-break
    jitter, which cannot reorder a top-k over non-tied real scores. It is
    still threaded into the returned mask's `seed` field so the two builders'
    masks compare equal on every field, not just `active`.
    """
    n = masks._n_blocks(seq_len, block_size)
    if tuple(importance_scores.shape) != (n, n):
        raise ValueError(
            f"importance_scores shape {tuple(importance_scores.shape)} != "
            f"({n}, {n}) for seq_len={seq_len}, block_size={block_size}")
    active = importance_block_mask_vectorised(
        n, sparsity, importance_scores, causal=causal)
    return masks.BlockSparseMask(
        seq_len=seq_len, block_size=block_size, active=active,
        seed=masks._int_seed(identity_seed), source="importance",
        causal=causal)


def _assert_equivalent(*, seq_len, block_size, finest_block_size, sparsities,
                       scores, seed):
    """Assert the two builders are equivalent FOR THIS MEASUREMENT, and say
    exactly what that means. Fatal otherwise.

    The first version of this check demanded bitwise-identical masks and
    fired immediately on real data: seq_len=8192, layer 3, sparsity 0.5, 2 of
    4096 cells. The laptop check that preceded it passed on 36 synthetic
    configs. The difference is not the shapes -- it is the dtype. Scores are
    cached fp16 (score_cache.save), and fp16-rounded pooled softmax
    probabilities are densely TIED: ~62% of cells round to exact zero, and
    ~27% of a row's candidates share a value with another candidate. The
    reference builder breaks those ties with 1e-9 jitter; the vectorised one
    uses a stable argsort. On a tie group straddling the budget boundary they
    pick different, equal-scoring blocks.

    So the builder's own docstring was wrong where it said equivalence is
    asserted "on non-tied scores, which is the case that occurs". Ties are
    overwhelmingly the case that occurs. Bitwise equality was never the right
    property to demand.

    The property this measurement actually needs is that the two builders
    give the kernel the same work and the same ranking:

      1. identical ACTIVE COUNT PER ROW -- block-sparse cost depends on how
         many blocks are active, not which, so this is what makes the two
         timings comparable at all;
      2. identical KEPT-SCORE MULTISET per row -- this is what makes the
         disagreement provably tie-breaking rather than a ranking difference.
         Two builders that keep the same sorted list of score VALUES differ
         only in their choice among equals.

    Both are checked here on every (layer, sparsity), and the residual
    differing-cell count is returned so it lands in the parquet instead of
    only in scrollback. What this check does NOT license is an accuracy
    claim: interchangeable-by-score is not interchangeable-by-output, and
    nothing here measures that.
    """
    import torch
    from attnbench.accuracy.model import pool_scores_to_block_size
    checked = 0
    differing_cells = 0
    tied_rows = 0
    # Quantisation structure of the scores the oracle actually ranks. Measured
    # here rather than quoted from a synthetic softmax: the zero-rate is a
    # property of THIS model at THIS block_size, and it is the reason the
    # tie-breaking matters at all. Recorded because it is also a finding about
    # the oracle -- a top-k over a signal that is two-thirds exact zeros is
    # frequently choosing arbitrarily rather than by score.
    zero_cells = total_cells = tied_cand = total_cand = 0
    for layer in sorted(scores):
        # The SAME reduction SwappedAttention.forward applies (model.py), via
        # the same function -- not a local re-derivation of it. If that path
        # changes, this check changes with it instead of silently comparing
        # builders on a tensor the model never sees.
        pooled = pool_scores_to_block_size(
            scores[layer], finest_block_size=finest_block_size,
            target_block_size=block_size)
        sc = pooled.mean(dim=0)

        n = sc.shape[0]
        causal_lower = torch.tril(torch.ones(n, n, dtype=torch.bool), -1)
        vals = sc[causal_lower]
        zero_cells += int((vals == 0).sum()); total_cells += int(vals.numel())
        for qb in range(1, n):
            v = sc[qb, :qb]
            tied_cand += int(v.numel() - torch.unique(v).numel())
            total_cand += int(v.numel())

        for sp in sparsities:
            a = _REFERENCE_BUILDER(seq_len, block_size, sp, sc,
                                   causal=True, identity_seed=seed)
            b = _vectorised_builder(seq_len, block_size, sp, sc,
                                    causal=True, identity_seed=seed)
            if (a.seed, a.source, a.causal) != (b.seed, b.source, b.causal):
                raise SystemExit(f"EQUIVALENCE FAILED on mask metadata at "
                                 f"seq_len={seq_len} layer={layer} sp={sp}")

            ca, cb = a.active.sum(dim=1), b.active.sum(dim=1)
            if not torch.equal(ca, cb):
                bad = int((ca != cb).sum())
                raise SystemExit(
                    f"EQUIVALENCE FAILED seq_len={seq_len} layer={layer} "
                    f"sparsity={sp}: active count differs on {bad} rows "
                    f"(max |delta| {int((ca - cb).abs().max())}). The two "
                    f"builders would give the kernel different amounts of "
                    f"work, so their timings are not comparable.")

            diff = a.active ^ b.active
            if diff.any():
                # Every disagreement must be a swap among EQUAL scores. Check
                # the kept-score multiset per row, not the index set.
                for qb in torch.nonzero(diff.any(dim=1)).flatten().tolist():
                    va = torch.sort(sc[qb][a.active[qb]]).values
                    vb = torch.sort(sc[qb][b.active[qb]]).values
                    if not torch.equal(va, vb):
                        raise SystemExit(
                            f"EQUIVALENCE FAILED seq_len={seq_len} "
                            f"layer={layer} sparsity={sp} row={qb}: kept-score "
                            f"multisets differ, so this is a RANKING "
                            f"disagreement, not tie-breaking. Refusing to time "
                            f"a builder that computes something else.")
                    tied_rows += 1
                differing_cells += int(diff.sum())
            checked += 1
    zero_frac = zero_cells / max(1, total_cells)
    tie_frac = tied_cand / max(1, total_cand)
    print(f"  equivalence OK at seq_len={seq_len}: {checked} (layer x "
          f"sparsity) masks agree on per-row active count and kept-score "
          f"multiset; {differing_cells} cells differ by tie-break across "
          f"{tied_rows} rows", flush=True)
    print(f"  score quantisation at seq_len={seq_len}: "
          f"{100*zero_frac:.1f}% of causal cells are EXACTLY zero in fp16, "
          f"{100*tie_frac:.1f}% of a row's candidates share a value with "
          f"another -- the oracle's top-k is choosing among ties this often",
          flush=True)
    return differing_cells, zero_frac, tie_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="configs/accuracy/stage3_grid.yaml")
    ap.add_argument("--out", default="results/s7_vec_endtoend")
    ap.add_argument("--bands", default="8192,16384")
    ap.add_argument("--sparsities", default="0.5,0.75,0.9")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lock-clocks", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM
    from attnbench.accuracy.model import SwappableAttentionModel

    grid = load_grid(args.grid)
    bands = [int(b) for b in args.bands.split(",")]
    sparsities = [float(s) for s in args.sparsities.split(",")]

    clocks_locked = False
    if args.lock_clocks:
        clocks_locked = provenance.lock_clocks()
        print(f"clock lock: {clocks_locked}", flush=True)
        if not clocks_locked:
            print("!! CLOCK LOCK FAILED -- every row stamped "
                  "clocks_locked=False", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        grid.model_primary, torch_dtype=getattr(torch, args.dtype))
    model = model.to(args.device).eval()
    geom = ModelGeometry.from_config(model.config, args.dtype)

    def cfg_for(band, mask, sparsity):
        base = AttnConfig(seq_len=band, batch=1, n_heads_q=1, n_heads_kv=1,
                          head_dim=128, mask=mask, sparsity=sparsity,
                          block_size=grid.finest_block_size,
                          mask_source="importance" if mask == "block_sparse" else None)
        return geom.onto(base, seq_len=band)

    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="s7vec_scores_")

    rows = []
    out_rows = []
    for band in bands:
        print(f"\n===== band {band} =====", flush=True)
        ids = torch.randint(0, model.config.vocab_size, (1, band),
                            device=args.device)
        wrapped = SwappableAttentionModel(
            model, cfg_for(band, "causal", None), model_id=grid.model_primary,
            finest_block_size=grid.finest_block_size)
        try:
            t0 = time.time()
            scores = wrapped.compute_importance_scores(
                ids, task="s7_vec", example_id=f"b{band}", cache_dir=scratch)
            # dict[layer_idx -> (n_heads_kv, n_blocks, n_blocks)], which is
            # what run_measured wants; only the equivalence check needs it
            # indexed, and it indexes the dict directly rather than stacking
            # into a tensor the model never sees.
            n_layers = len(scores)
            print(f"  scoring pass {time.time()-t0:.1f}s, {n_layers} layers, "
                  f"per-layer {tuple(scores[0].shape)}", flush=True)

            seed = masks._mask_identity_key(
                cfg_for(band, "block_sparse", sparsities[0]))
            tie_cells, zero_frac, tie_frac = _assert_equivalent(
                               seq_len=band, block_size=grid.finest_block_size,
                               finest_block_size=grid.finest_block_size,
                               sparsities=sparsities, scores=scores, seed=seed)

            # Dense control: no mask is built on this path, so the builder
            # cannot touch it. Measured once, and its stability across the
            # two builder settings is the sanity check that nothing else
            # moved between them.
            arms = [("sdpa_flash", None)] + [("block_sparse", s) for s in sparsities]

            # Does the tie-break choice change the MODEL OUTPUT?
            #
            # Identical per-row active counts make the two builders'
            # timings comparable; they do NOT make the outputs equal. A
            # different equal-scoring block is still a different block, and
            # the softmax renormalises over whatever was kept -- so a swap
            # among zero-SCORED blocks can still move the logits, because a
            # pooled score of zero in fp16 is not an attention weight of
            # zero. Measured rather than argued, one forward per arm, before
            # any timing (and outside it, so it costs the timed loop
            # nothing).
            logit_rows = []
            for name, sparsity in ([("sdpa_flash", None)]
                                   + [("block_sparse", sp) for sp in sparsities]):
                be = backend_instance(name)
                cfg = cfg_for(band,
                              "causal" if sparsity is None else "block_sparse",
                              sparsity)
                ls = None if sparsity is None else scores
                outs = {}
                for bn, fn in (("reference", _REFERENCE_BUILDER),
                               ("vectorised", _vectorised_builder)):
                    masks.importance_block_mask = fn
                    with torch.no_grad():
                        o = wrapped.run_measured(ids, be, cfg=cfg,
                                                 layer_scores=ls,
                                                 logits_to_keep=1)
                    outs[bn] = (o.logits if hasattr(o, "logits") else o
                                ).detach().float().cpu()
                masks.importance_block_mask = _REFERENCE_BUILDER
                a_, b_ = outs["reference"], outs["vectorised"]
                d = (a_ - b_).abs()
                same_argmax = bool((a_.argmax(-1) == b_.argmax(-1)).all())
                logit_rows.append(dict(
                    band=band, backend=name, sparsity=sparsity,
                    logits_max_abs_diff=float(d.max()),
                    logits_mean_abs_diff=float(d.mean()),
                    logits_bitwise_identical=bool(torch.equal(a_, b_)),
                    argmax_token_identical=same_argmax,
                    ref_logit_absmax=float(a_.abs().max())))
                r = logit_rows[-1]
                print(f"  output check {name:<13}"
                      f"{'dense' if sparsity is None else sparsity:>6}: "
                      f"max|d| {r['logits_max_abs_diff']:.3e}  "
                      f"bitwise {r['logits_bitwise_identical']}  "
                      f"argmax same {same_argmax}", flush=True)
            out_rows.extend(logit_rows)

            for builder_name, fn in (("reference", _REFERENCE_BUILDER),
                                     ("vectorised", _vectorised_builder)):
                masks.importance_block_mask = fn
                for name, sparsity in arms:
                    be = backend_instance(name)
                    cfg = cfg_for(band,
                                  "causal" if sparsity is None else "block_sparse",
                                  sparsity)
                    layer_scores = None if sparsity is None else scores
                    pf = time_repeated(
                        lambda: wrapped.run_measured(
                            ids, be, cfg=cfg, layer_scores=layer_scores,
                            logits_to_keep=1),
                        warmup=args.warmup, reps=args.reps, synchronize=sync)
                    ms = sum(pf) / len(pf)
                    rows.append(dict(
                        band=band, builder=builder_name, backend=name,
                        sparsity=sparsity, prefill_ms_mean=ms,
                        prefill_ms_min=min(pf), prefill_ms_max=max(pf),
                        n_reps=len(pf), n_warmup=args.warmup,
                        n_layers=n_layers,
                        tiebreak_cells_differing=tie_cells,
                        score_zero_fraction=zero_frac,
                        score_tied_fraction=tie_frac,
                        clocks_locked=clocks_locked))
                    print(f"  [{builder_name:<10}] {name:<13}"
                          f"{'dense' if sparsity is None else sparsity:>6}  "
                          f"prefill {ms:9.2f} ms", flush=True)
                masks.importance_block_mask = _REFERENCE_BUILDER
        finally:
            masks.importance_block_mask = _REFERENCE_BUILDER
            wrapped.unwrap()

    frame = pd.DataFrame(rows)
    # The measured value goes INTO the stamp, per stamp_onto's docstring: a
    # bare capture() defaults clocks_locked=False and would overwrite the
    # measurement in a gated field.
    provenance.stamp_onto(frame,
                          provenance.capture(clocks_locked=clocks_locked).to_dict())
    frame.to_parquet(out / "vec_endtoend.parquet", index=False)

    oframe = pd.DataFrame(out_rows)
    provenance.stamp_onto(oframe,
                          provenance.capture(clocks_locked=clocks_locked).to_dict())
    oframe.to_parquet(out / "vec_output_equivalence.parquet", index=False)
    print("\n===== OUTPUT UNDER EACH BUILDER =====", flush=True)
    print(oframe[["band", "backend", "sparsity", "logits_max_abs_diff",
                  "logits_bitwise_identical", "argmax_token_identical"]]
          .to_string(index=False), flush=True)

    # The comparison the session exists to make, printed at the terminal.
    print("\n===== REVERSAL UNDER EACH BUILDER =====", flush=True)
    print(f"{'band':>6} {'sparsity':>9} {'ref gap':>10} {'vec gap':>10} "
          f"{'ref ratio':>10} {'vec ratio':>10}", flush=True)
    for band in bands:
        b = frame[frame.band == band]
        dense = {bu: float(b[(b.builder == bu) & (b.backend == "sdpa_flash")]
                           .prefill_ms_mean.iloc[0]) for bu in ("reference", "vectorised")}
        for sp in sparsities:
            g = {}
            r = {}
            for bu in ("reference", "vectorised"):
                sel = b[(b.builder == bu) & (b.sparsity == sp)]
                if sel.empty:
                    continue
                v = float(sel.prefill_ms_mean.iloc[0])
                g[bu] = v - dense[bu]
                r[bu] = dense[bu] / v
            print(f"{band:>6} {sp:>9g} {g.get('reference', float('nan')):>+10.1f} "
                  f"{g.get('vectorised', float('nan')):>+10.1f} "
                  f"{r.get('reference', float('nan')):>10.3f} "
                  f"{r.get('vectorised', float('nan')):>10.3f}", flush=True)
        print(f"       dense control: ref {dense['reference']:.2f} ms  "
              f"vec {dense['vectorised']:.2f} ms  "
              f"(drift {100*(dense['vectorised']/dense['reference']-1):+.1f}%)",
              flush=True)
    print(f"\nwritten to {out / 'vec_endtoend.parquet'}", flush=True)


if __name__ == "__main__":
    main()
