"""Minimal vectorisation of importance_block_mask, FOR MEASUREMENT ONLY.

Not a contribution and not proposed as a replacement. It exists to answer one
counterfactual: if the ~8,000 per-mask scalar tensor assignments were replaced
by a batched top-k and a single scatter, would the A100 prefill reversal
survive? The original implementation's numbers stand as measured.

Semantics reproduced exactly, per attnbench/masks.py:
  - candidates per query block qb: causal -> [1..qb-1]; non-causal ->
    all kv != qb and kv != 0   (kv 0 is the sink, granted free)
  - budget = round((1 - sparsity) * len(candidates))
  - keep the top `budget` candidates by score
  - diagonal and column 0 always active
Jitter is omitted: masks.py documents it as tie-breaking only ("identity_seed
only breaks exact score ties"), at 1e-9 against real-valued scores, so it
cannot change a top-k on non-tied input.

The sentence that used to follow -- "equivalence is asserted below on
non-tied scores, which is the case that occurs" -- was WRONG, and measured
wrong on 2026-09-17. Ties are the case that occurs. score_cache stores fp16
(score_cache.save), and fp16-rounded pooled softmax probabilities are densely
tied: ~62% of cells round to exact zero and ~27% of a row's candidates share
a value with another candidate. Against the real model this builder and the
reference disagree on a handful of cells per layer -- always by choosing
different members of an equal-scoring group, never by ranking them
differently.

That makes this builder valid for the LATENCY counterfactual it exists for
(identical per-row active counts, so identical kernel work) and NOT a
drop-in for accuracy (a different, equal-scoring block is still a different
block). run_vectorised_endtoend._assert_equivalent checks exactly those two
properties and rejects a genuine ranking difference; a negative control
confirms it is not vacuous.

The claim this docstring made was the project's fourth instance of an
equivalence argument validated only on synthetic continuous data and then
applied to quantised real data.
"""
import torch


def importance_block_mask_vectorised(n: int, sparsity: float,
                                      importance_scores: torch.Tensor, *,
                                      causal: bool) -> torch.Tensor:
    ar = torch.arange(n)
    qb = ar.unsqueeze(1)          # (n,1) query block index
    kv = ar.unsqueeze(0)          # (1,n) key block index

    if causal:
        cand = (kv < qb) & (kv != 0)
    else:
        cand = (kv != qb) & (kv != 0)

    n_cand = cand.sum(dim=1)                                  # (n,)
    budgets = torch.round((1.0 - sparsity) * n_cand.double()).long()

    # Rank every candidate within its row, descending by score, in one pass.
    s = importance_scores.masked_fill(~cand, float("-inf"))
    order = s.argsort(dim=1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, ar.expand(n, n).contiguous())

    active = cand & (ranks < budgets.unsqueeze(1))
    active.fill_diagonal_(True)
    active[:, 0] = True
    return active
