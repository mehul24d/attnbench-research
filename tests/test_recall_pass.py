"""The recall pass (estimator-frontier pre-registration sec. 2.3), on CPU,
against a token-by-token computation of the same quantities."""
from __future__ import annotations

import math

import pytest
import torch

from attnbench import masks
from attnbench.accuracy import model as acc_model
from attnbench.accuracy import recall as R
from attnbench.analysis import frontier_prereg as fp
from attnbench.analysis import frontier_recall as fr


def _qk(h=2, s=400, d=8, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(h, s, d, generator=g).to(dtype),
            torch.randn(h, s, d, generator=g).to(dtype))


def _slow_mass(q, k, block):
    """M_b from the definition, one query token at a time."""
    h, s, d = q.shape
    n = -(-s // block)
    out = torch.zeros(h, n, n, dtype=torch.float64)
    for t in range(s):
        logits = (q[:, t:t + 1].double() @ k[:, :t + 1].double().transpose(-2, -1))[:, 0]
        p = torch.softmax(logits / math.sqrt(d), dim=-1)
        for u in range(t + 1):
            out[:, t // block, u // block] += p[:, u]
    counts = torch.tensor([min(block, s - i * block) for i in range(n)], dtype=torch.float64)
    return out / counts.view(1, -1, 1)


@pytest.mark.parametrize("s", [128, 400, 250])
def test_the_exact_mass_is_the_definition_and_each_row_sums_to_one(s):
    q, k = _qk(s=s)
    sums, counts = R.exact_block_sums(q, k, chunk_tokens=64)
    mass = R.block_mass(sums, counts)
    assert torch.allclose(mass.double(), _slow_mass(q, k, 16), atol=1e-5)
    assert torch.allclose(mass.sum(-1), torch.ones_like(mass.sum(-1)), atol=1e-5)
    assert not bool(mass.triu(1).any())                       # nothing from the future


@pytest.mark.parametrize("b", [32, 64, 128])
def test_coarser_blocks_are_sums_of_the_finest(b):
    """Including a prompt that ends inside a block."""
    q, k = _qk(s=400)
    fine = R.exact_block_sums(q, k, chunk_tokens=64)
    mass = R.block_mass(*R.coarsen(*fine, b // 16))
    assert torch.allclose(mass.double(), _slow_mass(q, k, b), atol=1e-5)


def test_the_chunk_size_does_not_change_the_result_and_bf16_input_is_scored_in_fp32():
    q, k = _qk(s=400)
    a = R.exact_block_sums(q, k, chunk_tokens=32)[0]
    b = R.exact_block_sums(q, k, chunk_tokens=4096)[0]
    assert torch.allclose(a, b, atol=1e-5)
    q16, k16 = _qk(s=400, dtype=torch.bfloat16)
    s16, _ = R.exact_block_sums(q16, k16)
    assert s16.dtype == torch.float32
    assert torch.allclose(s16, R.exact_block_sums(q16.float(), k16.float())[0], atol=1e-5)


def _mass(h=2, n=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(h, n, n, generator=g).tril()
    return m / m.sum(-1, keepdim=True)


def test_recall_against_a_row_by_row_computation():
    mass = _mass()
    active = masks.importance_block_mask_per_head(40 * 128, 128, 0.75, torch.rand(2, 40, 40),
                                                  causal=True, identity_seed="r").active
    raw, norm = R.recall(mass, active)
    for h in range(2):
        for i in range(40):
            r = float(mass[h, i][active[h, i]].sum())
            free = {0, i}
            n_cand = int(active[h, i].sum()) - len(free)
            cand = sorted((float(mass[h, i, j]) for j in range(1, i)), reverse=True)
            star = sum(float(mass[h, i, j]) for j in free) + sum(cand[:n_cand])
            assert float(raw[h, i]) == pytest.approx(r, abs=1e-6)
            assert float(norm[h, i]) == pytest.approx(r / star, abs=1e-5)
    assert float(norm.max()) <= 1 + 1e-6


def test_the_oracle_has_normalised_recall_one_and_a_worse_ranking_has_less():
    mass = _mass()
    best = masks.importance_block_mask_per_head(40 * 128, 128, 0.75, mass, causal=True,
                                                identity_seed="r").active
    worst = masks.importance_block_mask_per_head(40 * 128, 128, 0.75, -mass, causal=True,
                                                 identity_seed="r").active
    assert torch.allclose(R.recall(mass, best)[1], torch.ones(2, 40), atol=1e-6)
    assert float(R.recall(mass, worst)[1][:, 10:].max()) < 1.0


def test_a_mask_without_its_free_blocks_or_with_a_future_block_is_refused():
    mass = _mass()
    ok = torch.ones(2, 40, 40, dtype=torch.bool).tril()
    no_sink = ok.clone()
    no_sink[0, 5, 0] = False
    with pytest.raises(ValueError, match="free block"):
        R.recall(mass, no_sink)
    future = ok.clone()
    future[0, 5, 9] = True
    with pytest.raises(ValueError, match="above the diagonal"):
        R.recall(mass, future)
    with pytest.raises(ValueError):
        R.recall(mass, ok.float())


def test_the_pass_scores_every_arm_block_size_and_budget_and_writes_split_rows():
    q, k = _qk(h=2, s=1100)
    p = R.RecallPass(scorers={"MP": R.meanpool_scorer()}, identity_seed="e")
    for layer in range(2):
        p.set_layer(layer)
        p.score_layer(q, k)
    assert set(p.cells) == {(a, b, d) for a in ("O", "MP") for b in R.BLOCK_SIZES
                            for d in fr.D_NOMS}
    kw = dict(model="m", task="qa_1", band=16384, n_layers=2, git_commit="abc", git_dirty=False)
    rows = p.rows(example_id="qa_1_16384_1003", **kw)
    assert len(rows) == 32 and {r.split for r in rows} == {"selection"}
    assert {r.mask_selector for r in rows} == {"per_head"}
    for r in rows:
        assert 0 < r.recall_raw <= r.recall_norm <= 1 + 1e-6
        if r.arm == "O":
            assert r.recall_norm == pytest.approx(1.0, abs=1e-5)
    mp = {(r.block_size, r.d_nom): r for r in rows if r.arm == "MP"}
    assert mp[(128, 0.5)].recall_raw > mp[(128, 0.05)].recall_raw
    # MP ranks with its own scores, not the oracle's: it loses some mass.
    # (At b = 16; at b = 128 this short prompt has 9 blocks and little to lose.)
    assert all(mp[(16, d)].recall_norm < 0.999 for d in fr.D_NOMS)
    # The rows go straight into budget selection, which reads the split.
    assert set(fr.to_frame(rows)["split"]) == {"selection"}
    assert p.rows(example_id="qa_1_16384_3003", **kw)[0].split == "evaluation"


def test_the_realised_density_on_a_row_is_the_masks():
    q, k = _qk(h=2, s=128 * 20)
    p = R.RecallPass(scorers={}, identity_seed="e", block_sizes=(128,), d_noms=(0.25,))
    p.set_layer(0)
    p.score_layer(q, k)
    (row,) = p.rows(model="m", task="qa_1", band=16384, example_id="qa_1_16384_1000",
                    n_layers=1, git_commit="abc", git_dirty=False)
    want = sum(fp.kept_128(i, 0.25) for i in range(20)) / (20 * 21 / 2)
    assert row.realised_density == pytest.approx(want)


@pytest.mark.parametrize("b", [16, 64])
def test_below_128_the_pass_uses_the_matched_budgets(b):
    s, d_nom = 128 * 24, 0.25
    n = s // b
    q, k = _qk(h=2, s=s)
    p = R.RecallPass(scorers={}, identity_seed="e", block_sizes=(b,), d_noms=(d_nom,))
    p.set_layer(0)
    p.score_layer(q, k)
    (row,) = p.rows(model="m", task="qa_1", band=16384, example_id="qa_1_16384_1000",
                    n_layers=1, git_commit="abc", git_dirty=False)
    valid = n * (n + 1) / 2
    matched = sum(min(i + 1, 2) + fp.matched_budget(i, b, d_nom) for i in range(n)) / valid
    plain = sum(min(i + 1, 2) + (masks.era_budget(d_nom, i - 1) if i > 1 else 0)
                for i in range(n)) / valid
    assert row.realised_density == pytest.approx(matched)
    assert abs(matched - plain) > 1e-3


def test_a_layer_scored_twice_or_missing_is_refused():
    q, k = _qk(s=300)
    p = R.RecallPass(scorers={}, identity_seed="e", block_sizes=(128,), d_noms=(0.5,))
    with pytest.raises(RuntimeError, match="set_layer"):
        p.score_layer(q, k)
    p.set_layer(0)
    p.score_layer(q, k)
    with pytest.raises(RuntimeError, match="twice"):
        p.score_layer(q, k)
    with pytest.raises(RuntimeError, match="scored layers"):
        p.rows(model="m", task="qa_1", band=1, example_id="qa_1_1_1000", n_layers=2,
               git_commit="a", git_dirty=False)
    with pytest.raises(ValueError):
        R.RecallPass(scorers={"O": R.meanpool_scorer()}, identity_seed="e")


def test_per_head_mean_pool_is_each_heads_own_scores_and_its_group_mean_is_the_old_function():
    g = torch.Generator().manual_seed(4)
    hq, hkv, s, d, b = 6, 2, 512, 16, 128
    q = torch.randn(1, hq, s, d, generator=g)
    k = torch.randn(1, hkv, s, d, generator=g)
    per = acc_model.meanpool_scores_per_head_on_device(q, k, n_heads_kv=hkv, block_size=b)
    assert per.shape == (hq, 4, 4)
    for h in range(hq):
        qh = q[0, h].view(4, b, d).mean(1)
        kg = k[0, h // (hq // hkv)].view(4, b, d).mean(1)
        logits = (qh @ kg.T) / math.sqrt(d)
        logits = logits.masked_fill(torch.ones(4, 4, dtype=torch.bool).triu(1), float("-inf"))
        assert torch.allclose(per[h], torch.softmax(logits, -1), atol=1e-6)
    assert not torch.allclose(per[0], per[1])
    old = acc_model.minference_meanpool_scores_on_device(q, k, n_heads_kv=hkv, block_size=b)
    assert torch.equal(old, per.view(hkv, hq // hkv, 4, 4).mean(dim=1))
