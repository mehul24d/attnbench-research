"""Calibrated XAttention: per-(layer, head) thresholds, end to end.

The authors' RULER evaluation runs XAttention with a profiled 32x32 table,
not a scalar (docs/t4_xattention_calibrated.md). What these tests pin:

  - a table is identified by the digest of its values, and a file whose
    digest does not match is refused;
  - each layer's estimate receives that layer's row, as a per-head tensor,
    exactly as `Xattention_prefill` receives `threshold[layer_idx]`;
  - the profiler hands the official `xattn_prefill_profile` each layer's
    q and repeated k, keeps one row per layer, takes the maximum over
    texts, and returns dense attention so later layers are undisturbed;
  - a calibrated row names its table, and one table cannot resume into
    another's rows;
  - the analysis tests each calibration once, separately, and a
    descriptive calibration never claims.

The CPU tests stand in for the official modules; the two CUDA tests at the
end compare against the real ones on the instance.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import replace as dc_replace
from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from attnbench import provenance
from attnbench.accuracy import t4_pilot
from attnbench.accuracy.generation import ModelGeometry, StopTokens, generate_one
from attnbench.accuracy.model import SwappableAttentionModel
from attnbench.accuracy.ruler import RulerExample
from attnbench.accuracy.runner import ThresholdMismatch, build_cells, run_accuracy
from attnbench.accuracy.schema import Generated
from attnbench.backends.impls import SDPABackend
from attnbench.backends.xattention import (ThresholdProfiler, ThresholdTable,
                                           XAttentionBackend)
from attnbench.config import AttnConfig

ROOT = Path(__file__).resolve().parents[1]
XATTN = t4_pilot.XATTN_SCORE_SOURCE
LAYERS, HQ = 3, 4


def _script(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table_file(tmp_path, name="authors", values=None, sha=None):
    values = values or [[0.5 + 0.1 * l + 0.01 * h for h in range(HQ)] for l in range(LAYERS)]
    doc = dict(name=name, thresholds=values,
               sha256=sha if sha is not None else ThresholdTable.digest(values))
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(doc))
    return p


# ---- the table --------------------------------------------------------------

def test_a_table_is_its_values_digest(tmp_path):
    t = ThresholdTable.load(_table_file(tmp_path))
    assert t.label == f"authors:{t.sha256[:12]}" and len(t.values) == LAYERS
    with pytest.raises(ValueError, match="does not match"):
        ThresholdTable.load(_table_file(tmp_path, sha="0" * 64))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ThresholdTable.load(_table_file(tmp_path, values=[[1.5] * HQ] * LAYERS))
    with pytest.raises(ValueError, match="rectangular"):
        ThresholdTable.load(_table_file(tmp_path, values=[[0.5] * HQ, [0.5]]))


# ---- the backend with a table ----------------------------------------------

@pytest.fixture
def fake_estimate(monkeypatch):
    seen = []

    def xattn_estimate(q, k, **kw):
        seen.append(kw["threshold"])
        nb = -(-q.shape[2] // 128)
        m = torch.ones(1, k.shape[1], nb, nb, dtype=torch.bool).tril()
        return torch.zeros(1), m

    def block_sparse_attn_func(q, k, v, *a, **kw):
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=True)
        return out.transpose(0, 1).contiguous()

    pkg = types.ModuleType("xattn"); src = types.ModuleType("xattn.src")
    mod = types.ModuleType("xattn.src.Xattention"); mod.xattn_estimate = xattn_estimate
    bsa = types.ModuleType("block_sparse_attn"); bsa.block_sparse_attn_func = block_sparse_attn_func
    for name, m in (("xattn", pkg), ("xattn.src", src), ("xattn.src.Xattention", mod),
                    ("block_sparse_attn", bsa)):
        monkeypatch.setitem(sys.modules, name, m)
    return seen


def _qkv(s=300):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(1, HQ, s, 8, generator=g), torch.randn(1, 2, s, 8, generator=g),
            torch.randn(1, 2, s, 8, generator=g))


def _causal(s=300):
    return AttnConfig(seq_len=s, batch=1, n_heads_q=HQ, n_heads_kv=2, head_dim=8,
                      dtype="float32", mask="causal")


def test_each_layer_gets_its_own_row_as_a_per_head_tensor(fake_estimate, tmp_path):
    table = ThresholdTable.load(_table_file(tmp_path))
    b = XAttentionBackend(threshold=table)
    with pytest.raises(RuntimeError, match="set_layer"):
        b.forward(*_qkv(), _causal())
    for layer in (2, 0):
        b.set_layer(layer)
        b.forward(*_qkv(), _causal())
        got = fake_estimate[-1]
        assert isinstance(got, torch.Tensor) and got.shape == (HQ,)
        assert torch.allclose(got, torch.tensor(table.values[layer], dtype=torch.float32))


def test_a_table_for_another_head_count_is_refused(fake_estimate, tmp_path):
    table = ThresholdTable.load(_table_file(tmp_path, values=[[0.5] * 6] * LAYERS))
    b = XAttentionBackend(threshold=table)
    b.set_layer(0)
    with pytest.raises(ValueError, match="query heads"):
        b.forward(*_qkv(), _causal())


# ---- the profiler -----------------------------------------------------------

@pytest.fixture
def fake_profiler(monkeypatch):
    """Stands in for xattn.threshold.profile_threshold.profile_threshold: each
    call records its inputs and appends the next per-head value from a fixed
    script, the way the official function appends `threshold_head`."""
    script = iter([0.1, 0.5, 0.2,      # text 1, layers 0..2
                   0.3, 0.4, 0.9])     # text 2
    calls = []

    class ProfileConfig:
        def __init__(self, stride=8, causal=True):
            self.stride, self.causal, self.history_threshold = stride, causal, []

    def xattn_prefill_profile(self, query_states, key_states, value_states,
                              block_size, stride, chunk_size=16384, causal=True):
        calls.append((self.layer_idx, query_states.shape, key_states.shape, block_size, stride))
        th = torch.full((1, query_states.shape[1]), next(script))
        self.profile_config.history_threshold.append(th)
        return None

    names = ["xattn", "xattn.threshold", "xattn.threshold.profile_threshold",
             "xattn.threshold.profile_threshold.profile_threshold"]
    mods = [types.ModuleType(n) for n in names]
    mods[-1].ProfileConfig = ProfileConfig
    mods[-1].xattn_prefill_profile = xattn_prefill_profile
    for n, m in zip(names, mods):
        monkeypatch.setitem(sys.modules, n, m)
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    return calls


def _toy():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=LAYERS, num_attention_heads=HQ,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=512, attn_implementation="eager")
    model = LlamaForCausalLM(cfg).eval()
    geometry = ModelGeometry.from_config(model.config, "float32")
    template = geometry.onto(_causal(), seq_len=300)
    return SwappableAttentionModel(model, template, model_id="toy", score_source=XATTN), geometry


def test_the_profiler_collects_one_row_per_layer_and_keeps_the_max(fake_profiler):
    wrapped, geometry = _toy()
    ids = torch.randint(0, 64, (1, 300), generator=torch.Generator().manual_seed(1))
    cfg = geometry.onto(_causal(), seq_len=300)
    prof = ThresholdProfiler(n_layers=LAYERS, stride=8)
    for _ in range(2):
        prof.start_text()
        out = wrapped.run_measured(ids, prof, cfg=cfg, logits_to_keep=1)
        prof.end_text()
    assert [c[0] for c in fake_profiler] == [0] * 6          # a fresh shim per layer
    assert all(c[2][1] == HQ and c[3:] == (128, 8) for c in fake_profiler)  # k repeated
    assert prof.table()[:, 0].tolist() == pytest.approx([0.3, 0.5, 0.9])
    assert prof.flash_attn_stubbed is False
    # the layer output is dense causal attention: the same logits as SDPA math
    dense = wrapped.run_measured(ids, SDPABackend("math"), cfg=cfg, logits_to_keep=1)
    assert torch.allclose(out.logits, dense.logits, atol=1e-5)


def test_a_text_cut_short_is_refused(fake_profiler):
    prof = ThresholdProfiler(n_layers=LAYERS)
    prof.start_text()
    with pytest.raises(RuntimeError, match="every layer"):
        prof.end_text()


# ---- rows -------------------------------------------------------------------

class _Tok:
    eos_token_id = 63

    def __len__(self):
        return 64

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return type("E", (), {"input_ids": torch.randint(
            0, 64, (1, 300), generator=torch.Generator().manual_seed(1))})()

    def decode(self, ids, skip_special_tokens=False):
        return "".join("\n" if i == 5 else f"t{i}" for i in ids)

    def batch_decode(self, batches, skip_special_tokens=False):
        return [self.decode(b) for b in batches]


def test_a_calibrated_row_names_its_table(fake_estimate, tmp_path):
    table = ThresholdTable.load(_table_file(tmp_path))
    wrapped, geometry = _toy()
    ex = RulerExample(task="qa_1", example_id="e0", context="c", question="",
                      answer=["1"], context_length=300)
    tok = _Tok()
    gen = generate_one(wrapped, tok, cfg=_causal(2048), backend=XAttentionBackend(threshold=table),
                       example=ex, geometry=geometry, stop_tokens=StopTokens.from_tokenizer(tok),
                       score_cache_dir=str(tmp_path), device="cpu",
                       pinned_fallback_decode="sdpa_math")
    assert gen.xattn_calibration == table.label and gen.xattn_threshold is None
    assert len(gen.realised_density_by_layer) == LAYERS
    assert gen.realised_density == pytest.approx(1.0)
    assert len(fake_estimate) == LAYERS                       # one estimate per layer


def _run(tmp_path, label):
    cfg = AttnConfig(seq_len=1024, batch=1, n_heads_q=8, n_heads_kv=8, head_dim=64,
                     mask="causal")
    ex = RulerExample(task="qa_1", example_id="e0", context="c", question="",
                      answer=["1"], context_length=1024)
    cells = build_cells(configs_by_backend={"xattention": [cfg]},
                        examples_by_task_length={("qa_1", 1024): [ex]})
    prov = lambda: dc_replace(provenance.capture(), git_commit="a" * 40, git_dirty=False)
    gen = Generated("1", latency_ms=1.0, xattn_calibration=label, realised_density=0.3,
                    realised_density_by_layer=[0.3] * LAYERS)
    return run_accuracy(cells, out_dir=tmp_path, examples_by_id={("qa_1", "e0"): ex},
                        generate_fn=lambda c, b, e: gen, provenance_fn=prov,
                        score_source=XATTN, xattn_calibration=label)


def test_one_table_cannot_resume_into_anothers_rows(tmp_path):
    _run(tmp_path, "authors:aaaaaaaaaaaa")
    row = pd.read_parquet(tmp_path / "accuracy.parquet").iloc[0]
    assert row.xattn_calibration == "authors:aaaaaaaaaaaa" and pd.isna(row.xattn_threshold)
    assert list(row.realised_density_by_layer) == [0.3] * LAYERS
    with pytest.raises(ThresholdMismatch):
        _run(tmp_path, "ruler_heldout:bbbbbbbbbbbb")


# ---- the CLI ----------------------------------------------------------------

def test_the_cli_plans_a_calibrated_run_and_refuses_off_plan_tables(monkeypatch, capsys, tmp_path):
    sel = _script("run_accuracy")

    def fake_build(grid, seed, *, count_tokens, tasks, seq_lens):
        return {(t, b): [RulerExample(task=t, example_id=f"{t}_{b}_{i}", context="c",
                                      question="", answer=["a"], context_length=b,
                                      sizing="approximate") for i in range(n)]
                for t in tasks for b, n in seq_lens.items()}

    monkeypatch.setattr(sel, "build_examples_by_task_length", fake_build)

    def dry(*argv):
        monkeypatch.setattr(sys, "argv", ["run_accuracy.py", "--dry-run", "--out",
                                          str(tmp_path / "o"), *argv])
        sel.main()
        return capsys.readouterr().out

    out = dry("--t4-xattn-pilot", "--only-backends", "xattention",
              "--xattn-calibration", str(_table_file(tmp_path)))
    assert "total cells   : 400" in out and "calibration   : authors:" in out
    for argv in (["--t4-xattn-pilot", "--xattn-calibration", str(_table_file(tmp_path, "mine"))],
                 ["--t4-xattn-pilot", "--xattn-threshold", "0.9",
                  "--xattn-calibration", str(_table_file(tmp_path))],
                 ["--xattn-calibration", str(_table_file(tmp_path))]):
        with pytest.raises(SystemExit):
            dry(*argv)


# ---- the analysis -----------------------------------------------------------

def _ni_rows(losses: dict, digest="aaaaaaaaaaaa"):
    rows = []
    for task in t4_pilot.SPARSE_PILOT_TASKS:
        for band in t4_pilot.PILOT_BANDS:
            for i in range(t4_pilot.SPARSE_PILOT_N[task]):
                base = dict(task=task, example_id=f"{task}_{band}_{i}", context_length=band - 5,
                            git_commit="c" * 40, git_dirty=False, sparsity=None,
                            xattn_threshold=None)
                rows.append({**base, "backend": "sdpa_flash", "backend_role": "dense_reference",
                             "score_source": None, "xattn_calibration": None,
                             "realised_density": None, "correct": True})
                for name, k in losses.items():
                    rows.append({**base, "backend": "xattention", "backend_role": "block_sparse",
                                 "score_source": XATTN, "xattn_calibration": f"{name}:{digest}",
                                 "realised_density": 0.3, "correct": i >= k})
    return pd.DataFrame(rows)


def _ni(tmp_path, df):
    ni = _script("run_t4_noninferiority")
    p = tmp_path / "in.parquet"
    df.to_parquet(p)
    old = sys.argv
    sys.argv = ["x", str(p), "--out", str(tmp_path / "out")]
    try:
        ni.main()
    finally:
        sys.argv = old
    return pd.read_parquet(tmp_path / "out" / "t4_noninferiority.parquet")


def test_each_calibration_is_one_test_and_the_descriptive_one_never_claims(tmp_path):
    res = _ni(tmp_path, _ni_rows({"authors": 0, "ruler_heldout": 0}))
    q = res[(res.task == "qa_1") & (res.band == 16384)].set_index("xattn_calibration")
    assert q.tested.all() and q.non_inferior.all()
    assert bool(q.loc["authors", "claim"]) and not bool(q.loc["ruler_heldout", "claim"])
    assert q.sparsity.isna().all() and q.xattn_threshold.isna().all()


def test_two_tables_under_one_name_are_refused(tmp_path):
    df = pd.concat([_ni_rows({"authors": 0}, "aaaaaaaaaaaa"),
                    _ni_rows({"authors": 0}, "bbbbbbbbbbbb")
                    .query("backend == 'xattention'")
                    .assign(example_id=lambda d: d.example_id + "_x")])
    with pytest.raises((ValueError, SystemExit)):
        _ni(tmp_path, df)


# ---- the calibration script's own checks -------------------------------------

def test_chat_markers_are_stripped_and_overlap_is_refused():
    cal = _script("calibrate_xattn_thresholds")
    assert cal.strip_chat_markers(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nAnswer") == "user\n\nAnswer"
    ex = lambda i, c: RulerExample(task="qa_1", example_id=str(i), context=c, question="",
                                   answer=["a"], context_length=1)
    cal.assert_disjoint({("qa_1", 1): [ex(0, "a")]}, {("qa_1", 1): [ex(1, "b")]})
    with pytest.raises(SystemExit, match="leaks into the test"):
        cal.assert_disjoint({("qa_1", 1): [ex(0, "a")]}, {("qa_1", 1): [ex(9, "a")]})


def test_a1_a_shared_qa_question_under_another_seed_is_refused(monkeypatch, tmp_path):
    """T4 amendment A1's break-test (pre-registration sec. 5). Seed 1 at
    offset 0 asks the test's own qa_1 questions with other distractors, so
    whole contexts differ and the old check passed it. The new one refuses
    it; at the registered offset (2000) the same seed is disjoint."""
    from attnbench.accuracy import ruler, sizing
    from attnbench.accuracy import ruler_data
    cal = _script("calibrate_xattn_thresholds")
    from attnbench.accuracy import t4_pilot

    def make(seed, offset):
        return {("qa_1", 400): ruler.generate_examples(
            "qa_1", [400], n_per_length=3, seed=seed, index_offset=offset,
            count_tokens=sizing.approximate_token_count)}
    try:
        test = make(0, 0)
    except Exception as e:  # the QA corpora are not committed
        pytest.skip(f"qa_1 resources unavailable here: {type(e).__name__}")
    leaky = make(t4_pilot.XATTN_CALIBRATION_SEED, 0)
    assert all(a.context != b.context for a, b in zip(leaky[("qa_1", 400)], test[("qa_1", 400)]))
    # The ids collide too, which the check also refuses. Give the leaky set
    # fresh ids (same trailing question index) so only the question leaks.
    import dataclasses
    leaky = {k: [dataclasses.replace(e, example_id="cal_" + e.example_id) for e in v]
             for k, v in leaky.items()}
    with pytest.raises(SystemExit, match="qa_question"):
        cal.assert_disjoint(leaky, test)
    cal.assert_disjoint(make(t4_pilot.XATTN_CALIBRATION_SEED,
                             t4_pilot.XATTN_CALIBRATION_INDEX_OFFSET), test)


# ---- on the instance --------------------------------------------------------

def _real_xattn():
    if not torch.cuda.is_available():
        return False
    try:
        import block_sparse_attn  # noqa: F401
        import xattn.src.Xattention  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _real_xattn(), reason="needs CUDA + x-attention + block-sparse-attn")
def test_a_table_row_is_bitwise_xattention_prefill_with_that_row_on_cuda(tmp_path):
    """The authors' RULER path passes `threshold=table[layer_idx]`, a per-head
    tensor, to Xattention_prefill. This backend, given the table and the
    layer, must produce exactly its output."""
    from xattn.src.Xattention import Xattention_prefill
    s, hq = 16384, 12
    row = [0.6 + 0.03 * h for h in range(hq)]
    values = [[0.9] * hq, row]
    table = ThresholdTable.load(_table_file(tmp_path, values=values))
    g = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(1, hq, s, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, 2, s, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, 2, s, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    cfg = AttnConfig(seq_len=s, batch=1, n_heads_q=hq, n_heads_kv=2, head_dim=128,
                     dtype="bfloat16", mask="causal")
    b = XAttentionBackend(threshold=table)
    b.set_layer(1)
    ours = b.forward(q, k, v, cfg)
    kr, vr = k.repeat_interleave(6, dim=1), v.repeat_interleave(6, dim=1)
    theirs = Xattention_prefill(q, kr, vr, stride=8, norm=1,
                                threshold=torch.tensor(row, device="cuda"),
                                use_triton=True, keep_sink=True, keep_recent=True)
    assert torch.equal(ours, theirs)


@pytest.mark.skipif(not _real_xattn(), reason="needs CUDA + x-attention + block-sparse-attn")
def test_the_profiler_records_what_the_official_profile_computes_on_cuda():
    from attnbench.backends.xattention import _official_profiler
    mod, _ = _official_profiler()
    s, hq = 4096, 12
    g = torch.Generator(device="cuda").manual_seed(2)
    q = torch.randn(1, hq, s, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, 2, s, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, 2, s, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    cfg = AttnConfig(seq_len=s, batch=1, n_heads_q=hq, n_heads_kv=2, head_dim=128,
                     dtype="bfloat16", mask="causal")
    prof = ThresholdProfiler(n_layers=1)
    prof.start_text(); prof.set_layer(0); prof.forward(q, k, v, cfg); prof.end_text()
    shim = types.SimpleNamespace(layer_idx=0, profile_config=mod.ProfileConfig(stride=8))
    mod.xattn_prefill_profile(shim, query_states=q, key_states=k.repeat_interleave(6, dim=1),
                              value_states=v.repeat_interleave(6, dim=1), block_size=128, stride=8)
    want = shim.profile_config.history_threshold[0].reshape(-1).float().cpu()
    assert torch.equal(prof.table()[0], want)
    assert ((want >= 0) & (want <= 1)).all()


def test_a_stubbed_flash_attn_is_reported_after_the_first_layer(fake_profiler, monkeypatch):
    """The stand-in is registered on the first call; every later import then
    succeeds, and the flag must still say the first call found none."""
    monkeypatch.delitem(sys.modules, "flash_attn")
    real_import = __import__

    def no_flash(name, *a, **k):
        if name == "flash_attn" and "flash_attn" not in sys.modules:
            raise ImportError("no flash_attn")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", no_flash)
    wrapped, geometry = _toy()
    ids = torch.randint(0, 64, (1, 300), generator=torch.Generator().manual_seed(1))
    prof = ThresholdProfiler(n_layers=LAYERS)
    prof.start_text()
    wrapped.run_measured(ids, prof, cfg=geometry.onto(_causal(), seq_len=300), logits_to_keep=1)
    prof.end_text()
    assert prof.flash_attn_stubbed is True
