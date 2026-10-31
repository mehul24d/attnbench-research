"""GPU variant of the GQA test in test_swappable_attention_model.py (T2.26).

UNRUN: written in a container without torch or a GPU. Same assertion as the
CPU test, on the device and dtype `_gpu_env` reads from the environment; it
skips without that device. The hidden-state bound is the CPU test's 1e-4
raised to the dtype's rounding (`_gpu_env.tolerance`), still far below the
~1.8 drift of the zeroed-output bug it exists to catch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from attnbench.accuracy.model import SwappableAttentionModel  # noqa: E402
from attnbench.config import AttnConfig  # noqa: E402

import _gpu_env  # noqa: E402

pytestmark = _gpu_env.needs_gpu


def test_gpu_cheap_scoring_pass_propagates_faithful_hidden_states(tmp_path):
    tmpl = AttnConfig(seq_len=64, batch=1, n_heads_q=4, n_heads_kv=2, head_dim=8,
                      dtype=_gpu_env.dtype_name(), mask="causal", block_size=8)
    ids = torch.randint(0, 64, (1, 64), generator=torch.Generator().manual_seed(7))
    ids = ids.to(_gpu_env.device())

    def hidden_states_per_layer(score_source: str):
        torch.manual_seed(0)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=4, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=8,
                             max_position_embeddings=128, attn_implementation="eager")
        model = LlamaForCausalLM(config).eval().to(device=_gpu_env.device(),
                                                    dtype=_gpu_env.dtype())
        swapped = SwappableAttentionModel(model, cfg_template=tmpl, model_id="toy",
                                          score_source=score_source)
        seen = []
        for idx, layer in enumerate(model.model.layers):
            layer.self_attn.register_forward_pre_hook(
                (lambda i: (lambda mod, args, kwargs: seen.append(
                    (i, (args[0] if args else kwargs["hidden_states"]).detach().clone())
                )))(idx),
                with_kwargs=True)
        # The production entry point, as in the CPU test.
        with torch.no_grad():
            swapped.compute_importance_scores(ids, task="t", example_id="e",
                                              cache_dir=str(tmp_path / score_source))
        return seen

    cheap = hidden_states_per_layer("minference_meanpool")
    dense = hidden_states_per_layer("dense_softmax_fp32")
    assert [i for i, _ in cheap] == [0, 1, 2, 3]
    bound = _gpu_env.tolerance(1e-4)
    for (i, a), (j, b) in zip(cheap, dense):
        assert i == j
        drift = (a.float() - b.float()).abs().max().item()
        assert drift < bound, (
            f"layer {i} hidden states diverge by {drift:.3e} (bound {bound:.1e}) "
            f"between the cheap and the dense scoring pass")
