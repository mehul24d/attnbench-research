"""--model must refuse a model the device cannot hold, before the boot is spent.

The failure being prevented is not an OOM. It is an OOM discovered after the
tokenizer has downloaded, the examples have been built and the clocks have
been locked, on a rented instance -- the same shape as the Stage 1 gate and
the GCS write gate, which all check a precondition while failing is free.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "_run_accuracy", Path(__file__).resolve().parents[1] / "scripts" / "run_accuracy.py")


def _load():
    mod = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)
    return mod


def _fake_config(hidden, layers, vocab, inter):
    c = types.SimpleNamespace()
    c.hidden_size, c.num_hidden_layers = hidden, layers
    c.vocab_size, c.intermediate_size = vocab, inter
    return c


def _patched(monkeypatch, mod, *, total_gb, cfg):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda i: types.SimpleNamespace(
                total_memory=int(total_gb * 1e9)),
            get_device_name=lambda: "FAKE-GPU",
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda _id: cfg)))
    return mod


def test_refuses_a_7b_on_a_24gb_card(monkeypatch):
    mod = _load()
    qwen7b = _fake_config(3584, 28, 152064, 18944)
    _patched(monkeypatch, mod, total_gb=24.0, cfg=qwen7b)
    with pytest.raises(SystemExit) as e:
        mod._guard_model_fits("Qwen/Qwen2.5-7B-Instruct", "cuda")
    assert "REFUSING" in str(e.value)


def test_accepts_the_same_7b_on_an_80gb_card(monkeypatch):
    mod = _load()
    qwen7b = _fake_config(3584, 28, 152064, 18944)
    _patched(monkeypatch, mod, total_gb=80.0, cfg=qwen7b)
    mod._guard_model_fits("Qwen/Qwen2.5-7B-Instruct", "cuda")   # must not raise


def test_cpu_device_is_not_guarded(monkeypatch):
    """--dry-run and CPU planning must stay runnable without a GPU."""
    mod = _load()
    mod._guard_model_fits("anything/at-all", "cpu")
