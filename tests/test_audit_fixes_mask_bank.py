"""Fix for audit C4, finding 2 (mask_bank.py): a manifest's per-call density
and causal fields are checked on load, against the masks and against a hash
over the call entries. Written before the fix.

`mask_bank` imports torch at module level; these tests use only numpy and
skip without torch. They were run here against an import-only stand-in for
torch (kept out of the repository), and are UNRUN where torch is installed.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from attnbench.components import mask_bank as mb  # noqa: E402
from attnbench.components.op_sequence import CodePath  # noqa: E402

CP = CodePath(arm="xattn", gpu="NVIDIA L4", implementation="xattn:torch")


def _mask(a, causal=True):
    packed = np.packbits(a.reshape(-1))
    return mb.BankedMask(tuple(a.shape), causal, mb.density(a, causal),
                         hashlib.sha256(packed.tobytes()).hexdigest(), packed)


def _bank(tmp_path, masks=None):
    tril = np.tril(np.ones((1, 2, 4, 4), dtype=bool))
    half = tril.copy()
    half[..., 3, 0] = False
    masks = masks or (_mask(tril), _mask(half))
    base = tmp_path / "bank"
    mb.save(mb.MaskRecord(CP, 512, 1, "prompt-0", tuple(masks)), base)
    return base


def _edit(base, fn):
    manifest = base.with_suffix(".json")
    meta = json.loads(manifest.read_text())
    fn(meta)
    manifest.write_text(json.dumps(meta, indent=1))


def test_a_bank_round_trips(tmp_path):
    rec = mb.load(_bank(tmp_path))
    assert [m.density for m in rec.masks] == [1.0, pytest.approx(9 / 10)]


@pytest.mark.parametrize("field, value", [("density", 0.5), ("causal", False)])
def test_a_manifest_whose_fields_do_not_follow_from_the_masks_is_refused(tmp_path, field, value):
    base = _bank(tmp_path)
    _edit(base, lambda m: m["calls"][0].__setitem__(field, value))
    with pytest.raises(mb.MaskBankError):
        mb.load(base)


def test_a_rehashed_manifest_with_a_wrong_density_is_still_refused(tmp_path):
    base = _bank(tmp_path)

    def edit(m):
        m["calls"][1]["density"] = 0.75
        m["calls_sha256"] = mb.calls_sha256(m["calls"])
    _edit(base, edit)
    with pytest.raises(mb.MaskBankError, match="density"):
        mb.load(base)


def test_flipping_causal_and_density_together_is_caught_by_the_calls_hash(tmp_path):
    base = _bank(tmp_path)
    a = np.tril(np.ones((1, 2, 4, 4), dtype=bool))

    def edit(m):
        m["calls"][0]["causal"] = False
        m["calls"][0]["density"] = mb.density(a, False)    # consistent with the mask
    _edit(base, edit)
    with pytest.raises(mb.MaskBankError, match="call entries"):
        mb.load(base)


def test_a_manifest_without_the_calls_hash_is_refused(tmp_path):
    base = _bank(tmp_path)
    _edit(base, lambda m: m.pop("calls_sha256"))
    with pytest.raises(mb.MaskBankError, match="calls_sha256"):
        mb.load(base)


def test_a_shape_that_does_not_fit_the_packed_bits_is_refused(tmp_path):
    base = _bank(tmp_path)

    def edit(m):
        m["calls"][0]["shape"] = [1, 2, 8, 8]
        m["calls_sha256"] = mb.calls_sha256(m["calls"])
    _edit(base, edit)
    with pytest.raises(mb.MaskBankError, match="shape"):
        mb.load(base)
