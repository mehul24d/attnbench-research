"""A capability must agree with the implementation it describes.

`Capability` is what a backend CLAIMS; `probe()` is what it does. The project
already treats a backend claiming more than it delivers as a first-class
failure -- `gates.probe` records `claim_mismatch`, and the whole "verified,
never trusted" line in `backends/base.py` is about that direction.

**The other direction is not symmetric, and it is the dangerous one.** A
backend declaring LESS than it supports raises nothing. `claims_support()` is
a pre-filter: it answers "no decode support" and the config is never run, so
the row is recorded as unsupported and the sweep looks complete. There is no
mismatch to detect, because the kernel was never asked.

That is exactly what `sdpa` did until 2026-09-21. `SDPABackend` defines
`state_from_prefill`, so `SDPABackend.supports_decode()` -- which reads the
class dict -- returned True, and Stage 3 has always decoded the dense arm
through it. Its `Capability.supports_decode` was False, so the Stage 0/1
probe skipped every sdpa decode cell. Two answers to one question, one
consulted by the accuracy stage and the other by the probe, disagreeing for
as long as nobody asked them in the same breath.

This file asks them in the same breath.
"""

from __future__ import annotations

import pytest

from attnbench.backends import all_backends
from attnbench.backends.base import Capability
from attnbench.config import AttnConfig


REGISTERED = sorted(all_backends().items())


def test_there_are_backends_to_check():
    """Anti-vacuity: every parametrised test below is over this list."""
    assert len(REGISTERED) >= 6, f"only {len(REGISTERED)} backends registered"


@pytest.mark.parametrize("name,cls", REGISTERED)
def test_declared_decode_support_matches_the_implementation(name, cls):
    declared = cls.capability.supports_decode
    actual = cls.supports_decode()
    assert declared == actual, (
        f"{name}: Capability.supports_decode={declared} but "
        f"{cls.__name__}.supports_decode()={actual}.\n"
        f"These are asked by different callers -- the probe reads the "
        f"capability, accuracy/model.py and grid_configs.py call the "
        f"classmethod -- so a disagreement means one stage runs decode "
        f"through this backend while another records it as unsupported.\n"
        f"Declaring LESS than you implement raises nothing anywhere: the "
        f"config is filtered out before the kernel is reached, and the "
        f"exclusion is silent.")


@pytest.mark.parametrize("name,cls", REGISTERED)
def test_a_decode_capable_backend_is_not_filtered_out_of_decode(name, cls):
    """The consequence, asserted at the gate rather than inferred.

    `claims_support` is where the declaration turns into a decision, so that
    is where the two answers have to meet."""
    if not cls.supports_decode():
        return
    cap = cls.capability
    cfg = AttnConfig(seq_len=1024, batch=1, n_heads_q=8, n_heads_kv=8,
                     head_dim=cap.head_dims[0], dtype=cap.dtypes[0],
                     mask="causal", pass_kind="fwd", regime="decode")
    ok, why = cls().claims_support(cfg)
    assert why != "no decode support", (
        f"{name} implements decode and its capability gate refuses it with "
        f"{why!r}. Every decode cell for this backend is skipped by the "
        f"probe and never appears as a failure.")


def test_the_check_would_catch_a_backend_that_declares_less_than_it_does():
    """The break-test, against the shape the real defect had.

    Kept rather than run once: the real instance of this was found by an
    audit comparing two files, not by anything in the suite, and the way it
    went unnoticed for so long is that nothing ever compared them.
    """
    class _Understated:
        capability = Capability(name="understated", family="dense_exact",
                                supports_decode=False)

        @classmethod
        def supports_decode(cls) -> bool:
            return True

    with pytest.raises(AssertionError, match="Capability.supports_decode"):
        test_declared_decode_support_matches_the_implementation(
            "understated", _Understated)


def test_the_check_would_catch_a_backend_that_declares_more_than_it_does():
    class _Overstated:
        capability = Capability(name="overstated", family="dense_exact",
                                supports_decode=True)

        @classmethod
        def supports_decode(cls) -> bool:
            return False

    with pytest.raises(AssertionError, match="Capability.supports_decode"):
        test_declared_decode_support_matches_the_implementation(
            "overstated", _Overstated)
