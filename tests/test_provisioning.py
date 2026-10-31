"""One provisioning vocabulary (`attnbench/provisioning.py`, researcher's
decision of 31 Oct 2026, after Shakedown 2's t21_t220): launch names in the
tags, row names on rows and in every preflight."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from attnbench import provenance, provisioning as P

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_the_map():
    assert P.ROW_NAME == {"STANDARD": "on-demand", "SPOT": "spot",
                          "FLEX_START": "dws-flex-start", "RESERVED": "reserved"}
    assert P.row_name(" flex_start ") == "dws-flex-start"
    with pytest.raises(KeyError, match="launch names"):
        P.row_name("on-demand")                    # a row name is not a launch name
    assert P.is_preemptible("spot") and not P.is_preemptible("dws-flex-start")


def test_provenance_reads_the_module_not_a_copy():
    assert provenance.PROVISIONING_MODELS is P.ROW_NAME
    assert provenance.PROVISIONING_TAG == P.TAG


def test_the_aws_standard_tag_is_on_demand():
    text = (ROOT / "scripts" / "aws_launch_session.sh").read_text()
    (model,) = re.findall(r'^MODEL="([A-Z_]+)"', text, re.M)
    assert f"Key={P.TAG},Value=$MODEL" in text
    assert P.row_name(model) == "on-demand"
    got = provenance._provisioning_model(model, P.AWS_LIFE_CYCLE["on-demand"],
                                         checked_against="instance-life-cycle")
    assert got == ("on-demand", None)


def test_every_preflight_compares_row_names():
    import t227_steps
    import t61_session
    import t64_session
    assert t227_steps.EXPECTED_PROVISIONING in P.ROW_NAMES
    for cfg in (t61_session.T61, t64_session.T64, t64_session.T64_GATE):
        row = {"provisioning_model": P.row_name(cfg.provisioning[0])}
        assert not any(p.startswith("provisioning_model ")
                       for p in t61_session.check_provenance_row(cfg, row, []))
        launch = {"provisioning_model": cfg.provisioning[0]}
        assert any(p.startswith("provisioning_model ")
                   for p in t61_session.check_provenance_row(cfg, launch, []))


@pytest.mark.parametrize("script", sorted(p.name for p in (ROOT / "scripts").glob("*.sh")))
def test_every_literal_tag_in_a_launcher_is_a_launch_name(script):
    text = (ROOT / "scripts" / script).read_text()
    for value in re.findall(rf"{P.TAG}=([A-Z_]+)", text):
        assert value in P.ROW_NAME, (script, value)
