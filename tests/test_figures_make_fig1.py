"""scripts/figures/make_fig1.py (C5) on fixtures made by the repo's own
converter (`fig1_events`) and metadata generator (`figure_meta`)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from attnbench.figures import fig1_events as F
from attnbench.figures import meta as M
from attnbench.predict.commit import CodeStamp

import _figure_fixtures as FX

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "figures" / "make_fig1.py"
CLEAN = CodeStamp(git_commit="c" * 40, git_dirty=False)

sys.path.insert(0, str(SCRIPT.parent))
import make_fig1  # noqa: E402


@pytest.fixture
def inputs(tmp_path):
    """events.csv (layer 0 of the hand trace, the L4 panel) and meta.json
    for layer 0 of the committed xattn cell on L4: predicted end 15000 ms."""
    doc = FX.trace()
    events = F.write_events([F.layer_events(doc, FX.parse(doc), 0, gpu="L4")],
                            tmp_path / "events.csv")
    cp, pred, _, _ = FX.loaded_p1(tmp_path)
    res = tmp_path / "ground_truth.json"
    res.write_text(json.dumps(FX.ground_truth()))
    meta = M.figure_meta(figure="fig1", prediction=cp, prediction_path=pred, cell=FX.XATTN,
                         result=FX.ground_truth(), result_path=res,
                         inputs=[("events", events)], code=CLEAN, layer=0)
    meta_path = M.write_meta(meta, tmp_path / "meta.json")
    return dict(tmp=tmp_path, events=events, meta=meta_path, meta_dict=meta)


def _texts(fig):
    import matplotlib.text
    return [t.get_text() for t in fig.findobj(matplotlib.text.Text)]


def _capture(monkeypatch):
    figs = []
    real = make_fig1.draw

    def spy(*a, **k):
        figs.append(real(*a, **k))
        return figs[-1]
    monkeypatch.setattr(make_fig1, "draw", spy)
    return figs


def test_the_script_writes_a_pdf_with_truetype_fonts(inputs):
    out = inputs["tmp"] / "fig1.pdf"
    r = subprocess.run([sys.executable, str(SCRIPT), "--events", str(inputs["events"]),
                        "--meta", f"L4={inputs['meta']}", "--layer", "0", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    data = out.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 1000
    # pdf.fonttype 42: TrueType embedded, no Type 3 fonts.
    assert b"/FontFile2" in data and b"/Type3" not in data
    assert make_fig1.matplotlib.rcParams["pdf.fonttype"] == 42


def test_the_real_path_never_draws_the_watermark(inputs, monkeypatch):
    figs = _capture(monkeypatch)
    assert make_fig1.main(["--events", str(inputs["events"]), "--meta",
                           f"L4={inputs['meta']}", "--layer", "0",
                           "--out", str(inputs["tmp"] / "real.pdf")]) == 0
    assert make_fig1.main(["--events", str(inputs["events"]),
                           "--out", str(inputs["tmp"] / "bare.pdf")]) == 0
    for fig in figs:
        assert not any(make_fig1.WATERMARK in t for t in _texts(fig))
    # The check is not vacuous: the demo draws it, on every panel.
    assert make_fig1.main(["--demo", "--out", str(inputs["tmp"] / "demo.pdf")]) == 0
    assert sum(make_fig1.WATERMARK in t for t in _texts(figs[-1])) == 2


def test_demo_mode_takes_no_real_inputs(inputs):
    for extra in (["--events", str(inputs["events"])], ["--meta", f"L4={inputs['meta']}"],
                  ["--note", "L4=x"]):
        with pytest.raises(SystemExit, match="synthetic bars only"):
            make_fig1.main(["--demo", "--out", str(inputs["tmp"] / "d.pdf")] + extra)


def test_the_predicted_end_is_read_through_predicted_end_ms(inputs, monkeypatch):
    figs = _capture(monkeypatch)
    calls = []
    real = make_fig1.predicted_end_ms
    monkeypatch.setattr(make_fig1, "predicted_end_ms",
                        lambda m, layer: calls.append(layer) or real(m, layer))
    make_fig1.main(["--events", str(inputs["events"]), "--meta", f"L4={inputs['meta']}",
                    "--layer", "0", "--out", str(inputs["tmp"] / "f.pdf")])
    assert calls == [0]
    ax = figs[0].axes[0]
    dashed = [l.get_xdata()[0] for l in ax.get_lines() if l.get_linestyle() == "--"]
    assert dashed == [15000.0]
    assert "predicted end" in _texts(figs[0])


def test_a_whole_prefill_value_is_refused_by_the_script(inputs):
    m = json.loads(json.dumps(inputs["meta_dict"]))
    m["predicted_layer"]["predicted_end_ms"] = m["predicted_layer"]["whole_prefill_ms"]
    bad = inputs["tmp"] / "whole.json"
    bad.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="whole prefill"):
        make_fig1.main(["--events", str(inputs["events"]), "--meta", f"L4={bad}",
                        "--layer", "0", "--out", str(inputs["tmp"] / "w.pdf")])
    assert not (inputs["tmp"] / "w.pdf").exists()


@pytest.mark.parametrize("argv, match", [
    (["--meta", "L4={meta}"], "needs --layer"),
    (["--meta", "L4={meta}", "--layer", "3"], "not layer 3"),
    (["--meta", "A100={meta}", "--layer", "0"], "has no events"),
    (["--meta", "L4={meta}", "--meta", "L4={meta}", "--layer", "0"], "twice"),
    (["--note", "A100=x"], "has no events"),
])
def test_metadata_that_does_not_fit_the_panels_is_refused(inputs, argv, match):
    argv = [a.format(meta=inputs["meta"]) for a in argv]
    with pytest.raises(SystemExit, match=match):
        make_fig1.main(["--events", str(inputs["events"]), "--out",
                        str(inputs["tmp"] / "x.pdf")] + argv)


def test_metadata_for_another_gpu_is_refused(inputs, tmp_path):
    a100 = FX.trace(gpu="NVIDIA A100-SXM4-80GB")
    ev = F.write_events([F.layer_events(a100, FX.parse(a100), 0, gpu="A100")],
                        tmp_path / "a100_events.csv")
    with pytest.raises(SystemExit, match="not the A100 panel"):
        make_fig1.main(["--events", str(ev), "--meta", f"A100={inputs['meta']}",
                        "--layer", "0", "--out", str(tmp_path / "x.pdf")])


def test_a_note_is_printed_as_given(inputs, monkeypatch):
    figs = _capture(monkeypatch)
    make_fig1.main(["--events", str(inputs["events"]), "--note", "L4=as typed, 1.0x",
                    "--out", str(inputs["tmp"] / "n.pdf")])
    assert "as typed, 1.0x" in _texts(figs[0])
