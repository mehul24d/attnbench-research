"""Render docs/RESEARCH_PLAN.md to docs/research_plan.pdf.

    python tools/build_plan_pdf.py

The Markdown file is the source of truth; the PDF is generated from it and
committed alongside it so the plan can be read where Markdown is not rendered
(Claude chat project files, email to the supervisor). Never edit the PDF by
hand.

Needs `pip install markdown` and Playwright with a Chromium build (in cloud
sessions Chromium is preinstalled under /opt/pw-browsers).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "docs" / "RESEARCH_PLAN.md"
OUT = REPO / "docs" / "research_plan.pdf"

CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm; }
body { font-family: "DejaVu Sans", Arial, sans-serif; color: #1a1a1a;
       background: #fff; font-size: 9.4pt; line-height: 1.42; margin: 0; }
h1 { font-size: 18pt; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 16pt 0 5pt; page-break-after: avoid;
     border-bottom: 0.6pt solid #bdbdbd; padding-bottom: 2pt; }
h3 { font-size: 10.8pt; margin: 11pt 0 4pt; page-break-after: avoid; }
p { margin: 0 0 6pt; }
ul, ol { margin: 0 0 7pt; padding-left: 17pt; }
li { margin: 0 0 2.5pt; }
hr { border: 0; border-top: 0.6pt solid #bdbdbd; margin: 10pt 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 9pt;
        font-size: 8.1pt; line-height: 1.32; }
th, td { border: 0.6pt solid #bdbdbd; padding: 3pt 4.5pt; vertical-align: top;
         text-align: left; }
th { background: #e6e6e6; }
tr { page-break-inside: avoid; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.2pt; }
pre { background: #f4f4f4; padding: 6pt 8pt; font-size: 8.2pt;
      line-height: 1.5; white-space: pre-wrap; page-break-inside: avoid; }
"""

RENDER_JS = """
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + process.argv[2], { waitUntil: 'load' });
  await p.pdf({ path: process.argv[3], format: 'A4', displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="width:100%;font-family:DejaVu Sans,Arial;' +
      'font-size:7.5pt;color:#888;text-align:right;padding-right:16mm">' +
      'Research plan, page <span class="pageNumber"></span> of ' +
      '<span class="totalPages"></span></div>',
    margin: { top: '17mm', bottom: '18mm', left: '16mm', right: '16mm' } });
  await b.close();
})();
"""


def main() -> int:
    body = markdown.markdown(SRC.read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "sane_lists"])
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Sparse Prefill Research Plan</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "plan.html"
        page.write_text(html, encoding="utf-8")
        script = Path(tmp) / "render.js"
        script.write_text(RENDER_JS, encoding="utf-8")
        node_path = subprocess.run(["npm", "root", "-g"], capture_output=True,
                                   text=True, check=True).stdout.strip()
        subprocess.run(["node", str(script), str(page), str(OUT)], check=True,
                       env={**os.environ, "NODE_PATH": node_path})
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
