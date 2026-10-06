"""Render docs/brief/brief.html to docs/research_brief.pdf.

    python tools/build_brief_pdf.py

The HTML is the source of the professor brief; the PDF is generated from it
and committed alongside it. Never edit the PDF by hand. House style for the
brief: plain language, short sentences, no em dashes (this script refuses to
build if it finds one).

Needs Playwright with a Chromium build (in cloud sessions Chromium is
preinstalled under /opt/pw-browsers).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "docs" / "brief" / "brief.html"
OUT = REPO / "docs" / "research_brief.pdf"

RENDER_JS = """
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + process.argv[2], { waitUntil: 'load' });
  await p.pdf({ path: process.argv[3], format: 'A4', displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="width:100%;font-family:DejaVu Sans,Arial;' +
      'font-size:7.5pt;color:#888;text-align:right;padding-right:18mm">' +
      'Page <span class="pageNumber"></span></div>',
    margin: { top: '18mm', bottom: '20mm', left: '18mm', right: '18mm' } });
  await b.close();
})();
"""


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    if "—" in text:
        print(f"refusing to build: {SRC.relative_to(REPO)} contains an em dash",
              file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "render.js"
        script.write_text(RENDER_JS, encoding="utf-8")
        node_path = subprocess.run(["npm", "root", "-g"], capture_output=True,
                                   text=True, check=True).stdout.strip()
        subprocess.run(["node", str(script), str(SRC), str(OUT)], check=True,
                       env={**os.environ, "NODE_PATH": node_path})
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
