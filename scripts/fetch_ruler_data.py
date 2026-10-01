#!/usr/bin/env python
"""Build the RULER data directory the T4 tasks need, the way RULER builds it.

    python scripts/fetch_ruler_data.py [--out DIR]

Writes into DIR (default: $ATTNBENCH_RULER_DATA, else
~/.cache/attnbench/ruler_data):

  PaulGrahamEssays.json  RULER's scripts/data/synthetic/json/
                         download_paulgraham_essay.py, reproduced: the URL
                         list from the pinned RULER commit, .txt files fetched
                         raw, .html pages through BeautifulSoup('font') and
                         html2text with RULER's settings, repo files then html
                         files, each group sorted by filename, concatenated.
  squad.json             SQuAD 2.0 dev, from the URL RULER's
                         download_qa_dataset.sh uses.
  hotpotqa.json          HotpotQA dev distractor, RULER's URL with its
                         Hugging Face fallback (pinned revision).
  nltk_data/             NLTK's punkt_tab, for sent_tokenize.
  MANIFEST.json          sha256 and byte size of each file, and the URLs.

It prints the sha256 of each data file. Those go into
`attnbench/accuracy/ruler_data.PINNED_SHA256`, once, and the built directory
goes to the project bucket so every instance reads the same bytes: the essay
pages can change upstream, and an essay that changed is a haystack that
changed. Nothing here is committed to the repository (the essays are Paul
Graham's; the QA sets are CC BY-SA 4.0).

Needs network, plus `html2text` and `beautifulsoup4` (data-build only; not
runtime dependencies of the harness).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attnbench.accuracy import ruler_data  # noqa: E402

RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
ESSAY_URL_LIST = (f"https://raw.githubusercontent.com/NVIDIA/RULER/{RULER_COMMIT}/"
                  "scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt")
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
HOTPOT_URLS = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
    "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
    "7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json",
)


def _ssl_context():
    """Verification stays on. python.org's macOS build has no CA bundle until
    its 'Install Certificates' step runs, so use certifi's where present."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url: str, timeout: int = 60, tries: int = 4) -> bytes:
    """With retries and backoff. RULER's script has none; on 2026-10-01 five
    consecutive paulgraham.com pages returned HTTP 500 in one pass and served
    fine on retry, and a corpus with holes is a different haystack."""
    import time
    req = urllib.request.Request(url, headers={"User-Agent": "attnbench-ruler-data"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
                return r.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(5 * 2 ** attempt)


def build_essays(out: Path) -> list[str]:
    import html2text
    from bs4 import BeautifulSoup

    h = html2text.HTML2Text()
    h.ignore_images = True
    h.ignore_tables = True
    h.escape_all = True
    h.reference_links = False
    h.mark_code = False

    urls = [u.strip() for u in _get(ESSAY_URL_LIST).decode().splitlines() if u.strip()]
    repo, html, failed = {}, {}, []
    for url in urls:
        try:
            if ".html" in url:
                name = url.split("/")[-1].replace(".html", ".txt")
                # RULER's own decode, kept as is.
                content = _get(url).decode("unicode_escape", "utf-8")
                tag = BeautifulSoup(content, "html.parser").find("font")
                html[name] = h.handle(str(tag))
            else:
                repo[url.split("/")[-1]] = _get(url).decode("utf-8")
        except Exception as e:                      # RULER prints and continues
            failed.append(f"{url}: {e}")
    text = "".join(repo[k] for k in sorted(repo)) + "".join(html[k] for k in sorted(html))
    (out / "PaulGrahamEssays.json").write_text(json.dumps({"text": text}))
    print(f"essays: {len(repo)} repo + {len(html)} html of {len(urls)} URLs, "
          f"{len(failed)} failed")
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ruler_data.data_dir())
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    failed = build_essays(out)
    if failed:
        print("essay fetch failures (not pinning a corpus with holes):\n  "
              + "\n  ".join(failed))
        return 1
    (out / "squad.json").write_bytes(_get(SQUAD_URL, timeout=300))
    for url in HOTPOT_URLS:
        try:
            (out / "hotpotqa.json").write_bytes(_get(url, timeout=600))
            hotpot_url = url
            break
        except Exception as e:
            print(f"hotpotqa: {url} failed ({e})")
    else:
        print("hotpotqa: every URL failed")
        return 1

    import nltk
    try:
        import certifi                          # nltk's downloader reads this
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    if not nltk.download("punkt_tab", download_dir=str(out / "nltk_data"), quiet=True):
        print("nltk punkt_tab download failed")
        return 1

    manifest = {"ruler_commit": RULER_COMMIT, "essay_url_list": ESSAY_URL_LIST,
                "squad_url": SQUAD_URL, "hotpotqa_url": hotpot_url,
                "essay_fetch_failures": failed, "files": {}}
    for name in ruler_data.PINNED_SHA256:
        p = out / name
        manifest["files"][name] = {"sha256": ruler_data.sha256_of(p),
                                   "bytes": p.stat().st_size}
        print(f"{name}  {manifest['files'][name]['sha256']}  "
              f"{manifest['files'][name]['bytes']} bytes")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
