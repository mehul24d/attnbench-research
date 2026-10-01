"""External resources for the RULER tasks that need them (audit T4, 2026-10-01).

The noise- and needle-haystack tasks are self-contained. RULER's essay
haystack, word needles and QA tasks are not:

  - `PaulGrahamEssays.json`: the corpus RULER's own script builds by fetching
    the essays (`scripts/fetch_ruler_data.py` reproduces that script). The
    essays are Paul Graham's; RULER does not redistribute them and neither
    does this repository.
  - `squad.json` (SQuAD 2.0 dev) and `hotpotqa.json` (HotpotQA dev,
    distractor setting): CC BY-SA 4.0, fetched from the URLs RULER's
    `download_qa_dataset.sh` uses.
  - NLTK's `punkt_tab` sentence model, for the essay haystack's
    `sent_tokenize`.
  - wonderwords' adjective and noun lists, for word needles (the package
    ships them).

None of the data files is committed. They live in a data directory
(`ATTNBENCH_RULER_DATA`, default `~/.cache/attnbench/ruler_data`), reach the
instance from the project bucket, and are checked against the SHA-256 pins in
`PINNED_SHA256` on every load: an example built from a different corpus is a
different example, and nothing downstream could tell. A file with no pin yet
is refused, not trusted.

Package versions are pinned too (`PINNED_VERSIONS`), because the word lists
and the sentence splitter are part of what an example IS. A different
wonderwords release with a different noun list would generate different
needles under the same seed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

DATA_ENV = "ATTNBENCH_RULER_DATA"
DEFAULT_DIR = Path.home() / ".cache" / "attnbench" / "ruler_data"

# Filled in by scripts/fetch_ruler_data.py's output once the files are built,
# and never edited by hand afterwards. None means "not yet pinned": refused.
# Pinned 2026-10-01 from the first complete build: all 218 of RULER's essay
# URLs (49 repo .txt + 169 paulgraham.com pages, 0 failures; an earlier pass
# that day lost 5 pages to HTTP 500 and was not pinned). SQuAD and HotpotQA
# (Hugging Face fallback; the CMU URL timed out) hashed identically on both
# downloads. MANIFEST.json in the data directory records the URLs.
PINNED_SHA256: dict[str, Optional[str]] = {
    "PaulGrahamEssays.json": "58e352531a80cef2d22c205dbebfbfd64a8afe55a32434de845f200718756c65",
    "squad.json": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
    "hotpotqa.json": "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8",
}
PINNED_VERSIONS = {"wonderwords": "2.2.0", "nltk": "3.9.1"}


class RulerDataError(RuntimeError):
    """A resource is missing, unpinned, or does not match its pin."""


def data_dir() -> Path:
    return Path(os.environ.get(DATA_ENV, DEFAULT_DIR))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verified(name: str, pins: Optional[dict] = None) -> Path:
    pins = PINNED_SHA256 if pins is None else pins
    path = data_dir() / name
    if not path.exists():
        raise RulerDataError(
            f"{path} missing. Build it with scripts/fetch_ruler_data.py, or "
            f"point {DATA_ENV} at a directory that has it.")
    pin = pins.get(name)
    if pin is None:
        raise RulerDataError(
            f"{name} has no SHA-256 pin in ruler_data.PINNED_SHA256. Pin it "
            f"from fetch_ruler_data.py's output before generating examples "
            f"from it.")
    got = sha256_of(path)
    if got != pin:
        raise RulerDataError(f"{name}: sha256 {got} != pinned {pin}")
    return path


def _check_version(pkg: str, version: str) -> None:
    want = PINNED_VERSIONS[pkg]
    if version != want:
        raise RulerDataError(f"{pkg} {version} installed; examples are pinned to {want}")


@lru_cache(maxsize=1)
def essay_words() -> tuple[str, ...]:
    """Upstream: `re.sub(r'\\s+', " ", essay).split(" ")`."""
    text = json.loads(_verified("PaulGrahamEssays.json").read_text())["text"]
    return tuple(re.sub(r"\s+", " ", text).split(" "))


@lru_cache(maxsize=1)
def sent_tokenizer() -> Callable[[str], list]:
    import nltk
    _check_version("nltk", nltk.__version__)
    nltk_dir = data_dir() / "nltk_data"
    if not (nltk_dir / "tokenizers" / "punkt_tab").exists():
        raise RulerDataError(f"{nltk_dir}/tokenizers/punkt_tab missing (fetch_ruler_data.py)")
    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))
    from nltk.tokenize import sent_tokenize
    return sent_tokenize


def word_pool_from(adjs, nouns) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Sorted, de-duplicated lists, and a refusal if any two (adj, noun) pairs
    join to the same string -- the condition under which choosing them
    independently stops being upstream's uniform choice over joined strings."""
    adjs, nouns = tuple(sorted(set(adjs))), tuple(sorted(set(nouns)))
    adj_set, noun_set = set(adjs), set(nouns)
    # (adj, noun) and (adj + "-" + head, tail) join identically whenever
    # noun == head + "-" + tail. Every hyphen split of every noun is checked.
    for noun in nouns:
        parts = noun.split("-")
        for k in range(1, len(parts)):
            head, tail = "-".join(parts[:k]), "-".join(parts[k:])
            if tail not in noun_set:
                continue
            for adj in adjs:
                if f"{adj}-{head}" in adj_set:
                    raise RulerDataError(f"word-pair collision: {adj}-{noun}")
    return adjs, nouns


@lru_cache(maxsize=1)
def word_pool() -> tuple[tuple[str, ...], tuple[str, ...]]:
    import wonderwords
    from wonderwords import random_word
    _check_version("wonderwords", getattr(wonderwords, "__version__", "?"))
    return word_pool_from(random_word._get_words_from_text_file("adjectivelist.txt"),
                          random_word._get_words_from_text_file("nounlist.txt"))


@lru_cache(maxsize=2)
def qa_dataset(name: str) -> tuple[list, list]:
    from .._vendor.ruler import qa
    if name == "squad":
        return qa.read_squad(json.loads(_verified("squad.json").read_text()))
    if name == "hotpotqa":
        return qa.read_hotpotqa(json.loads(_verified("hotpotqa.json").read_text()))
    raise ValueError(f"unknown QA dataset {name!r}")
