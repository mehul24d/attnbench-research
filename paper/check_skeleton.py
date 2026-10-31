#!/usr/bin/env python3
"""Static checks on the paper sources. Needs no LaTeX.

  python3 paper/check_skeleton.py           # the rules below
  python3 paper/check_skeleton.py --final   # also refuses any remaining \\stub

Checks (plan Section 6):
  - every \\input{sections/...} in main.tex exists;
  - main.tex loads the generated numbers file through \\NumbersFile and stops
    with an error when it is missing;
  - typeset text has no digits except in identifiers (H1, RQ2, T7.1, P4 and
    the like) and in \\stub{...} markers: numbers come from macros (C10);
  - no em dashes;
  - no "no accuracy loss" or "zero accuracy cost" (rule 1);
  - no assistant or model identifiers anywhere under paper/ (rule 9);
  - every refs.bib entry carries readinfull = {YYYY-MM-DD} (rule 5).
Exit status 1 on any failure, with every failure listed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()

# Rule 9. Lower-case word fragments; matched case-insensitively as words.
BANNED_IDENTIFIERS = ("claude", "anthropic", "opus", "sonnet", "haiku",
                      "fable", "chatgpt", "gpt-", "gemini", "copilot")
BANNED_PHRASES = ("no accuracy loss", "zero accuracy cost")
IDENTIFIER = re.compile(r"\b(?:RQ|[HGKPTC])\d+(?:\.\d+)*\b")


def strip_comments(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def strip_braced(tex: str, command: str) -> str:
    """Remove \\command{...} (balanced braces), with any [options] before it."""
    out, i = [], 0
    pat = re.compile(r"\\" + command + r"\s*(\[[^\]]*\])?\s*\{")
    while True:
        m = pat.search(tex, i)
        if m is None:
            out.append(tex[i:])
            return "".join(out)
        out.append(tex[i:m.start()])
        depth, j = 1, m.end()
        while j < len(tex) and depth:
            depth += {"{": 1, "}": -1}.get(tex[j], 0)
            j += 1
        i = j


def typeset_text(tex: str) -> str:
    tex = strip_comments(tex)
    for cmd in ("stub", "label", "ref", "eqref", "cite", "input",
                "paperfigure", "includegraphics", "usepackage",
                "documentclass", "graphicspath", "providecommand",
                "newcommand", "IfFileExists", "begin", "end", "url"):
        tex = strip_braced(tex, cmd)
    tex = re.sub(r"\\begin\{[^}]*\}\[[^\]]*\]", "", tex)
    tex = re.sub(r"\\[A-Za-z@]+", " ", tex)       # remaining command names
    return IDENTIFIER.sub(" ", tex)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true",
                    help="also refuse any \\stub left in the sources")
    args = ap.parse_args()
    errors: list[str] = []

    main_tex = PAPER / "main.tex"
    main_src = main_tex.read_text()
    tex_files = [main_tex] + sorted((PAPER / "sections").glob("*.tex"))

    for name in re.findall(r"\\input\{(sections/[^}]+)\}", strip_comments(main_src)):
        path = PAPER / (name if name.endswith(".tex") else name + ".tex")
        if not path.is_file():
            errors.append(f"main.tex inputs {name}, which does not exist")

    body = strip_comments(main_src)
    if r"\input{\NumbersFile}" not in body or r"\errmessage" not in body:
        errors.append("main.tex must input \\NumbersFile and stop with "
                      "\\errmessage when it is missing")

    for path in tex_files:
        rel = path.relative_to(PAPER)
        src = path.read_text()
        # In main.tex only the document body is typeset text.
        body_src = src.split(r"\begin{document}", 1)[-1] if path == main_tex else src
        text = typeset_text(body_src)
        for lineno, line in enumerate(text.splitlines(), 1):
            if re.search(r"\d", line):
                errors.append(f"{rel}: digit in typeset text near "
                              f"'{line.strip()[:60]}'; use a macro from the numbers file")
        if "---" in strip_comments(src) or "\u2014" in src:
            errors.append(f"{rel}: em dash")
        low = src.lower()
        for phrase in BANNED_PHRASES:
            if phrase in low:
                errors.append(f"{rel}: banned phrase '{phrase}' (rule 1)")
        if args.final and re.search(r"\\stub\s*\{", strip_comments(src)):
            errors.append(f"{rel}: \\stub left in a final build")

    for path in sorted(p for p in PAPER.rglob("*")
                       if p.is_file() and p != SELF and "build" not in p.parts
                       and p.suffix in {".tex", ".bib", ".md", ""}):
        low = path.read_text(errors="replace").lower()
        for word in BANNED_IDENTIFIERS:
            if re.search(r"(?<![a-z])" + re.escape(word) + (r"(?![a-z])" if not word.endswith("-") else ""), low):
                errors.append(f"{path.relative_to(PAPER)}: identifier '{word}' (rule 9)")

    bib = strip_comments((PAPER / "refs.bib").read_text())
    entries = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", bib, re.S)
    if bib.count("@") != len(entries):
        errors.append("refs.bib: could not parse every entry; close each with '}' on its own line")
    for key, fields in entries:
        if not re.search(r"\breadinfull\s*=\s*\{\d{4}-\d{2}-\d{2}\}", fields):
            errors.append(f"refs.bib: entry {key} has no readinfull = {{YYYY-MM-DD}} (rule 5)")

    for e in errors:
        print("error:", e, file=sys.stderr)
    if errors:
        return 1
    print(f"paper checks passed: {len(tex_files)} TeX files, {len(entries)} bib entries"
          + (", no stubs" if args.final else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
