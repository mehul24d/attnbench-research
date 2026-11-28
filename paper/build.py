#!/usr/bin/env python3
"""Build the paper and check it for ISPASS 2027 (paper skeleton wave, 29 Nov 2026).

    python3 paper/build.py                  # draft: \\stub and \\fact allowed
    python3 paper/build.py --submission     # also refuses any \\stub or \\fact

Steps, in order; the report goes to paper/build/BUILD.md and to stdout:

  1. the static checks (check_skeleton.py, with --final for a submission);
  2. the generated tables and figures are current: scripts/paper_figures.py is
     rerun on the committed inputs into a scratch directory and every output
     must equal the committed copy byte for byte;
  3. compile with Tectonic (XeTeX); a draft is compiled twice, with the
     drafting facts shown and hidden, so the layout of everything else can be
     measured on its own;
  4. page count: the page of the label paper:bodyend (set after the last body
     section, before the references) against ISPASS's nine pages, references
     extra;
  5. identifier scan (scripts/artifact_scan.py, literals from
     ARTIFACT_SCAN_LITERALS, never printed):
       - ADDED LINES ONLY for the branch's diff against its base (default: the
         merge base with origin/main), working tree included, plus untracked
         files whole, plus the branch's commit messages;
       - WHOLE FILE for what is handed out: the built PDF's text and metadata,
         and paper/generated and paper/figures;
     The artifact keeps its own whole-file scan (scripts/artifact_scan.py on
     the artifact directory);
  6. identity hints for anonymisation, counted with their pages: repository
     and owner names, branch and tag names, repository paths, the artifact
     folder name, commit hashes. Counts and pages only, never the owner name.

The exit code is 0 only if every check passed (a draft's page count over the
limit is reported, not refused; a submission's is refused).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
BUILD = PAPER / "build"
PAGE_LIMIT = 9


def git(*args, check=True) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=check).stdout


def gitb(*args) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pdf_text(path: Path) -> list[str]:
    import pypdf
    return [p.extract_text() or "" for p in pypdf.PdfReader(str(path)).pages]


def pdf_meta(path: Path) -> dict:
    import pypdf
    return {k: str(v) for k, v in (pypdf.PdfReader(str(path)).metadata or {}).items()}


# ------------------------------------------------------------------ steps

def step_checks(final: bool) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, str(PAPER / "check_skeleton.py")] + (["--final"] if final else []),
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    return r.returncode == 0, out[-1] if r.returncode == 0 else f"{len(out)} errors, first: {out[0]}"


def step_current() -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, str(ROOT / "scripts/paper_figures.py"), "--out", d],
                           capture_output=True, text=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if r.returncode:
            return False, f"paper_figures.py refused: {(r.stderr or r.stdout).strip()[:200]}"
        diff = []
        for f in sorted(Path(d).rglob("*")):
            if f.is_file():
                rel = f.relative_to(d)
                mine = PAPER / rel
                if not mine.is_file() or mine.read_bytes() != f.read_bytes():
                    diff.append(str(rel))
        n = sum(1 for f in Path(d).rglob("*") if f.is_file())
    return not diff, (f"{n} files identical to a fresh run" if not diff
                      else f"stale or missing: {', '.join(diff)}")


def compile_once(show_facts: bool, tag: str) -> tuple[Path | None, str]:
    exe = shutil.which("tectonic")
    if exe is None:
        return None, "tectonic not found (brew install tectonic), or use the Makefile with TeX Live"
    out = BUILD / tag
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    src = PAPER / "main.tex"
    if not show_facts:
        # A wrapper beside main.tex, so relative inputs resolve the same way.
        src = PAPER / "_build_nofacts.tex"
        src.write_text("\\def\\ShowFacts{0}\\input{main.tex}\n")
    try:
        r = subprocess.run([exe, "--keep-intermediates", "--keep-logs", "--outdir", str(out), src.name],
                           cwd=PAPER, capture_output=True, text=True)
    finally:
        if not show_facts:
            src.unlink(missing_ok=True)
    pdf = out / (src.stem + ".pdf")
    if r.returncode or not pdf.is_file():
        tail = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith(("!", "error"))]
        return None, f"compile failed: {tail[:3]}"
    warn = len(re.findall(r"LaTeX Warning: (?:Reference|Citation).*undefined", (out / (src.stem + ".log")).read_text(errors="replace")))
    return pdf, f"compiled{'' if not warn else f', {warn} undefined references'}"


def page_count(pdf: Path) -> tuple[int, int]:
    """(last body page, total pages). Floats can land after the last body
    text, so without references the body is every page; with references it
    ends on the page where they start (the label paper:bodyend sits just
    before them; a float LaTeX holds past that point is not counted)."""
    aux = pdf.with_suffix(".aux").read_text(errors="replace")
    m = re.search(r"\\newlabel\{paper:bodyend\}\{\{[^}]*\}\{(\d+)\}", aux)
    if not m:
        raise SystemExit("paper:bodyend label not found in the .aux file")
    import pypdf
    total = len(pypdf.PdfReader(str(pdf)).pages)
    has_refs = "\\bibdata" in aux
    return (int(m.group(1)) if has_refs else total), total


def added_lines_tree(base: str, dst: Path) -> int:
    """The branch's added lines, untracked files whole, and its commit messages."""
    n = 0
    diff = gitb("diff", "--unified=0", "--no-color", "--no-ext-diff", base)
    cur, buf = None, {}
    for line in diff.split(b"\n"):
        if line.startswith(b"+++ "):
            cur = line[4:].decode(errors="replace").removeprefix("b/")
        elif line.startswith(b"+") and cur:
            buf.setdefault(cur, []).append(line[1:])
    # Binary files git cannot diff line by line: the new copy whole (PDFs as text).
    for row in git("diff", "--numstat", "--diff-filter=AM", base).splitlines():
        added, _, rel = row.split("\t", 2)
        p = ROOT / rel
        if added == "-" and p.is_file():
            buf[rel] = [("\n".join(pdf_text(p))).encode() if p.suffix == ".pdf" else p.read_bytes()]
    for rel in git("ls-files", "--others", "--exclude-standard").split():
        p = ROOT / rel
        if p.is_file():
            buf[rel] = ["\n".join(pdf_text(p)).encode()] if p.suffix == ".pdf" else [p.read_bytes()]
    for rel, lines in buf.items():
        f = dst / "added" / (rel + ".added")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"\n" + b"\n".join(lines) + b"\n")
        n += 1
    msgs = gitb("log", "--format=%B%n", f"{base}..HEAD")
    (dst / "commit_messages.txt").write_bytes(b"\n" + msgs)
    return n


def whole_file_tree(pdf: Path, dst: Path) -> int:
    (dst / "pdf").mkdir(parents=True)
    (dst / "pdf" / "main.pdf.txt").write_text("\n".join(pdf_text(pdf)))
    (dst / "pdf" / "main.pdf.meta.txt").write_text("\n".join(f"{k}: {v}" for k, v in pdf_meta(pdf).items()))
    n = 2
    for sub in ("generated", "figures"):
        for f in sorted((PAPER / sub).iterdir()):
            if f.is_file():
                t = dst / sub / (f.name + (".txt" if f.suffix == ".pdf" else ""))
                t.parent.mkdir(parents=True, exist_ok=True)
                t.write_bytes("\n".join(pdf_text(f)).encode() if f.suffix == ".pdf" else f.read_bytes())
                n += 1
    return n


def step_scan(pdf: Path, base: str) -> tuple[bool, list[str]]:
    scan = load("artifact_scan", ROOT / "scripts/artifact_scan.py")
    lits = [x.strip().encode() for x in os.environ.get("ARTIFACT_SCAN_LITERALS", "").splitlines() if x.strip()]
    lines, ok = [f"Literals given: {len(lits)} (values never printed)."], bool(lits)
    if not lits:
        lines.append("**No literals given: patterns only. Set ARTIFACT_SCAN_LITERALS.**")
    with tempfile.TemporaryDirectory() as d:
        a, w = Path(d) / "added", Path(d) / "whole"
        na = added_lines_tree(base, a)
        nw = whole_file_tree(pdf, w)
        for name, root, n in (("added lines since base", a, na), ("whole files handed out", w, nw)):
            res = scan.scan(root, lits)
            hits = {k: v for k, v in res["hits"].items() if v}
            ok &= not hits
            lines.append(f"- {name} ({n} files): " + ("CLEAN" if not hits else
                         "HITS " + "; ".join(f"{k} in {sorted(v)}" for k, v in hits.items())))
    return ok, lines


def step_hints(pdf: Path) -> list[str]:
    """Where the typeset paper could identify its authors. Counts and pages."""
    pages = pdf_text(pdf)
    remote = git("remote", "get-url", "origin", check=False).strip()
    m = re.search(r"[:/]([^/:]+)/([^/]+?)(?:\.git)?$", remote)
    owner, repo = (m.group(1), m.group(2)) if m else (None, None)
    branches = {b.strip().removeprefix("origin/") for b in
                git("for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes").split()}
    branches = {b for b in branches if len(b) > 6 and b not in {"main", "origin", "HEAD"}}
    tags = set(git("tag").split())
    cats = {
        "repository name": [repo, "attnbench"] if repo else ["attnbench"],
        "GitHub owner name (not printed)": [owner] if owner else [],
        "branch names": sorted(branches),
        "lock tag names": sorted(tags),
        "artifact folder name": ["attnbench_artifact_"],
    }
    rx = {
        "repository paths": re.compile(r"\b(?:docs|scripts|attnbench|paper|results|tests|tools|configs)/[\w./-]+"),
        "commit hashes (7-40 hex, resolvable)": re.compile(r"\b[0-9a-f]{7,40}\b"),
    }
    out = []
    for cat, needles in cats.items():
        found = {}
        for i, t in enumerate(pages, 1):
            flat = re.sub(r"\s+", "", t)
            for nd in needles:
                if nd and (nd in t or nd in flat):
                    found.setdefault(nd if "owner" not in cat else "<owner>", set()).add(i)
        out.append(f"- {cat}: " + ("none" if not found else
                   "; ".join(f"`{k}` p.{','.join(map(str, sorted(v)))}" for k, v in sorted(found.items()))))
    for cat, r in rx.items():
        found = {}
        for i, t in enumerate(pages, 1):
            for mm in r.findall(t):
                if "hash" in cat and subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", mm + "^{commit}"],
                                                    capture_output=True).returncode:
                    continue
                found.setdefault(mm, set()).add(i)
        out.append(f"- {cat}: {len(found)} distinct" + ("" if not found else ", e.g. " + "; ".join(
            f"`{k}` p.{','.join(map(str, sorted(v)))}" for k, v in sorted(found.items())[:6])))
    meta = pdf_meta(pdf)
    out.append("- PDF metadata keys: " + ", ".join(sorted(meta)) +
               (" (no author field)" if "/Author" not in meta else " (**has an author field**)"))
    return out


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--submission", action="store_true")
    ap.add_argument("--base", help="diff base for the added-lines scan (default: merge base with origin/main)")
    a = ap.parse_args(argv)
    base = a.base or git("merge-base", "HEAD", "origin/main").strip()
    BUILD.mkdir(exist_ok=True)
    rep = ["# Paper build report", "",
           f"Commit `{git('rev-parse', '--short', 'HEAD').strip()}`"
           f"{' (uncommitted changes)' if git('status', '--porcelain').strip() else ''}; "
           f"mode {'submission' if a.submission else 'draft'}; scan base `{base[:7]}`.", ""]
    ok = True

    good, msg = step_checks(a.submission)
    ok &= good
    rep.append(f"1. Static checks: {'PASS' if good else 'FAIL'}: {msg}")
    good, msg = step_current()
    ok &= good
    rep.append(f"2. Generated tables and figures current: {'PASS' if good else 'FAIL'}: {msg}")

    pdf, msg = compile_once(True, "main")
    if pdf is None:
        rep.append(f"3. Compile: FAIL: {msg}")
        print("\n".join(rep))
        (BUILD / "BUILD.md").write_text("\n".join(rep) + "\n")
        return 1
    shutil.copy(pdf, PAPER / "main.pdf")
    body, total = page_count(pdf)
    rep.append(f"3. Compile: PASS: {msg}; paper/main.pdf")
    over = body > PAGE_LIMIT
    line = (f"4. Pages: body ends on page {body} of {PAGE_LIMIT} allowed; {total} pages in all "
            f"(references {'none yet' if total == body else f'on pages {body}-{total}'})")
    if not a.submission:
        nf, msg2 = compile_once(False, "nofacts")
        if nf is not None:
            b2, _ = page_count(nf)
            line += f". Without the drafting facts: body ends on page {b2}"
        else:
            line += f". Facts-hidden build failed: {msg2}"
    rep.append(line + (" **OVER THE LIMIT**" if over else "") + ".")
    if over and a.submission:
        ok = False

    good, lines = step_scan(pdf, base)
    ok &= good
    rep.append(f"5. Identifier scan: {'PASS' if good else 'FAIL'}")
    rep += ["   " + l for l in lines]
    rep.append("6. Identity hints in the typeset PDF (for anonymisation):")
    rep += ["   " + l for l in step_hints(pdf)]
    rep += ["", f"Result: {'PASS' if ok else 'FAIL'}"]
    (BUILD / "BUILD.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
