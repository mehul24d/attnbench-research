"""Which mask era a result file belongs to, and whether two may be compared.

**The exposure this closes.** `scripts/run_scale_comparison.py` and
`scripts/run_scorer_comparison.py` refuse on `score_source`, `mask_source`,
`block_size`, band, tasks, n and `git_dirty`. Neither looked at `git_commit`.
So the one incomparability the project has *named* -- the mask era -- was the
one the comparison scripts could not see, and a 2026-09-21 audit found three
documents asserting that `analysis/composition.py` "refuses cross-era
comparisons". It does not and cannot: it refuses on *facet composition* (the
batch levels present in each group) and has no concept of a commit. The
protection was cited in the S12 disposition and did not exist.

**Both boundaries are commits, and ancestry decides both.** Era 1 ends at the
sink fix `37675a0` and era 2 ends at the jitter fix `5cc3a40`; a row's era is
fixed by where its own `git_commit` sits relative to those two. This module
said the opposite until 2026-09-21 -- that era 2 and era 3 "split on date
only", so a row could not be assigned between them from its own contents --
which was copied out of `limitations.md` rather than derived, one day after
the guard against exactly that kind of copying was built. It is false twice
over: `5cc3a40` is an ordinary commit, and the date rule it offered instead
does not even work, because three era-2 commits share 2026-09-20 with the
single era-3 one.

`COMMIT_ERA` below is therefore a CACHE of `era_from_git`, not an independent
fact. It exists because the comparison scripts must work in a checkout with
banked results and no git history, and `tests/test_eras.py` requires the two
to agree wherever git is available.

**Why still no `mask_rule` column.** Not because the information is
unavailable -- it is, from `git_commit` -- but because adding the column now
would stamp future rows and leave every banked row unlabelled, and a
half-populated provenance field is worse than none (`provenance.stamp_onto`).
That argument survives. The one about unavailability did not.

**Commits, not files, because the scripts take paths.** A comparison script is
handed two parquets and must decide from their contents alone. Every banked
accuracy file carries `git_commit`, so commit is the join key.
`tests/test_eras.py` derives this mapping from the documented table plus the
banked parquets and asserts it agrees, so the table stays the source of truth
and this dictionary cannot drift from it in silence.

**One commit set is not one era, and one era is not one commit.**
`results/s1a/accuracy_all_bands.parquet` carries TWO commits (`179c894` and
`44ab65c`) because it is the concatenation of two sessions -- both era 2. A
check that refused on "the commit sets differ" alone would refuse that file
against either of its own halves. The question is whether the ERAS differ.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

# The mask rule in force at each commit, from the era table in
# `docs/limitations.md` ("Which mask era each banked accuracy file belongs
# to"). Keys are the short forms the documents use; lookup is by prefix so a
# full 40-character hash from a parquet resolves.
#
#   1  pre-sink     kv block 0 is an ordinary candidate
#   2  forced-sink  kv block 0 granted free (`37675a0`, 2026-09-16)
#   3  post-jitter  as 2, plus jitter drawn unconditionally (2026-09-20)
# Cache of `era_from_git`, not an independent list. Verified against the
# commit graph by `tests/test_eras.py::test_the_register_agrees_with_git_
# ancestry`, which skips only where git is unavailable.
COMMIT_ERA: dict[str, int] = {
    "d27c650": 1,   # stage3_s1
    "1126bd2": 1,   # stage3_s1b
    "8e9e0fd": 1,   # stage3_16384
    "3421f89": 1,   # stage3_32768
    "35d726a": 1,   # stage3_flashdecode
    "166b2df": 2,   # accuracy_forced_sink
    "c525f75": 2,   # accuracy_forced_sink_cheap
    "35ac080": 2,   # s7_7b_16384
    "0397c60": 2,   # s9_7b_cheap_16384
    "179c894": 2,   # s1a/accuracy_band2048
    "44ab65c": 2,   # s1a/accuracy_bands2
    "33598b4": 2,   # sink_control
    "39e1d6d": 3,   # s7_jitter
}

# The two commits the eras are defined BY. Era 1 is everything that is not a
# descendant-or-self of the sink fix; era 3 is everything that is a
# descendant-or-self of the jitter fix; era 2 is what lies between. Both
# boundary commits are themselves in the era they open -- `37675a0` builds
# forced-sink masks, so it is era 2, and `5cc3a40` draws the jitter
# unconditionally, so it is era 3.
SINK_FIX_COMMIT = "37675a0"      # 2026-09-16, kv block 0 granted free
JITTER_FIX_COMMIT = "5cc3a40"    # 2026-09-20, jitter drawn unconditionally

ERA_LABELS = {1: "era1", 2: "era2", 3: "era3"}
LABEL_ERAS = {v: k for k, v in ERA_LABELS.items()}

ERA_RULES = {
    1: "pre-sink (kv block 0 an ordinary candidate)",
    2: "forced-sink (kv block 0 free; jitter after the budget check)",
    3: "post-jitter (forced-sink, plus jitter drawn unconditionally)",
}


class EraRefusal(SystemExit):
    """Raised instead of printing a comparison. A SystemExit so a script that
    forgets to catch it still exits non-zero rather than continuing."""


def era_from_git(commit: str, repo: str | None = None) -> int | None:
    """The era of a commit, derived from the commit graph. None if it cannot
    be decided here -- no git, no repository, or a commit this checkout does
    not contain.

    This is the definition. `COMMIT_ERA` is a cache of it, kept because the
    comparison scripts have to run against banked results in a tree with no
    history, which is the condition both 2026-09 audits ran under and the
    reason the era-2/era-3 boundary went unverified for so long.
    """
    import subprocess

    def run(*args: str) -> int:
        try:
            return subprocess.run(["git", *args], cwd=repo,
                                  capture_output=True).returncode
        except (OSError, ValueError):
            return 128

    if run("rev-parse", "--git-dir") != 0:
        return None
    for c in (commit, SINK_FIX_COMMIT, JITTER_FIX_COMMIT):
        if run("cat-file", "-e", f"{c}^{{commit}}") != 0:
            return None
    if run("merge-base", "--is-ancestor", SINK_FIX_COMMIT, commit) != 0:
        return 1
    if run("merge-base", "--is-ancestor", JITTER_FIX_COMMIT, commit) != 0:
        return 2
    return 3


def era_of(commit: str) -> int | None:
    """The era of a commit, or None if it is not in the register.

    None is the honest answer for a commit produced after this table was
    written, and the callers below treat it as "cannot decide" rather than as
    "same era" -- an unknown commit must not be able to license a comparison
    by being unrecognised.
    """
    c = str(commit).strip()
    for prefix, era in COMMIT_ERA.items():
        if c.startswith(prefix):
            return era
    return None


@dataclass(frozen=True)
class Side:
    label: str
    commits: tuple[str, ...]

    @property
    def eras(self) -> set[int]:
        return {e for e in (era_of(c) for c in self.commits) if e is not None}

    @property
    def unknown(self) -> tuple[str, ...]:
        return tuple(c for c in self.commits if era_of(c) is None)

    def describe(self) -> str:
        shown = ", ".join(sorted({c[:7] for c in self.commits})) or "none recorded"
        eras = "/".join(ERA_LABELS[e] for e in sorted(self.eras)) or "unknown"
        return f"{self.label}: {shown} ({eras})"


def add_cross_era_flags(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--cross-era", metavar="LEFT:RIGHT", default=None,
        help="declare that the two inputs are from different mask eras and "
             "that the comparison is wanted anyway. Must name both eras, e.g. "
             "--cross-era era2:era3. Requires --cross-era-reason.")
    ap.add_argument(
        "--cross-era-reason", metavar="TEXT", default=None,
        help="why a cross-era comparison is admissible here. Printed with "
             "the result, because the number cannot be read without it.")


def _parse_declaration(args: argparse.Namespace) -> tuple[int, int, str]:
    raw = args.cross_era
    parts = raw.split(":")
    if len(parts) != 2 or not all(p.strip() for p in parts):
        raise EraRefusal(
            f"--cross-era must name both eras as LEFT:RIGHT (got {raw!r}). "
            f"Known labels: {', '.join(sorted(LABEL_ERAS))}.")
    left, right = (p.strip().lower() for p in parts)
    for p in (left, right):
        if p not in LABEL_ERAS:
            raise EraRefusal(
                f"--cross-era: {p!r} is not an era label. "
                f"Known labels: {', '.join(sorted(LABEL_ERAS))}. See the era "
                f"table in docs/limitations.md.")
    reason = (args.cross_era_reason or "").strip()
    if len(reason) < 12:
        raise EraRefusal(
            "--cross-era requires --cross-era-reason: a sentence saying why "
            "the comparison is admissible across the boundary. A declaration "
            "with no reason is a flag that gets pasted from the last command "
            "line, which is the failure this check exists to prevent.")
    return LABEL_ERAS[left], LABEL_ERAS[right], reason


def licence(left: Side, right: Side, args: argparse.Namespace) -> list[str]:
    """Header lines for a permitted comparison; raises `EraRefusal` otherwise.

    The order of the checks matters. A declaration is verified against the
    register BEFORE it is allowed to license anything, so `--cross-era` cannot
    be used to mislabel a pair whose eras are known -- otherwise the flag
    would be a way of asserting provenance rather than of disclosing it.
    """
    head = ["commits : " + left.describe(), "          " + right.describe()]

    same_commits = set(left.commits) == set(right.commits)
    both_known = not left.unknown and not right.unknown
    same_era = both_known and left.eras == right.eras and len(left.eras) == 1

    if args.cross_era is None:
        if same_commits:
            return head
        if same_era:
            era = next(iter(left.eras))
            return head + [f"          different commits, both {ERA_LABELS[era]} "
                           f"-- {ERA_RULES[era]}; permitted"]
        if both_known:
            raise EraRefusal(
                f"\nREFUSED: the two inputs are from different mask eras.\n"
                f"  {left.describe()}\n  {right.describe()}\n"
                f"A difference read across an era boundary is not drift and "
                f"is not an effect of the variable under study -- it is the "
                f"mask rule changing underneath the comparison. If the "
                f"comparison is wanted anyway, say so explicitly:\n"
                f"  --cross-era {ERA_LABELS[min(left.eras)]}:"
                f"{ERA_LABELS[min(right.eras)]} "
                f"--cross-era-reason '...'\n"
                f"See the era table in docs/limitations.md.")
        raise EraRefusal(
            f"\nREFUSED: the two inputs carry different commits and at least "
            f"one is not in the era register, so their eras cannot be "
            f"decided.\n  {left.describe()}\n  {right.describe()}\n"
            f"Unknown: {', '.join(sorted(c[:7] for c in left.unknown + right.unknown))}\n"
            f"Add them to attnbench/analysis/eras.py (and to the era table in "
            f"docs/limitations.md, which is where that mapping is derived "
            f"from), or declare the comparison with --cross-era.")

    claimed_l, claimed_r, reason = _parse_declaration(args)

    for side, claimed in ((left, claimed_l), (right, claimed_r)):
        if side.unknown or not side.eras:
            continue
        if side.eras != {claimed}:
            actual = "/".join(ERA_LABELS[e] for e in sorted(side.eras))
            raise EraRefusal(
                f"\nREFUSED: --cross-era declares {side.label} as "
                f"{ERA_LABELS[claimed]}, but its commits "
                f"({', '.join(sorted(c[:7] for c in side.commits))}) are "
                f"{actual} in the register. Declaring the wrong era is worse "
                f"than not declaring one: the output carries the label.")

    if same_commits and claimed_l == claimed_r:
        head.append("          note: --cross-era declared, but both inputs "
                    "carry the same commit. The flag is doing nothing.")

    unverified = [s.label for s in (left, right) if s.unknown]
    stamp = (f"CROSS-ERA: {ERA_LABELS[claimed_l]} vs {ERA_LABELS[claimed_r]} "
             f"-- {reason}")
    head += ["", "*" * 72, stamp,
             f"  {ERA_LABELS[claimed_l]} = {ERA_RULES[claimed_l]}",
             f"  {ERA_LABELS[claimed_r]} = {ERA_RULES[claimed_r]}"]
    if unverified:
        head.append(f"  NOT VERIFIED against the register for: "
                    f"{', '.join(unverified)} (unrecognised commit). The "
                    f"label above is the operator's claim, not a check.")
    head += ["Every number below is read across that boundary.", "*" * 72]
    return head


def sides(a_commits: Iterable[str], b_commits: Iterable[str],
          a_label: str, b_label: str) -> tuple[Side, Side]:
    return (Side(a_label, tuple(sorted(set(map(str, a_commits))))),
            Side(b_label, tuple(sorted(set(map(str, b_commits))))))


def commits_of(frame) -> tuple[str, ...]:
    """`git_commit` values in a result frame, as a tuple of strings."""
    if "git_commit" not in frame:
        return ()
    return tuple(sorted({str(c) for c in frame.git_commit.dropna().unique()}))
