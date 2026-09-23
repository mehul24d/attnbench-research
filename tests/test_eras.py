"""The mask-era register, and the refusal the comparison scripts now carry.

**What was wrong.** `docs/audit_register.md` (S12), `docs/claims.md` and
`docs/limitations.md` all said a cross-era comparison is "precisely what
`analysis/composition.py` refuses". It is not. `composition.py` refuses on
facet composition -- whether both groups contain the same batch levels -- and
has no concept of a commit, a date or a mask rule. The S12 disposition rested
partly on a guard that did not exist.

**What this file holds to.** Two things, and the second is the one that rots:

  1. The refusal fires. `licence()` must refuse the real era-2/era-3 pair, and
     must refuse a declaration that names the eras wrongly -- otherwise
     `--cross-era` would be a way of asserting provenance rather than
     disclosing it.
  2. The register agrees with the documented table. `eras.COMMIT_ERA` is a
     hand-written dictionary, and a hand-written dictionary beside a prose
     table is the exact shape of every defect the 2026-09-21 audit found. So
     it is not trusted: the table in `limitations.md` names the FILES in each
     era, the banked parquets carry the COMMITS, and the join of the two is
     what the register has to equal.

The derivation runs only where the full banked tree is present, and the
cross-check of the refusal itself does not -- so a fresh clone still proves
the guard fires, and a working tree additionally proves it is pointed at the
right commits.

**Not `results/` existing (instance #52).** `results/` is gitignored but
carries a handful of force-committed evidence files (`git ls-files results/`),
so it exists in every clone. The first version of this guard checked bare
existence, passed here because the full tree happens to also be present
locally, and failed the first time it ran anywhere else -- see
`_full_results_tree_present()` below.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from attnbench.analysis import eras  # noqa: E402

LIMITATIONS = REPO / "docs" / "limitations.md"
RESULTS = REPO / "results"


def _args(cross_era=None, reason=None) -> argparse.Namespace:
    return argparse.Namespace(cross_era=cross_era, cross_era_reason=reason)


def _side(label: str, *commits: str) -> eras.Side:
    return eras.Side(label, tuple(commits))


# --- the refusal, shown to fire --------------------------------------------

REAL_ERA3 = "39e1d6de0d72ee587dd585d74eedb628884079bf"   # results/s7_jitter
REAL_ERA2 = "35ac0808d37736b49fafaa2c18943e6dbcdaecd7"   # results/s7_7b_16384


def test_the_real_era_2_era_3_pair_is_refused():
    """The pair the S12 row is about, by full hash as banked."""
    with pytest.raises(SystemExit) as e:
        eras.licence(_side("small", REAL_ERA3), _side("large", REAL_ERA2),
                     _args())
    assert "different mask eras" in str(e.value)
    assert "--cross-era era3:era2" in str(e.value), (
        "the refusal must name the flag that would license it, with the "
        "labels filled in; a refusal that only says no gets worked around "
        "rather than declared")


def test_the_declared_comparison_is_permitted():
    head = eras.licence(_side("small", REAL_ERA3), _side("large", REAL_ERA2),
                        _args("era3:era2", "S12 scoping: how far the gap moves"))
    joined = "\n".join(head)
    assert "CROSS-ERA: era3 vs era2" in joined
    assert "S12 scoping" in joined, "the reason must reach the output"
    assert "read across that boundary" in joined


def test_a_declaration_that_names_the_eras_wrongly_is_refused():
    """The flag discloses; it does not assert. Swapping the two labels on a
    pair whose eras are both known must not pass."""
    with pytest.raises(SystemExit) as e:
        eras.licence(_side("small", REAL_ERA3), _side("large", REAL_ERA2),
                     _args("era2:era3", "swapped on purpose, must be caught"))
    assert "declares small as era2" in str(e.value)
    assert "era3 in the register" in str(e.value)


def test_a_declaration_without_a_reason_is_refused():
    with pytest.raises(SystemExit) as e:
        eras.licence(_side("small", REAL_ERA3), _side("large", REAL_ERA2),
                     _args("era3:era2"))
    assert "--cross-era-reason" in str(e.value)


@pytest.mark.parametrize("bad", ["era3", "era3:", "era3:era9", "3:2", "yes"])
def test_a_malformed_declaration_is_refused(bad):
    with pytest.raises(SystemExit):
        eras.licence(_side("small", REAL_ERA3), _side("large", REAL_ERA2),
                     _args(bad, "a reason long enough to pass that check"))


def test_same_commit_is_permitted_silently():
    head = eras.licence(_side("small", REAL_ERA2), _side("large", REAL_ERA2),
                        _args())
    assert not any("REFUS" in line for line in head)
    assert not any("CROSS-ERA" in line for line in head)


def test_different_commits_in_the_same_era_are_permitted():
    """`results/s1a/accuracy_all_bands.parquet` carries two commits because it
    is two sessions concatenated, and the published scorer comparison sets
    `accuracy_forced_sink` against `accuracy_forced_sink_cheap`. Both are
    era 2 throughout. A check that refused on "the commit sets differ" would
    refuse the very comparison the study publishes."""
    head = eras.licence(_side("oracle", "166b2df"), _side("cheap", "c525f75"),
                        _args())
    assert any("both era2" in line for line in head)


def test_an_unknown_commit_cannot_license_by_being_unrecognised():
    """The failure mode of every allowlist: the thing not on the list is
    treated as fine. An unknown commit means the era cannot be decided, which
    is a refusal, not a pass."""
    with pytest.raises(SystemExit) as e:
        eras.licence(_side("small", "cafef00dcafef00d"),
                     _side("large", REAL_ERA2), _args())
    assert "cannot be decided" in str(e.value)
    assert "cafef00" in str(e.value)


def test_an_unknown_commit_may_still_be_declared_but_is_flagged_unverified():
    head = eras.licence(_side("small", "cafef00dcafef00d"),
                        _side("large", REAL_ERA2),
                        _args("era3:era2", "a later run not yet registered"))
    assert any("NOT VERIFIED" in line for line in head), (
        "a label that could not be checked must say so, or the stamp claims "
        "more than the register knows")


# --- the register against the documented table -----------------------------

def documented_eras() -> dict[str, int]:
    """file key -> era, parsed from the era table in `limitations.md`."""
    text = LIMITATIONS.read_text()
    block = re.search(
        r"### Which mask era each banked accuracy file belongs to\n(.*?)\n#",
        text, re.S)
    assert block, "the era table's heading is no longer in limitations.md"
    out: dict[str, int] = {}
    for line in block.group(1).splitlines():
        m = re.match(r"\|\s*\*\*(\d)\.", line)
        if not m:
            continue
        era = int(m.group(1))
        # Only the file list, which runs to the first em dash. Era 3's cell
        # continues past one into prose that names `accuracy_forced_sink`
        # (an era-2 file whose niah cells it supersedes) and two task names.
        # Reading the whole cell assigns that file to era 3, which is how the
        # first run of this test failed -- correctly, on its own parsing.
        cell = line.split("|")[3].split("\u2014")[0]
        for name in re.findall(r"`([A-Za-z0-9_/]+)`", cell):
            if out.get(name, era) != era:
                pytest.fail(
                    f"the era table lists `{name}` under both era "
                    f"{out[name]} and era {era}. A file has one era; if it "
                    f"is genuinely split, the split belongs in prose and the "
                    f"cell should name only the file that IS that era.")
            out[name] = era
    return out


def _banked_files(root: Path | None = None) -> dict[str, list[Path]]:
    """file key -> the parquet paths that contribute rows to it.

    `root` is injectable so the break-test below can build the exact partial
    checkout that defeated the previous guard, rather than relying on someone
    having constructed one by hand once.

    The single source of both the skip condition and the assertion below, so
    the two cannot describe different quantities. That is not hypothetical:
    until 2026-09-23 the skip counted `rglob("accuracy.parquet")` PATHS while
    the assertion counted `banked_commits()` KEYS, and the tree holds more
    paths than keys because `results/gpu_session_2026090{6,7,8}*/results/`
    carries archived copies of `stage3_s1`, `stage3_s1b`, `stage3_32768`,
    `stage3_flashdecode` and `stage3_16384` that collapse onto the canonical
    key. 19 paths, 14 keys. A partial checkout holding only the session
    archives reaches 9 paths -- clearing a `> 8` PATH floor -- while offering
    6 keys, so the guard passed and the assertion failed. See instance #53.
    """
    base = RESULTS if root is None else root
    found: dict[str, list[Path]] = {}
    paths = list(base.rglob("accuracy.parquet"))
    paths += list((base / "s1a").glob("accuracy*.parquet"))
    for p in sorted(paths):
        key = (f"s1a/{p.stem}" if p.parent.name == "s1a" else p.parent.name)
        found.setdefault(key, []).append(p)
    return found


# The floor, named once. Both the skip condition and the assertion inside
# `test_the_register_agrees_...` read it, so "enough banked data to run" and
# "enough banked data to have proved anything" are the same number.
MIN_BANKED_KEYS = 8


def banked_commits() -> dict[str, set[str]]:
    """file key -> the commits its rows carry."""
    pd = pytest.importorskip("pandas")
    found: dict[str, set[str]] = {}
    for key, paths in _banked_files().items():
        for p in paths:
            d = pd.read_parquet(p, columns=["git_commit"])
            found.setdefault(key, set()).update(
                str(c) for c in d.git_commit.dropna().unique())
    return found


def _full_results_tree_present() -> bool:
    """True only when the banked accuracy data itself is present, not merely
    when `results/` exists (instance #52) and not merely when enough FILES
    named `accuracy.parquet` exist (instance #53).

    Counts the same keys `banked_commits()` will build, from paths alone so it
    needs no pandas and is cheap enough to call from a `skipif` condition.
    """
    return len(_banked_files()) > MIN_BANKED_KEYS


def _partial_checkout(root: Path) -> None:
    """The checkout that defeated the previous guard, built rather than
    described: a clone's nine force-committed evidence files plus ONLY the
    archived `gpu_session_*` copies -- what you get from a partial bucket sync
    or one session's tarball. Nine paths named `accuracy.parquet`, six keys.
    """
    for rel in (
        # the one tracked accuracy.parquet a clone always has
        "s7_7b_16384/accuracy.parquet",
        # archived duplicates: five distinct keys across eight paths
        "gpu_session_20260906_stage3_s1/results/stage3_s1/accuracy.parquet",
        "gpu_session_20260907/results/stage3_s1/accuracy.parquet",
        "gpu_session_20260907/results/stage3_s1b/accuracy.parquet",
        "gpu_session_20260908/results/stage3_s1/accuracy.parquet",
        "gpu_session_20260908/results/stage3_s1b/accuracy.parquet",
        "gpu_session_20260908/results/stage3_32768/accuracy.parquet",
        "gpu_session_20260908/results/stage3_flashdecode/accuracy.parquet",
        "gpu_session_20260908_band16384/stage3_16384/accuracy.parquet",
    ):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")


def test_the_skip_condition_counts_keys_not_paths(tmp_path):
    """Break-test for instance #53, permanent rather than performed once.

    The previous guard counted paths and would have run the derivation against
    six keys; this one counts keys and skips. Both halves are asserted, so
    reverting `_banked_files` to a path count turns this red.
    """
    _partial_checkout(tmp_path)

    paths = list(tmp_path.rglob("accuracy.parquet"))
    keys = _banked_files(tmp_path)
    assert len(paths) == 9, [str(p) for p in paths]
    assert len(keys) == 6, sorted(keys)

    # The defeated guard: a PATH count clears the floor on this tree.
    assert len(paths) > MIN_BANKED_KEYS, (
        "the fixture no longer reproduces instance #53 -- it must hold MORE "
        "paths than the floor, or it is not the state that broke the guard")
    # The guard that holds: a KEY count does not.
    assert not len(keys) > MIN_BANKED_KEYS, (
        f"{len(keys)} keys clears the {MIN_BANKED_KEYS} floor, so this tree "
        f"would RUN the era derivation against a partial checkout instead of "
        f"skipping it. That is instance #53.")


def test_the_skip_condition_admits_the_real_tree():
    """Anti-vacuity for the break-test above: a guard that skipped everywhere
    would also pass it. Runs only where the real tree is present."""
    if not RESULTS.exists() or len(_banked_files()) <= MIN_BANKED_KEYS:
        pytest.skip("no full banked tree here to admit")
    keys, paths = _banked_files(), list(RESULTS.rglob("accuracy.parquet"))
    assert len(keys) > MIN_BANKED_KEYS
    assert len(paths) > len(keys), (
        f"{len(paths)} paths, {len(keys)} keys -- if these are ever equal the "
        f"archived-duplicate shape is gone and the distinction this guard "
        f"draws no longer has a live example in the tree")


@pytest.mark.skipif(not _full_results_tree_present(),
                    reason="only a handful of force-committed evidence files, "
                           "not the full banked results/ tree, is present in "
                           "this checkout")
def test_the_register_agrees_with_the_documented_era_table():
    """The join that makes `eras.COMMIT_ERA` a derivation rather than a
    restatement: the table says which era each FILE is in, the parquet says
    which COMMIT it was produced at."""
    documented, banked = documented_eras(), banked_commits()
    assert len(documented) > 10, f"only {len(documented)} files in the table"
    assert len(banked) > MIN_BANKED_KEYS, (
        f"only {len(banked)} banked files found, at or below the "
        f"{MIN_BANKED_KEYS} floor the skip condition uses. If this fires, the "
        f"skip condition and this assertion have drifted apart again -- they "
        f"read the same `_banked_files()` precisely so they cannot.")

    checked = 0
    for key, commits in sorted(banked.items()):
        if key not in documented:
            pytest.fail(
                f"{key} carries banked accuracy rows and is in no row of the "
                f"era table in limitations.md. Every banked file has an era; "
                f"an unlisted one is a comparison waiting to happen.")
        for c in commits:
            got = eras.era_of(c)
            assert got == documented[key], (
                f"{key} is era {documented[key]} in limitations.md, but "
                f"commit {c[:7]} resolves to "
                f"{'nothing' if got is None else f'era {got}'} in "
                f"attnbench/analysis/eras.py.")
            checked += 1
    assert checked >= 13, f"only {checked} commit/era pairs cross-checked"


@pytest.mark.skipif(not _full_results_tree_present(),
                    reason="only a handful of force-committed evidence files, "
                           "not the full banked results/ tree, is present in "
                           "this checkout")
def test_no_registered_commit_is_absent_from_the_banked_data():
    """The other direction: a register entry for a commit no file carries is
    either a typo or a file that has been deleted, and both make the register
    look more complete than it is."""
    banked = banked_commits()
    assert len(banked) > MIN_BANKED_KEYS, (
        f"only {len(banked)} banked files found; this direction of the check "
        f"gets STRICTER as data goes missing, so without a floor a partial "
        f"tree makes it fail rather than skip -- which is exactly how "
        f"instance #53 presented.")
    seen = {c[:7] for commits in banked.values() for c in commits}
    stale = sorted(k for k in eras.COMMIT_ERA if k not in seen)
    assert not stale, (
        f"eras.COMMIT_ERA registers {stale}, which no banked accuracy file "
        f"carries.")


# --- the scripts actually call it ------------------------------------------

ERA3_FILE = RESULTS / "s7_jitter" / "accuracy.parquet"
ERA2_FILE = RESULTS / "s7_7b_16384" / "accuracy.parquet"

SCRIPTS = [
    ("run_scale_comparison.py", "--small", "--large"),
    ("run_scorer_comparison.py", "--oracle", "--cheap"),
]


@pytest.mark.skipif(not (ERA3_FILE.exists() and ERA2_FILE.exists()),
                    reason="the era-2/era-3 pair is not present in this checkout")
@pytest.mark.parametrize("script,left,right", SCRIPTS)
def test_the_script_refuses_the_era_boundary_end_to_end(script, left, right):
    """Not the module -- the command line. A guard wired into a helper and
    never called from the script is the vacuous-by-construction shape this
    project has now been caught by twice."""
    cmd = [sys.executable, str(REPO / "scripts" / script),
           left, str(ERA3_FILE), right, str(ERA2_FILE), "--band", "16384"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0, (
        f"{script} exited 0 on an era-2/era-3 pair:\n{r.stdout[-2000:]}")
    assert "different mask eras" in (r.stdout + r.stderr), (
        f"{script} failed for some other reason:\n{r.stderr[-2000:]}")

    ok = subprocess.run(
        cmd + ["--cross-era", "era3:era2",
               "--cross-era-reason", "break-test of the era guard"],
        capture_output=True, text=True, cwd=REPO)
    assert "CROSS-ERA: era3 vs era2" in ok.stdout, (
        f"{script} did not stamp the declaration onto its output:\n"
        f"{ok.stdout[-2000:]}{ok.stderr[-2000:]}")


# --- the boundary, derived rather than restated -----------------------------
#
# `COMMIT_ERA` was hand-written and checked against the era TABLE in
# limitations.md joined to the banked parquets. Both are prose-and-data; the
# eras are defined by two commits, and nothing compared the register to the
# commit graph. It happened to be correct in all thirteen entries -- verified
# 2026-09-21 -- but correct by care is not the same as correct by
# construction, and the docstring sitting above it was copied from a
# paragraph that was wrong. This closes that.

HAS_GIT = eras.era_from_git(REAL_ERA3, repo=str(REPO)) is not None


def disagreements(table: dict) -> list[tuple[str, int, int]]:
    """(commit, era claimed by the table, era the commit graph gives)."""
    out = []
    for commit, claimed in table.items():
        actual = eras.era_from_git(commit, repo=str(REPO))
        if actual is not None and actual != claimed:
            out.append((commit, claimed, actual))
    return out


@pytest.mark.skipif(not HAS_GIT, reason="no usable git history in this checkout")
def test_the_register_agrees_with_git_ancestry():
    """The definition is ancestry of the two boundary commits. The table is a
    cache of it, and a cache that disagrees with its source is worse than no
    cache -- it is the guard confidently returning the wrong era."""
    bad = disagreements(eras.COMMIT_ERA)
    assert not bad, "\n".join(
        f"{c}: register says era {claimed}, the commit graph says era {actual}"
        for c, claimed, actual in bad)


@pytest.mark.skipif(not HAS_GIT, reason="no usable git history in this checkout")
def test_every_registered_commit_is_resolvable_from_the_graph():
    """Anti-vacuity for the check above: `era_from_git` returns None for a
    commit this checkout does not contain, and None is skipped as
    'cannot decide'. If every entry resolved to None the agreement test
    would pass over an empty comparison."""
    resolved = [c for c in eras.COMMIT_ERA
                if eras.era_from_git(c, repo=str(REPO)) is not None]
    assert len(resolved) == len(eras.COMMIT_ERA), (
        f"only {len(resolved)}/{len(eras.COMMIT_ERA)} registered commits are "
        f"in this checkout: "
        f"{sorted(set(eras.COMMIT_ERA) - set(resolved))}")


@pytest.mark.skipif(not HAS_GIT, reason="no usable git history in this checkout")
def test_the_agreement_check_fires_on_a_wrong_entry():
    """Break-test, against the exact misclassification the era-2/era-3
    confusion would produce: the one era-3 file filed as era 2."""
    bad = disagreements({REAL_ERA3[:7]: 2})
    assert bad == [(REAL_ERA3[:7], 2, 3)], bad
    assert disagreements({REAL_ERA3[:7]: 3}) == []


@pytest.mark.skipif(not HAS_GIT, reason="no usable git history in this checkout")
def test_each_boundary_commit_belongs_to_the_era_it_opens():
    """`37675a0` builds forced-sink masks, so it is era 2, not the last of
    era 1. `5cc3a40` draws the jitter unconditionally, so it is era 3."""
    assert eras.era_from_git(eras.SINK_FIX_COMMIT, repo=str(REPO)) == 2
    assert eras.era_from_git(eras.JITTER_FIX_COMMIT, repo=str(REPO)) == 3


@pytest.mark.skipif(not HAS_GIT, reason="no usable git history in this checkout")
def test_the_era_2_era_3_split_is_not_decidable_by_date():
    """Why the paragraph this module used to carry was not merely imprecise.

    It said era 2 and era 3 "split on date only". Three era-2 commits share
    2026-09-20 with the single era-3 commit, so a date rule at day
    granularity gets them wrong; ancestry of `5cc3a40` gets all four right.
    """
    import subprocess
    def day(c):
        return subprocess.run(
            ["git", "log", "-1", "--format=%ad", "--date=short", c],
            cwd=str(REPO), capture_output=True, text=True).stdout.strip()
    era3_day = day(REAL_ERA3)
    same_day_era2 = [c for c, e in eras.COMMIT_ERA.items()
                     if e == 2 and day(c) == era3_day]
    assert same_day_era2, (
        "no era-2 commit shares a date with the era-3 one any more; if the "
        "register changed, re-derive whether a date rule would now work "
        "rather than deleting this test")
    for c in same_day_era2:
        assert eras.era_from_git(c, repo=str(REPO)) == 2


# --- a provenance column present on one side only ---------------------------
#
# Added 2026-09-23. Both comparison scripts checked their comparability columns
# behind `if col in a and col in b`, so a file recording `mask_source` compared
# against one that does not passed in silence. In run_scale_comparison.py the
# check was additionally preceded by a loop whose entire body was `continue`.
# Silence reads as agreement, and these are the columns that exist to stop two
# different experiments being averaged together.

@pytest.mark.skipif(not (ERA3_FILE.exists() and ERA2_FILE.exists()),
                    reason="the era-2/era-3 pair is not present in this checkout")
@pytest.mark.parametrize("script,left,right", SCRIPTS)
def test_the_script_refuses_a_provenance_column_present_on_one_side_only(
        script, left, right, tmp_path):
    """End-to-end, from the command line, against a real banked file with one
    column dropped -- not against the helper."""
    pd = pytest.importorskip("pandas")

    full = pd.read_parquet(ERA3_FILE)
    col = next((c for c in ("mask_source", "block_size", "model_id")
                if c in full.columns), None)
    assert col is not None, f"{ERA3_FILE} records none of the checked columns"

    other = full.drop(columns=[col]).copy()
    if script == "run_scorer_comparison.py":
        # That script refuses two inputs sharing a score_source BEFORE it looks
        # at the comparability columns, and rightly so -- it exists to compare
        # two different scorers. Give it a difference there so the refusal under
        # test is the one being tested and not a louder earlier one.
        other["score_source"] = "cheap_estimator_fp16"

    both = tmp_path / "both.parquet"
    one = tmp_path / "missing.parquet"
    full.to_parquet(both, index=False)
    other.to_parquet(one, index=False)

    cmd = [sys.executable, str(REPO / "scripts" / script),
           left, str(both), right, str(one), "--band", "16384",
           # the same commit on both sides, so the era check permits the pair
           # and whatever refuses has to be the column check
           "--allow-n-mismatch"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    out = r.stdout + r.stderr
    assert r.returncode != 0, (
        f"{script} exited 0 with {col!r} recorded on one side only:\n"
        f"{out[-2000:]}")
    assert "side only" in out and col in out, (
        f"{script} refused for some other reason than the one-sided "
        f"{col!r}:\n{out[-2000:]}")
