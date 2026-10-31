"""The order between the prediction commit and the first clocked block (T3.1).

Plan Section 5.1 and Appendix A (25 Oct 2026): the prediction file and its
sha256 are synced before the timed workload starts, the workload starts at
least 60 s after the sync, and the analysis allows 60 s of clock skew between
the store's clock and the instance's. Appendix A (27 Oct 2026, strict rule):
the stamp the late-prediction check reads is the start of the first clocked
block of any kind, the dense control's clocked warm-up forwards included.

`commit.load_committed` refuses a file when `synced_at + skew >= start`, with
`synced_at` by the store's clock (an upper bound) and `start` by the
instance's. So nothing clocked may start until the instance clock reads
`synced_at + skew + guard`. The session runner holds there (`hold`), and the
ground-truth process checks the same condition again itself
(`ready_to_clock`), so a process started by hand, or out of order, refuses
too.

**Which stamp.** `ground_truth.time_arms` still stamps the first timed block,
not the clocked warm-up before it (the strict rule's change to T2.13 is
owed). Until it lands, each ground-truth process records `clock_not_before`,
read before its first forward of any kind, and the session's start for the
late check is the earliest stamp of either kind (`session_start`). That is
never later than the strict rule's stamp, so it can only make the check
stricter.

**The guard** is a margin on the instance's own clock, so that the hold's
release is strictly past what the analysis refuses. It is not a skew
estimate: the 60 s allowance is the plan's.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Mapping

from .commit import (PLANNED_SYNC_WAIT_S, REQUIRED_SYNC_GAP_S, SYNC_WAIT_MARGIN_S, SyncReceipt,
                     ground_truth_start)

# The one source of the numbers is `commit` (Appendix A, 25 and 31 Oct 2026).
REQUIRED_SKEW = timedelta(seconds=REQUIRED_SYNC_GAP_S)
PLANNED_GUARD = timedelta(seconds=SYNC_WAIT_MARGIN_S)
PLANNED_WAIT = timedelta(seconds=PLANNED_SYNC_WAIT_S)


class HoldError(RuntimeError):
    """A clocked block would start too soon after the prediction's sync."""


def _aware(t: datetime, name: str) -> datetime:
    if not isinstance(t, datetime) or t.tzinfo is None:
        raise HoldError(f"{name} must be a timezone-aware datetime, got {t!r}")
    return t


def receipt_from_dict(d: Mapping) -> SyncReceipt:
    """`SyncReceipt.to_dict` read back."""
    return SyncReceipt(remote=d["remote"], remote_sidecar=d["remote_sidecar"],
                       sha256=d["sha256"],
                       synced_at=_aware(datetime.fromisoformat(d["synced_at"]),
                                        "synced_at"))


def check_session_skew(clock_skew: timedelta, guard: timedelta, *, dry_run: bool) -> None:
    """A real session holds the planned wait (the gap plus the margin, `commit`);
    a dry run may shorten it."""
    if guard < timedelta(0):
        raise HoldError(f"the guard must be >= 0, got {guard}")
    if clock_skew < timedelta(0):
        raise HoldError(f"the clock skew must be >= 0, got {clock_skew}")
    if not dry_run and clock_skew < REQUIRED_SKEW:
        raise HoldError(
            f"a session holds at least {REQUIRED_SKEW.total_seconds():.0f} s after the "
            f"sync (Appendix A, 25 Oct 2026); {clock_skew.total_seconds():.0f} s is "
            f"allowed only in a dry run")
    if not dry_run and clock_skew + guard < PLANNED_WAIT:
        raise HoldError(
            f"a session holds the planned {PLANNED_WAIT.total_seconds():.0f} s after the "
            f"sync; {(clock_skew + guard).total_seconds():.0f} s is allowed only in a dry run")


def release_at(synced_at: datetime, *, clock_skew: timedelta,
               guard: timedelta) -> datetime:
    """The first instance-clock time at which a clocked block may start."""
    return _aware(synced_at, "synced_at") + clock_skew + guard


def ready_to_clock(synced_at: datetime, now: datetime, *, clock_skew: timedelta,
                   guard: timedelta) -> None:
    """Refuse unless `now` (instance clock) is at or past the release."""
    rel = release_at(synced_at, clock_skew=clock_skew, guard=guard)
    if _aware(now, "now") < rel:
        raise HoldError(
            f"the prediction was synced by {synced_at.isoformat()}; no clocked block "
            f"before {rel.isoformat()} (instance clock), and it is {now.isoformat()}")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def hold(synced_at: datetime, *, clock_skew: timedelta, guard: timedelta,
         now: Callable[[], datetime] = utc_now,
         sleep: Callable[[float], None] = time.sleep,
         max_wait: timedelta = timedelta(minutes=10)) -> datetime:
    """Wait until a clocked block may start, and return that instance time.

    `max_wait` bounds a hold whose release is implausibly far ahead (a clock
    that jumped, or a receipt from another session)."""
    rel = release_at(synced_at, clock_skew=clock_skew, guard=guard)
    t = _aware(now(), "now")
    if rel - t > max_wait:
        raise HoldError(f"the release {rel.isoformat()} is more than {max_wait} away "
                        f"({t.isoformat()}); is this the session's receipt?")
    while t < rel:
        sleep(min(5.0, max(0.05, (rel - t).total_seconds())))
        t = _aware(now(), "now")
    return t


def session_start(clock_not_before: Iterable, first_timed_block: Iterable = ()) -> datetime:
    """The start the late-prediction check reads: the earliest of every
    ground-truth process's pre-forward stamp and every first-timed-block
    stamp (epoch seconds or aware datetimes)."""
    return ground_truth_start([*clock_not_before, *first_timed_block])
