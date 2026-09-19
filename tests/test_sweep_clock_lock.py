"""Stage 2 stamps the clock-lock OUTCOME, never the request (audit S4)."""
from attnbench.sweep import clock_locked_provenance_fn


def _capture_spy(calls):
    def capture(clocks_locked=False):
        calls.append(clocks_locked)
        return clocks_locked
    return capture


def test_lock_that_succeeds_is_stamped_true():
    calls = []
    locked, fn = clock_locked_provenance_fn(
        lock_requested=True, dry_run=False, lock=lambda: True,
        capture=_capture_spy(calls))
    assert locked is True and fn() is True and calls == [True]


def test_lock_that_fails_is_stamped_false_not_the_request():
    # The failure mode #16 names: asked for a lock, did not get one.
    calls = []
    locked, fn = clock_locked_provenance_fn(
        lock_requested=True, dry_run=False, lock=lambda: False,
        capture=_capture_spy(calls))
    assert locked is False and fn() is False


def test_no_request_never_calls_the_lock():
    def lock():
        raise AssertionError("lock attempted without --lock-clocks")
    locked, fn = clock_locked_provenance_fn(
        lock_requested=False, dry_run=False, lock=lock,
        capture=_capture_spy([]))
    assert locked is False and fn() is False


def test_dry_run_attempts_nothing():
    def lock():
        raise AssertionError("dry run touched the clocks")
    locked, _ = clock_locked_provenance_fn(
        lock_requested=True, dry_run=True, lock=lock,
        capture=_capture_spy([]))
    assert locked is False
