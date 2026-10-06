# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the event-loop watchdog.

Uses short thresholds/intervals so the test suite stays fast.
"""
import asyncio
import logging
import threading
import time

import pytest

from async_utils.event_loop_watchdog import (
    EventLoopWatchdog,
    _format_stalled_loop_diagnostics,
    _reset_global_watchdog_for_tests,
    start_global_watchdog,
)


@pytest.fixture(autouse=True)
def _isolate_global_watchdog():
    """Reset the process-wide watchdog between tests to avoid bleed-through."""
    _reset_global_watchdog_for_tests()
    yield
    _reset_global_watchdog_for_tests()


@pytest.mark.asyncio
async def test_healthy_loop_does_not_trigger(caplog):
    """A loop that yields regularly should not be flagged as stalled."""
    watchdog = EventLoopWatchdog(
        stall_threshold_sec=0.5,
        probe_interval_sec=0.05,
        check_interval_sec=0.05,
    )
    with caplog.at_level(logging.ERROR,
                         logger='temporalio_commons.event_loop_watchdog'):
        watchdog.start()
        # Yield often enough that the probe runs and the watchdog never sees
        # a gap larger than the threshold.
        for _ in range(20):
            await asyncio.sleep(0.05)
        watchdog.stop()

    stall_errors = [
        r for r in caplog.records if 'STALL DETECTED' in r.getMessage()
    ]
    assert stall_errors == [], (
        f'Watchdog falsely flagged a healthy loop as stalled: {stall_errors}')


@pytest.mark.asyncio
async def test_blocked_loop_triggers_stall_and_captures_stacks(
        caplog, tmp_path):
    """Blocking the loop past the threshold should log ERROR with stacks
    and write a dump file under ``dump_dir``."""
    watchdog = EventLoopWatchdog(
        stall_threshold_sec=0.3,
        probe_interval_sec=0.05,
        check_interval_sec=0.05,
        dump_dir=str(tmp_path),
    )
    with caplog.at_level(logging.DEBUG,
                         logger='async_utils.event_loop_watchdog'):
        watchdog.start()
        # Let the probe arm itself once.
        await asyncio.sleep(0.1)
        # Block the loop synchronously. time.sleep blocks the thread (and
        # hence the asyncio loop, since the loop runs on this thread).
        time.sleep(0.6)
        # Give the watchdog a check tick to react now that we're yielding
        # again.
        await asyncio.sleep(0.15)
        watchdog.stop()

    stall_records = [
        r for r in caplog.records if 'STALL DETECTED' in r.getMessage()
    ]
    assert len(stall_records) >= 1, (
        f'Expected stall detection log, got: '
        f'{[r.getMessage() for r in caplog.records]}')

    msg = stall_records[0].getMessage()
    assert 'asyncio loop blocked' in msg
    assert 'event_id=' in msg
    assert 'full_dump=' in msg
    assert 'EVENT LOOP THREAD' in msg

    # A dump file should have been written to tmp_path.
    dump_files = list(tmp_path.glob('event-loop-stall-*.txt'))
    assert len(dump_files) >= 1, (
        f'Expected a dump file in {tmp_path}, got: {list(tmp_path.iterdir())}')
    dump_content = dump_files[0].read_text()
    assert 'event_id:' in dump_content
    assert 'EVENT LOOP THREAD' in dump_content
    assert 'other thread(s)' in dump_content


@pytest.mark.asyncio
async def test_stall_resolution_is_logged(caplog):
    """When the loop recovers, a 'stall resolved' WARNING should fire."""
    watchdog = EventLoopWatchdog(
        stall_threshold_sec=0.2,
        probe_interval_sec=0.05,
        check_interval_sec=0.05,
    )
    with caplog.at_level(logging.WARNING,
                         logger='temporalio_commons.event_loop_watchdog'):
        watchdog.start()
        await asyncio.sleep(0.1)
        time.sleep(0.4)
        # Let the watchdog observe the recovery.
        await asyncio.sleep(0.2)
        watchdog.stop()

    resolved = [
        r for r in caplog.records if 'stall resolved' in r.getMessage()
    ]
    assert len(resolved) >= 1, (
        f'Expected stall-resolved log, got: '
        f'{[r.getMessage() for r in caplog.records]}')


@pytest.mark.asyncio
async def test_disabled_watchdog_is_noop():
    """``enabled=False`` should skip thread creation entirely."""
    watchdog = EventLoopWatchdog(enabled=False)
    watchdog.start()
    # No thread was started.
    assert watchdog._thread is None  # pylint: disable=protected-access
    # Stop is safe even when never started.
    watchdog.stop()


def test_start_outside_async_context_raises():
    """Calling start() with no running loop is a programming error → raise."""
    watchdog = EventLoopWatchdog(
        stall_threshold_sec=0.1,
        probe_interval_sec=0.05,
        check_interval_sec=0.05,
    )
    with pytest.raises(RuntimeError):
        watchdog.start()
    assert watchdog._thread is None  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_double_start_is_idempotent(caplog):
    """Calling start() twice should not spawn a second thread."""
    watchdog = EventLoopWatchdog(
        stall_threshold_sec=0.5,
        probe_interval_sec=0.05,
        check_interval_sec=0.05,
    )
    with caplog.at_level(logging.WARNING,
                         logger='temporalio_commons.event_loop_watchdog'):
        watchdog.start()
        thread_after_first = watchdog._thread  # pylint: disable=protected-access
        watchdog.start()
        thread_after_second = watchdog._thread  # pylint: disable=protected-access
        watchdog.stop()

    assert thread_after_first is thread_after_second, (
        'Double start should reuse the same thread')
    second_start_warning = [
        r for r in caplog.records if 'already started' in r.getMessage()
    ]
    assert len(second_start_warning) == 1


def test_format_diagnostics_prioritizes_event_loop_thread():
    """The formatter must lead with the event loop thread's stack."""
    # Use the current thread as the "event loop" for this synchronous test.
    ident = threading.get_ident()
    formatted = _format_stalled_loop_diagnostics(ident)
    # The event loop thread section comes first.
    assert formatted.startswith('=== EVENT LOOP THREAD'), formatted[:200]
    # And contains an entry for the test function frame.
    assert 'test_format_diagnostics_prioritizes_event_loop_thread' in formatted


def test_format_diagnostics_dedupes_other_threads():
    """Identical idle thread stacks should collapse into a single section."""
    # Spawn several threads parked on the same Event.wait() — they all share
    # the same stack signature and should dedupe.
    parked = threading.Event()
    parked_threads = []
    for i in range(4):
        t = threading.Thread(target=parked.wait, daemon=True,
                             name=f'parked-{i}')
        t.start()
        parked_threads.append(t)
    try:
        # Give the threads a moment to reach the wait() frame.
        time.sleep(0.05)
        formatted = _format_stalled_loop_diagnostics(threading.get_ident())
        # The grouped header [4x] should appear for the parked threads,
        # OR [Nx] for some N ≥ 2 if other deduped library threads exist too.
        # Minimum requirement: the 4 parked threads collapse, not appear as
        # 4 separate sections.
        assert '[4x]' in formatted or any(
            f'[{n}x]' in formatted for n in (5, 6, 7, 8)
        ), formatted
    finally:
        parked.set()
        for t in parked_threads:
            t.join(timeout=1.0)


def test_format_diagnostics_handles_missing_loop_thread():
    """If the loop thread id doesn't match any live thread, don't crash."""
    formatted = _format_stalled_loop_diagnostics(999_999_999)
    assert 'EVENT LOOP THREAD' in formatted
    # And we still show the other threads.
    assert 'unique stack signature' in formatted


def test_watchdog_stops_when_monitored_loop_exits():
    ready = threading.Event()
    holder = {}

    def run_short_lived_loop():
        async def main():
            watchdog = EventLoopWatchdog(
                stall_threshold_sec=0.1,
                probe_interval_sec=0.01,
                check_interval_sec=0.01,
            )
            watchdog.start()
            holder["watchdog"] = watchdog
            ready.set()

        asyncio.run(main())

    loop_thread = threading.Thread(target=run_short_lived_loop)
    loop_thread.start()
    assert ready.wait(timeout=1.0)
    loop_thread.join(timeout=1.0)
    assert not loop_thread.is_alive()

    watchdog = holder["watchdog"]
    watchdog_thread = watchdog._thread  # pylint: disable=protected-access
    assert watchdog_thread is not None
    watchdog_thread.join(timeout=1.0)
    assert not watchdog_thread.is_alive()

    watchdog.stop(timeout=1.0)
    assert watchdog._thread is None  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_global_watchdog_is_singleton():
    """``start_global_watchdog()`` returns the same instance across calls."""
    first = start_global_watchdog(stall_threshold_sec=0.5,
                                  probe_interval_sec=0.05,
                                  check_interval_sec=0.05)
    second = start_global_watchdog(stall_threshold_sec=0.01)  # kwargs ignored
    assert first is second
    # kwargs of the second call are ignored — the first call's config wins.
    assert first.stall_threshold_sec == 0.5
