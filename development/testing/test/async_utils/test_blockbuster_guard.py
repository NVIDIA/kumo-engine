# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the blockbuster blocking-call guard.

Uses ``time.sleep(0)`` as the canonical blocking primitive: blockbuster patches
it with no exemption predicate, so it is reliably flagged when invoked on a
running event loop. The guard is catch-all (no scoping), so a blocking call on
the loop is flagged regardless of which library sits between it and kumo code.
"""
import logging
import threading
import time

import pytest
from blockbuster import BlockingError

from async_utils.blockbuster_guard import (
    BlockBusterMode,
    _reset_blockbuster_guard_for_tests,
    _resolve_mode,
    activate_blockbuster_guard,
)

_LOGGER = 'async_utils.blockbuster_guard'


def _sleep_warnings(caplog):
    return [r for r in caplog.records
            if 'blocking call on event loop: time.sleep' in r.message]


@pytest.fixture(autouse=True)
def _isolate_guard():
    """Reset the process-wide guard between tests to avoid bleed-through."""
    _reset_blockbuster_guard_for_tests()
    yield
    _reset_blockbuster_guard_for_tests()


@pytest.mark.asyncio
async def test_raise_mode_flags_blocking_call_on_loop():
    activate_blockbuster_guard(mode=BlockBusterMode.RAISE)
    with pytest.raises(BlockingError):
        time.sleep(0)


@pytest.mark.asyncio
async def test_warn_mode_logs_full_stack_with_dedup_key(caplog):
    activate_blockbuster_guard(mode=BlockBusterMode.WARN)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        time.sleep(0)  # must NOT raise in warn mode
    warnings = _sleep_warnings(caplog)
    assert len(warnings) == 1
    # First occurrence: full stack (this test frame appears) + a dedup_key.
    assert __file__ in warnings[0].message
    assert 'test_warn_mode_logs_full_stack_with_dedup_key' in warnings[0].message
    assert 'dedup_key=' in warnings[0].message


@pytest.mark.asyncio
async def test_warn_mode_first_full_then_rollup_only(caplog, monkeypatch):
    # window=0 => every repeat is immediately due for a rollup line.
    monkeypatch.setenv('BLOCKBUSTER_SUMMARY_WINDOW_SEC', '0')
    activate_blockbuster_guard(mode=BlockBusterMode.WARN)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        for _ in range(3):
            time.sleep(0)
    warnings = _sleep_warnings(caplog)
    # 1 full first-occurrence + rollups for the 2 repeats; only the first
    # carries the full stack, the rest are compact rollups.
    assert len(warnings) == 3
    assert 'Stack:' in warnings[0].message
    assert all('more in the last' in w.message for w in warnings[1:])
    # All share one dedup_key (same call stack).
    keys = {w.message.split('dedup_key=')[1].split(')')[0] for w in warnings}
    assert len(keys) == 1


@pytest.mark.asyncio
async def test_warn_mode_no_repeat_spam_within_window(caplog):
    # Default window (300s): 5 rapid repeats -> only the first occurrence logs.
    activate_blockbuster_guard(mode=BlockBusterMode.WARN)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        for _ in range(5):
            time.sleep(0)
    assert len(_sleep_warnings(caplog)) == 1


@pytest.mark.asyncio
async def test_thread_start_offload_not_flagged_in_warn(caplog):
    """Thread-pool offload machinery (run_in_executor / anyio.run_sync) starts a
    worker thread, and Thread.start() briefly acquires an internal lock.
    blockbuster allowlists threading.py:start via can_block_functions, so warn
    mode must NOT flag it — it's the correct offload pattern, not a violation."""
    activate_blockbuster_guard(mode=BlockBusterMode.WARN)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        t = threading.Thread(target=lambda: None)
        t.start()
        t.join()
    lock_warnings = [r for r in caplog.records
                     if 'blocking call on event loop: lock.acquire' in r.message]
    assert lock_warnings == []


@pytest.mark.asyncio
async def test_off_mode_is_noop():
    assert activate_blockbuster_guard(mode=BlockBusterMode.OFF) is None
    time.sleep(0)  # not patched -> no raise


@pytest.mark.asyncio
async def test_blocking_call_off_loop_is_not_flagged():
    """A blocking call from a thread with no running loop must never trip."""
    activate_blockbuster_guard(mode=BlockBusterMode.RAISE)
    outcome = {}

    def worker():
        try:
            time.sleep(0)
            outcome['raised'] = False
        except BlockingError:
            outcome['raised'] = True

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert outcome['raised'] is False


@pytest.mark.asyncio
async def test_idempotent_activation():
    first = activate_blockbuster_guard(mode=BlockBusterMode.WARN)
    second = activate_blockbuster_guard(mode=BlockBusterMode.RAISE)
    assert first is second  # second call is a no-op, mode unchanged (warn)
    time.sleep(0)  # warn stuck: logs, does not raise


@pytest.mark.asyncio
async def test_rlock_acquire_not_yet_intercepted():
    """Documents the known gap (kumo#29025): blockbuster patches threading.Lock
    but not threading.RLock, and wrapt's @synchronized uses an RLock. If this
    starts raising, blockbuster gained RLock coverage -- revisit the guard."""
    activate_blockbuster_guard(mode=BlockBusterMode.RAISE)
    lock = threading.RLock()
    lock.acquire()  # not intercepted today
    lock.release()


def test_resolve_mode_explicit_env(monkeypatch):
    monkeypatch.setenv('BLOCKBUSTER_MODE', 'warn')
    assert _resolve_mode() is BlockBusterMode.WARN


def test_resolve_mode_invalid_env_disables(monkeypatch):
    monkeypatch.setenv('BLOCKBUSTER_MODE', 'bogus')
    assert _resolve_mode() is BlockBusterMode.OFF


def test_resolve_mode_auto_dev(monkeypatch):
    monkeypatch.delenv('BLOCKBUSTER_MODE', raising=False)
    monkeypatch.setenv('DEPLOY_STAGE', 'dev')
    assert _resolve_mode() is BlockBusterMode.WARN


def test_resolve_mode_auto_test_is_off(monkeypatch):
    monkeypatch.delenv('BLOCKBUSTER_MODE', raising=False)
    monkeypatch.setenv('DEPLOY_STAGE', 'test')
    assert _resolve_mode() is BlockBusterMode.OFF


def test_resolve_mode_auto_prod(monkeypatch):
    monkeypatch.delenv('BLOCKBUSTER_MODE', raising=False)
    monkeypatch.setenv('DEPLOY_STAGE', 'prod')
    assert _resolve_mode() is BlockBusterMode.WARN
