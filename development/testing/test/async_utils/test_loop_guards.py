# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""install_loop_guards wires up both event-loop guards from one call."""
import pytest

from async_utils import event_loop_watchdog
from async_utils.blockbuster_guard import _reset_blockbuster_guard_for_tests
from async_utils.event_loop_watchdog import _reset_global_watchdog_for_tests
from async_utils.loop_guards import install_loop_guards


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # off -> activate_blockbuster_guard no-ops (no stdlib patching in this test);
    # we only assert that install_loop_guards drives both installers.
    monkeypatch.setenv('BLOCKBUSTER_MODE', 'off')
    _reset_global_watchdog_for_tests()
    _reset_blockbuster_guard_for_tests()
    yield
    _reset_global_watchdog_for_tests()
    _reset_blockbuster_guard_for_tests()


@pytest.mark.asyncio
async def test_install_loop_guards_starts_watchdog_and_runs_blockbuster():
    assert event_loop_watchdog._GLOBAL_WATCHDOG is None
    install_loop_guards()  # must not raise; drives both installers
    assert event_loop_watchdog._GLOBAL_WATCHDOG is not None
