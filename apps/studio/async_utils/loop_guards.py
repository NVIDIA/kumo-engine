# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single entry point for event-loop health instrumentation.

Call :func:`install_loop_guards` once per asyncio main, after startup init.
It installs both loop guards together:

* the event-loop **stall watchdog** — reactive: fires after the loop has
  already blocked past a threshold, cause-agnostic
  (:mod:`async_utils.event_loop_watchdog`);
* the **blockbuster guard** — proactive: fires the instant a known-blocking
  primitive runs on the loop, even when it was fast this time
  (:mod:`async_utils.blockbuster_guard`).

Bundling them behind one call means every asyncio entry point gets both, and a
new main needs only this single line — there is no second place to forget.
Both installers are idempotent, so this is safe to call more than once.
"""
from async_utils.blockbuster_guard import activate_blockbuster_guard
from async_utils.event_loop_watchdog import start_global_watchdog


def install_loop_guards() -> None:
    """Install the stall watchdog and the blocking-call guard. Idempotent."""
    start_global_watchdog()
    activate_blockbuster_guard()
