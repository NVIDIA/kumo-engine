# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect when the asyncio event loop is blocked.

Spawns a daemon thread that monitors a monotonic counter bumped by the loop
every ``probe_interval_sec`` via ``loop.call_later``. If the counter hasn't
advanced for longer than ``stall_threshold_sec`` (default 2 s), the watchdog
captures stacks of every live thread and emits a single ERROR log line.
One log per stall episode, plus a WARNING on recovery.

Motivating use case: the Temporal SDK's ``_heartbeat_processor`` schedules
heartbeats via ``asyncio.run_coroutine_threadsafe(...).result(10)`` on the
main loop. When the loop is blocked for >10 s, that call raises
``TimeoutError``; the SDK's broad ``except Exception: ... return``
permanently terminates the processor thread, which then cascades into
activity-slot exhaustion. We want to know what blocked the loop *before* the
SDK gives up — so we fire at 2 s rather than waiting for the SDK's 10 s
cliff. The module itself is generic and not Temporal-specific.

Usage::

    async def main():
        watchdog = EventLoopWatchdog()
        watchdog.start()  # daemon thread; survives until process exit
        ...
"""
import asyncio
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from types import FrameType
from typing import Dict, List, Optional, Tuple

from service_framework.env_var import EnvVar

log = logging.getLogger(__name__)

_STALL_THRESHOLD_SEC = EnvVar(
    'EVENT_LOOP_WATCHDOG_STALL_THRESHOLD_SEC',
    float,
    default_value=2.0,
    description=('Seconds the asyncio loop can go without bumping the probe '
                 'counter before the watchdog declares a stall.'),
)
_PROBE_INTERVAL_SEC = EnvVar(
    'EVENT_LOOP_WATCHDOG_PROBE_INTERVAL_SEC',
    float,
    default_value=0.5,
    description='How often the loop bumps the tick counter via call_later.',
)
_CHECK_INTERVAL_SEC = EnvVar(
    'EVENT_LOOP_WATCHDOG_CHECK_INTERVAL_SEC',
    float,
    default_value=0.25,
    description=('How often the watchdog thread evaluates the tick counter. '
                 'Smaller values narrow the detection edge window at the '
                 'cost of slightly more wakeups; default ~halves the '
                 'probe_interval so we never miss a fresh tick.'),
)
_ENABLED = EnvVar(
    'EVENT_LOOP_WATCHDOG_ENABLED',
    bool,
    default_value=True,
    description='Master enable/disable switch for the watchdog.',
)
_DUMP_DIR = EnvVar(
    'EVENT_LOOP_WATCHDOG_DUMP_DIR',
    str,
    default_value='/tmp',
    description=('Directory where per-stall full stack dumps are written. '
                 'One file per detection event; pod restart clears them.'),
)


def _format_all_thread_stacks_verbose(loop_thread_ident: int) -> str:
    """Format every thread's stack individually — no dedup, no truncation.

    Intended for the on-disk dump that pairs with a stall log line. We pay
    the full size cost here because the file is for deep investigation and
    only written once per stall episode.
    """
    # pylint: disable=protected-access
    frames: Dict[int, FrameType] = sys._current_frames()
    name_by_id = {t.ident: t.name for t in threading.enumerate()}
    sections: List[str] = []

    # Event loop thread first.
    loop_frame = frames.get(loop_thread_ident)
    if loop_frame is not None:
        loop_name = name_by_id.get(loop_thread_ident,
                                   f'<id={loop_thread_ident}>')
        loop_stack = ''.join(traceback.format_stack(loop_frame))
        sections.append(
            f'=== EVENT LOOP THREAD {loop_name!r} (id={loop_thread_ident}) '
            f'===\n{loop_stack}')
    else:
        sections.append(
            f'=== EVENT LOOP THREAD (id={loop_thread_ident}) not found ===')

    # Everyone else in deterministic order (sorted by thread id).
    others = sorted((tid, fr) for tid, fr in frames.items()
                    if tid != loop_thread_ident)
    sections.append(f'=== {len(others)} other thread(s) ===')
    for thread_id, frame in others:
        name = name_by_id.get(thread_id, f'<unknown-{thread_id}>')
        stack = ''.join(traceback.format_stack(frame))
        sections.append(f'Thread {name!r} (id={thread_id}):\n{stack}')

    return '\n'.join(sections)


def _generate_event_id() -> str:
    """Sortable, unique-per-process event id."""
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return f'{ts}-{uuid.uuid4().hex[:8]}'


def _write_dump_file(dump_dir: str, event_id: str, header: str,
                     verbose_stacks: str) -> Optional[str]:
    """Write the full stack dump for a stall event. Returns the file path,
    or None if the write failed (best-effort)."""
    try:
        os.makedirs(dump_dir, exist_ok=True)
        file_path = os.path.join(dump_dir,
                                 f'event-loop-stall-{event_id}.txt')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write('\n')
            f.write(verbose_stacks)
        return file_path
    except OSError:
        log.exception(
            'event-loop watchdog: failed to write stall dump file in %s',
            dump_dir)
        return None


def _format_stalled_loop_diagnostics(loop_thread_ident: int) -> str:
    """Format thread stacks for a stall report.

    Prioritizes the event loop thread (the one we actually care about — its
    stack is the smoking gun for what blocked the loop). Other threads are
    deduped by stack signature and reported in descending count order; this
    collapses the typical 20+ idle thread-pool workers (all parked on
    ``queue.get``) into a single line per unique stack.
    """
    # pylint: disable=protected-access
    frames: Dict[int, FrameType] = sys._current_frames()
    name_by_id = {t.ident: t.name for t in threading.enumerate()}
    sections: List[str] = []

    # 1) Event loop thread first — verbose, this is the diagnostic value.
    loop_frame = frames.get(loop_thread_ident)
    if loop_frame is not None:
        loop_name = name_by_id.get(loop_thread_ident,
                                   f'<id={loop_thread_ident}>')
        loop_stack = ''.join(traceback.format_stack(loop_frame))
        sections.append(
            f'=== EVENT LOOP THREAD {loop_name!r} (id={loop_thread_ident}) '
            f'===\n{loop_stack}')
    else:
        sections.append(
            f'=== EVENT LOOP THREAD (id={loop_thread_ident}) not found in '
            f'sys._current_frames(); thread may have died ===')

    # 2) Other threads — dedupe by stack signature.
    groups: Dict[str, List[Tuple[int, str]]] = {}
    for thread_id, frame in frames.items():
        if thread_id == loop_thread_ident:
            continue
        name = name_by_id.get(thread_id, f'<unknown-{thread_id}>')
        stack = ''.join(traceback.format_stack(frame))
        groups.setdefault(stack, []).append((thread_id, name))

    if groups:
        total_others = sum(len(members) for members in groups.values())
        sections.append(
            f'=== {total_others} other thread(s), '
            f'{len(groups)} unique stack signature(s) ===')
        # Most-common stacks first; ties broken by stack content for stability.
        ordered = sorted(groups.items(),
                         key=lambda kv: (-len(kv[1]), kv[0]))
        for stack, members in ordered:
            count = len(members)
            if count == 1:
                tid, tname = members[0]
                header = f'[1x] {tname!r} (id={tid})'
            else:
                # Show a couple of representative names so the reader can
                # cross-reference with logs.
                sample = ', '.join(
                    f'{n!r} (id={i})' for i, n in members[:3])
                more = '' if count <= 3 else f', +{count - 3} more'
                header = f'[{count}x] {sample}{more}'
            sections.append(f'{header}\n{stack}')

    return '\n'.join(sections)


class EventLoopWatchdog:
    """Asyncio event-loop stall detector.

    Lifecycle:
      - ``start()`` must be called from within an asyncio context so that
        ``asyncio.get_running_loop()`` returns the loop to monitor.
      - The watchdog thread is daemon=True; it dies with the process.
      - ``stop()`` is optional (for tests / clean shutdown).
    """
    def __init__(
        self,
        *,
        stall_threshold_sec: Optional[float] = None,
        probe_interval_sec: Optional[float] = None,
        check_interval_sec: Optional[float] = None,
        enabled: Optional[bool] = None,
        dump_dir: Optional[str] = None,
    ) -> None:
        self.stall_threshold_sec = (stall_threshold_sec
                                    if stall_threshold_sec is not None else
                                    _STALL_THRESHOLD_SEC.parse_value())
        self.probe_interval_sec = (probe_interval_sec
                                   if probe_interval_sec is not None else
                                   _PROBE_INTERVAL_SEC.parse_value())
        self.check_interval_sec = (check_interval_sec
                                   if check_interval_sec is not None else
                                   _CHECK_INTERVAL_SEC.parse_value())
        self.enabled = (enabled
                        if enabled is not None else _ENABLED.parse_value())
        self.dump_dir = (dump_dir
                         if dump_dir is not None else _DUMP_DIR.parse_value())

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # OS-level thread ident of the event loop thread, captured at start().
        # We need this so the stall report can prioritize that thread's stack.
        self._loop_thread_ident: Optional[int] = None
        # Monotonic timestamp of the most recent probe execution. Written by
        # the event loop (single writer), read by the watchdog thread. No
        # lock: float assignment is atomic under the GIL and there's no
        # compound state to keep consistent.
        self._last_tick: float = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Track whether we're currently mid-stall so we only log once per
        # stall episode (not once per check_interval).
        self._in_stall = False

    def start(self) -> None:
        if not self.enabled:
            log.info('event-loop watchdog disabled via env var, not starting')
            return
        if self._thread is not None:
            log.warning('event-loop watchdog already started, ignoring')
            return
        # asyncio.get_running_loop() raises RuntimeError if there's no running
        # loop. Calling start() outside an async context is a programming
        # error — let it propagate so the caller fixes the call site.
        self._loop = asyncio.get_running_loop()
        # asyncio.get_running_loop() returns the loop bound to the calling
        # thread; therefore the calling thread *is* the event loop thread.
        self._loop_thread_ident = threading.get_ident()

        self._last_tick = time.monotonic()
        # Schedule first probe; subsequent probes self-reschedule.
        self._loop.call_soon(self._probe)
        self._thread = threading.Thread(
            target=self._watchdog_loop,
            name='event-loop-watchdog',
            daemon=True,
        )
        self._thread.start()
        log.info(
            'event-loop watchdog started '
            '(stall_threshold=%.2fs, probe_interval=%.2fs, '
            'check_interval=%.2fs)',
            self.stall_threshold_sec,
            self.probe_interval_sec,
            self.check_interval_sec,
        )

    def stop(self, *, timeout: float = 2.0) -> None:
        """Signal the watchdog to stop. Safe to call multiple times."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    # --- internals -------------------------------------------------------

    def _probe(self) -> None:
        """Bump the tick counter; re-schedule self after probe_interval_sec.

        Runs on the asyncio loop. Must be cheap — no I/O, no awaits.
        """
        self._last_tick = time.monotonic()
        if self._stop_event.is_set() or self._loop is None:
            return
        self._loop.call_later(self.probe_interval_sec, self._probe)

    def _watchdog_loop(self) -> None:
        """Periodically compare ``now`` against the last tick.

        Runs on a dedicated daemon thread. Detects (and on resolution
        clears) the in-stall state. One log line per stall episode.
        """
        while not self._stop_event.wait(self.check_interval_sec):
            if self._monitored_loop_is_gone():
                self._stop_event.set()
                break
            elapsed = time.monotonic() - self._last_tick
            if elapsed > self.stall_threshold_sec:
                if not self._in_stall:
                    self._on_stall_start(elapsed)
                    self._in_stall = True
            else:
                if self._in_stall:
                    self._on_stall_end(elapsed)
                    self._in_stall = False

    def _monitored_loop_is_gone(self) -> bool:
        loop = self._loop
        if loop is None or loop.is_closed():
            return True
        loop_thread_ident = self._loop_thread_ident
        return (loop_thread_ident is not None
                and loop_thread_ident not in sys._current_frames())

    def _on_stall_start(self, elapsed: float) -> None:
        """Emit one ERROR line with stalled-loop diagnostic.

        Captures both a deduped summary (inline in the log) and a full
        per-thread dump (written to a unique file in ``dump_dir``). The log
        line references the file's event_id so investigators can locate the
        verbose dump on the affected pod.
        """
        event_id = _generate_event_id()
        loop_thread_ident = self._loop_thread_ident
        try:
            if loop_thread_ident is None:
                summary = '<event loop thread ident not captured>'
                verbose = summary
            else:
                summary = _format_stalled_loop_diagnostics(loop_thread_ident)
                verbose = _format_all_thread_stacks_verbose(loop_thread_ident)
        except Exception:  # pylint: disable=broad-except
            summary = '<failed to capture stack traces>'
            verbose = summary
            log.exception('event-loop watchdog: stack capture failed')

        header = (f'event_id: {event_id}\n'
                  f'pid: {os.getpid()}\n'
                  f'stall_duration_so_far_sec: {elapsed:.2f}\n'
                  f'stall_threshold_sec: {self.stall_threshold_sec:.2f}\n'
                  f'detected_at: '
                  f'{datetime.now(timezone.utc).isoformat()}\n')
        file_path = _write_dump_file(self.dump_dir, event_id, header, verbose)

        log.error(
            'event-loop watchdog: STALL DETECTED — asyncio loop blocked '
            '%.2fs (threshold %.2fs). event_id=%s full_dump=%s '
            'Deduped summary follows:\n%s',
            elapsed,
            self.stall_threshold_sec,
            event_id,
            file_path or '<dump write failed>',
            summary,
        )

    def _on_stall_end(self, last_elapsed: float) -> None:
        log.warning(
            'event-loop watchdog: stall resolved '
            '(last measured gap %.2fs, threshold %.2fs)',
            last_elapsed,
            self.stall_threshold_sec,
        )


# --- Process-wide singleton --------------------------------------------------
#
# Most call sites should use ``start_global_watchdog()`` rather than
# instantiating ``EventLoopWatchdog`` directly. This guarantees one watchdog
# per process even if multiple entry points (e.g., FastAPI ``startup_event``
# firing twice on reload) try to start it. Direct instantiation is preserved
# for tests that need custom thresholds.

_GLOBAL_WATCHDOG: Optional[EventLoopWatchdog] = None
_GLOBAL_WATCHDOG_LOCK = threading.Lock()


def start_global_watchdog(**kwargs) -> EventLoopWatchdog:
    """Start the process-wide event-loop watchdog. Idempotent.

    First call instantiates and starts the watchdog. Subsequent calls are
    no-ops (the existing instance is returned). Thread-safe.

    ``kwargs`` are forwarded to :class:`EventLoopWatchdog` on the first call
    only; subsequent calls' kwargs are ignored.
    """
    global _GLOBAL_WATCHDOG
    with _GLOBAL_WATCHDOG_LOCK:
        if _GLOBAL_WATCHDOG is None:
            _GLOBAL_WATCHDOG = EventLoopWatchdog(**kwargs)
        _GLOBAL_WATCHDOG.start()  # idempotent per-instance
        return _GLOBAL_WATCHDOG


def _reset_global_watchdog_for_tests(  # pyright: ignore[reportUnusedFunction]
) -> None:
    """Test-only helper: stop and clear the process-wide singleton."""
    global _GLOBAL_WATCHDOG
    with _GLOBAL_WATCHDOG_LOCK:
        if _GLOBAL_WATCHDOG is not None:
            _GLOBAL_WATCHDOG.stop()
        _GLOBAL_WATCHDOG = None
