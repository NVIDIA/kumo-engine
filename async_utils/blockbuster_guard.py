# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect synchronous blocking calls made on the asyncio event loop.

Thin policy layer over the ``blockbuster`` library, which monkeypatches stdlib
blocking primitives (file I/O, ``os.*``, ``socket``, ``time.sleep``,
``threading.Lock.acquire``) and flags them when invoked while an asyncio loop
is running on the calling thread.

Every blocking call on the loop is a bug in *our* code: kumo invoked something
synchronous — directly or through a third-party library — without offloading it
to a thread. The library in between is irrelevant; the fix is always on the
kumo side (wrap it with ``asyncio.to_thread`` / a ``@blocking_method`` offload).
So the guard is **catch-all** — it does not try to decide which calls are
"first-party"; it flags them all and lets the full stack show the chain.

Two modes:

* ``warn`` (default off-prod surfaces): the **first** occurrence of each
  distinct call stack is logged in full (the whole chain, kumo frame -> library
  -> blocking primitive) with a short ``dedup_key``; thereafter the guard emits
  at most one rollup line per window per key ("N more in the last Ws") instead
  of re-logging — so it never spams while still reporting frequency. The call is
  allowed to proceed; a running service is never broken.
* ``raise``: raise ``BlockingError`` at the call site (the exception traceback
  is the same full chain) — the fail-fast gate.

Relationship to :mod:`async_utils.event_loop_watchdog`:

* The watchdog is a *reactive latency* detector — it fires only once the loop
  has *already* stalled past a threshold, but is cause-agnostic (catches any
  blocker, including primitives blockbuster does not patch).
* This guard is a *proactive call-site* detector — it fires the instant a
  known-blocking primitive runs on the loop, even if it was fast this time
  (e.g. an IMDS lookup that is usually ms but 5 s on timeout).

They are complementary; we run both.

Known gap: blockbuster patches ``threading.Lock`` but not ``threading.RLock``.
wrapt's ``@synchronized`` (the data-backend cache lock implicated in kumo#29025)
uses an RLock, so this guard does not yet intercept that specific lock — the
watchdog remains the backstop for it. RLock coverage is a tracked follow-up.
"""
import hashlib
import logging
import sys
import threading
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from blockbuster import BlockBuster

from service_framework.deploy_stage import deploy_stage
from service_framework.env_var import EnvVar

log = logging.getLogger(__name__)


class BlockBusterMode(str, Enum):
    OFF = 'off'
    WARN = 'warn'
    RAISE = 'raise'


_MODE = EnvVar(
    'BLOCKBUSTER_MODE',
    str,
    default_value='',
    description=("Blocking-call guard mode: 'off', 'warn', or 'raise'. Empty "
                 '(default) derives from DEPLOY_STAGE: off in test, warn '
                 'everywhere else. raise (the fail-fast hard gate) is opt-in.'),
)
_SUMMARY_WINDOW_SEC = EnvVar(
    'BLOCKBUSTER_SUMMARY_WINDOW_SEC',
    float,
    default_value=300.0,
    description=('Warn-mode rollup window. The first occurrence of each distinct '
                 'call stack is logged in full with a dedup_key; thereafter at '
                 'most one rollup line per window per key reports how many more '
                 'times it fired.'),
)

# Cap distinct tracked stacks so a pathological variety cannot grow memory
# unbounded; once full, novel stacks stop being tracked.
_MAX_TRACKED_SITES = 4096

_GUARD_LOCK = threading.Lock()
_ACTIVE_GUARD: Optional[BlockBuster] = None


class _SiteStat:
    """Per-distinct-stack occurrence stats, keyed by stack fingerprint."""
    __slots__ = ('total', 'since_last_summary', 'last_summary_ts')

    def __init__(self, now: float) -> None:
        self.total = 1
        self.since_last_summary = 0
        self.last_summary_ts = now


# fingerprint -> _SiteStat
_SITES: Dict[str, _SiteStat] = {}
_SITES_LOCK = threading.Lock()


def _resolve_mode() -> BlockBusterMode:
    raw = _MODE.parse_value().strip().lower()
    if raw:
        try:
            return BlockBusterMode(raw)
        except ValueError:
            log.warning('invalid BLOCKBUSTER_MODE=%r; disabling guard', raw)
            return BlockBusterMode.OFF
    stage = deploy_stage()
    if stage.is_test:
        # Off in the test suite: once a main's startup activates the guard it
        # patches stdlib process-wide, adding overhead/noise to every async
        # test. The deliberate CI gate (raise + a baseline allowlist) is a
        # separate opt-in via BLOCKBUSTER_MODE, not the suite-wide default.
        return BlockBusterMode.OFF
    # dev + preprod + staging + prod: observe via warn. raise (the fail-fast
    # hard gate) stays opt-in until the known offenders are offloaded/baselined.
    return BlockBusterMode.WARN


def _offending_stack(max_frames: int = 30
                     ) -> Tuple[Tuple[Tuple[str, int], ...], str]:
    """Full call stack at the blocking call, with the guard's own frames and
    blockbuster's wrapper frames trimmed (noise at the top). Returns a
    (signature, formatted-text) pair; the chain ends at the immediate caller of
    the blocking primitive and runs back up through kumo code."""
    frames = [
        f for f in traceback.extract_stack()
        if f.filename != __file__ and '/blockbuster/' not in f.filename
    ]
    frames = frames[-max_frames:]
    signature = tuple((f.filename, f.lineno or 0) for f in frames)
    text = ''.join(traceback.format_list(frames)).rstrip()
    return signature, text


def _fingerprint(signature: Tuple[Tuple[str, int], ...]) -> str:
    """Short stable key for a distinct call stack (10 hex chars)."""
    return hashlib.blake2b(repr(signature).encode(), digest_size=5).hexdigest()


def _record_occurrence(func_name: str,
                       signature: Tuple[Tuple[str, int], ...],
                       text: str) -> None:
    """Log the first occurrence of each distinct stack in full (with its
    dedup_key), then at most one rollup line per window per key — so a hot
    offender does not spam the log while still reporting its frequency."""
    fp = _fingerprint(signature)
    now = time.monotonic()
    window = _SUMMARY_WINDOW_SEC.parse_value()

    first = False
    rollup: Optional[Tuple[int, int]] = None  # (count_since_last, total)
    with _SITES_LOCK:
        stat = _SITES.get(fp)
        if stat is None:
            if len(_SITES) >= _MAX_TRACKED_SITES:
                return
            _SITES[fp] = _SiteStat(now)
            first = True
        else:
            stat.total += 1
            stat.since_last_summary += 1
            if now - stat.last_summary_ts >= window:
                rollup = (stat.since_last_summary, stat.total)
                stat.since_last_summary = 0
                stat.last_summary_ts = now

    if first:
        log.warning(
            'blocking call on event loop: %s (dedup_key=%s) — offload it to a '
            'thread. Stack:\n%s', func_name, fp, text)
    elif rollup is not None:
        count_since, total = rollup
        log.warning(
            'blocking call on event loop: %s (dedup_key=%s) — %d more in the '
            'last %.0fs (total %d)', func_name, fp, count_since, window, total)


def _stack_allows(can_block_functions: List[Tuple[str, Iterable[str]]],
                  max_depth: int = 80) -> bool:
    """Mirror blockbuster's ``can_block_functions`` allowlist: a call is benign
    if any stack frame matches an allowed (filename-suffix, function) pair.

    This is how blockbuster exempts thread-pool/offload machinery — e.g.
    ``run_in_executor`` / ``anyio.run_sync`` start a worker thread, and
    ``threading.Thread.start`` briefly acquires an internal lock
    (``threading.py``:``start``). Those are correct offloads, not violations."""
    if not can_block_functions:
        return False
    frame: Optional[object] = sys._getframe(2)  # noqa: SLF001 (skip self + predicate)
    depth = 0
    while frame is not None and depth < max_depth:
        code = frame.f_code  # type: ignore[attr-defined]
        fname = Path(code.co_filename).as_posix()
        func = code.co_name
        for filename, functions in can_block_functions:
            if fname.endswith(filename) and func in functions:
                return True
        frame = frame.f_back  # type: ignore[attr-defined]
        depth += 1
    return False


def _make_warn_predicate(
    func_name: str,
    orig_predicate: Callable[..., bool],
    can_block_functions: List[Tuple[str, Iterable[str]]],
) -> Callable[..., bool]:
    """Warn-mode ``can_block_predicate``: log the full stack once, then allow.

    Honors the same filters blockbuster would in raise mode:
    * the primitive's original predicate (uncontended ``Lock.acquire``,
      non-blocking-fd ``os.read``/``os.write`` are not real blocks), and
    * the ``can_block_functions`` allowlist (thread-pool offload machinery).
    Anything that *would* still have been flagged is logged with its full stack
    and allowed to proceed (warn never blocks).
    """
    def predicate(*args: object, **kwargs: object) -> bool:
        try:
            if orig_predicate(*args, **kwargs):
                return True
        except Exception:  # pragma: no cover - defensive; predicate must not raise
            pass
        if _stack_allows(can_block_functions):
            return True
        signature, text = _offending_stack()
        _record_occurrence(func_name, signature, text)
        return True

    return predicate


def activate_blockbuster_guard(
    mode: Optional[BlockBusterMode] = None,
) -> Optional[BlockBuster]:
    """Activate the process-wide blocking-call guard. Idempotent, catch-all.

    First call installs the guard; subsequent calls return the existing
    instance. ``mode`` defaults to env/stage resolution. Returns the active
    :class:`BlockBuster`, or None when mode is ``off``.
    """
    global _ACTIVE_GUARD
    with _GUARD_LOCK:
        if _ACTIVE_GUARD is not None:
            return _ACTIVE_GUARD

        resolved_mode = mode or _resolve_mode()
        if resolved_mode is BlockBusterMode.OFF:
            log.info('blockbuster guard disabled (mode=off)')
            return None

        # No scanned_modules: catch every blocking call on the loop regardless
        # of which library sits between kumo and the primitive.
        guard = BlockBuster()
        if resolved_mode is BlockBusterMode.WARN:
            for fn in guard.functions.values():
                fn.can_block_predicate = _make_warn_predicate(
                    fn.full_name, fn.can_block_predicate,
                    fn.can_block_functions)
        # RAISE: leave blockbuster's native behaviour — it raises BlockingError
        # whose traceback is the same full chain.
        guard.activate()
        _ACTIVE_GUARD = guard
        log.info('blockbuster guard active (mode=%s, %d primitives, catch-all)',
                 resolved_mode.value, len(guard.functions))
        return guard


def _reset_blockbuster_guard_for_tests(  # pyright: ignore[reportUnusedFunction]
) -> None:
    """Test-only: deactivate and clear the process-wide guard + dedup state."""
    global _ACTIVE_GUARD
    with _GUARD_LOCK:
        if _ACTIVE_GUARD is not None:
            _ACTIVE_GUARD.deactivate()
            _ACTIVE_GUARD = None
    with _SITES_LOCK:
        _SITES.clear()
