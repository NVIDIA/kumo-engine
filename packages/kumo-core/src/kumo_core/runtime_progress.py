# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional progress observation for an in-process Kumo workload.

The ordinary local runtime does not install an observer, so calls in this
module are deliberately no-ops.  An outer execution environment may install
one for the duration of a user-owned function without changing how Kumo's
training or prediction code executes.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Protocol

logger = logging.getLogger(__name__)


class RuntimeProgressObserver(Protocol):
    """Receives progress notifications from one process."""
    def operation_started(
        self,
        kind: str,
        operation_id: str,
        stage: str,
    ) -> None:
        raise NotImplementedError

    def stage_changed(self, stage: str, detail: str | None = None) -> None:
        raise NotImplementedError

    def progress_updated(self, kind: str, progress: Any) -> None:
        raise NotImplementedError

    def operation_finished(
        self,
        kind: str,
        operation_id: str,
        error: BaseException | None,
    ) -> None:
        raise NotImplementedError


_OBSERVER: RuntimeProgressObserver | None = None
_LOCK = threading.Condition(threading.RLock())
_IN_FLIGHT = 0
_RESETTING = False


def install(observer: RuntimeProgressObserver, ) -> RuntimeProgressObserver:
    """Install ``observer`` for the current process and return its token.

    Prediction callbacks may run on a worker thread, so thread-local or
    context-local storage would lose native progress.  The Snowflake wrapper
    owns one process and permits only one active top-level Kumo operation.
    """
    global _OBSERVER
    with _LOCK:
        if _OBSERVER is not None:
            raise RuntimeError('a Kumo runtime progress observer is active')
        _OBSERVER = observer
        return observer


def reset(token: RuntimeProgressObserver) -> None:
    """Remove the observer represented by ``token`` after callbacks finish."""
    global _OBSERVER, _RESETTING
    with _LOCK:
        if _OBSERVER is not token or _RESETTING:
            raise RuntimeError('Kumo runtime progress observer token changed')
        _RESETTING = True
        try:
            while _IN_FLIGHT:
                _LOCK.wait()
            _OBSERVER = None
        finally:
            _RESETTING = False


def has_observer() -> bool:
    """Whether optional runtime progress needs to be collected."""
    with _LOCK:
        return _OBSERVER is not None and not _RESETTING


@contextmanager
def _delivery() -> Iterator[RuntimeProgressObserver | None]:
    """Reserve a callback without holding the process lock during I/O."""
    global _IN_FLIGHT
    with _LOCK:
        observer = None if _RESETTING else _OBSERVER
        if observer is not None:
            _IN_FLIGHT += 1
    try:
        yield observer
    finally:
        if observer is not None:
            with _LOCK:
                _IN_FLIGHT -= 1
                _LOCK.notify_all()


def stage_changed(stage: str, detail: str | None = None) -> None:
    """Publish a stage transition when an observer is installed."""
    with _delivery() as observer:
        if observer is not None:
            observer.stage_changed(stage, detail)


def progress_updated(kind: str, progress: Any) -> None:
    """Best-effort publish a native Kumo progress value when available."""
    with _delivery() as observer:
        if observer is not None:
            try:
                observer.progress_updated(kind, progress)
            except Exception:
                # Progress is observability.  An observer failure must not
                # turn a successful training or prediction workload into a
                # failure.
                logger.warning(
                    'Could not publish %s runtime progress',
                    kind,
                    exc_info=True,
                )


@contextmanager
def operation(
    kind: str,
    operation_id: str,
    stage: str,
) -> Iterator[None]:
    """Bracket one synchronous top-level Kumo operation.

    Observers may reject overlapping operations.  Calls remain free when no
    observer is installed, which is the normal :class:`LocalBackend` path.
    """
    with _delivery() as observer:
        if observer is not None:
            observer.operation_started(kind, operation_id, stage)
    if observer is None:
        yield
        return
    error: BaseException | None = None
    try:
        yield
    except BaseException as caught:
        error = caught
        raise
    finally:
        with _delivery() as current:
            try:
                if current is observer:
                    observer.operation_finished(kind, operation_id, error)
            except Exception:
                # Completion reporting is observability.  Once the operation
                # has produced a result (or its own exception), an observer
                # failure must not replace that outcome.
                logger.warning(
                    'Could not finish %s runtime progress',
                    kind,
                    exc_info=True,
                )
