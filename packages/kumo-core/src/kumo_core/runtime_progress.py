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
    """Receives serial progress notifications from one process."""
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
_LOCK = threading.RLock()


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
    global _OBSERVER
    with _LOCK:
        if _OBSERVER is not token:
            raise RuntimeError('Kumo runtime progress observer token changed')
        _OBSERVER = None


def stage_changed(stage: str, detail: str | None = None) -> None:
    """Publish a stage transition when an observer is installed."""
    with _LOCK:
        observer = _OBSERVER
        if observer is not None:
            observer.stage_changed(stage, detail)


def progress_updated(kind: str, progress: Any) -> None:
    """Best-effort publish a native Kumo progress value when available."""
    with _LOCK:
        observer = _OBSERVER
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
    with _LOCK:
        observer = _OBSERVER
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
        with _LOCK:
            observer.operation_finished(kind, operation_id, error)
