# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snapshot lifecycle helper errors."""
import errno
from typing import Optional

from statedb.utils.errors import DiskgraphError


class InsufficientDataplaneCapacityError(DiskgraphError):
    """A fresh local dataplane mirror cannot fit before ingestion starts."""


class DiskgraphCapacityError(DiskgraphError):
    """No configured durable host can satisfy a snapshot placement."""


class ExistingWorkerUnavailableError(DiskgraphError):
    """A pinned worker in an unavailable tier cannot be safely reused."""


class EnsureFailedNoSpace(RuntimeError):
    """Snapshot materialization ran out of local SSD space."""


class EnsureFailedSource(RuntimeError):
    """Fresh ingest from source failed permanently."""


class DataplaneBackupNotFound(RuntimeError):
    """Dataplane restore source was missing."""


class HomeStaleError(RuntimeError):
    """The snapshot's assigned home worker is STALE (heartbeats lapsed past
    the stale threshold but not yet DEAD).

    Raised by ``route`` instead of placing on a worker that may be mid
    self-heal. The caller (``run_pooled_activity``) should wait for the FSM to
    settle — the worker either re-heartbeats back to READY (re-route succeeds)
    or ages to DEAD (re-route mints a fresh worker) — and then re-route.
    """


class SnapshotEvictionInProgressError(RuntimeError):
    """The snapshot's current local copy is being physically evicted.

    Placement must retry after deletion finishes so it cannot lease files that
    are about to disappear. ``RuntimeError`` keeps the AutoRPC failure
    retryable through the existing Temporal placement policy.
    """


def is_no_space_error(exc: Exception) -> bool:
    current: Optional[BaseException] = exc
    while current is not None:
        if isinstance(current, EnsureFailedNoSpace):
            return True
        if isinstance(current, OSError) and current.errno == errno.ENOSPC:
            return True
        current = current.__cause__
    return False


# ``ApplicationError.type`` of the retryable disk-full error raised by
# ``kumo_core.diskgraph.runner.DiskgraphNoSpaceError``. A string
# constant (not an import) because workflow code matches on it and the
# workflow sandbox must not import the subprocess-driving ingest_runner;
# a test pins it to the class name.
NO_SPACE_ERROR_TYPE = 'DiskgraphNoSpaceError'


def is_no_space_activity_error(err: BaseException) -> bool:
    """Whether a (workflow-side) activity error was caused by the ingest
    running out of local SSD space.

    Activity failures cross the Temporal serialization boundary as
    ``ActivityError -> ApplicationError(type=<class name>)``, so this
    matches on ``.type`` rather than isinstance (``is_no_space_error``
    only works on raw in-process exceptions).
    """
    cause: Optional[BaseException] = err.__cause__
    while cause is not None:
        if getattr(cause, 'type', None) == NO_SPACE_ERROR_TYPE:
            return True
        cause = cause.__cause__
    return False
