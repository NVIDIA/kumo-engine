# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Run a Python script as a scheduled job: :func:`job`."""

import contextlib
import json
import logging
import os
import signal
import sys
import threading
import traceback
import warnings
from typing import Any, Dict, Iterator, Mapping, NoReturn, Optional

_MAX_MESSAGE = 500
_MAX_WARNING = 300
_MAX_WARNINGS = 20
_DEPRECATIONS = (DeprecationWarning, PendingDeprecationWarning, FutureWarning)


class Terminated(BaseException):
    r"""The process received SIGTERM."""


class JobRun:
    r"""What a job records for its summary line; see :func:`job`."""
    def __init__(self) -> None:
        self.summary: Dict[str, Any] = {
            'bundle_id': None,
            'snapshot_id': None,
            'run_id': None,
            'outputs': {},
            'warnings': [],
        }

    def record(self, *, bundle_id: Optional[str] = None,
               snapshot_id: Optional[str] = None, run_id: Optional[str] = None,
               outputs: Optional[Mapping[str, Any]] = None) -> None:
        r"""Adds ids and outputs (name to absolute URI or table) to the
        summary; ``None`` leaves a field."""
        for key, value in (('bundle_id', bundle_id),
                           ('snapshot_id', snapshot_id), ('run_id', run_id)):
            if value is not None:
                self.summary[key] = value
        self.summary['outputs'].update(outputs or {})


def _error_kind(error: BaseException) -> str:
    from kumo_core.sources.errors import (
        PermanentSourceError,
        TransientSourceError,
    )

    cause: Optional[BaseException] = error
    while cause is not None:
        if isinstance(cause, PermanentSourceError):
            return 'permanent'
        if isinstance(cause,
                      (TransientSourceError, ConnectionError, TimeoutError)):
            return 'transient'
        cause = cause.__cause__
    return 'unknown'


def _exit_code(error: Optional[BaseException]) -> int:
    if error is None:
        return 0
    if isinstance(error, SystemExit):
        code = error.code
        return code if isinstance(code, int) else 0 if code is None else 1
    if isinstance(error, KeyboardInterrupt):
        return 130
    if isinstance(error, Terminated):
        return 143
    return 1


@contextlib.contextmanager
def job(*, job_id: Optional[str] = None,
        job_run_id: Optional[str] = None) -> Iterator[JobRun]:
    r"""Runs the body as one scheduled job.

    ``job_id``/``job_run_id`` (else ``KUMO_JOB_ID``/``KUMO_JOB_RUN_ID`` or the
    detected scheduler) are recorded in the manifests and tracking tags the
    body writes. Logs go to stderr. On exit, however the body ends, one JSON
    line goes to stdout: ``status``, ``exit_code``, the scheduler context, what
    the body passed to :meth:`JobRun.record`, the first 20 distinct warnings
    other than deprecations (300 characters each), and ``error``
    (``type``, ``kind``: ``transient``/``permanent``/``unknown``, ``message``)
    on failure. A failed body exits the process with 1; SIGTERM with 143.

    Example::

        with kumo.job() as run:
            result = Trainer(plan).fit(table, snapshot=snapshot)
            run.record(bundle_id=result.bundle_id, snapshot_id=snapshot.id)
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    ids = {'KUMO_JOB_ID': job_id, 'KUMO_JOB_RUN_ID': job_run_id}
    previous = {name: os.environ.get(name) for name in ids}
    for name, value in ids.items():
        if value is not None:
            os.environ[name] = value
    run = JobRun()
    shown = warnings.showwarning
    showing = threading.local()

    def show(message: Any, category: Any, *args: Any, **kwargs: Any) -> None:
        if getattr(showing, 'active', False):
            warnings._showwarnmsg_impl(  # type: ignore[attr-defined]
                warnings.WarningMessage(message, category, *args, **kwargs))
            return
        text = str(message)[:_MAX_WARNING]
        recorded = run.summary['warnings']
        if (not issubclass(category, _DEPRECATIONS) and text not in recorded
                and len(recorded) < _MAX_WARNINGS):
            recorded.append(text)
        showing.active = True
        try:
            shown(message, category, *args, **kwargs)
        finally:
            showing.active = False

    warnings.showwarning = show
    error: Optional[BaseException] = None
    try:
        with _sigterm_raises():
            yield run
    except BaseException as raised:
        error = raised
        if isinstance(raised, (Exception, Terminated)):
            traceback.print_exception(raised, file=sys.stderr)
            raise SystemExit(_exit_code(raised)) from raised
        raise
    finally:
        warnings.showwarning = shown
        try:
            _print_summary(run, error)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _print_summary(run: JobRun, error: Optional[BaseException]) -> None:
    from kumo.engine._provenance import scheduler_context

    code = _exit_code(error)
    line = {
        'event': 'kumo.engine.job',
        'status': 'succeeded' if code == 0 else 'failed',
        'exit_code': code,
        'scheduler': scheduler_context(),
        **run.summary,
    }
    if code != 0 and error is not None:
        line['error'] = {
            'type': type(error).__name__,
            'kind': _error_kind(error),
            'message': str(error)[:_MAX_MESSAGE],
        }
    print(json.dumps(line, default=str), flush=True)


def _terminate(signum: int, frame: Any) -> NoReturn:
    raise Terminated('Terminated by SIGTERM')


@contextlib.contextmanager
def _sigterm_raises() -> Iterator[None]:
    try:
        previous = signal.signal(signal.SIGTERM, _terminate)
    except ValueError:  # not the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
