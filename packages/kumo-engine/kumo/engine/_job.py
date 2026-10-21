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
import traceback
import warnings
from typing import Any, Dict, Iterator, Mapping, NoReturn, Optional

EXIT_OK = 0
EXIT_REFUSED = 10
EXIT_TRANSIENT = 20
EXIT_FAILED = 30
EXIT_TERMINATED = 143

_MAX_MESSAGE = 500


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
               snapshot_id: Optional[str] = None,
               run_id: Optional[str] = None,
               outputs: Optional[Mapping[str, Any]] = None) -> None:
        r"""Adds ids and outputs to the summary; ``None`` leaves a field."""
        for key, value in (('bundle_id', bundle_id),
                           ('snapshot_id', snapshot_id), ('run_id', run_id)):
            if value is not None:
                self.summary[key] = value
        self.summary['outputs'].update(outputs or {})


def exit_code(error: BaseException) -> int:
    r"""The exit code for a failure: 10 a source refused the request (do not
    retry), 20 a transient source, connection or timeout error (retry), 30
    anything else, 143 SIGTERM."""
    from kumo_core.sources.errors import (
        PermanentSourceError,
        TransientSourceError,
    )

    cause: Optional[BaseException] = error
    while cause is not None:
        if isinstance(cause, Terminated):
            return EXIT_TERMINATED
        if isinstance(cause, PermanentSourceError):
            return EXIT_REFUSED
        if isinstance(cause,
                      (TransientSourceError, ConnectionError, TimeoutError)):
            return EXIT_TRANSIENT
        cause = cause.__cause__
    return EXIT_FAILED


@contextlib.contextmanager
def job(*, job_id: Optional[str] = None,
        job_run_id: Optional[str] = None) -> Iterator[JobRun]:
    r"""Runs the body as one scheduled job.

    ``job_id``/``job_run_id`` (else ``KUMO_JOB_ID``/``KUMO_JOB_RUN_ID`` or the
    detected scheduler) are recorded in the manifests and tracking tags the
    body writes. Logs go to stderr. On exit one JSON line goes to stdout:
    ``status``, ``exit_code``, the scheduler context, what the body passed to
    :meth:`JobRun.record`, warnings raised, and ``error`` on failure. A failed
    body exits the process with :func:`exit_code`; SIGTERM exits with 143.

    Example::

        with kumo.job() as run:
            result = Trainer(plan).fit(table, snapshot=snapshot)
            run.record(bundle_id=result.bundle_id, snapshot_id=snapshot.id)
    """
    from kumo.engine._provenance import scheduler_context

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    ids = {'KUMO_JOB_ID': job_id, 'KUMO_JOB_RUN_ID': job_run_id}
    previous = {name: os.environ.get(name) for name in ids}
    for name, value in ids.items():
        if value is not None:
            os.environ[name] = value
    run = JobRun()
    shown = warnings.showwarning

    def show(message: Any, *args: Any, **kwargs: Any) -> None:
        run.summary['warnings'].append(str(message))
        shown(message, *args, **kwargs)

    warnings.showwarning = show
    error: Optional[BaseException] = None
    try:
        with _sigterm_raises():
            yield run
    except (Exception, Terminated) as raised:
        error = raised
    finally:
        warnings.showwarning = shown
    code = EXIT_OK if error is None else exit_code(error)
    line = {
        'event': 'kumo.engine.job',
        'status': 'succeeded' if error is None else 'failed',
        'exit_code': code,
        'scheduler': scheduler_context(),
        **run.summary,
    }
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    if error is not None:
        traceback.print_exception(error, file=sys.stderr)
        line['error'] = {
            'type': type(error).__name__,
            'message': str(error)[:_MAX_MESSAGE]
        }
    print(json.dumps(line, default=str), flush=True)
    if error is not None:
        raise SystemExit(code) from error


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
