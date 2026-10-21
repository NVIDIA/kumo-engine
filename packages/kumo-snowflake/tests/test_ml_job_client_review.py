# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for dispatch ambiguity and long-lived observation."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from kumo_snowflake import ml_jobs
from kumo_snowflake.ml_job_contracts import (
    SnowflakeMLJobProgress,
    SnowflakeMLJobSubmissionError,
)
from kumoapi.common import JobStatus
from kumoapi.jobs import JobStatusReport, PredictionProgress


def entrypoint() -> None:
    pass


class MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, Any] = {}

    def put_json(self, location: str, payload: Any) -> None:
        self.objects[location] = payload

    def copy_file(self, *args: Any) -> None:
        pass

    def get_json(self, location: str, **kwargs: Any) -> Any:
        return self.objects.get(location)


def make_job() -> ml_jobs.SnowflakeMLJob:
    return ml_jobs.SnowflakeMLJob(
        'DB.PUBLIC.SERVICE', stage_name='@DB.PUBLIC.STAGE',
        execution_id='execution', progress_location='progress.json',
        result_location='result.json', session=object(),
        handle=SimpleNamespace(status='RUNNING'))


@pytest.mark.parametrize('failure', [
    ConnectionError('response lost'),
    ml_jobs.SnowflakeMLJobTransientError('SDK failed', retryable=True)
])
def test_dispatch_ambiguity_never_invites_retry(monkeypatch: Any,
                                                failure: Exception) -> None:
    store = MemoryStore()
    dispatched: list[str] = []

    def remote(pool: str, **options: Any) -> Any:
        def decorate(function: Any) -> Any:
            def invoke(*args: Any) -> None:
                # Simulate creation succeeding but acknowledgement being lost.
                dispatched.append(options['name'])
                raise failure

            return invoke

        return decorate

    monkeypatch.setattr(ml_jobs, 'StageStore', lambda *args: store)
    monkeypatch.setattr(ml_jobs, '_require_gpu_pool', lambda *args: None)
    monkeypatch.setattr(ml_jobs, '_preflight_stage', lambda *args: None)
    monkeypatch.setattr(ml_jobs, 'discover_installed_wheel', lambda: object())
    monkeypatch.setattr(
        ml_jobs, 'stage_wheel',
        lambda *args: SimpleNamespace(name='runtime.whl', sha256='digest'))
    monkeypatch.setattr(ml_jobs, '_load_jobs_api',
                        lambda: SimpleNamespace(remote=remote))

    with pytest.raises(SnowflakeMLJobSubmissionError) as caught:
        ml_jobs.submit_ml_job(entrypoint, compute_pool='GPU',
                              stage_name='@DB.PUBLIC.STAGE', session=object())

    assert len(dispatched) == 1
    assert not caught.value.retryable
    assert caught.value.__cause__ is failure
    manifest = next(iter(store.objects.values()))
    assert manifest['submission_state'] == 'dispatch_unknown'
    assert manifest['definition_name'] in str(caught.value)
    assert manifest['execution_id'] in str(caught.value)
    assert '@DB.PUBLIC.STAGE/' in str(caught.value)


def test_progress_falls_back_to_service_on_stage_outage(
        monkeypatch: Any) -> None:
    job = make_job()

    class UnavailableStore(MemoryStore):
        def get_json(self, *args: Any, **kwargs: Any) -> Any:
            raise ConnectionError('stage unavailable')

    monkeypatch.setattr(job, '_store', UnavailableStore)
    progress = job.progress()
    assert progress.status_report.status is JobStatus.RUNNING


def progress(status: JobStatus) -> SnowflakeMLJobProgress:
    return SnowflakeMLJobProgress(
        execution_id='execution', updated_at=datetime.now(timezone.utc),
        status_report=JobStatusReport(status=status, tracking_url='',
                                      start_time=datetime.now(timezone.utc),
                                      event_log=[]),
        prediction=PredictionProgress(total_iterations=10,
                                      completed_iterations=5))


class Bar:
    def __init__(self, **kwargs: Any) -> None:
        self.n = 0
        self.closed = False

    def update(self, increment: float) -> None:
        self.n += increment

    def refresh(self) -> None:
        pass

    def set_postfix_str(self, detail: str) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def test_attach_retries_progress_and_result_then_closes_bar(
        monkeypatch: Any) -> None:
    job = make_job()
    samples = iter([
        progress(JobStatus.RUNNING),
        ConnectionError('blip'),
        progress(JobStatus.DONE),
        progress(JobStatus.DONE)
    ])
    results = iter([TimeoutError('result read timed out'), 'durable result'])
    bars: list[Bar] = []
    sleeps: list[float] = []

    def sample() -> Any:
        value = next(samples)
        if isinstance(value, Exception):
            raise value
        return value

    def result() -> Any:
        value = next(results)
        if isinstance(value, Exception):
            raise value
        return value

    def new_bar(**kwargs: Any) -> Bar:
        bar = Bar(**kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(job, 'progress', sample)
    monkeypatch.setattr(job, 'result', result)
    monkeypatch.setattr('tqdm.auto.tqdm', new_bar)
    monkeypatch.setattr(ml_jobs.time, 'sleep', sleeps.append)
    assert job.attach(interval_s=0.1) == 'durable result'
    assert len(sleeps) == 3
    assert bars and all(bar.closed for bar in bars)


def test_attach_closes_bar_on_permanent_error(monkeypatch: Any) -> None:
    job = make_job()
    samples = iter([progress(JobStatus.RUNNING), ValueError('bad metadata')])
    bar = Bar()

    def sample() -> Any:
        value = next(samples)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(job, 'progress', sample)
    monkeypatch.setattr('tqdm.auto.tqdm', lambda **kwargs: bar)
    monkeypatch.setattr(ml_jobs.time, 'sleep', lambda interval: None)
    with pytest.raises(ValueError, match='bad metadata'):
        job.attach(interval_s=0.1)
    assert bar.closed


def test_permission_error_ignores_implicit_connection_context() -> None:
    try:
        raise ConnectionError('earlier connection failed')
    except ConnectionError:
        try:
            raise PermissionError('not allowed')
        except PermissionError as error:
            assert ml_jobs._exception_kind(error) == 'permission'


def test_permission_wins_over_explicit_transient_cause() -> None:
    error = type('SnowparkSQLException', (Exception, ),
                 {'sqlstate': '42501'})('denied')
    error.__cause__ = ConnectionError('transport problem')
    assert ml_jobs._exception_kind(error) == 'permission'


def test_explicit_cause_retries_but_implicit_context_does_not() -> None:
    error = RuntimeError('wrapper')
    error.__context__ = ConnectionError('unrelated')
    assert ml_jobs._exception_kind(error) == 'unexpected'
    error.__cause__ = ConnectionError('actual cause')
    assert ml_jobs._exception_kind(error) == 'transient'


def test_generic_snowpark_client_error_is_not_retryable() -> None:
    error = type('SnowparkClientException', (Exception, ), {})('invalid client')
    assert ml_jobs._exception_kind(error) == 'sql'


def test_message_substring_on_unrelated_exception_is_not_permission() -> None:
    assert ml_jobs._exception_kind(
        RuntimeError('not authorized')) == 'unexpected'


def test_tag_ddl_escapes_quotes_and_backslashes() -> None:
    statements: list[str] = []
    session = SimpleNamespace(
        sql=lambda sql: SimpleNamespace(collect=lambda: statements.append(sql)))
    ml_jobs._set_tags(
        session, 'DB.PUBLIC.SERVICE', {
            'DB.PUBLIC.REQUEST': "run' ; DROP SERVICE other; --\\tail",
        })
    assert statements == [
        'ALTER SERVICE "DB"."PUBLIC"."SERVICE" SET TAG DB.PUBLIC.REQUEST = '
        "'run'' ; DROP SERVICE other; --\\\\tail'"
    ]
