# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import threading
import time
from unittest import mock

import _snowflake_experiment_fake
import pytest
from _snowflake_experiment_fake import FakeExperimentTracking, FakeSession
from test_tracking_contract import (
    CONTEXT,
    EVENTS,
    RESULT,
    join_tracking_threads,
)

from kumo.engine._tracking import TrackingSession
from kumo.engine.trainer._run_spec import fit_spec, fitted_spec

FAKED = ('__init__', 'set_experiment', 'start_run', 'end_run', 'log_params',
         'log_metrics', 'log_artifact')


@pytest.mark.parametrize('method', FAKED)
def test_fake_matches_installed_experiment_tracking(method):
    experiment = pytest.importorskip('snowflake.ml.experiment')
    real = inspect.signature(getattr(experiment.ExperimentTracking, method))
    fake = inspect.signature(getattr(FakeExperimentTracking, method))
    assert [(p.name, p.kind, p.default) for p in fake.parameters.values()
            ] == [(p.name, p.kind, p.default) for p in real.parameters.values()]


def test_installed_singleton_exposes_what_the_adapter_reads(monkeypatch):
    experiment = pytest.importorskip('snowflake.ml.experiment')
    cls = experiment.ExperimentTracking
    monkeypatch.setattr(cls, '_instance', None)
    session = mock.MagicMock()
    session.get_current_database.return_value = 'DB'
    session.get_current_schema.return_value = 'PUBLIC'

    tracking = cls(session)

    assert cls(mock.MagicMock()) is tracking
    assert tracking._run is None
    assert tracking._session is session
    assert 'experiment_name' in inspect.signature(
        tracking._sql_client.show_runs_in_experiment).parameters


@pytest.fixture
def fake(monkeypatch):
    return _snowflake_experiment_fake.install(monkeypatch)


def _tracking(session=FakeSession, **kwargs):
    from kumo.engine.integrations.snowflake import SnowflakeTracking
    return SnowflakeTracking(
        session() if session is FakeSession else session,
        experiment_name='ents', **kwargs)


def _replay(tracking, result=RESULT, error=None, during=None):
    session = TrackingSession(tracking)
    session.start(lambda: fit_spec(CONTEXT))
    for event in EVENTS:
        session.on_event(event)
    if during is not None:
        caught_up = threading.Event()
        session._submit(caught_up.set)
        caught_up.wait(30)
        during()
    session.finish(
        None if result is None else
        lambda: fitted_spec(session.spec, result, EVENTS[-1].metrics['seconds']),
        error)


def test_native_model_lineage_holds_the_run_open_during_the_fit(fake):
    lineage = []
    _replay(
        _tracking(database='ml', schema='runs', run_name='fit',
                  native_model_lineage=True),
        during=lambda: lineage.extend(fake._instance.lineage_stack))

    assert lineage == [('ENTS', 'FIT')]
    assert fake._instance.lineage_stack == []
    assert fake._instance.calls[1] == ('set_experiment', 'ents', 'ml', 'runs')


def test_snowflake_errors_warn_once_and_end_the_run_failed(fake, caplog):
    session = FakeSession()
    fake(session).fail_on = {'log_metrics'}

    _replay(_tracking(session))

    tracking = fake._instance
    assert caplog.text.count('off for the rest of this run') == 1
    run, = tracking.runs.values()
    assert run['status'] == 'FAILED'
    assert tracking._run is None


def test_failed_fit_ends_the_run_failed(fake):
    _replay(_tracking(), None, ValueError('stop'))

    run, = fake._instance.runs.values()
    assert run['status'] == 'FAILED'


@pytest.mark.parametrize('busy', ['active run', 'other session'])
def test_an_instance_in_use_is_left_alone(fake, busy, caplog):
    session = FakeSession()
    user = fake(session, warnings_as_errors=True)
    user.set_experiment('mine')
    if busy == 'active run':
        user.start_run('mine_1')
    else:
        session = FakeSession()
    calls = len(user.calls)

    _replay(_tracking(session))

    assert user.calls[calls:] == []
    assert user.warnings_as_errors is True
    assert caplog.text.count('not tracked in Snowflake') == 1


def test_location_defaults_to_the_session_not_the_last_fit(fake):
    session = FakeSession(database='DEV', schema='WORK')
    _replay(_tracking(session, database='ml'))
    _replay(_tracking(session))

    assert [
        call[1:] for call in fake._instance.calls if call[0] == 'set_experiment'
    ] == [('ents', 'ml', 'PUBLIC'), ('ents', 'DEV', 'WORK')]
    assert not any(call[0].startswith('run.') for call in fake._instance.calls)


@pytest.mark.parametrize('taken', ['FIT', 'MYXRUN'])
def test_an_existing_run_is_never_resumed(fake, taken, caplog):
    session = FakeSession()
    user = fake(session)
    user.set_experiment('ents')
    user.start_run(taken)
    if taken == 'FIT':
        user.end_run()
    else:
        user._run = None  # still running, in another process
    run_name = 'fit' if taken == 'FIT' else 'my_run'

    tracking = _tracking(session, run_name=run_name)
    _replay(tracking)

    new = f"{run_name}_{CONTEXT.run_id.rsplit('-', 1)[-1][:8]}".upper()
    assert tracking.run_id == new
    assert set(user.runs) == {taken, new}
    assert user.runs[taken]['params'] == {}
    assert user.runs[new]['status'] == 'FINISHED'
    assert 'recording this fit as' in caplog.text


def test_metric_and_param_limits(fake, monkeypatch):
    import kumo.engine.integrations.snowflake as snowflake

    monkeypatch.setattr(snowflake, '_EPOCH_METRICS', 2)
    monkeypatch.setattr(snowflake, '_SUMMARY_METRICS', 3)
    monkeypatch.setattr(snowflake, '_MAX_METRICS', 5)
    result = type(RESULT)(metrics={**RESULT.metrics, 'r2': float('nan')},
                          id=RESULT.id, bundle_dir=RESULT.bundle_dir)
    _replay(_tracking(tags={'note': 'a$$b', 'level': 2}), result)

    run, = fake._instance.runs.values()
    assert sorted({name for name, _, _ in run['metrics']}) == [
        'test_mae', 'trial-1-attempt-1/best_val_mae',
        'trial-1-attempt-1/train_loss', 'trial-1-attempt-1/val_mae', 'val_mae'
    ]
    assert run['params']['kumo.dropped_metrics'] == '6'
    assert run['params']['kumo.dropped_params'] == '1'
    assert 'note' not in run['params']
    assert run['params']['level'] == '2'


def test_quoted_run_names_are_taken_only_by_their_exact_name():
    from kumo.engine.integrations.snowflake import _taken

    assert _taken('"MyRun"', ['MyRun'])
    assert not _taken('"MyRun"', ['MYRUN'])
    assert _taken('my_run', ['MYXRUN'])


def test_the_users_session_is_not_altered(fake):
    session = FakeSession()

    _replay(_tracking(session))

    assert session.statements == []


def test_a_hung_snowflake_call_does_not_stall_the_fit(fake, monkeypatch,
                                                      caplog):
    import kumo.engine._tracking as tracking

    monkeypatch.setattr(tracking, 'FINISH_SECONDS', 0.5)
    release = threading.Event()
    monkeypatch.setattr(FakeExperimentTracking, 'log_metrics',
                        lambda *args, **kwargs: release.wait(30))
    began = time.monotonic()
    try:
        _replay(_tracking())
        elapsed = time.monotonic() - began
    finally:
        release.set()
        join_tracking_threads()

    assert elapsed < 1.5
    assert 'did not finish within' in caplog.text
    run, = fake._instance.runs.values()
    assert run['status'] == 'FINISHED'


class _SchemaSession:
    def __init__(self, schemas):
        self.schemas = schemas
        self.statements = []

    def get_current_database(self):
        return 'DB'

    def get_current_schema(self):
        return 'RUNS'

    def sql(self, statement):
        self.statements.append(statement)
        return mock.Mock(collect=lambda: self.schemas)


def test_a_connector_opens_the_session_and_the_schema_is_checked_first(
        fake):
    import kumo.engine as kumo

    session = _SchemaSession([('RUNS', )])
    tracking = _tracking(None, connector=mock.Mock(session=lambda: session))
    tracking.check()
    assert tracking.session is session
    assert session.statements == ["SHOW SCHEMAS LIKE 'RUNS' IN DATABASE DB"]

    tracking.session = _SchemaSession([])
    with pytest.raises(ValueError, match='needs schema DB.RUNS'):
        kumo.MLOps('/tmp/root', tracking=tracking)
    with pytest.raises(ValueError, match='a session or a connector'):
        _tracking(None)
