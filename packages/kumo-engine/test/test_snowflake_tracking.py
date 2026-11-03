# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import threading
from datetime import datetime, timezone
from unittest import mock

import _snowflake_experiment_fake
import pytest
from _snowflake_experiment_fake import FakeExperimentTracking, FakeSession
from test_tracking_contract import CONTEXT, EVENTS, RESULT

from kumo.engine._manifest import TrackingRef
from kumo.engine._tracking import RunRecord, TrackingSession, tracking_ref
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
        lambda: fitted_spec(session.spec, result),
        error)
    return session


def test_native_model_lineage_holds_the_run_open_during_the_fit(fake):
    lineage = []
    _replay(
        _tracking(database='ml', schema_name='runs', run_name='fit',
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
def test_an_instance_in_use_is_left_alone_and_fails_strict_mode(
        fake, busy, table_and_snapshot, quick_plan, tmp_path, monkeypatch):
    import kumo.engine as kumo
    from kumo.engine.integrations.snowflake import SnowflakeTracking

    monkeypatch.setattr(SnowflakeTracking, 'check', lambda *args: None)
    session = FakeSession()
    user = fake(session, warnings_as_errors=True)
    user.set_experiment('mine')
    if busy == 'active run':
        user.start_run('mine_1')
    else:
        session = FakeSession()
    calls, run = len(user.calls), user._run
    table, snapshot = table_and_snapshot
    trainer = kumo.Trainer(quick_plan)
    in_use = 'Snowflake ExperimentTracking is already in use'

    with pytest.raises(RuntimeError, match=in_use):
        trainer.fit(table, snapshot=snapshot, mlops=kumo.MLOps(
            str(tmp_path / 'root'), tracking=_tracking(session),
            store=False, on_error='raise'))

    assert trainer.result is not None
    assert user.calls[calls:] == []
    assert (user._run, user.warnings_as_errors) == (run, True)


def test_location_defaults_to_the_session_not_the_last_fit(fake):
    session = FakeSession(database='DEV', schema='WORK')
    _replay(_tracking(session, database='ml'))
    _replay(_tracking(session))

    assert [
        call[1:] for call in fake._instance.calls if call[0] == 'set_experiment'
    ] == [('ents', 'ml', 'PUBLIC'), ('ents', 'DEV', 'WORK')]
    assert not any(call[0].startswith('run.') for call in fake._instance.calls)


@pytest.mark.parametrize('taken,run_name,new', [
    ('FIT', 'fit', 'FIT_2'), ('FIT_MYXRUN', 'my_run', 'FIT_MY_RUN_2')])
def test_an_existing_run_is_never_resumed(fake, taken, run_name, new, caplog):
    session = FakeSession()
    user = fake(session)
    user.set_experiment('ents')
    user.start_run(taken)
    if taken == 'FIT':
        user.end_run()
    else:
        user._run = None

    tracking = _tracking(session, run_name=run_name)
    _replay(tracking)

    assert tracking.run_id == tracking.logged_run_name == new
    assert tracking_ref(tracking) == TrackingRef(
        platform='snowflake', run_id=new, experiment='ENTS', run_name=new)
    assert set(user.runs) == {taken, new}
    assert user.runs[taken]['params'] == {}
    assert user.runs[new]['status'] == 'FINISHED'
    assert 'recording this run as' in caplog.text


def test_metric_and_param_limits(fake, monkeypatch):
    import kumo.engine.integrations.snowflake as snowflake

    monkeypatch.setattr(snowflake, '_EPOCH_METRICS', 2)
    monkeypatch.setattr(snowflake, '_SUMMARY_METRICS', 3)
    monkeypatch.setattr(snowflake, '_MAX_METRICS', 5)
    result = type(RESULT)(**{**vars(RESULT), 'metrics': {
        **RESULT.metrics, 'r2': float('nan')}})
    _replay(_tracking(tags={'note': 'a$$b', 'level': 2}), result)

    run, = fake._instance.runs.values()
    assert sorted({name for name, _, _ in run['metrics']}) == [
        'TRIAL_1/val/best_mae', 'TRIAL_2_ATTEMPT_2/val/best_mae',
        'duration_seconds', 'test/mae'
    ]
    assert run['params']['kumo.dropped_metrics'] == '2'
    assert run['params']['kumo.dropped_params'] == '1'
    assert 'note' not in run['params']
    assert run['params']['level'] == '2'


def test_the_users_session_is_not_altered(fake):
    session = FakeSession()

    _replay(_tracking(session))

    assert session.statements == []


def _connected():
    from kumo.engine.connector.base import Connector

    session = FakeSession(schema='RUNS', schemas=[('RUNS', )])
    connector = mock.Mock(spec=Connector, session=lambda: session)
    return _tracking(connector), connector, session


def test_a_connector_opens_the_session_and_the_schema_is_checked_first(
        fake):
    import kumo.engine as kumo

    tracking, connector, session = _connected()
    tracking.check()
    assert tracking.session is session and tracking.connector is connector
    assert session.statements == ["SHOW SCHEMAS LIKE 'RUNS' IN DATABASE DB"]

    tracking.session = FakeSession(schema='RUNS')
    with pytest.raises(ValueError, match='schema DB.RUNS does not exist'):
        kumo.MLOps('/tmp/root', tracking=tracking)
    with pytest.raises(ValueError, match='or a Snowpark session'):
        _tracking(None)


def test_mlops_loads_a_registered_model_through_the_tracker(
        fake, monkeypatch, tmp_path):
    import kumo.engine as kumo

    loaded = []
    original = kumo.Trainer.load_from_connector.__func__
    monkeypatch.setattr(
        kumo.Trainer, 'load_from_connector',
        classmethod(lambda cls, source, **kwargs: loaded.append(
            (source, kwargs)) if not isinstance(source, kumo.MLOps) else
                    original(cls, source, **kwargs)))
    tracking, connector, _ = _connected()
    cfg = kumo.MLOps(str(tmp_path), tracking=tracking)
    location = kumo.ModelRegistryLocation('CHURN', alias='champion')
    kumo.Trainer.load_from_connector(cfg, location=location)
    assert loaded == [(connector, {
        'location': location, 'embedding_models': None,
        'accelerator': 'auto'
    })]
    with pytest.raises(ValueError, match='SnowflakeTracking\\(connector'):
        _tracking().load_model(location)


@pytest.mark.parametrize('run_name, record, name', [
    (None, RunRecord(tags={'kumo.kind': 'fit'}, model_label='kumo-demo-churn',
                     started=datetime(2026, 10, 29, 6, 18, 5,
                                      tzinfo=timezone.utc)),
     'FIT_KUMO_DEMO_CHURN_20261029_061805'),
    ('nightly', RunRecord(tags={'kumo.kind': 'fit'}), 'FIT_NIGHTLY'),
])
def test_runs_are_named_by_kind_model_and_date_or_the_users_name(
        fake, run_name, record, name):
    tracking = _tracking(run_name=run_name)
    tracking.start(record)
    tracking.finish(record, None)

    assert list(fake._instance.runs) == [name]
