# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import _snowflake_experiment_fake
import pytest
from _snowflake_experiment_fake import FakeExperimentTracking, FakeSession
from kumo_core.summaries import Note, Series, Table
from kumoapi.model_plan import ModelPlan
from kumoapi.plugin import PublishedModel
from test_fit_events import table_and_snapshot  # noqa: F401
from test_tracking_contract import (
    CONTEXT,
    EVENTS,
    PREDICTION_MANIFEST,
    QUERY,
    RESULT,
    join_tracking_threads,
)

from kumo.engine._manifest import TrackingRef
from kumo.engine._tracking import RunRecord, TrackingSession, prediction_record
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
        fake, busy, table_and_snapshot, tmp_path, monkeypatch):
    import kumo.engine as kumo
    from kumo.engine._tracking import log_run
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
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 2
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(plan)
    in_use = 'Snowflake ExperimentTracking is in use in this process'

    with pytest.raises(RuntimeError, match=in_use):
        trainer.fit(table, snapshot=snapshot, mlops=kumo.MLOps(
            str(tmp_path / 'root'), tracking=_tracking(session),
            store=False, on_error='raise'))
    failure = log_run(_tracking(session), 'log_prediction_run',
                      prediction_record(PREDICTION_MANIFEST))

    assert trainer.result is not None
    assert in_use in str(failure)
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

    new = tracking.run_id
    assert re.fullmatch(f'{run_name}_[0-9a-f]{{8}}'.upper(), new)
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
        'duration_seconds', 'mae', 'trial-1-attempt-1/train/loss',
        'trial-1-attempt-1/val/best_mae', 'trial-1-attempt-1/val/mae'
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


def test_a_hung_snowflake_call_does_not_stall_the_fit(fake, monkeypatch):
    import kumo.engine._tracking as tracking

    monkeypatch.setattr(tracking, 'FINISH_SECONDS', 0.5)
    release = threading.Event()
    monkeypatch.setattr(FakeExperimentTracking, 'log_metrics',
                        lambda *args, **kwargs: release.wait(30))
    began = time.monotonic()
    try:
        session = _replay(_tracking())
        elapsed = time.monotonic() - began
    finally:
        release.set()
        join_tracking_threads()

    assert elapsed < 1.5
    assert isinstance(session.error, TimeoutError)
    run, = fake._instance.runs.values()
    assert run['status'] == 'FINISHED'


def test_prediction_runs_are_never_resumed(fake, caplog):
    tracking = _tracking()
    record = prediction_record(PREDICTION_MANIFEST)
    tracking.log_prediction_run(record)
    tracking.log_prediction_run(record)

    first, second = fake._instance.runs
    assert first == 'PREDICT_20261021_0000'
    assert re.fullmatch('PREDICT_20261021_0000_[0-9A-F]{8}', second)
    assert {run['status'] for run in fake._instance.runs.values()} == {
        'FINISHED'}
    assert 'already exists' in caplog.text


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
    from kumo.engine.connector.base import Connector

    session = _SchemaSession([('RUNS', )])
    connector = mock.Mock(spec=Connector, session=lambda: session)
    tracking = _tracking(connector)
    tracking.check()
    assert tracking.session is session and tracking.connector is connector
    assert session.statements == ["SHOW SCHEMAS LIKE 'RUNS' IN DATABASE DB"]

    tracking.session = _SchemaSession([])
    with pytest.raises(ValueError, match='needs schema DB.RUNS'):
        kumo.MLOps('/tmp/root', tracking=tracking)
    with pytest.raises(ValueError, match='or a Snowpark session'):
        _tracking(None)


def test_mlops_loads_a_registered_model_through_the_tracker(
        fake, monkeypatch, tmp_path):
    import kumo.engine as kumo
    from kumo.engine.connector.base import Connector

    loaded = []
    original = kumo.Trainer.load_from_connector.__func__
    monkeypatch.setattr(
        kumo.Trainer, 'load_from_connector',
        classmethod(lambda cls, source, **kwargs: loaded.append(
            (source, kwargs)) if not isinstance(source, kumo.MLOps) else
                    original(cls, source, **kwargs)))
    session = _SchemaSession([('RUNS', )])
    connector = mock.Mock(spec=Connector, session=lambda: session)
    tracking = _tracking(connector)
    cfg = kumo.MLOps(str(tmp_path), tracking=tracking)
    location = kumo.ModelRegistryLocation('CHURN', alias='champion')
    kumo.Trainer.load_from_connector(cfg, location=location)
    assert loaded == [(connector, {
        'location': location, 'embedding_models': None,
        'accelerator': 'auto'
    })]
    with pytest.raises(ValueError, match='SnowflakeTracking\\(connector'):
        _tracking().load_model(location)


@dataclass(frozen=True)
class _Timed(RunRecord):
    started: Optional[datetime] = None
    finished: Optional[datetime] = None


STARTED = datetime(2026, 10, 29, 6, 18, tzinfo=timezone.utc)
REGISTERED = SimpleNamespace(
    registered_model=PublishedModel('churn', '3', 'DB.ML.CHURN/3'),
    registered_alias=None,
    manifest=SimpleNamespace(holdout=None, tracking=TrackingRef(
        platform='snowflake', run_id='CHURN_FIT_20261029_061800', url=None)))


def test_a_fit_run_gets_a_card_and_its_duration(fake, monkeypatch):
    import kumo.engine.integrations.snowflake as snowflake

    monkeypatch.setattr(snowflake, '_now',
                        lambda: STARTED + timedelta(seconds=90))
    tags = {'kumo.kind': 'fit', 'kumo.task_type': 'regression',
            'kumo.tune_metric': 'mae'}
    start = _Timed(name='fit', tags=tags, description=f'`{QUERY}`',
                   started=STARTED)
    tracking = _tracking()
    tracking.start(start)
    tracking.finish(replace(start, metrics={'mae': 0.32}, tags={
        **tags, 'kumo.data_warnings': '1'}, summaries=(
            Note('problems', 'link `ents.ID → users`: 14 % of rows match'),
            Note('locations', 'bundle `models/b/`'),
            Table('data/splits', ('split', 'rows'),
                  (('train', 8), ('val', 1), ('test', 1)), brief=True),
            Series('holdout/by_k/map', ((1, 0.5), (10, 0.25)), x_label='k'),
        )), None)

    run, = fake._instance.runs.values()
    assert run['artifacts']['run_card.md'] == '\n'.join([
        'mae 0.32 · 1 data problem · regression of ents',
        f'`{QUERY}`', '',
        '**Problems:** link `ents.ID → users`: 14 % of rows match', '',
        '**Training data:**', '', '| split | rows |', '| --- | ---: |',
        '| train | 8 |', '| val | 1 |', '| test | 1 |', '',
        '**Where things are**', '- bundle `models/b/`', '',
        'Full lineage: `_provenance.json`. Tables: `holdout/by_k.json`.'])
    assert ('duration_seconds', 90.0, 0) in run['metrics']


def test_a_prediction_run_names_its_model_and_lasts_as_long_as_it_ran(fake):
    record = prediction_record(PREDICTION_MANIFEST, model=REGISTERED)
    timed = _Timed(**vars(record), started=STARTED,
                   finished=STARTED + timedelta(seconds=60))

    _tracking().log_prediction_run(timed)

    run, = fake._instance.runs.values()
    assert run['params']['kumo.model'] == 'churn/3'
    assert run['artifacts']['run_card.md'] == (
        'v3 churn\n'
        'Model: `churn` v3, trained by `CHURN_FIT_20261029_061800`.\n\n'
        'Full lineage: `_provenance.json`.')
    assert ('duration_seconds', 60.0, 0) in run['metrics']
