# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import _snowflake_experiment_fake
import numpy as np
import pytest
from _snowflake_experiment_fake import FakeExperimentTracking, FakeSession
from kumo_core.summaries import Note, Scalar, Series, Table, Tag
from kumoapi.plugin import PublishedModel
from test_tracking_contract import (
    CONTEXT,
    EVENTS,
    PREDICTION_MANIFEST,
    QUERY,
    RESULT,
)

from kumo.engine._manifest import TrackingRef
from kumo.engine._tracking import (
    RunRecord,
    TrackingSession,
    prediction_record,
    tracking_ref,
)
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
    trainer = kumo.Trainer(quick_plan)
    in_use = 'Snowflake ExperimentTracking is already in use'

    with pytest.raises(RuntimeError, match=in_use):
        trainer.fit(table, snapshot=snapshot, mlops=kumo.MLOps(
            str(tmp_path / 'root'), tracking=_tracking(session),
            store=False, on_error='raise'))
    tracker = _tracking(session)
    failure = log_run(tracker, tracker.log_prediction_run,
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


def test_metric_names_are_the_ones_mlflow_logs(fake):
    record = RunRecord(name='fit', tags={'kumo.kind': 'fit'})
    tracking = _tracking()
    tracking.start(record)
    tracking.finish(replace(record, metrics={'test/map@10': 0.5}), None)

    run, = fake._instance.runs.values()
    assert ('test/map_at_10', 0.5, 0) in run['metrics']


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


STARTED = datetime(2026, 10, 29, 6, 18, tzinfo=timezone.utc)
REGISTERED = SimpleNamespace(
    registered_model=PublishedModel('churn', '3', 'DB.ML.CHURN/3'),
    registered_alias=None,
    manifest=SimpleNamespace(holdout=None, tracking=TrackingRef(
        platform='snowflake', run_id='FIT_CHURN_20261029_061800', url=None)))


def test_a_fit_run_gets_a_card_and_its_duration(fake, monkeypatch):
    import kumo.engine.integrations.snowflake as snowflake

    monkeypatch.setattr(snowflake, '_now',
                        lambda: STARTED + timedelta(seconds=90))
    tags = {'kumo.kind': 'fit', 'kumo.task_type': 'regression',
            'kumo.tune_metric': 'mae', 'kumo.entity': 'ents'}
    start = RunRecord(name='fit', tags=tags, description=f'`{QUERY}`',
                      started=STARTED)
    tracking = _tracking()
    tracking.start(start)
    tracking.finish(replace(start, metrics={'test/mae': 0.32}, tags={
        **tags, 'kumo.data_warnings': '1', 'kumo.bundle_id': 'b' * 64},
        summaries=(
            Note('problems', 'link `ents.ID → users`: 14 % of rows match'),
            Note('lineage/model', '`ents` v1 → graph snapshot g → training '
                 'table t'),
            Table('locations', ('kind', 'id', 'location'),
                  (('model', 'b' * 64, 's3://b/models/b'),), brief=True),
            Table('data/splits', ('split', 'rows'),
                  (('train', 8), ('val', 1), ('test', 1)), brief=True),
            Series('holdout/by_k/map', ((1, 0.5), (10, 0.25)), x_label='k'),
            Series('pred/score_histogram', ((0, 1.0), (1, 2.0)),
                   'score_pct', chart=True),
            Table('data_preparation', ('step', 'seconds', 'bytes', 'reused'),
                  (('training_table', 12.5, 4300, False),), brief=True),
            Tag('prep.training_table.seconds', '12.5'),
            Tag('prep.training_table.bytes', '4300'),
        )), None)

    run, = fake._instance.runs.values()
    assert run['artifacts']['run_card.md'] == '\n'.join([
        'holdout mae 0.32, 1 data problem', '', '',
        'Fit for regression of ents.', '', f'`{QUERY}`', '', '**Problems**',
        '', '- link `ents.ID → users`: 14 % of rows match', '',
        '**Holdout metrics**', '', '| metric | value |', '| --- | ---: |',
        '| test/mae | 0.32 |', '', '**Training data**', '', '| split | rows |',
        '| --- | ---: |', '| train | 8 |', '| val | 1 |', '| test | 1 |', '',
        '**Data preparation**', '', '| step | time (s) | disk |',
        '| --- | ---: | --- |', '| Training table | 12.5 | 4.2 KiB |', '',
        '**Notes**', '', '- Charts: `pred/score_histogram`: step = score × '
        '100 (step 37 = scores 0.37–0.38).', '', '**Lineage**', '', '- `ents` v1 → graph snapshot g → training table '
        't → model `bbbbbbbb`', '', '**Where things are**', '', '| Artifact | Location |', '| --- | --- |',
        '| model bundle `bbbbbbbb` | [`s3://b/models/b`](https://s3.console.'
        'aws.amazon.com/s3/buckets/b?prefix=models/b/) |', '',
        'Full lineage: `_provenance.json`. Tables: `holdout/by_k.json`.'])
    assert ('duration_seconds', 90.0, 0) in run['metrics']
    assert {('pred/score_histogram', 1.0, 0),
            ('pred/score_histogram', 2.0, 1)} <= set(run['metrics'])
    assert not any(name.startswith('charts/') for name in run['artifacts'])
    assert run['params']['prep.training_table.seconds'] == '12.5'
    assert run['params']['prep.training_table.bytes'] == '4300'


def test_a_prediction_run_names_its_model_and_lasts_as_long_as_it_ran(fake):
    record = prediction_record(PREDICTION_MANIFEST, model=REGISTERED)
    timed = replace(record, started=STARTED,
                    finished=STARTED + timedelta(seconds=60))

    _tracking().log_prediction_run(timed)

    run, = fake._instance.runs.values()
    assert run['params']['kumo.model'] == 'churn/3'
    assert run['artifacts']['run_card.md'] == (
        '50 rows as of 2024-04-29\n\n\n'
        'Scores of `churn` v3, trained by `FIT_CHURN_20261029_061800`.\n\n'
        'Full lineage: `_provenance.json`.')
    assert ('duration_seconds', 60.0, 0) in run['metrics']


def test_a_prediction_run_logs_its_skipped_rows(fake, tmp_path):
    (tmp_path / 'part-0.parquet').write_bytes(b'PAR1\xff')
    (tmp_path / '_SUCCESS').touch()
    record = prediction_record(PREDICTION_MANIFEST, model=REGISTERED,
                               files={'skipped': str(tmp_path)})

    _tracking().log_prediction_run(record)

    run, = fake._instance.runs.values()
    assert run['artifacts']['skipped/part-0.parquet'] == b'PAR1\xff'
    assert run['artifacts']['skipped/_SUCCESS'] == ''


def _prediction(as_of, version=None, label='churn'):
    tags = {'kumo.kind': 'prediction', 'kumo.as_of': as_of,
            'kumo.bundle_id': 'b' * 64,
            **({'kumo.model_version': version} if version else {})}
    return RunRecord(tags=tags, model_label=label)


def test_a_repeated_prediction_run_gets_a_number(fake, caplog):
    tracking = _tracking()
    for _ in range(3):
        tracking.log_prediction_run(_prediction('2026-10-29', '6'))

    assert list(fake._instance.runs) == [
        'PREDICT_CHURN_V6_2026_10_29', 'PREDICT_CHURN_V6_2026_10_29_2',
        'PREDICT_CHURN_V6_2026_10_29_3']
    assert {run['status'] for run in fake._instance.runs.values()} == {
        'FINISHED'}
    assert ('Snowflake run PREDICT_CHURN_V6_2026_10_29 already exists in '
            'experiment ents; recording this run as '
            'PREDICT_CHURN_V6_2026_10_29_3') in caplog.text


def _with_runs(fake, count):
    session = FakeSession()
    fake(session).runs = {f'R{i}': {'experiment': 'ENTS'}
                          for i in range(count)}
    return session


def test_an_experiment_near_its_run_limit_warns(fake, caplog):
    session = _with_runs(fake, 450)

    _tracking(session).log_prediction_run(_prediction('2026-10-29', '6'))

    assert 'PREDICT_CHURN_V6_2026_10_29' in fake._instance.runs
    assert ('Snowflake experiment ents holds 450 of the 500 runs it can; '
            'track into a new experiment before it is full') in caplog.text


def test_a_full_experiment_is_not_tracked_into(fake, caplog):
    from kumo.engine._tracking import log_run

    session = _with_runs(fake, 500)

    tracker = _tracking(session)
    failure = log_run(tracker, tracker.log_prediction_run,
                      _prediction('2026-10-29', '6'))

    assert len(fake._instance.runs) == 500
    assert fake._instance._run is None
    assert str(failure) == (
        'Snowflake experiment ents holds 500 runs, the most it can (500), so '
        'this prediction run is not tracked; track into a new experiment, '
        "e.g. SnowflakeTracking(..., experiment_name='ents_2')")
    assert 'the prediction run is not tracked' in caplog.text


def _monitoring(monitor):
    return RunRecord(
        tags={'kumo.kind': 'monitoring', 'kumo.monitor': monitor,
              'kumo.as_of': '2026-10-29', 'kumo.model_name': 'churn',
              'kumo.model_version': '6', 'kumo.bundle_id': 'b' * 64},
        model_label='churn', summaries=(
            Scalar('drift/psi_prev', 0.01),
            Table('drift/inputs/orders', ('column', 'psi'),
                  (('AMOUNT', 0.12),)),
            Table('drift/table_rows', ('table', 'rows at training', 'rows now'),
                  (('orders', 100, 104),), brief=True),
            Note('notes', 'Labels mature 2026-11-05.')))


@pytest.mark.parametrize('run_name, record, name', [
    (None, RunRecord(tags={'kumo.kind': 'fit'}, model_label='kumo-demo-churn',
                     started=datetime(2026, 10, 29, 6, 18, 5,
                                      tzinfo=timezone.utc)),
     'FIT_KUMO_DEMO_CHURN_20261029_061805'),
    (None, _prediction('2026-10-29', '6', 'kumo-demo-churn'),
     'PREDICT_KUMO_DEMO_CHURN_V6_2026_10_29'),
    (None, _prediction('2026-10-29', 'V2'), 'PREDICT_CHURN_V2_2026_10_29'),
    (None, _prediction('2026-10-28..2026-10-29',
                       label='users-binary_classification'),
     'PREDICT_USERS_BINARY_CLASSIFICATION_BBBBBBBB_2026_10_28_2026_10_29'),
    ('nightly', RunRecord(tags={'kumo.kind': 'fit'}), 'FIT_NIGHTLY'),
    ('nightly', _prediction('2026-10-29', '6'), 'PREDICT_NIGHTLY_2026_10_29'),
    ('nightly', _monitoring('drift'), 'DRIFT_NIGHTLY_2026_10_29'),
    ('nightly', _monitoring('evaluate'), 'EVAL_NIGHTLY_2026_10_29'),
])
def test_runs_are_named_by_kind_model_and_date_or_the_users_name(
        fake, run_name, record, name):
    tracking = _tracking(run_name=run_name)
    tracking.start(record)
    tracking.finish(record, None)

    assert list(fake._instance.runs) == [name]


def test_each_monitoring_call_is_a_run_named_like_its_prediction(fake):
    tracking = _tracking()
    for monitor in ('drift', 'evaluate', 'drift'):
        tracking.log_monitoring(_monitoring(monitor))

    assert list(fake._instance.runs) == [
        'DRIFT_CHURN_V6_2026_10_29', 'EVAL_CHURN_V6_2026_10_29',
        'DRIFT_CHURN_V6_2026_10_29_2']
    run = fake._instance.runs['DRIFT_CHURN_V6_2026_10_29']
    assert run['status'] == 'FINISHED'
    assert ('drift/psi_prev', 0.01, 0) in run['metrics']
    assert set(run['artifacts']) == {'drift/inputs/orders.json', 'run_card.md'}
    assert run['artifacts']['run_card.md'] == '\n'.join([
        'as of 2026-10-29', '', '',
        'Drift check of the scores of `churn` v6.', '', '**Table rows**', '',
        '| table | rows at training | rows now |', '| --- | ---: | ---: |',
        '| orders | 100 | 104 |', '', '**Notes**', '',
        '- Labels mature 2026-11-05.', '',
        'Full lineage: `_provenance.json`. Tables: `drift/inputs/orders.json`.'])


@pytest.mark.parametrize('seconds', [
    0, -1, float('nan'), float('inf'), True, None, '300',
    pytest.param(threading.TIMEOUT_MAX * 2, id='over-thread-limit'),
    pytest.param(10**1000, id='overflow'),
])
def test_invalid_tracking_timeout_fails_before_provider_import(
        monkeypatch, seconds):
    import kumo.engine.integrations.snowflake as module

    def unexpected(*args):
        raise AssertionError('Provider import must follow timeout validation')

    monkeypatch.setattr(module, 'import_provider', unexpected)
    with pytest.raises(ValueError, match='finite and positive'):
        module.SnowflakeTracking(None, experiment_name='test',
                                 finish_timeout_seconds=seconds)


def test_snowflake_tracking_finish_budget(fake):
    assert _tracking().finish_timeout_seconds == 30.0
    assert _tracking(finish_timeout_seconds=90.0).finish_timeout_seconds == 90.0


@pytest.mark.parametrize('seconds', [np.int64(120), np.float64(120), threading.TIMEOUT_MAX])
def test_snowflake_tracking_accepts_supported_real_budgets(fake, seconds):
    assert _tracking(finish_timeout_seconds=seconds).finish_timeout_seconds == float(seconds)


@pytest.mark.parametrize('pause_at', ['log_metrics', 'start_run', 'end_run'])
def test_reuse_after_timeout_preserves_background_run(fake, monkeypatch, pause_at):
    import kumo.engine._tracking as module
    from kumo.engine._tracking import RunSpec

    entered, release, finished, rejected = (threading.Event() for _ in range(4))
    original = getattr(fake, pause_at)

    def paused(self, *args, **kwargs):
        if pause_at == 'end_run':
            result = original(self, *args, **kwargs)
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        if pause_at != 'end_run':
            result = original(self, *args, **kwargs)
        return result

    monkeypatch.setattr(fake, pause_at, paused)
    monkeypatch.setattr(module, 'FINISH_SECONDS', 2.0)
    adapter = _tracking(finish_timeout_seconds=0.01)
    original_finish = adapter.finish

    def finish(*args):
        try:
            return original_finish(*args)
        finally:
            if threading.current_thread().name == 'kumo-tracking':
                (finished if release.is_set() else rejected).set()

    monkeypatch.setattr(adapter, 'finish', finish)
    spec = RunSpec(kind='fit', name='FIRST', metrics={'loss': 1.0})
    first = TrackingSession(adapter)
    first.start(lambda: spec)
    if pause_at != 'start_run':
        assert first.pointer() is not None
    else:
        assert entered.wait(2)
    first.finish(None, None)
    try:
        assert entered.wait(2)
        assert isinstance(first.error, TimeoutError)
        second = TrackingSession(adapter)
        second.start(lambda: replace(spec, name='SECOND'))
        assert rejected.wait(2)
        assert second.pointer() is None
        second.finish(None, None)
        assert 'already in use' in str(second.error)
    finally:
        release.set()
    assert finished.wait(3)
    # Join only public worker threads so final ref publication has completed.
    for worker in threading.enumerate():
        if worker.name == 'kumo-tracking':
            worker.join(3)
    assert first.final is not None
    assert fake._instance.runs[first.final.run_id]['status'] == 'FINISHED'
    assert sum(c[0] == 'end_run' for c in fake._instance.calls) == 1
    adapter.log_prediction_run(_prediction('2026-10-30'))
    assert len(fake._instance.runs) == 2
    assert all(run['status'] == 'FINISHED' for run in fake._instance.runs.values())


@pytest.mark.parametrize('kind', ['direct', 'prediction', 'session'])
def test_failed_start_releases_adapter_and_preserves_provider_error(fake, monkeypatch, kind):
    from kumo.engine._tracking import RunSpec

    adapter = _tracking()
    failure = ValueError('provider rejected parameters')
    original = fake.log_params

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(fake, 'log_params', fail)
    if kind == 'session':
        session = TrackingSession(adapter)
        session.start(lambda: RunSpec(kind='fit', name='FIRST'))
        assert session.pointer() is None
        session.finish(None, None)
        assert session.error is failure
    else:
        with pytest.raises(ValueError) as caught:
            if kind == 'direct':
                adapter.start(RunRecord(tags={'kumo.kind': 'fit'}))
            else:
                adapter.log_prediction_run(_prediction('2026-10-30'))
        assert caught.value is failure
    assert next(iter(fake._instance.runs.values()))['status'] == 'FAILED'
    monkeypatch.setattr(fake, 'log_params', original)
    adapter.log_prediction_run(_prediction('2026-10-31'))
    assert list(fake._instance.runs.values())[-1]['status'] == 'FINISHED'


def test_final_ref_belongs_to_completed_run_when_adapter_is_immediately_reused(fake, monkeypatch):
    from kumo.engine._tracking import RunSpec

    adapter = _tracking()
    original = adapter.finish
    second = RunRecord(tags={'kumo.kind': 'fit'}, name='SECOND')

    def finish(record, error):
        original(record, error)
        adapter.start(second)
        original(second, None)

    monkeypatch.setattr(adapter, 'finish', finish)
    session = TrackingSession(adapter)
    session.start(lambda: RunSpec(kind='fit', name='FIRST'))
    first = session.pointer()
    assert first is not None
    session.finish(None, None)
    assert session.error is None
    assert session.final == first
    assert adapter.run_id != first.run_id


@pytest.mark.parametrize('during_start', [False, True])
def test_context_cleanup_failure_still_releases_adapter(fake, monkeypatch, during_start):
    from contextlib import ExitStack

    adapter = _tracking()
    original_close = ExitStack.close
    failed = False

    def close(self):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError('context cleanup failed')
        original_close(self)

    monkeypatch.setattr(ExitStack, 'close', close)
    if during_start:
        provider_error = ValueError('parameters rejected')
        original_params = fake.log_params

        def params(*args, **kwargs):
            raise provider_error

        monkeypatch.setattr(fake, 'log_params', params)
        with pytest.raises(ValueError) as caught:
            adapter.log_prediction_run(_prediction('2026-10-30'))
        assert caught.value is provider_error
        monkeypatch.setattr(fake, 'log_params', original_params)
    else:
        with pytest.raises(RuntimeError, match='context cleanup failed'):
            adapter.log_prediction_run(_prediction('2026-10-30'))
    adapter.log_prediction_run(_prediction('2026-10-31'))
    assert list(fake._instance.runs.values())[-1]['status'] == 'FINISHED'


@pytest.mark.parametrize('state', ['busy', 'full_queue', 'idle'])
def test_model_timeout_defers_subclass_cleanup_to_owner(fake, monkeypatch, tmp_path, state):
    import kumo.engine._tracking as module
    from kumo.engine._tracking import RunSpec
    from kumo.engine.integrations.snowflake import SnowflakeTracking

    entered, release, returned, idle = (threading.Event() for _ in range(4))
    owner_threads, ended_threads, results = [], [], []
    original_end = fake.end_run
    original_put = module.queue.Queue.put_nowait
    admissions = []
    producer = threading.current_thread()
    overflow = threading.Event()
    caller = None

    def put(self, item):
        try:
            result = original_put(self, item)
        except module.queue.Full:
            if threading.current_thread() is caller:
                overflow.set()
            raise
        if (threading.current_thread() is producer
                and entered.is_set() and not release.is_set()):
            admissions.append(self.qsize())
        return result

    def end(self, *args, **kwargs):
        ended_threads.append(threading.current_thread())
        return original_end(self, *args, **kwargs)

    class ModelTracking(SnowflakeTracking):
        def log_model(self, *args):
            owner_threads.append(threading.current_thread())
            entered.set()
            assert release.wait(5)
            returned.set()
            if state == 'busy':
                raise ValueError('model logging failed after timeout')

    monkeypatch.setattr(fake, 'end_run', end)
    monkeypatch.setattr(module, 'MODEL_SECONDS', 0.01)
    monkeypatch.setattr(module, 'QUEUE_SIZE', 2 if state == 'full_queue' else 10)
    if state == 'full_queue':
        monkeypatch.setattr(module.queue.Queue, 'put_nowait', put)
    adapter = ModelTracking(FakeSession(), experiment_name='ents',
                            finish_timeout_seconds=2)
    session = TrackingSession(adapter, warn=False)
    session.start(lambda: RunSpec(kind='fit', name='FIRST'))
    reference = session.pointer()
    assert reference is not None
    caller = threading.Thread(target=lambda: results.append(
        session.log_model(tmp_path, None, None)))
    if state == 'idle':
        original_get = module.queue.Queue.get
        original_clock = module.time.monotonic

        def get(self, *args, **kwargs):
            if threading.current_thread() in owner_threads and returned.is_set():
                idle.set()
            return original_get(self, *args, **kwargs)

        def clock():
            if threading.current_thread() is caller and entered.is_set():
                release.set()
                assert idle.wait(2)
            return original_clock()

        monkeypatch.setattr(module.queue.Queue, 'get', get)
        monkeypatch.setattr(module.time, 'monotonic', clock)
    caller.start()
    try:
        assert entered.wait(2)
        if state == 'full_queue':
            session.on_event(EVENTS[0])
            session.on_event(EVENTS[0])
            assert admissions == [1, 2]
        caller.join(3)
        assert not caller.is_alive()
        if state == 'full_queue':
            assert overflow.is_set()
        published, error = results[0]
        assert published is None
        assert isinstance(error, TimeoutError)
        assert session.error is error
        if state != 'idle':
            assert fake._instance.runs[reference.run_id]['status'] == 'RUNNING'
            assert ended_threads == []
            rejected = TrackingSession(adapter)
            rejected.start(lambda: RunSpec(kind='fit', name='SECOND'))
            assert rejected.pointer() is None
            rejected.finish(None, None)
            assert 'already in use' in str(rejected.error)
        else:
            assert idle.is_set()
    finally:
        release.set()
        caller.join(3)
        session.finish(None, None)
    owner_threads[0].join(3)
    assert not owner_threads[0].is_alive()
    assert ended_threads == owner_threads
    assert fake._instance.runs[reference.run_id]['status'] == 'FAILED'
    adapter.log_prediction_run(_prediction('2026-11-01'))
    assert list(fake._instance.runs.values())[-1]['status'] == 'FINISHED'


def test_finish_survives_full_queue_and_preserves_fifo(fake, monkeypatch):
    import kumo.engine._tracking as module
    from kumo.engine._events import Event
    from kumo.engine._tracking import RunSpec
    from kumo.engine.integrations.snowflake import SnowflakeTracking

    entered, release = threading.Event(), threading.Event()
    calls, owners = [], []

    class QueuedTracking(SnowflakeTracking):
        def on_trial(self, record, event):
            calls.append(event.trial)
            if event.trial == 1:
                owners.append(threading.current_thread())
                entered.set()
                assert release.wait(5)

        def finish(self, record, error):
            if record is not None:
                calls.append(('finish', record.name))
            super().finish(record, error)

    monkeypatch.setattr(module, 'QUEUE_SIZE', 2)
    adapter = QueuedTracking(FakeSession(), experiment_name='ents',
                             finish_timeout_seconds=0.01)
    session = TrackingSession(adapter)
    session.start(lambda: RunSpec(kind='fit', name='FIRST'))
    reference = session.pointer()
    assert reference is not None

    def event(trial):
        return Event('trial_started', 0, trial=trial, params={'attempt': 1})

    session.on_event(event(1))
    try:
        assert entered.wait(2)
        session.on_event(event(2))
        session.on_event(event(3))
        began = module.time.monotonic()
        session.finish(None, None)
        assert module.time.monotonic() - began < 1
        error = session.error
        assert isinstance(error, TimeoutError)
        session.on_event(event(4))
        session.finish(lambda: RunSpec(kind='fit', name='REPLACEMENT'), None)
        assert session.error is error
        assert calls == [1]
    finally:
        release.set()
    owners[0].join(3)
    assert not owners[0].is_alive()
    assert calls == [1, 2, 3, ('finish', 'FIRST')]
    assert session.final == reference
    assert fake._instance.runs[reference.run_id]['status'] == 'FINISHED'
    adapter.log_prediction_run(_prediction('2026-11-02'))
    assert list(fake._instance.runs.values())[-1]['status'] == 'FINISHED'


@pytest.mark.parametrize('phase', ['start', 'event'])
@pytest.mark.parametrize('exception_type', [SystemExit, KeyboardInterrupt])
def test_unexpected_worker_exit_closes_its_owned_run(fake, monkeypatch, phase, exception_type):
    from kumo.engine._tracking import RunSpec
    from kumo.engine.integrations.snowflake import SnowflakeTracking

    failure = exception_type('callback terminated worker')
    exited = threading.Event()
    exceptions = []

    def unhandled(args):
        exceptions.append(args)
        exited.set()

    class InterruptedTracking(SnowflakeTracking):
        def start(self, record):
            super().start(record)
            if phase == 'start' and record.tags.get('kumo.kind') == 'fit':
                raise failure

        def on_trial(self, record, event):
            raise failure

    monkeypatch.setattr(threading, 'excepthook', unhandled)
    adapter = InterruptedTracking(FakeSession(), experiment_name='ents')
    session = TrackingSession(adapter)
    session.start(lambda: RunSpec(kind='fit', name='INTERRUPTED'))
    reference = session.pointer()
    if phase == 'event':
        assert reference is not None
        session.on_event(EVENTS[0])
    else:
        assert reference is None
    assert exited.wait(2)
    worker = exceptions[0].thread
    worker.join(3)
    assert not worker.is_alive()
    assert exceptions[0].exc_value is failure
    assert session.error is failure
    assert not session.enabled
    assert next(iter(fake._instance.runs.values()))['status'] == 'FAILED'
    # The failed callback is specific to fit; prediction still reuses this adapter.
    adapter.log_prediction_run(_prediction('2026-11-03'))
    assert list(fake._instance.runs.values())[-1]['status'] == 'FINISHED'


def test_pending_cleanup_preserves_worker_exception(fake, monkeypatch, tmp_path):
    import kumo.engine._tracking as module
    from kumo.engine._tracking import RunSpec
    from kumo.engine.integrations.snowflake import SnowflakeTracking

    entered, release, exited = (threading.Event() for _ in range(3))
    primary, secondary = SystemExit('callback stopped'), KeyboardInterrupt('cleanup stopped')
    exceptions = []
    original_end = fake.end_run

    def unhandled(args):
        exceptions.append(args)
        exited.set()

    def end(self, *args, **kwargs):
        original_end(self, *args, **kwargs)
        raise secondary

    class InterruptedTracking(SnowflakeTracking):
        def on_trial(self, record, event):
            entered.set()
            assert release.wait(5)
            raise primary

        def log_model(self, *args):
            pytest.fail('The model hook is queued behind the interrupted event')

    monkeypatch.setattr(threading, 'excepthook', unhandled)
    monkeypatch.setattr(module, 'MODEL_SECONDS', 0.01)
    adapter = InterruptedTracking(FakeSession(), experiment_name='ents')
    session = TrackingSession(adapter, warn=False)
    session.start(lambda: RunSpec(kind='fit', name='INTERRUPTED'))
    reference = session.pointer()
    assert reference is not None
    session.on_event(EVENTS[0])
    try:
        assert entered.wait(2)
        published, timeout = session.log_model(tmp_path, None, None)
        assert published is None
        assert isinstance(timeout, TimeoutError)
        monkeypatch.setattr(fake, 'end_run', end)
    finally:
        release.set()
    assert exited.wait(2)
    worker = exceptions[0].thread
    worker.join(3)
    assert not worker.is_alive()
    assert exceptions[0].exc_value is primary
    assert session.error is timeout
    assert fake._instance.runs[reference.run_id]['status'] == 'FAILED'
    monkeypatch.setattr(fake, 'end_run', original_end)
    adapter.log_prediction_run(_prediction('2026-11-04'))
    assert list(fake._instance.runs.values())[-1]['status'] == 'FINISHED'


@pytest.mark.parametrize('mode', ['full_queue', 'admitted', 'admitted_fast_cleanup'])
@pytest.mark.parametrize('exception_type', [SystemExit, ValueError])
def test_model_requests_stop_before_worker_cleanup_returns(
        fake, monkeypatch, tmp_path, mode, exception_type):
    import kumo.engine._tracking as module
    from kumo.engine._tracking import RunSpec
    from kumo.engine.integrations.snowflake import SnowflakeTracking

    entered, fail, requested, cleanup, release = (
        threading.Event() for _ in range(5))
    primary = exception_type('event stopped tracking')
    owners, exceptions, results, admissions = [], [], [], []
    original_put, original_end = module.queue.Queue.put_nowait, fake.end_run
    producer, caller = threading.current_thread(), None

    def check_result(result):
        published, error = result
        assert published is None
        assert type(error) is RuntimeError
        assert error.__cause__ is primary
        assert 'tracking stopped after an earlier failure' in str(error)

    def put(self, item):
        try:
            result = original_put(self, item)
        except module.queue.Full:
            if threading.current_thread() is caller:
                requested.set()
            raise
        if threading.current_thread() is caller:
            requested.set()
        elif threading.current_thread() is producer and entered.is_set():
            admissions.append(self.qsize())
        return result

    def end(self, *args, **kwargs):
        result = original_end(self, *args, **kwargs)
        cleanup.set()
        if mode != 'admitted_fast_cleanup':
            assert release.wait(5)
        return result

    class InterruptedTracking(SnowflakeTracking):
        def on_trial(self, record, event):
            owners.append(threading.current_thread())
            entered.set()
            assert fail.wait(5)
            raise primary

        def log_model(self, *args):
            pytest.fail('A disabled worker must not call the model hook')

    monkeypatch.setattr(threading, 'excepthook', exceptions.append)
    monkeypatch.setattr(module, 'MODEL_SECONDS', 60)
    monkeypatch.setattr(module, 'QUEUE_SIZE', 2)
    monkeypatch.setattr(module.queue.Queue, 'put_nowait', put)
    monkeypatch.setattr(fake, 'end_run', end)
    adapter = InterruptedTracking(FakeSession(), experiment_name='ents')
    session = TrackingSession(adapter, warn=False)
    session.start(lambda: RunSpec(kind='fit', name='INTERRUPTED'))
    reference = session.pointer()
    assert reference is not None
    session.on_event(EVENTS[0])
    try:
        assert entered.wait(2)
        admissions.clear()
        if mode == 'full_queue':
            session.on_event(EVENTS[0])
            session.on_event(EVENTS[0])
            assert admissions == [1, 2]
        caller = threading.Thread(target=lambda: results.append(
            session.log_model(tmp_path, None, None)))
        caller.start()
        assert requested.wait(2)
        fail.set()
        assert cleanup.wait(2)
        caller.join(2)
        assert not caller.is_alive()
        assert len(results) == 1
        check_result(results[0])
        began = module.time.monotonic()
        session.on_event(EVENTS[0])
        check_result(session.log_model(tmp_path, None, None))
        assert module.time.monotonic() - began < 1
    finally:
        fail.set()
        release.set()
        if caller is not None:
            caller.join(3)
        session.finish(None, None)
    owners[0].join(3)
    assert not owners[0].is_alive()
    assert session.error is primary
    assert [item.exc_value for item in exceptions] == (
        [primary] if exception_type is SystemExit else [])
    assert fake._instance.runs[reference.run_id]['status'] == 'FAILED'
    check_result(session.log_model(tmp_path, None, None))
    monkeypatch.setattr(fake, 'end_run', original_end)
    adapter.log_prediction_run(_prediction('2026-11-05'))
    assert list(fake._instance.runs.values())[-1]['status'] == 'FINISHED'


def test_fatal_worker_collects_timeout_handoff_after_pending_drain(
        fake, monkeypatch, tmp_path):
    import kumo.engine._tracking as module
    from kumo.engine._tracking import RunSpec
    from kumo.engine.integrations.snowflake import SnowflakeTracking

    drained, release, exited = (threading.Event() for _ in range(3))
    owners, exceptions = [], []
    primary = SystemExit('event stopped tracking')
    gated = False
    original_queue = module.queue.SimpleQueue

    class GatedQueue(original_queue):
        def get_nowait(self):
            nonlocal gated
            try:
                return super().get_nowait()
            except module.queue.Empty:
                if threading.current_thread() in owners and not gated:
                    gated = True
                    drained.set()
                    assert release.wait(5)
                raise

    def unhandled(args):
        exceptions.append(args)
        exited.set()

    class InterruptedTracking(SnowflakeTracking):
        def on_trial(self, record, event):
            owners.append(threading.current_thread())
            raise primary

        def log_model(self, *args):
            pytest.fail('The model job cannot run after a fatal callback')

    monkeypatch.setattr(module.queue, 'SimpleQueue', GatedQueue)
    monkeypatch.setattr(module, 'MODEL_SECONDS', 0.01)
    monkeypatch.setattr(threading, 'excepthook', unhandled)
    adapter = InterruptedTracking(FakeSession(), experiment_name='ents')
    session = TrackingSession(adapter, warn=False)
    session.start(lambda: RunSpec(kind='fit', name='INTERRUPTED'))
    reference = session.pointer()
    assert reference is not None
    session.on_event(EVENTS[0])
    try:
        assert drained.wait(2)
        published, timeout = session.log_model(tmp_path, None, None)
        assert published is None
        assert isinstance(timeout, TimeoutError)
    finally:
        release.set()
    assert exited.wait(2)
    owners[0].join(3)
    assert not owners[0].is_alive()
    assert exceptions[0].exc_value is primary
    assert session.error is timeout
    assert fake._instance.runs[reference.run_id]['status'] == 'FAILED'
    adapter.log_prediction_run(_prediction('2026-11-06'))
    assert list(fake._instance.runs.values())[-1]['status'] == 'FINISHED'


def test_model_hook_base_exception_is_returned_unchanged(fake, tmp_path):
    from kumo.engine._tracking import RunSpec
    from kumo.engine.integrations.snowflake import SnowflakeTracking

    failure = SystemExit('model SDK stopped')

    class InterruptedTracking(SnowflakeTracking):
        def log_model(self, *args):
            raise failure

    adapter = InterruptedTracking(FakeSession(), experiment_name='ents')
    session = TrackingSession(adapter, warn=False)
    session.start(lambda: RunSpec(kind='fit', name='INTERRUPTED'))
    reference = session.pointer()
    assert reference is not None
    assert session.log_model(tmp_path, None, None) == (None, failure)
    assert session.error is failure
    session.finish(None, failure)
    assert fake._instance.runs[reference.run_id]['status'] == 'FAILED'
