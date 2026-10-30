# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import math
import os
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
from kumo_core.summaries import Dataset, Note, Scalar, Table, Tag
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._manifest import ArtifactStats, TrackingRef
from kumo.engine._monitoring import evaluate, score_drift
from kumo.engine._scratch import get_scratch_dir
from kumo.engine._version import __version__
from kumo.engine.trainer.trainer import PredictionResult, TrainingResult

CUTOFF = pd.Timestamp('2024-03-31')
ANCHOR = CUTOFF - pd.Timedelta(days=1)
PREDICTED_AT = pd.Timestamp('2024-05-01', tz='UTC')


@pytest.fixture(scope='module')
def snapshots(tmp_path_factory):
    root = tmp_path_factory.mktemp('monitoring')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    rng = np.random.default_rng(0)
    data = root / 'data'
    data.mkdir()
    flag = pd.Series(rng.integers(0, 2, 60).astype(bool), dtype='boolean')
    orders = pd.DataFrame({
        'EID': np.arange(1500),
        'USER_ID': rng.integers(0, 60, 1500),
        'AMT': rng.gamma(2.0, 10.0, 1500),
        'TS': pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 1500), unit='D'),
    })

    def write(users_flag, rows):
        pd.DataFrame({
            'ID': np.arange(60),
            'FLAG': users_flag
        }).to_parquet(data / 'users.parquet')
        rows.to_parquet(data / 'orders.parquet')

    hidden = flag.copy()
    hidden[:12] = pd.NA
    write(hidden, orders[orders.TS < CUTOFF])
    # File versions are modification times in seconds.
    for path in data.iterdir():
        os.utime(path, (time.time() - 3600, ) * 2)
    connector = kumo.FileConnector(root_dir=str(data))
    users = kumo.Table.from_source_table(connector['users.parquet'],
                                         primary_key='ID').infer_metadata()
    events = kumo.Table.from_source_table(
        connector['orders.parquet'], primary_key='EID',
        time_column='TS').infer_metadata().with_column('USER_ID',
                                                       stype=Stype.ID)
    graph = kumo.Graph(
        tables={
            'users': users,
            'orders': events
        }, edges=[Edge(src_table='orders', fkey='USER_ID', dst_table='users')])
    before = graph.snapshot()
    write(flag, orders)
    after = graph.snapshot()
    yield SimpleNamespace(graph=graph, before=before, after=after,
                          orders=orders, flag=flag)
    kumo.configure(scratch_dir=previous)


@pytest.fixture(scope='module')
def classifier(snapshots):
    pquery = kumo.PredictiveQuery(
        graph=snapshots.graph, query='PREDICT COUNT(orders.*, 0, 7, days) > 0 '
        'FOR EACH users.ID')
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(model_plan=plan)
    trainer.fit(pquery.generate_training_table(snapshots.before),
                snapshot=snapshots.before)
    table = pquery.generate_prediction_table(snapshots.before)
    return trainer, trainer.predict(table, snapshot=snapshots.before,
                                    binary_classification_threshold=0.5)


@dataclass
class _Run:
    frame: pd.DataFrame
    artifact_type: Any = None
    manifest: Any = None
    id: str = 'run'
    bundle_id: Optional[str] = None

    @property
    def files(self):
        path = get_scratch_dir() / f'{id(self.frame)}.parquet'
        self.frame.to_parquet(path)
        return [path]


def _evaluate(snapshots, query, frame, anchors=()):
    trainer = SimpleNamespace(_training_query=query, result=None,
                              training_table_generation_plan=None)
    manifest = SimpleNamespace(
        prediction_table=SimpleNamespace(anchors=[str(a) for a in anchors]),
        timings=SimpleNamespace(started_at=PREDICTED_AT.isoformat()))
    return evaluate(trainer, _Run(frame, manifest=manifest), snapshots.after)


def _window(orders, anchor):
    rows = orders[(orders.TS > anchor)
                  & (orders.TS <= anchor + pd.Timedelta(days=7))]
    return rows.groupby('USER_ID')


def test_evaluate_matured_run(snapshots, classifier):
    trainer, run = classifier
    started = pd.Timestamp.now(tz='UTC')
    result = trainer.evaluate(run, labels_snapshot=snapshots.after)

    assert pd.Timestamp(result.anchors[0]).tz_convert(None) == ANCHOR
    assert (result.matured, result.bundle_matches,
            result.labelled_fraction) == (True, True, 1.0)
    assert result.rows == result.labelled == 60 and result.reasons == {}
    active = set(_window(snapshots.orders, ANCHOR).groups)
    rows = pd.read_parquet(result.actuals).set_index('ENTITY')
    labels = rows['TARGET']
    assert labels.to_dict() == {user: user in active for user in range(60)}
    expected = run.frame.set_index('ENTITY').loc[rows.index]
    assert (rows['True_PROB'] == expected['True_PROB']).all()
    assert (rows['TARGET_PRED'] == expected['TARGET_PRED']).all()
    assert rows['EVALUATED_AT'].nunique() == 1
    assert started <= rows['EVALUATED_AT'].iloc[0] <= pd.Timestamp.now(
        tz='UTC')

    from sklearn.metrics import roc_auc_score
    joined = run.frame.set_index('ENTITY').join(labels)
    assert result.metrics['auroc'] == pytest.approx(
        roc_auc_score(joined['TARGET'], joined['True_PROB']), abs=1e-6)

    with pytest.warns(UserWarning, match='labels are not mature yet'):
        immature = trainer.evaluate(run, labels_snapshot=snapshots.before)
    assert immature.matured is False
    assert immature.metrics == {'auroc': None, 'ap': None}
    assert 'undefined' in immature.reasons['metrics']


def test_evaluate_immature_run_returns_metrics_and_warns(
        snapshots, classifier, monkeypatch):
    import kumo_core.pquery

    trainer, run = classifier
    monkeypatch.setattr(kumo_core.pquery, '_target_maximum',
                        lambda *args: ANCHOR + pd.Timedelta(days=1))
    with pytest.warns(UserWarning, match='labels are not mature yet'):
        result = trainer.evaluate(run, labels_snapshot=snapshots.after)
    assert result.matured is False and result.labelled == 60
    assert result.metrics['auroc'] is not None and result.reasons == {}


def test_evaluate_joins_several_anchors(snapshots, classifier, monkeypatch):
    trainer, run = classifier
    anchors = [ANCHOR - pd.Timedelta(days=14), ANCHOR]
    stacked = _Run(pd.concat([
        run.frame.assign(ANCHOR_TIMESTAMP=anchor.tz_localize('UTC'))
        for anchor in anchors
    ], ignore_index=True))
    recorded = run.manifest
    monkeypatch.setattr(PredictionResult, 'files',
                        property(lambda _: stacked.files))
    monkeypatch.setattr(
        PredictionResult, 'manifest', property(lambda _: replace(
            recorded, prediction_table=replace(
                recorded.prediction_table, anchors=[
                    anchor.tz_localize('UTC').isoformat()
                    for anchor in anchors
                ]))))
    result = trainer.evaluate(run, labels_snapshot=snapshots.after)

    assert (result.rows, result.labelled, result.matured) == (120, 120, True)
    labels = pd.read_parquet(result.actuals).set_index(
        ['ANCHOR_TIMESTAMP', 'ENTITY'])['TARGET']
    for anchor in anchors:
        active = set(_window(snapshots.orders, anchor).groups)
        assert labels[anchor].to_dict() == {
            user: user in active
            for user in range(60)
        }

    monkeypatch.undo()
    monkeypatch.setattr(PredictionResult, 'manifest', property(lambda _: None))
    result = trainer.evaluate(run, labels_snapshot=snapshots.after)
    assert result.reasons['metrics'] == 'the run records no anchor timestamps'
    assert (result.metrics, result.actuals) == ({}, None)


def test_evaluate_regression_and_static_queries(snapshots):
    frame = pd.DataFrame({'ENTITY': np.arange(60), 'TARGET_PRED': 10.0})
    result = _evaluate(snapshots,
                       'PREDICT SUM(orders.AMT, 0, 7, days) FOR EACH users.ID',
                       frame, [ANCHOR])
    sums = _window(snapshots.orders, ANCHOR)['AMT'].sum()
    error = 10.0 - pd.Series({user: sums.get(user, 0.0) for user in range(60)})
    assert result.matured and result.labelled_fraction == 1.0
    assert result.metrics['mae'] == pytest.approx(error.abs().mean(), rel=1e-5)
    assert result.metrics['rmse'] == pytest.approx(
        math.sqrt((error**2).mean()), rel=1e-5)

    frame = pd.DataFrame({'ENTITY': np.arange(12), 'True_PROB': .5})
    result = _evaluate(snapshots, 'PREDICT users.FLAG FOR EACH users.ID',
                       frame)
    assert (result.matured, result.anchors) == (None, ())
    assert result.rows == 12 and result.labelled_fraction == 1.0
    rows = pd.read_parquet(result.actuals).set_index('ENTITY')
    assert all(rows['TARGET'][user] == snapshots.flag[user]
               for user in range(12))
    assert (rows['ANCHOR_TIMESTAMP'] == PREDICTED_AT).all()
    assert (rows['True_PROB'] == .5).all()

    unknown = pd.DataFrame({'ENTITY': np.arange(1000, 1012), 'True_PROB': .5})
    result = _evaluate(snapshots, 'PREDICT users.FLAG FOR EACH users.ID',
                       unknown)
    assert (result.rows, result.labelled, result.actuals) == (12, 0, None)
    assert result.reasons['metrics'] == 'no prediction has a label'
    with pytest.raises(ValueError, match='No evaluated rows'):
        result.export(None)


def test_evaluation_exports_its_rows(snapshots, classifier, tmp_path):
    from kumo.engine._monitoring import EvaluationResult
    from kumo.engine.trainer.output_config import PredictionOutputConfig

    trainer, run = classifier
    result = trainer.evaluate(run, labels_snapshot=snapshots.after)
    connector = kumo.FileConnector(root_dir=str(tmp_path))
    output = PredictionOutputConfig(connector,
                                    output_paths={'predictions': 'evaluated/'})
    result.export(output)
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / 'evaluated' / run.id), result.frame)

    loaded = EvaluationResult.load_from_connector(
        connector, path=f'evaluated/{run.id}/')
    assert replace(loaded, actuals=None) == replace(result, actuals=None)
    pd.testing.assert_frame_equal(pd.read_parquet(loaded.actuals),
                                  pd.read_parquet(result.actuals))

    written = (tmp_path / 'evaluated' / run.id / '_SUCCESS').stat().st_mtime_ns
    result.export(output)
    assert (tmp_path / 'evaluated' / run.id /
            '_SUCCESS').stat().st_mtime_ns == written


def test_evaluation_appends_to_a_table_with_its_ids(snapshots, classifier):
    from unittest.mock import Mock

    from kumoapi.jobs import PredictionArtifactType, WriteMode

    from kumo.engine.trainer.output_config import PredictionOutputConfig

    trainer, run = classifier
    result = trainer.evaluate(run, labels_snapshot=snapshots.after)
    sink = SimpleNamespace(write_files=Mock())
    output = PredictionOutputConfig(
        SimpleNamespace(sink_backend=sink), PredictionArtifactType.PREDICTIONS,
        'EVALUATED')
    result.export(output)
    (_, table, _, mode), kwargs = sink.write_files.call_args
    assert (table, mode) == ('EVALUATED', WriteMode.APPEND)
    assert kwargs['extra_columns'] == {'PREDICTION_ID': run.id,
                                       'BUNDLE_ID': trainer.result.id}


def test_a_snowflake_table_gets_the_types_model_monitor_needs(
        snapshots, classifier, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as parquet

    from kumo.engine.trainer.output_config import PredictionOutputConfig

    trainer, run = classifier
    result = trainer.evaluate(run, labels_snapshot=snapshots.after)
    written = []
    sink = SimpleNamespace(write_files=lambda connector, table, files, mode,
                           **_: written.extend(
                               parquet.read_table(f) for f in files))
    monkeypatch.setattr(kumo.SnowflakeConnector, 'sink_backend', sink)
    connector = kumo.SnowflakeConnector(warehouse='WH', database='ML',
                                        schema_name='S')
    result.export(PredictionOutputConfig(connector, path='EVALUATED'))
    schema = pa.concat_tables(written).schema
    assert schema.field('TARGET').type == pa.int8()
    assert schema.field('EVALUATED_AT').type == pa.timestamp('ns')
    original = pd.read_parquet(result.actuals)
    assert original.TARGET.dtype == bool
    assert pa.concat_tables(written).column('TARGET').to_pylist() == (
        original.TARGET.astype(int).tolist())


@pytest.mark.parametrize('threshold', [0.5, None])
def test_the_holdout_baseline_loads_with_the_evaluated_rows_types(
        snapshots, classifier, monkeypatch, threshold):
    import pyarrow as pa
    import pyarrow.parquet as parquet
    from kumo_core.artifact_export import prepared_artifact_file

    from kumo.engine.trainer.output_config import PredictionOutputConfig

    def write_files(connector, table, files, mode, extra_columns, **_):
        with prepared_artifact_file(files[0], extra_columns) as loaded:
            loads[table] = parquet.read_schema(loaded).remove_metadata()

    loads = {}
    monkeypatch.setattr(kumo.SnowflakeConnector, 'sink_backend',
                        SimpleNamespace(write_files=write_files))
    connector = kumo.SnowflakeConnector(warehouse='WH', database='ML',
                                        schema_name='S')
    trainer, _ = classifier
    run = trainer.predict(snapshots.before,
                          binary_classification_threshold=threshold)
    trainer.evaluate(run, labels_snapshot=snapshots.after).export(
        PredictionOutputConfig(connector, path='EVALUATED'))
    baseline = trainer.result.holdout_evaluation(
        binary_classification_threshold=threshold)
    baseline.export(PredictionOutputConfig(connector, path='BASELINE'))
    assert loads['EVALUATED'] == loads['BASELINE']
    assert loads['BASELINE'].field('TARGET_PRED').type == pa.bool_()

    rows = pd.read_parquet(baseline.actuals)
    holdout = trainer.result.holdout_df()
    assert rows.TARGET.tolist() == holdout.TARGET.tolist()
    assert rows.TARGET_PRED.tolist() == (
        (holdout.True_PROB > threshold).tolist() if threshold else
        [pd.NA] * len(holdout))


def _join(tmp_path, task, predictions, labels):
    import json

    import diskgraph

    predictions.to_parquet(tmp_path / 'predictions.parquet',
                           row_group_size=97)
    labels.to_parquet(tmp_path / 'labels.parquet', row_group_size=89)
    spec = {'task': task, 'prediction_keys': ['ENTITY'],
            'label_keys': ['ENTITY1'], 'label_target': 'TARGET1',
            'spill_dir': str(tmp_path / 'spill'), 'partition_rows': 500}
    return json.loads(diskgraph.evaluate_prediction_parquet(
        [str(tmp_path / 'predictions.parquet')],
        [str(tmp_path / 'labels.parquet')], json.dumps(spec)))


def test_realized_metrics_match_fit_time_definitions(tmp_path):
    import torch
    from kumo_core.evaluation.classification import (
        BinaryAUROC,
        BinaryAveragePrecision,
    )
    from torch_geometric.metrics import LinkPredMAP, LinkPredPrecision

    rng = np.random.default_rng(0)
    n = 2000
    score = np.round(rng.random(n), 2).astype(np.float32)
    target = rng.random(n) < score
    joined = _join(tmp_path, 'binary',
                   pd.DataFrame({'ENTITY': np.arange(n), 'True_PROB': score}),
                   pd.DataFrame({'ENTITY1': np.arange(n)[::-1],
                                 'TARGET1': target[::-1]}))
    for name, metric in (('auroc', BinaryAUROC()),
                         ('ap', BinaryAveragePrecision())):
        metric.update(torch.tensor(score), torch.tensor(target).long())
        assert joined['metrics'][name] == pytest.approx(
            float(metric.compute()), abs=1e-6)

    entities, k, items = 300, 5, 40
    ranked = -np.sort(-rng.random((entities, k)), axis=1)
    classes = np.stack([rng.choice(items, k, replace=False)
                        for _ in range(entities)])
    truth = [rng.choice(items, rng.integers(0, 4), replace=False)
             for _ in range(entities)]
    joined = _join(
        tmp_path, 'link',
        pd.DataFrame({'ENTITY': np.repeat(np.arange(entities), k),
                      'CLASS': classes.ravel(), 'SCORE': ranked.ravel()}),
        pd.DataFrame({'ENTITY1': np.arange(entities),
                      'TARGET1': [list(t) for t in truth]}))
    index = torch.tensor(np.stack([
        np.repeat(np.arange(entities), [len(t) for t in truth]),
        np.concatenate(truth)]))
    for name, metric in ((f'map@{k}', LinkPredMAP(k)),
                         (f'precision@{k}', LinkPredPrecision(k))):
        metric.update(torch.tensor(classes), index)
        assert joined['metrics'][name] == pytest.approx(
            float(metric.compute()), abs=1e-6)
    assert joined['labelled'] == entities


def _counted(scores, edges=(0.0, 0.5, 1.0), id='r'):
    counts = {'edges': {'score': list(edges)},
              'arrays': {'True_PROB': list(scores)}}
    return SimpleNamespace(id=id, bundle_id='b', manifest=SimpleNamespace(
        stats={'predictions': ArtifactStats(rows=sum(scores), counts=counts)}))


def test_score_drift_is_exact_on_the_counted_bins():
    holdout = _counted((0, 3, 1, 0)).manifest.stats['predictions'].counts
    trainer = SimpleNamespace(result=SimpleNamespace(
        id='b', artifact_paths={}, manifest=SimpleNamespace(
            holdout=SimpleNamespace(counts=holdout, truncated=False))))
    run = _counted((0, 1, 3, 0))

    report = score_drift(trainer, run, previous=run, snapshot=None,
                         training_snapshot=None)
    assert report.score == 'True_PROB' and report.bundle_matches
    assert (report.psi, report.ks, report.wasserstein) == (
        pytest.approx(math.log(3)), 0.5, 0.25)
    assert (report.previous_psi, report.previous_ks,
            report.previous_wasserstein) == (0, 0, 0)

    report = score_drift(trainer, run, previous=_counted((1, 3), (0.5,)),
                         snapshot=None, training_snapshot=None)
    assert report.previous_psi is report.previous_ks is None
    assert report.reasons['previous'] == (
        'the run and the previous run did not count the same score on the '
        'same bins')
    report = score_drift(SimpleNamespace(result=None),
                         SimpleNamespace(manifest=None), previous=run,
                         snapshot=None, training_snapshot=None)
    assert (report.prediction_id, report.score, report.psi) == (None, None,
                                                                None)
    assert report.reasons['holdout'] == report.reasons['previous'] == (
        'the run records no score counts')


def test_score_drift(snapshots, classifier, monkeypatch):
    trainer, run = classifier
    report = trainer.score_drift(run, previous=run, snapshot=snapshots.after,
                                 training_snapshot=snapshots.before)

    assert report.score == 'True_PROB' and math.isfinite(report.psi)
    assert math.isfinite(report.ks) and math.isfinite(report.wasserstein)
    assert report.holdout_truncated is False and report.bundle_matches
    assert (report.previous_psi, report.previous_ks,
            report.previous_wasserstein) == (0, 0, 0)
    sizes = (len(snapshots.orders[snapshots.orders.TS < CUTOFF]),
             len(snapshots.orders))
    assert report.input_changes == {'orders': sizes,
                                    'orders.USER_ID → users': sizes}
    assert report.reasons == {}
    amount = report.features['orders']['AMT']
    assert amount['type'] == 'numerical'
    assert amount['delta']['num_unique'] > 0

    found = trainer.score_drift(run)
    assert found.features == trainer.score_drift(
        run, snapshot=snapshots.before,
        training_snapshot=snapshots.before).features
    assert (report.prediction_id, report.previous_prediction_id) == (run.id,
                                                                     run.id)

    monkeypatch.setattr(TrainingResult, 'manifest', property(lambda self: None))
    report = trainer.score_drift(run)
    assert (report.psi, report.ks) == (None, None)
    assert report.reasons['holdout'] == (
        "the model's holdout records no score counts")


def test_monitoring_a_run_loaded_from_its_export(snapshots, classifier,
                                                 tmp_path):
    from kumo.engine.trainer.output_config import PredictionOutputConfig

    trainer, run = classifier
    connector = kumo.FileConnector(root_dir=str(tmp_path))
    output = PredictionOutputConfig(connector, path='p/')
    run.export(output)
    success = tmp_path / 'p' / run.id / '_SUCCESS'
    written = success.stat().st_mtime_ns
    run.export(output)
    assert success.stat().st_mtime_ns == written

    path = f'p/{run.id}/'
    loaded = PredictionResult.load_from_connector(connector, path=path)
    parts = loaded._exported.local / 'parts'
    assert loaded.manifest.id == run.id and not parts.exists()
    assert trainer.score_drift(loaded).psi is not None
    assert not parts.exists()

    assert (loaded.id, loaded.bundle_id, loaded.snapshot_id,
            loaded.num_rows) == (run.id, run.bundle_id, run.snapshot_id,
                                 run.num_rows)
    assert trainer.evaluate(loaded, labels_snapshot=snapshots.after).metrics == \
        trainer.evaluate(run, labels_snapshot=snapshots.after).metrics
    assert trainer.score_drift(loaded).psi == trainer.score_drift(run).psi

    local = loaded._exported.local
    assert local.is_dir()
    del loaded
    gc.collect()
    assert not local.exists()

    (tmp_path / path / '_SUCCESS').unlink()
    with pytest.raises(FileNotFoundError, match='_SUCCESS'):
        PredictionResult.load_from_connector(connector, path=path)

    from kumo.engine.trainer import _run_manifest

    cfg = kumo.MLOps(str(tmp_path / 'root'))
    _run_manifest.store(run, run.manifest, cfg)
    stored = PredictionResult.load_from_connector(cfg,
                                                  path=f'predictions/{run.id}')
    assert (stored.id, stored.num_rows) == (run.id, run.num_rows)


class _Tracker:
    platform = 'fake'

    def __init__(self):
        self.records = []

    def log_monitoring(self, record):
        self.records.append(record)
        self.run_id = f'run-{len(self.records)}'


def test_tracked_monitoring_is_tagged_like_its_prediction_run(
        snapshots, classifier, tmp_path):
    trainer, run = classifier
    tracker = _Tracker()
    cfg = kumo.MLOps(artifact_root=str(tmp_path / 'root'), tracking=tracker)
    report = trainer.score_drift(run, snapshot=snapshots.after,
                                 training_snapshot=snapshots.before,
                                 mlops=cfg)
    with kumo.mlops(cfg):
        evaluation = trainer.evaluate(run, labels_snapshot=snapshots.after)

    drift, evaluated = tracker.records
    assert (report.tracking, evaluation.tracking) == (
        TrackingRef(platform='fake', run_id='run-1'),
        TrackingRef(platform='fake', run_id='run-2'))
    for record, monitor in ((drift, 'drift'), (evaluated, 'evaluate')):
        assert record.tags.items() >= {
            'kumo.kind': 'monitoring', 'kumo.monitor': monitor,
            'kumo.engine_version': __version__, 'kumo.prediction_id': run.id,
            'kumo.bundle_id': run.bundle_id}.items()
        assert record.prediction_run is None
        assert record.started <= record.finished
    before = len(snapshots.orders[snapshots.orders.TS < CUTOFF])
    after = len(snapshots.orders)
    changes, = [s for s in drift.summaries
                if getattr(s, 'name', None) == 'drift/input_changes']
    assert ('orders', before, after, (after - before) / before * 100
            ) in changes.rows and changes.brief
    features = drift.summaries[-2]
    amount = report.features['orders']['AMT']
    assert features.name == 'drift/features'
    assert ('orders', 'AMT', 'numerical', None,
            amount['standardized_mean_shift'],
            amount['delta']['null_rate']) in features.rows
    labels = snapshots.after.manifest
    name = (f'snapshot-{labels.created_at[:10]}-'
            f"{labels.id.partition('@')[2][:6]}")
    auroc, ap = evaluation.metrics['auroc'], evaluation.metrics['ap']
    assert evaluated.summaries == (
        Tag('kumo.labels_snapshot_id', snapshots.after.id),
        Scalar('realized/auroc', auroc), Scalar('realized/ap', ap),
        Scalar('realized/labelled_share', 1.0),
        Note('notes', f'Realized auroc {auroc:.3g} · ap {ap:.3g} · 100 % '
             f'labelled (60 of 60 scored rows), with labels from {name}.'))


def test_drift_summaries_log_the_previous_run_and_explain_gaps():
    from kumo.engine._monitoring import DriftReport, track

    report = DriftReport(
        prediction_id='p', bundle_id='b', score='True_PROB', psi=0.5, ks=0.25,
        wasserstein=0.125, holdout_truncated=True, bundle_matches=True,
        previous_prediction_id='p0', previous_psi=0.01, previous_ks=0.02,
        previous_wasserstein=0.03, input_changes={
            'orders': (100, 104), 'orders.USER_ID → users': (100, None)},
        reasons={'features': 'no drift statistics for the type of users.TS '
                 '(timestamp)'},
        features={'users': {
            'AGE': {'type': 'numerical', 'standardized_mean_shift': -2.5,
                    'delta': {'null_rate': 0.0}},
            'PLAN': {'type': 'categorical', 'psi': 0.75,
                     'standardized_mean_shift': None, 'delta': {}},
            'ID': {'type': 'categorical', 'type_changed': ['numerical',
                                                           'categorical']}}})
    tracker = _Tracker()
    tracked = track(report, SimpleNamespace(
        id='p', bundle_id='b', manifest=None,
        tracking=TrackingRef(platform='fake', run_id='run-7')),
        SimpleNamespace(tracking=tracker, on_error='raise'),
        SimpleNamespace(result=None), PREDICTED_AT.to_pydatetime(),
        previous=SimpleNamespace(tracking=TrackingRef(
            platform='fake', run_id='run-0', run_name='2026-10-28-v6-churn')))

    assert tracked == replace(report, tracking=TrackingRef(platform='fake',
                                                           run_id='run-1'))
    record, = tracker.records
    assert record.prediction_run == 'run-7'
    assert record.summaries == (
        Tag('kumo.drift_previous', 'p0'),
        Scalar('drift/psi_prev', 0.01), Scalar('drift/ks_prev', 0.02),
        Scalar('drift/wasserstein_prev', 0.03),
        Note('notes', 'Drift vs training: psi 0.5 · ks 0.25 · vs '
             '2026-10-28-v6-churn: psi 0.01 · ks 0.02.'),
        Table('drift/input_changes', ('table or link', 'at training', 'now',
                                      'change %'),
              (('orders', 100, 104, 4.0),
               ('orders.USER_ID → users', 100, None, None)), brief=True),
        Table('drift/features', ('table', 'column', 'type', 'psi',
                                 'mean shift', 'null-rate change'),
              (('users', 'PLAN', 'categorical', 0.75, None, None),
               ('users', 'AGE', 'numerical', None, -2.5, 0.0),
               ('users', 'ID', 'numerical → categorical', None, None, None))),
        Note('notes', 'Most moved columns: `users.PLAN` (psi 0.75), '
             '`users.AGE` (mean shift -2.5 sd); all in '
             '`drift/features.json`.'),
        Note('notes', "The model's holdout file was truncated to fit its "
             'size limit.'),
        Note('notes', 'Column drift: no drift statistics for the type of '
             'users.TS (timestamp).'))

    tracker.records.clear()
    track(replace(report, bundle_matches=False),
          SimpleNamespace(id='p', bundle_id='b', manifest=None),
          SimpleNamespace(tracking=tracker, on_error='raise'),
          SimpleNamespace(result=None), PREDICTED_AT.to_pydatetime())
    record, = tracker.records
    assert record.prediction_run is None
    assert record.summaries[4:] == (
        Note('notes', 'Drift vs p0: psi 0.01 · ks 0.02.'),
        Note('notes', "This trainer's model did not score this run, so its "
             'drift since training is not logged here; against its '
             'holdout: psi 0.5 · ks 0.25 · wasserstein 0.125.'))


def test_evaluation_summaries_name_the_labels_snapshot(tmp_path):
    from kumo.engine._manifest import SnapshotManifest
    from kumo.engine._monitoring import (
        EvaluationResult,
        _evaluation_summaries,
        track,
    )

    result = EvaluationResult(
        prediction_id='p', bundle_id='b', snapshot_id='g@023b25a5',
        task_type='binary_classification', anchors=(),
        metrics={'ap': 0.5, 'auroc': 0.75, 'brier': None}, rows=4,
        labelled=3, labelled_fraction=0.75, matured=False,
        bundle_matches=True,
        reasons={'metrics': 'a metric is undefined for these labels'})
    plan = tmp_path / 'resolved-model-plan.json'
    plan.write_text('{"training_job": {"tune_metric": "auroc"}}')
    tracker = _Tracker()
    track(result, SimpleNamespace(id='p', bundle_id='b', manifest=None),
          SimpleNamespace(tracking=tracker, on_error='raise'),
          SimpleNamespace(result=SimpleNamespace(
              artifact_paths={'resolved_model_plan': str(plan)})),
          PREDICTED_AT.to_pydatetime(),
          SimpleNamespace(manifest=SnapshotManifest(
              id='g@023b25a5', location='s3://b/snapshots/g/2',
              created_at='2024-06-08T10:00:00+00:00')))

    record, = tracker.records
    assert record.tags['kumo.monitor'] == 'evaluate'
    assert record.summaries == (
        Tag('kumo.labels_snapshot_id', 'g@023b25a5'),
        Dataset('snapshot-2024-06-08-023b25', {
            'connector': 'S3', 'location': 's3://b/snapshots/g/2',
            'version': 'g@023b25a5', 'version_kind': 'exact'}, {}, {},
            'labels'),
        Scalar('realized/auroc', 0.75), Scalar('realized/ap', 0.5),
        Scalar('realized/labelled_share', 0.75),
        Note('notes', 'Realized auroc 0.75 · ap 0.5 · 75 % labelled (3 of 4 '
             'scored rows), with labels from snapshot-2024-06-08-023b25.'),
        Note('notes', 'Labels are not mature yet: the label window ends '
             'after the snapshot data.'),
        Note('notes', 'Realized metrics: a metric is undefined for these '
             'labels.'))

    assert _evaluation_summaries(replace(
        result, metrics={}, labelled=0, labelled_fraction=0.0), None)[1:] == [
        Scalar('realized/labelled_share', 0.0),
        Note('notes', 'Labelled 0 of 4 scored rows.'),
        Note('notes', 'Labels are not mature yet: the label window ends '
             'after the snapshot data.'),
        Note('notes', 'Realized metrics: a metric is undefined for these '
             'labels.')]
    assert _evaluation_summaries(replace(
        result, task_type='forecasting', metrics={}, rows=0, labelled=0,
        labelled_fraction=None, matured=None,
        reasons={'metrics': 'forecasting metrics are not supported'}), None,
        scored=2000) == [
        Tag('kumo.labels_snapshot_id', 'g@023b25a5'),
        Note('notes', "Realized quality isn't computed for forecasting yet "
             '(2,000 rows scored).')]


def test_a_monitoring_tracking_failure_warns_or_raises(classifier, tmp_path,
                                                       caplog):
    trainer, run = classifier

    class Tracker:
        def log_monitoring(self, record):
            raise ConnectionError('tracking server down')

    for on_error in ('warn', 'raise'):
        cfg = kumo.MLOps(str(tmp_path / 'root'), tracking=Tracker(),
                         on_error=on_error)
        if on_error == 'warn':
            report = trainer.score_drift(run, mlops=cfg)
            assert report.psi is not None and report.tracking is None
        else:
            with pytest.raises(RuntimeError, match='tracking server down'
                               ) as raised:
                trainer.score_drift(run, mlops=cfg)
            assert isinstance(raised.value.__cause__, ConnectionError)
    assert ('Experiment tracking failed: ConnectionError: tracking server '
            'down; the monitoring run is not tracked') in caplog.messages


def test_drift_reads_snapshot_statistics_from_the_artifact_root(
        snapshots, classifier, tmp_path):
    trainer, _ = classifier
    pquery = kumo.PredictiveQuery(
        graph=snapshots.graph, query='PREDICT COUNT(orders.*, 0, 7, days) > 0 '
        'FOR EACH users.ID')
    run = trainer.predict(pquery.generate_prediction_table(snapshots.after),
                          snapshot=snapshots.after,
                          binary_classification_threshold=0.5)
    passed = trainer.score_drift(run, snapshot=snapshots.after,
                                 training_snapshot=snapshots.before)
    root = kumo.FileConnector(root_dir=str(tmp_path / 'root'))
    for snapshot in (snapshots.before, snapshots.after):
        snapshot.export(kumo.SnapshotOutputConfig(root, 'snapshots'))

    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'fresh')
    try:
        alone = trainer.score_drift(run)
        found = trainer.score_drift(
            run, mlops=kumo.MLOps(artifact_root=str(tmp_path / 'root')))
    finally:
        kumo.configure(scratch_dir=previous)

    assert (alone.features, alone.input_changes) == ({}, None)
    for name in ('features', 'input_changes'):
        assert alone.reasons[name].endswith(
            'training and scoring snapshot: pass training_snapshot='
            'GraphSnapshot.load_from_connector(store, path=...), snapshot='
            'GraphSnapshot.load_from_connector(store, path=...), or turn MLOps '
            'on with the artifact root that stores them')
    assert found.reasons == passed.reasons == {}
    assert found.features == passed.features
    assert found.features['orders']['AMT']['delta']['num_unique'] > 0
    assert found.input_changes == passed.input_changes
    assert found.input_changes['orders'] == (len(
        snapshots.orders[snapshots.orders.TS < CUTOFF]), len(snapshots.orders))
    assert not (tmp_path / 'fresh' / 'snapshots').exists()


@pytest.fixture(scope='module')
def shifted(tmp_path_factory):
    root = tmp_path_factory.mktemp('features')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    rng = np.random.default_rng(1)
    data = root / 'data'
    data.mkdir()

    def users(mean, plans):
        pd.DataFrame({
            'ID': np.arange(400),
            'AGE': rng.normal(mean, 5.0, 400),
            'PLAN': rng.choice(['free', 'pro', 'team'], 400, p=plans),
            'BIO': rng.choice(['likes red shoes', 'reads books'], 400),
        }).to_parquet(data / 'users.parquet')

    connector = kumo.FileConnector(root_dir=str(data))
    users(40.0, [0.6, 0.3, 0.1])
    graph = kumo.Graph(tables={
        'users': kumo.Table.from_source_table(
            connector['users.parquet'], primary_key='ID').infer_metadata()
        .with_column('PLAN', stype=Stype.categorical)
        .with_column('BIO', stype=Stype.text)
    }, edges=[])
    before = graph.snapshot()
    time.sleep(1.1)
    users(55.0, [0.1, 0.3, 0.6])
    after = graph.snapshot()
    yield before, after, root
    kumo.configure(scratch_dir=previous)


def _compare(trainer, snapshot, training_snapshot, reasons):
    from kumo.engine._monitoring import _features, _snapshot_stats

    return _features(trainer, _snapshot_stats(
        trainer, SimpleNamespace(), snapshot, training_snapshot, None),
        reasons)


def _encoding(root, columns):
    from kumo_core.encoders.config import EncodersConfig
    from kumo_core.io.json import save_json
    from kumoapi.encoder import Index, Numerical

    path = root / f"encoders-{'-'.join(columns)}.json"
    save_json(EncodersConfig(encoders={'users': {
        column: Index() if column == 'PLAN' else Numerical()
        for column in columns
    }, 'train_table': {'LABEL': Numerical()}}), str(path))
    return SimpleNamespace(result=SimpleNamespace(
        manifest=None, artifact_paths={'encoders': str(path)}))


def test_feature_drift_detects_a_numeric_and_a_categorical_shift(shifted):
    before, after, root = shifted
    reasons = {}
    features = _compare(_encoding(root, ['AGE', 'PLAN']), after, before,
                        reasons)

    assert reasons == {}
    age, plan = features['users']['AGE'], features['users']['PLAN']
    assert age['type'] == 'numerical'
    assert age['standardized_mean_shift'] > 2
    assert age['delta']['mean'] == pytest.approx(15.0, abs=2.0)
    assert set(age['delta']) >= {'mean', 'std', 'min', 'max', 'q25', 'q50',
                                 'q75', 'skewness', 'kurtosis', 'null_rate',
                                 'num_unique'}
    assert plan['type'] == 'categorical' and plan['psi'] > 0.5
    assert 0 < plan['js'] <= 1 and plan['category_counts_capped'] is False

    same = _compare(_encoding(root, ['AGE', 'PLAN']), before, before, {})
    assert same['users']['AGE']['standardized_mean_shift'] == 0
    assert same['users']['PLAN']['psi'] == pytest.approx(0, abs=1e-9)


def test_feature_drift_explains_what_it_cannot_compare(shifted):
    before, after, root = shifted
    reasons = {}
    features = _compare(_encoding(root, ['AGE', 'BIO', 'GONE']), after,
                        before, reasons)
    assert list(features['users']) == ['AGE', 'BIO']
    assert reasons['features'] == (
        'column statistics missing on a side for users.GONE')
    bio = features['users']['BIO']
    assert bio['type'] == 'text'
    assert set(bio['delta']) == {'null_rate', 'num_unique', 'avg_tokens',
                                 'max_tokens', 'glove_overlap'}

    reasons = {}
    assert _compare(_encoding(root, ['AGE']), None, None, reasons) == {}
    assert reasons['features'].startswith(
        "no column statistics for the training and scoring snapshot: the "
        "model's manifest and the run's manifest name no snapshot; pass ")


def test_feature_drift_reads_the_training_snapshot_of_a_derived_model(
        shifted):
    from kumo.engine._manifest import (
        ModelParents,
        TrainingTableManifest,
        TrainingTableParents,
    )

    before, after, root = shifted
    trained = kumo.ModelManifest(id='m', parents=ModelParents(
        training_table=TrainingTableManifest(
            id='tt', parents=TrainingTableParents(
                snapshot=kumo.SnapshotManifest(id=before.id)))))
    trainer = _encoding(root, ['AGE'])
    trainer.result.manifest = kumo.ModelManifest(
        id='d', parents=ModelParents(derived_from=trained))
    reasons = {}
    features = _compare(trainer, after, None, reasons)
    assert reasons == {}
    assert features['users']['AGE']['standardized_mean_shift'] > 2
