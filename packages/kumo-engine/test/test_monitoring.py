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
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._manifest import ArtifactStats
from kumo.engine._monitoring import _ks, _psi, _wasserstein, evaluate
from kumo.engine._scratch import get_scratch_dir
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
    trainer = SimpleNamespace(_training_query=query, bundle_id=None,
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
    result = trainer.evaluate(run, snapshot=snapshots.after)

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

    from sklearn.metrics import roc_auc_score
    joined = run.frame.set_index('ENTITY').join(labels)
    assert result.metrics['auroc'] == pytest.approx(
        roc_auc_score(joined['TARGET'], joined['True_PROB']), abs=1e-6)

    immature = trainer.evaluate(run, snapshot=snapshots.before)
    assert immature.matured is False
    assert immature.metrics == {'auroc': None, 'ap': None}
    assert 'undefined' in immature.reasons['metrics']


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
    result = trainer.evaluate(run, snapshot=snapshots.after)

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
    result = trainer.evaluate(run, snapshot=snapshots.after)
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


def test_evaluation_exports_its_rows(snapshots, classifier, tmp_path):
    from kumo.engine.trainer.output_config import PredictionOutputConfig

    trainer, run = classifier
    result = trainer.evaluate(run, snapshot=snapshots.after)
    result.export(PredictionOutputConfig(
        kumo.FileConnector(root_dir=str(tmp_path)),
        output_paths={'predictions': 'evaluated/'}))
    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / 'evaluated'),
        pd.read_parquet(result.actuals))


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


def _summary(values):
    return {'count': len(values), 'min': float(np.min(values)),
            'max': float(np.max(values)),
            'percentiles': list(np.quantile(values, np.arange(1, 100) / 100))}


def test_psi_reads_percentile_curves():
    rng = np.random.default_rng(0)
    baseline = _summary(rng.beta(2, 5, 50_000))

    assert _psi(baseline, baseline) == 0
    assert _psi(_summary(rng.beta(2, 5, 50_000)), baseline) < .01
    assert _psi(_summary(rng.beta(5, 2, 50_000)), baseline) > .2
    tied = _summary(np.r_[np.zeros(40_000), rng.random(10_000)])
    assert _psi(tied, tied) == 0
    assert _psi({'quantiles': {'q50': .5}}, baseline) is None


def test_ks_and_wasserstein_from_percentile_curves():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 100_000), rng.normal(0.5, 2, 100_000)
    ks = max(abs(np.searchsorted(np.sort(a), x, 'right') / len(a) -
                 np.searchsorted(np.sort(b), x, 'right') / len(b))
             for x in np.linspace(-6, 6, 4001))
    levels = (np.arange(10_000) + .5) / 10_000
    w1 = np.mean(np.abs(np.quantile(a, levels) - np.quantile(b, levels)))

    assert _ks(_summary(a), _summary(b)) == pytest.approx(ks, abs=0.02)
    assert _wasserstein(_summary(a), _summary(b)) == pytest.approx(w1, rel=.02)
    assert _ks(_summary(a), _summary(a)) == _wasserstein(
        _summary(a), _summary(a)) == 0
    # Point masses: 30% more ones moves the CDF jump at 0 by 0.3.
    zeros = _summary(np.r_[np.zeros(500), np.ones(500)])
    ones = _summary(np.r_[np.zeros(200), np.ones(800)])
    assert _ks(zeros, ones) == pytest.approx(0.3, abs=0.02)
    assert _wasserstein(zeros, ones) == pytest.approx(0.3, abs=0.02)
    assert _ks(zeros, None) is None and _wasserstein({}, zeros) is None


def test_score_drift(snapshots, classifier, monkeypatch):
    trainer, run = classifier
    report = trainer.score_drift(run, previous=run, snapshot=snapshots.after,
                                 training_snapshot=snapshots.before)

    assert report.score == 'True_PROB' and math.isfinite(report.psi)
    assert math.isfinite(report.ks) and math.isfinite(report.wasserstein)
    assert report.holdout_truncated is False and report.bundle_matches
    assert (report.previous_psi, report.previous_ks,
            report.previous_wasserstein) == (0, 0, 0)
    assert report.input_changes['tables.orders.rows'] == (len(
        snapshots.orders[snapshots.orders.TS < CUTOFF]), len(snapshots.orders))
    assert not any(key.endswith('_reverse') for key in report.input_changes)
    assert report.reasons == {}
    assert (report.prediction_id, report.previous_prediction_id) == (run.id,
                                                                     run.id)

    shifted = _summary(np.full(500, 0.999))
    report = trainer.score_drift(
        SimpleNamespace(id='r', bundle_id=None, manifest=SimpleNamespace(
            stats={'predictions': ArtifactStats(
                rows=500, columns={'True_PROB': shifted})})))
    assert report.psi > 0.2 and report.ks > 0.5
    assert report.bundle_matches is None
    assert 'input_changes' in report.reasons

    report = trainer.score_drift(SimpleNamespace(manifest=None))
    assert (report.prediction_id, report.psi) == (None, None)
    assert report.reasons
    monkeypatch.setattr(TrainingResult, 'manifest', property(lambda self: None))
    report = trainer.score_drift(run)
    assert (report.psi, report.ks) == (None, None)
    assert 'holdout' in report.reasons['holdout']


def test_monitoring_a_run_loaded_from_its_export(snapshots, classifier,
                                                 tmp_path):
    from kumo.engine.trainer.output_config import PredictionOutputConfig

    trainer, run = classifier
    connector = kumo.FileConnector(root_dir=str(tmp_path))
    run.export(
        PredictionOutputConfig(connector, output_paths={'predictions': 'p/'}))

    loaded = PredictionResult.load_from_connector(connector, path='p/')
    parts = loaded._exported.local / 'parts'
    assert loaded.manifest.id == run.id and not parts.exists()
    assert trainer.score_drift(loaded).psi is not None
    assert not parts.exists()

    assert (loaded.id, loaded.bundle_id, loaded.snapshot_id,
            loaded.num_rows) == (run.id, run.bundle_id, run.snapshot_id,
                                 run.num_rows)
    assert trainer.evaluate(loaded, snapshot=snapshots.after).metrics == \
        trainer.evaluate(run, snapshot=snapshots.after).metrics
    assert trainer.score_drift(loaded).psi == trainer.score_drift(run).psi

    local = loaded._exported.local
    assert local.is_dir()
    del loaded
    gc.collect()
    assert not local.exists()

    (tmp_path / 'p' / '_SUCCESS').unlink()
    with pytest.raises(FileNotFoundError, match='_SUCCESS'):
        PredictionResult.load_from_connector(connector, path='p/')


def test_tracked_monitoring_logs_a_new_run(snapshots, classifier, tmp_path):
    trainer, run = classifier

    class Tracker:
        records: list = []

        def start(self, record):
            pass

        def on_trial(self, record, event):
            pass

        def finish(self, record, error):
            pass

        def log_monitoring(self, record):
            self.records.append(record)

    cfg = kumo.MLOps(artifact_root=str(tmp_path / 'root'), tracking=Tracker())
    report = trainer.score_drift(run, mlops=cfg)
    with kumo.mlops(cfg):
        trainer.evaluate(run, snapshot=snapshots.after)

    drift, evaluation = Tracker.records
    for record in (drift, evaluation):
        assert record.tags['kumo.kind'] == 'monitoring'
        assert record.tags['kumo.prediction_run'] == run.id
        assert record.tags['kumo.bundle_id'] == run.bundle_id
    assert drift.metrics['psi'] == report.psi
    assert drift.metrics['wasserstein'] == report.wasserstein
    assert 'auroc' in evaluation.metrics and 'rows' in evaluation.metrics
