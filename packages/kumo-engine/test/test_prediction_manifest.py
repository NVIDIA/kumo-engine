# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.graph import Edge
from kumoapi.jobs import PredictionArtifactType
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._manifest import (
    ArtifactStats,
    OutputDestination,
    PredictionManifest,
    PredictionParents,
    PredictionTableRecord,
    RunScoring,
    embedded,
    read_manifest,
    stub,
    write_manifest,
)
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.trainer._provenance import embedded_model
from kumo.engine.trainer.output_config import PredictionOutputConfig


@pytest.fixture(scope='module')
def graph(tmp_path_factory):
    root = tmp_path_factory.mktemp('graph')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    rng = np.random.default_rng(0)
    data = root / 'data'
    data.mkdir()
    pd.DataFrame({'ID': np.arange(50)}).to_parquet(data / 'users.parquet')
    pd.DataFrame({
        'EID': np.arange(2000),
        'USER_ID': rng.integers(0, 50, 2000),
        'TS': pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 2000), unit='D'),
    }).to_parquet(data / 'orders.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    users = kumo.Table.from_source_table(connector['users.parquet'],
                                         primary_key='ID').infer_metadata()
    orders = kumo.Table.from_source_table(
        connector['orders.parquet'],
        primary_key='EID', time_column='TS').infer_metadata().with_column(
            'USER_ID', stype=Stype.ID)
    graph = kumo.Graph(
        tables={'users': users, 'orders': orders},
        edges=[Edge(src_table='orders', fkey='USER_ID', dst_table='users')])
    yield graph, graph.snapshot()
    kumo.configure(scratch_dir=previous)


@pytest.fixture(scope='module')
def classifier(graph):
    graph, snapshot = graph
    pquery = kumo.PredictiveQuery(
        graph=graph, query='PREDICT COUNT(orders.*, 0, 7, days) > 0 '
        'FOR EACH users.ID')
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(model_plan=plan)
    trainer.fit(pquery.generate_training_table(snapshot), snapshot=snapshot)
    return trainer, pquery.generate_prediction_table(snapshot)


def _quantiles_match(summary, values):
    values = np.sort(values)
    levels = np.arange(1, 100) / 100
    low, high = (np.quantile(values, np.clip(levels + d, 0, 1))
                 for d in (-.03, .03))
    assert ((low <= summary['percentiles'])
            & (summary['percentiles'] <= high)).all()
    assert summary['count'] == len(values)
    assert (summary['min'], summary['max']) == (values[0], values[-1])


def test_manifest_records_run_and_directory_export(graph, classifier,
                                                   tmp_path, recwarn):
    _, snapshot = graph
    trainer, table = classifier
    output = PredictionOutputConfig(
        connector=kumo.FileConnector(root_dir=str(tmp_path)),
        output_paths={'predictions': 'preds/', 'embeddings': 'embs/'})
    result = trainer.predict(table, snapshot=snapshot, output_config=output,
                             binary_classification_threshold=0.4)
    assert not [w for w in recwarn if 'location' in str(w.message)]

    manifest = result.manifest
    assert manifest.kind == 'prediction'
    assert manifest.id == result.id == result.selection.job_id
    assert manifest.parents == PredictionParents(
        model=embedded_model(trainer.result.manifest),
        snapshot=embedded(snapshot.manifest))
    assert manifest.parents.model.holdout is None
    assert (result.bundle_id, result.snapshot_id) == (trainer.result.id,
                                                      snapshot.id)
    anchor = pd.Timestamp(table.timestamps[0], unit='ns', tz='UTC')
    assert manifest.prediction_table == PredictionTableRecord(
        query_sha256=hashlib.sha256(table.query.encode()).hexdigest(),
        source='pql', anchors=[anchor.isoformat()], rows=table.row_count,
        requested_rows=table.row_count, scored_rows=table.row_count,
        skipped_rows=0, skipped_sample=[])
    assert manifest.scoring == RunScoring(
        binary_classification_threshold=0.4, threshold_source='argument',
        num_classes_to_return=None, count_source='none', num_workers=1)
    assert [o.destination for o in manifest.outputs] == [
        OutputDestination(connector='LOCAL_FILE',
                          location=str(tmp_path / name / result.id))
        for name in ('preds', 'embs')
    ]
    assert manifest.timings.started_at <= manifest.timings.finished_at
    assert manifest.status == 'succeeded'
    assert manifest.stats['embeddings'] == ArtifactStats(rows=table.row_count)

    frame = result.frame
    stats = manifest.stats['predictions']
    assert stats.rows == len(frame)
    _quantiles_match(stats.columns['True_PROB'], frame['True_PROB'])
    assert stats.predicted_positive_rate == pytest.approx(
        (frame['True_PROB'] > 0.4).mean())

    for name in ('preds', 'embs'):
        sidecar = json.loads(
            (tmp_path / name / result.id / '_provenance.json').read_text())
        assert sidecar['id'] == manifest.id
        assert sidecar['outputs'] == manifest.to_dict()['outputs']


def test_a_model_without_a_manifest_is_a_stub_parent(graph, classifier,
                                                     monkeypatch):
    _, snapshot = graph
    trainer, table = classifier
    monkeypatch.setattr(type(trainer.result), 'manifest',
                        property(lambda _: None))
    result = trainer.predict(table, snapshot=snapshot,
                             binary_classification_threshold=0.5)
    assert result.manifest.parents.model == stub('model', trainer.result.id)
    assert result.bundle_id == trainer.result.id


def test_mlops_stores_the_run_after_its_export(graph, classifier, tmp_path):
    _, snapshot = graph
    trainer, table = classifier
    output = PredictionOutputConfig(
        connector=kumo.FileConnector(root_dir=str(tmp_path / 'out')),
        output_paths={'predictions': 'preds/'})
    with kumo.mlops(kumo.MLOps(artifact_root=str(tmp_path / 'root'))):
        result = trainer.predict(table, snapshot=snapshot,
                                 output_config=output,
                                 binary_classification_threshold=0.5)
    stored = tmp_path / 'root' / 'predictions' / result.id / 'predictions'
    assert sorted(p.name for p in stored.glob('*.parquet')) == sorted(
        Path(p).name for p in result.files)
    copy = json.loads((stored / '_provenance.json').read_text())
    assert copy == result.manifest.to_dict()
    assert copy['location'] == str(tmp_path / 'root' / 'predictions' /
                                   result.id)
    assert [o['destination']['location'] for o in copy['outputs']] == [
        str(tmp_path / 'out' / 'preds' / result.id), str(stored)
    ]

    off = kumo.MLOps(artifact_root=str(tmp_path / 'off'), store={'model'})
    trainer.predict(table, snapshot=snapshot, mlops=off,
                    binary_classification_threshold=0.5)
    assert not (tmp_path / 'off' / 'predictions').exists()


def test_mlops_runs_record_their_exact_score_counts(graph, classifier,
                                                    tmp_path, monkeypatch):
    from kumo_core.prediction_counts import psi
    from kumo_core.score_counts import PROBABILITY_EDGES, Counts
    from kumo_core.summaries import Dataset, Histogram, Scalar, Tag

    _, snapshot = graph
    trainer, table = classifier
    plain = trainer.predict(table, snapshot=snapshot,
                            binary_classification_threshold=0.4)
    assert plain.manifest.stats['predictions'].counts is None

    def histogram(values):
        return np.bincount(np.searchsorted(PROBABILITY_EDGES, values, 'right'),
                           minlength=102)

    holdout = trainer.result.holdout_df()['True_PROB'].to_numpy()
    baseline = Counts({'score': PROBABILITY_EDGES})
    baseline.histogram('True_PROB', 'score', holdout)
    monkeypatch.setattr('kumo.engine.trainer._provenance.holdout_counts',
                        lambda model: baseline)
    recorder = _Recorder()
    result = trainer.predict(
        table, snapshot=snapshot, binary_classification_threshold=0.4,
        mlops=kumo.MLOps(str(tmp_path / 'root'), tracking=recorder))
    counts = Counts.from_dict(result.manifest.stats['predictions'].counts)
    frame = result.frame
    for column in ('False_PROB', 'True_PROB'):
        values = frame[column].to_numpy()
        assert counts.arrays[column].tolist() == histogram(values).tolist()
        assert counts.arrays[f'{column}/sum'] == pytest.approx(values.sum())
        assert counts.extents[column] == (values.min(), values.max())
    assert counts.arrays['predicted_positive'] == frame['TARGET_PRED'].sum()

    assert recorder.records[0].tags.items() >= {
        'kumo.task_type': 'binary_classification',
        'kumo.snapshot_id': snapshot.id}.items()
    logged = {summary.name: summary
              for summary in recorder.records[0].summaries}
    assert logged['pred/True_PROB/hist'] == Histogram(
        'pred/True_PROB/hist', PROBABILITY_EDGES,
        tuple(histogram(frame['True_PROB']).tolist()))
    assert logged['pred/positive_rate'] == Scalar(
        'pred/positive_rate', frame['TARGET_PRED'].mean())
    assert logged['drift/score_psi'].value == pytest.approx(
        psi(histogram(frame['True_PROB']), histogram(holdout)))
    output = str(tmp_path / 'root' / 'predictions' / result.id / 'predictions')
    assert logged['kumo.output_uri'] == Tag('kumo.output_uri', output)
    assert [summary.source['location']
            for summary in recorder.records[0].summaries
            if isinstance(summary, Dataset)
            and summary.context == 'output'] == [output]


@pytest.mark.parametrize('on_error', ['warn', 'raise'])
def test_a_failed_store_of_a_run_follows_on_error(graph, classifier,
                                                  tmp_path, monkeypatch,
                                                  on_error):
    _, snapshot = graph
    trainer, table = classifier
    monkeypatch.setattr(
        kumo.PredictionResult, '_export',
        lambda self, *args: (_ for _ in ()).throw(OSError('ACCESS_DENIED')))
    records = []
    tracking = SimpleNamespace(log_prediction_run=records.append)
    cfg = kumo.MLOps(str(tmp_path / 'root'), tracking=tracking,
                     on_error=on_error)
    if on_error == 'raise':
        with pytest.raises(RuntimeError, match='Could not store prediction'):
            trainer.predict(table, snapshot=snapshot, mlops=cfg,
                            binary_classification_threshold=0.5)
    else:
        with pytest.warns(UserWarning, match='ACCESS_DENIED'):
            trainer.predict(table, snapshot=snapshot, mlops=cfg,
                            binary_classification_threshold=0.5)
    [record] = records
    assert 'ACCESS_DENIED' in record.tags['kumo.error']


def test_prediction_manifest_round_trips_unchanged(graph, classifier,
                                                   tmp_path):
    _, snapshot = graph
    trainer, table = classifier
    result = trainer.predict(table, snapshot=snapshot,
                             binary_classification_threshold=0.4)
    newer = json.loads(result._manifest_path.read_text())
    newer['future'] = {'a': 1}
    newer['scoring']['calibration'] = 'none'
    newer['stats']['predictions']['predicted_positive_rate'] = None
    source = tmp_path / 'newer.json'
    source.write_text(json.dumps(newer, indent=2, sort_keys=True) + '\n')

    manifest = read_manifest(source, PredictionManifest)
    write_manifest(tmp_path / 'out.json', manifest)

    assert (tmp_path / 'out.json').read_bytes() == source.read_bytes()
    assert manifest.extra == {'future': {'a': 1}}
    assert manifest.scoring.extra == {'calibration': 'none'}
    assert manifest.stats['predictions'].predicted_positive_rate is None


@pytest.mark.parametrize('argument, source', [(None, 'bundle'),
                                              (0.4, 'argument')])
def test_manifest_records_bundle_scoring_defaults_as_bundle(
        graph, classifier, argument, source):
    _, snapshot = graph
    trainer, table = classifier
    child = trainer.result.with_scoring_defaults(
        kumo.ScoringDefaults(binary_classification_threshold=0.3))
    scoring = child.predict(
        table, snapshot=snapshot,
        binary_classification_threshold=argument).manifest.scoring
    assert (scoring.binary_classification_threshold,
            scoring.threshold_source) == (argument or 0.3, source)


def test_top_scores_per_entity_and_anchor_across_files(tmp_path):
    from kumo_core.types import TaskType

    from kumo.engine.trainer._run_manifest import output_stats

    rng = np.random.default_rng(1)
    entity = np.repeat(np.arange(200), 6)
    frame = pd.DataFrame({
        'ENTITY': entity,
        'ANCHOR_TIMESTAMP': pd.to_datetime(
            np.tile(np.repeat(['2024-01-01', '2024-02-01'], 3), 200)),
        'CLASS': rng.integers(0, 50, entity.size),
        'SCORE': rng.random(entity.size),
    })
    paths = []
    for index, part in enumerate((frame.iloc[:601], frame.iloc[601:])):
        paths.append(str(tmp_path / f'part-{index}.parquet'))
        pq.write_table(pa.Table.from_pandas(part, preserve_index=False),
                       paths[-1], row_group_size=100)

    stats = output_stats(paths, TaskType.MULTICLASS_CLASSIFICATION)
    link = output_stats(paths, TaskType.TEMPORAL_LINK_PREDICTION)

    tops = frame.groupby(['ENTITY', 'ANCHOR_TIMESTAMP'])['SCORE'].max()
    _quantiles_match(stats['columns']['top_score'], tops)
    assert link == stats


def test_plain_export_records_nothing_and_skips_non_parquet_sidecars(
        graph, classifier, tmp_path):
    _, snapshot = graph
    trainer, table = classifier
    result = trainer.predict(table, snapshot=snapshot,
                             binary_classification_threshold=0.5)
    before = result.manifest
    calls = []
    connector = SimpleNamespace(
        source_type='SNOWFLAKE', config=SimpleNamespace(),
        source_backend=SimpleNamespace(
            table_location=lambda _, name: f'DB.SCHEMA.{name}'),
        sink_backend=SimpleNamespace(
            write_files=lambda *args, **kwargs: calls.append(kwargs)))
    result.export(PredictionOutputConfig(connector,
                                         PredictionArtifactType.PREDICTIONS,
                                         'PREDS'))
    assert len(calls) == 1 and 'sidecars' not in calls[0]

    files = kumo.FileConnector(root_dir=str(tmp_path))
    result.export(PredictionOutputConfig(
        connector=files, output_paths={'predictions': 'preds/'}))
    result.export(PredictionOutputConfig(
        connector=files, output_paths={'predictions': 'preds.csv/'}))
    sidecar = json.loads(
        (tmp_path / 'preds' / result.id / '_provenance.json').read_text())
    assert sidecar == before.to_dict() and sidecar['outputs'] == []
    assert list((tmp_path / result.id / 'preds.csv').glob('*.csv'))
    assert not (tmp_path / result.id / 'preds.csv' / '_provenance.json'
                ).exists()
    assert result.manifest == before


def test_manifest_failures_warn_and_keep_outputs(graph, classifier, tmp_path,
                                                 monkeypatch):
    from kumo.engine.trainer import _run_manifest

    _, snapshot = graph
    trainer, table = classifier

    def run(name):
        output = PredictionOutputConfig(
            connector=kumo.FileConnector(root_dir=str(tmp_path)),
            output_paths={'predictions': f'{name}/'})
        result = trainer.predict(table, snapshot=snapshot,
                                 output_config=output,
                                 binary_classification_threshold=0.5)
        written = pd.concat(
            pd.read_parquet(p)
            for p in sorted((tmp_path / name).glob('*/*.parquet')))
        return result, written.sort_values('ENTITY').reset_index(drop=True)

    def fail(*args, **kwargs):
        raise RuntimeError('unavailable')

    _, exported = run('baseline')
    monkeypatch.setattr(_run_manifest, 'output_stats', fail)
    with pytest.warns(UserWarning, match='without output score statistics'):
        partial, partial_exported = run('partial')
    assert partial.manifest.stats == {
        'predictions': ArtifactStats(rows=table.row_count)
    }
    monkeypatch.setattr(_run_manifest, 'destination_record', fail)
    with pytest.warns(UserWarning, match='Exporting without a prediction'):
        unknown, unknown_exported = run('unknown')
    assert unknown.manifest.outputs is None
    monkeypatch.setattr(_run_manifest, '_manifest', fail)
    with pytest.warns(UserWarning, match='without a prediction manifest'):
        bare, bare_exported = run('bare')
    assert bare.manifest is None and bare.bundle_id is None
    assert not (tmp_path / 'bare' / '_provenance.json').exists()
    # Scores agree to float tolerance: GPU scoring is not bit-reproducible,
    # so a label may differ only where its probability sits at the threshold.
    clear = (exported['True_PROB'] - 0.5).abs() > 1e-3
    for other in (partial_exported, unknown_exported, bare_exported):
        pd.testing.assert_frame_equal(
            other.drop(columns='TARGET_PRED'),
            exported.drop(columns='TARGET_PRED'), check_exact=False,
            atol=1e-4)
        assert other['TARGET_PRED'][clear].equals(
            exported['TARGET_PRED'][clear])


class _Recorder:
    def __init__(self, raises=False):
        self.records, self.raises = [], raises

    def start(self, record):
        pass

    def on_trial(self, record, event):
        pass

    def finish(self, record, error):
        pass

    def log_prediction_run(self, record):
        if self.raises:
            raise RuntimeError('tracker down')
        self.records.append(record)


def test_failed_export_still_writes_and_tracks_the_run(graph, classifier,
                                                       tmp_path, monkeypatch):
    _, snapshot = graph
    trainer, table = classifier
    connector = kumo.FileConnector(root_dir=str(tmp_path))
    sink = type(connector.sink_backend)
    write_files = sink.write_files

    def fail_second(self, connector, name, *args, **kwargs):
        if name.startswith('embs'):
            raise OSError('disk full')
        return write_files(self, connector, name, *args, **kwargs)

    monkeypatch.setattr(sink, 'write_files', fail_second)
    output = PredictionOutputConfig(
        connector=connector,
        output_paths={'predictions': 'preds/', 'embeddings': 'embs/'})
    recorder = _Recorder()
    cfg = kumo.MLOps(artifact_root=str(tmp_path / 'root'), tracking=recorder)
    with pytest.raises(OSError, match='disk full'):
        trainer.predict(table, snapshot=snapshot, output_config=output,
                        binary_classification_threshold=0.5, mlops=cfg)

    manifest, = [
        json.loads(path.read_text()) for path in trainer.scratch_dir.glob(
            'predict_output/*/_provenance.json')
        if json.loads(path.read_text())['status'] == 'failed'
    ]
    assert manifest['outputs'] == [{
        'artifact': 'predictions',
        'destination': {'connector': 'LOCAL_FILE',
                        'location': str(tmp_path / 'preds' / manifest['id'])},
    }]
    assert manifest['timings']['finished_at'] is not None
    assert not (tmp_path / 'root' / 'predictions').exists()
    record, = recorder.records
    assert record.tags['kumo.error'] == 'OSError: disk full'
    assert record.tags['kumo.prediction_id'] == manifest['id']


def test_tracking_records_the_run_and_never_changes_it(graph, classifier,
                                                       tmp_path, caplog):
    from kumo.engine._tracking import prediction_record
    from kumo.engine.trainer._run_manifest import _scoring_summaries

    _, snapshot = graph
    trainer, table = classifier
    recorder = _Recorder()

    def predict(tracking):
        cfg = kumo.MLOps(artifact_root=str(tmp_path / 'root'),
                         tracking=tracking, store=set())
        return trainer.predict(table, snapshot=snapshot, mlops=cfg,
                               binary_classification_threshold=0.4)

    untracked = trainer.predict(table, snapshot=snapshot,
                                binary_classification_threshold=0.4)
    tracked = predict(recorder)
    failed = predict(_Recorder(raises=True))

    for result in (tracked, failed):
        pd.testing.assert_frame_equal(result.frame, untracked.frame)
    settings = kumo.MLOps(artifact_root=str(tmp_path / 'root'),
                          tracking=recorder, store=set())
    assert recorder.records == [prediction_record(
        tracked.manifest, model=trainer.result,
        summaries=_scoring_summaries(tracked.manifest, settings, snapshot))]
    assert any(summary.name == 'graph/tables'
               for summary in recorder.records[0].summaries)
    assert 'tracker down' in caplog.text


@pytest.mark.parametrize('destination, location', [
    ('predictions/r/predictions/', 's3://b/root/predictions/r/predictions/'),
    ('s3://b/root/predictions/r/predictions/',
     's3://b/root/predictions/r/predictions'),
    ('/Volumes/c/s/v/predictions/r', '/Volumes/c/s/v/predictions/r'),
    ('@ML.KUMO.STAGE/predictions/r/', '@ML.KUMO.STAGE/predictions/r'),
])
def test_an_output_location_joins_only_a_relative_path_to_the_root(
        destination, location):
    from kumo.engine.trainer._run_manifest import destination_record

    connector = SimpleNamespace(source_type='S3', config=SimpleNamespace(
        root_dir='s3://b/root'))

    assert destination_record(connector, destination).location == location
