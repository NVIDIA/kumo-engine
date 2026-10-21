# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
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
    PredictionOutput,
    PredictionParents,
    PredictionTableRecord,
    RunScoring,
    read_manifest,
    write_manifest,
)
from kumo.engine._scratch import get_scratch_dir
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
    assert manifest.kind == 'prediction_run'
    assert manifest.id == result.run_id == result.selection.job_id
    assert manifest.parents == PredictionParents(
        bundle=trainer.bundle_id, snapshot=snapshot.id,
        prediction_table=table.tt_id)
    assert manifest.source_tables == snapshot.manifest.parents.source_tables
    anchor = pd.Timestamp(table.timestamps[0], unit='ns', tz='UTC')
    assert manifest.prediction_table == PredictionTableRecord(
        query_sha256=hashlib.sha256(table.query.encode()).hexdigest(),
        source='pql', anchors=[anchor.isoformat()], rows=table.row_count)
    assert manifest.scoring == RunScoring(
        binary_classification_threshold=0.4, threshold_source='argument',
        num_classes_to_return=None, count_source='none', num_workers=1,
        lineage_columns=False)
    assert [o.destination for o in manifest.outputs] == [
        OutputDestination(connector='LOCAL_FILE',
                          location=str(tmp_path / name))
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
            (tmp_path / name / '_kumo_manifest.json').read_text())
        assert sidecar['id'] == manifest.id
        assert sidecar['outputs'] == manifest.to_dict()['outputs']
    assert table.query not in json.dumps(manifest.to_dict())


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
    child = trainer.with_scoring_defaults(
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


def test_export_records_outputs_and_skips_non_parquet_sidecars(
        graph, classifier, tmp_path, monkeypatch):
    _, snapshot = graph
    trainer, table = classifier
    result = trainer.predict(table, snapshot=snapshot,
                             binary_classification_threshold=0.5)
    calls = []
    connector = SimpleNamespace(
        source_type='SNOWFLAKE', config=SimpleNamespace(),
        source_backend=SimpleNamespace(
            table_location=lambda _, name: f'DB.SCHEMA.{name}'),
        sink_backend=SimpleNamespace(
            write_files=lambda *args, **kwargs: calls.append(kwargs)))
    warehouse = PredictionOutputConfig(connector,
                                       PredictionArtifactType.PREDICTIONS,
                                       'PREDS')
    result.export(warehouse)
    assert len(calls) == 1 and 'sidecars' not in calls[0]
    assert result.manifest.outputs == [
        PredictionOutput(
            artifact='predictions',
            destination=OutputDestination(connector='SNOWFLAKE',
                                          location='DB.SCHEMA.PREDS'))
    ]

    result.export(
        PredictionOutputConfig(
            connector=kumo.FileConnector(root_dir=str(tmp_path)),
            output_paths={'predictions': 'preds.csv/'}))
    assert not (tmp_path / 'preds.csv' / '_kumo_manifest.json').exists()
    assert len(result.manifest.outputs) == 2

    def fail(*args):
        raise OSError('read-only scratch')

    monkeypatch.setattr('kumo.engine._manifest.write_manifest', fail)
    with pytest.warns(UserWarning, match='read-only scratch'):
        result.export(warehouse)
    assert len(calls) == 2


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
        files = sorted(p.read_bytes()
                       for p in (tmp_path / name).glob('*.parquet'))
        return result, files

    def fail(*args, **kwargs):
        raise RuntimeError('unavailable')

    _, exported = run('baseline')
    monkeypatch.setattr(_run_manifest, 'output_stats', fail)
    with pytest.warns(UserWarning, match='without output score statistics'):
        partial, partial_exported = run('partial')
    assert partial.manifest.stats == {
        'predictions': ArtifactStats(rows=table.row_count)
    }
    monkeypatch.setattr(_run_manifest, 'output_record', fail)
    with pytest.warns(UserWarning, match='Exporting without a prediction'):
        unknown, unknown_exported = run('unknown')
    assert unknown.manifest.outputs is None
    monkeypatch.setattr(_run_manifest, 'run_manifest', fail)
    with pytest.warns(UserWarning, match='without a prediction manifest'):
        bare, bare_exported = run('bare')
    assert bare.manifest is None
    assert not (tmp_path / 'bare' / '_kumo_manifest.json').exists()
    assert (exported == partial_exported == unknown_exported ==
            bare_exported)


def test_failed_export_still_writes_manifest(graph, classifier, tmp_path,
                                             monkeypatch):
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
    with pytest.raises(OSError, match='disk full'):
        trainer.predict(table, snapshot=snapshot, output_config=output,
                        binary_classification_threshold=0.5)

    manifest, = [
        json.loads(path.read_text()) for path in trainer.scratch_dir.glob(
            'predict_output/*/prediction-manifest.json')
        if json.loads(path.read_text())['status'] == 'failed'
    ]
    assert manifest['outputs'] == [{
        'artifact': 'predictions',
        'destination': {'connector': 'LOCAL_FILE',
                        'location': str(tmp_path / 'preds')},
    }]
    assert manifest['timings']['finished_at'] is not None


def test_predict_tracking_logs_the_run_and_never_changes_it(
        graph, classifier, caplog):
    from kumo.engine._tracking import prediction_record

    _, snapshot = graph
    trainer, table = classifier

    class Recorder:
        def __init__(self):
            self.records = []

        def log_prediction_run(self, record):
            self.records.append(record)

    class Raising:
        def log_prediction_run(self, record):
            raise RuntimeError('tracker down')

    recorder = Recorder()
    untracked = trainer.predict(table, snapshot=snapshot,
                                binary_classification_threshold=0.4)
    tracked = trainer.predict(table, snapshot=snapshot,
                              binary_classification_threshold=0.4,
                              tracking=recorder)
    failed = trainer.predict(table, snapshot=snapshot,
                             binary_classification_threshold=0.4,
                             tracking=Raising())

    for result in (tracked, failed):
        pd.testing.assert_frame_equal(result.frame, untracked.frame)
    assert recorder.records == [prediction_record(tracked.manifest)]
    assert caplog.text.count('off for the rest of this run') == 1
