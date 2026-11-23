# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from kumo_core.io import filesystem
from kumoapi.explain.subgraph import SubgraphNode
from kumoapi.graph import Edge
from kumoapi.jobs import MetadataField
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir
from kumo.engine.trainer.output_config import PredictionOutputConfig

ANCHOR = MetadataField.ANCHOR_TIMESTAMP
JOB = MetadataField.JOB_TIMESTAMP
ARMS = [None, [], [JOB], [ANCHOR], [ANCHOR, JOB]]


@pytest.fixture(scope='module')
def graph(tmp_path_factory):
    root = tmp_path_factory.mktemp('graph')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    rng = np.random.default_rng(0)
    data = root / 'data'
    data.mkdir()
    pd.DataFrame({'ID': np.arange(50)}).to_parquet(data / 'ents.parquet')
    pd.DataFrame({'IID': np.arange(20)}).to_parquet(data / 'items.parquet')
    pd.DataFrame({
        'EID':
        np.arange(3000),
        'ENT_ID':
        rng.integers(0, 50, 3000),
        'ITEM_ID':
        rng.integers(0, 20, 3000),
        'TS':
        pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 3000), unit='D'),
        'V':
        rng.normal(size=3000),
    }).to_parquet(data / 'events.parquet')
    graph = _source_graph(data)
    yield graph, graph.snapshot(), root
    kumo.configure(scratch_dir=previous)


def _source_graph(data):
    connector = kumo.FileConnector(root_dir=str(data))
    ents = kumo.Table.from_source_table(connector['ents.parquet'],
                                        primary_key='ID').infer_metadata()
    items = kumo.Table.from_source_table(connector['items.parquet'],
                                         primary_key='IID').infer_metadata()
    events = kumo.Table.from_source_table(
        connector['events.parquet'],
        primary_key='EID', time_column='TS').infer_metadata().with_column(
            'ENT_ID', stype=Stype.ID).with_column('ITEM_ID', stype=Stype.ID)
    return kumo.Graph(
        tables={
            'ents': ents,
            'items': items,
            'events': events
        }, edges=[
            Edge(src_table='events', fkey='ENT_ID', dst_table='ents'),
            Edge(src_table='events', fkey='ITEM_ID', dst_table='items'),
        ])


def _fit(graph, query):
    graph, snapshot, _ = graph
    pquery = kumo.PredictiveQuery(graph=graph, query=query)
    plan = ModelPlan()
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    trainer = kumo.Trainer(model_plan=plan)
    trainer.fit(pquery.generate_training_table(snapshot), snapshot=snapshot)
    return trainer, pquery, pquery.generate_prediction_table(snapshot)


@pytest.fixture(scope='module')
def forecast(graph):
    return _fit(
        graph, 'PREDICT COUNT(events.*, 0, 7, days) '
        'FORECAST 3 TIMEFRAMES FOR EACH ents.ID')


@pytest.fixture(scope='module')
def classifier(graph):
    return _fit(graph, 'PREDICT COUNT(events.*, 0, 7, days) > 0 '
                'FOR EACH ents.ID')


@pytest.fixture(scope='module')
def link(graph):
    return _fit(
        graph, 'PREDICT LIST_DISTINCT(events.ITEM_ID, 0, 7, days) '
        'RANK TOP 3 FOR EACH ents.ID')


def _predict(graph, fitted, fields, kinds=('predictions', ), **options):
    """Return each exported artifact; ``fields=None`` exports nothing."""
    _, snapshot, root = graph
    trainer, _, table = fitted
    if fields is None:
        return {
            'predictions':
            trainer.predict(table, snapshot=snapshot, **options).frame
        }
    name = '-'.join(field.value for field in fields) or 'none'
    paths = {
        kind: f'{trainer.scratch_dir.name}/{name}/{kind}/'
        for kind in kinds
    }
    output = PredictionOutputConfig(
        connector=kumo.FileConnector(root_dir=str(root / 'out')),
        output_paths=paths, extra_fields=fields)
    trainer.predict(table, snapshot=snapshot, output_config=output, **options)
    return {
        kind:
        pd.concat(
            pd.read_parquet(path)
            for path in sorted((root / 'out' / paths[kind]).glob('*.parquet')))
        for kind in kinds
    }


def _metadata(fields):
    return [field.value for field in (ANCHOR, JOB) if field in (fields or [])]


def test_time_column_is_a_feature(graph, forecast):
    [path] = forecast[0].scratch_dir.glob(
        'runs/*-fit-*/task_dataset/feature_schema.json')
    schema = json.loads(path.read_text())['node_types']

    assert schema['events']['feature_columns'] == ['V', 'TS']


@pytest.mark.parametrize('fields', ARMS)
def test_forecast_output_columns(graph, forecast, fields):
    # #31405
    frame = _predict(graph, forecast, fields)['predictions']
    anchor = pd.Timestamp(forecast[2].timestamps[0], unit='ns')

    assert set(frame) == {
        'ENTITY', 'TIMESTAMP', 'TARGET_PRED', *_metadata(fields)
    }
    horizons = frame.groupby('ENTITY')['TIMESTAMP'].agg(sorted)
    assert horizons.map(len).eq(3).all()
    assert (horizons.map(lambda times: times[0]) == anchor).all()
    if ANCHOR in (fields or []):
        assert set(frame['ANCHOR_TIMESTAMP']) == {anchor}


@pytest.mark.parametrize('fields', ARMS)
def test_classification_output_columns(graph, classifier, fields):
    kinds = ('predictions', ) if fields is None else ('predictions',
                                                      'embeddings')
    frames = _predict(graph, classifier, fields, kinds,
                      binary_classification_threshold=0.5)
    anchor = pd.Timestamp(classifier[2].timestamps[0], unit='ns')
    metadata = _metadata(fields)

    assert list(frames['predictions']) == [
        'ENTITY', *metadata, 'TARGET_PRED', 'False_PROB', 'True_PROB'
    ]
    if fields is not None:
        assert list(frames['embeddings']) == ['ENTITY', *metadata, 'EMBEDDING']
    for frame in frames.values():
        if ANCHOR in metadata:
            assert set(frame['ANCHOR_TIMESTAMP']) == {anchor}
        if JOB in metadata:
            assert frame['JOB_TIMESTAMP'].nunique() == 1


@pytest.mark.parametrize('fields', [[], [ANCHOR, JOB]])
def test_link_prediction_output_columns(graph, link, fields):
    frames = _predict(graph, link, fields, ('predictions', 'embeddings'),
                      num_classes_to_return=3)
    metadata = _metadata(fields)

    assert list(
        frames['predictions']) == ['ENTITY', 'CLASS', 'SCORE', *metadata]
    assert list(frames['embeddings']) == ['ID', 'TYPE', 'EMBEDDING', *metadata]


@pytest.mark.parametrize('name', ['forecast', 'classifier', 'link'])
def test_prediction_task_matches_the_fitted_task(graph, name, request,
                                                 monkeypatch):
    from kumo_core.io.json import load_json
    from kumo_core.resolved_task import ResolvedNumericalTask

    trainer = request.getfixturevalue(name)[0]
    resolve = ResolvedNumericalTask.from_manifest
    tasks = []

    def spy(*args, **kwargs):
        tasks.append(resolve(*args, **kwargs))
        return tasks[-1]

    monkeypatch.setattr(ResolvedNumericalTask, 'from_manifest', spy)
    _predict(graph, request.getfixturevalue(name), None,
             num_classes_to_return=3 if name == 'link' else None)
    fitted = ResolvedNumericalTask.from_dict(
        load_json(trainer._training_result.artifact_paths['input_manifest'],
                  dict))

    assert fitted.unix_end_offset is not None
    for field in ('forecast_length', 'lag_timesteps', 'unix_start_offset',
                  'unix_end_offset', 'unix_step_size'):
        assert getattr(tasks[0], field) == getattr(fitted, field)


def test_forecast_rejects_a_table_with_other_lags(graph, forecast):
    trainer, _, table = forecast
    options = dict(table.compiler_options, lag_offsets_secs=(-86400, -172800))

    with pytest.raises(ValueError, match='table uses lag_timesteps = 2'):
        trainer.predict(replace(table, compiler_options=options),
                        snapshot=graph[1])


def test_forecast_rejects_a_table_with_another_horizon(graph, forecast):
    trainer, _, table = forecast
    future = table.compiler_options['future_offsets_secs']
    options = dict(table.compiler_options,
                   future_offsets_secs=(*future, future[-1] + 86400))

    expected = f'table uses forecast_length = {len(future) + 1}'
    with pytest.raises(ValueError, match=expected):
        trainer.predict(replace(table, compiler_options=options),
                        snapshot=graph[1])


@pytest.mark.parametrize('num_classes_to_return, expected', [
    (2, 2), (5, 5)
])
@pytest.mark.parametrize('export', [False, True])
def test_link_prediction_respects_explicit_count(
        graph, link, num_classes_to_return, expected, export, tmp_path):
    options = {'num_classes_to_return': num_classes_to_return}
    if export:
        options['output_config'] = PredictionOutputConfig(
            connector=kumo.FileConnector(root_dir=str(tmp_path)),
            output_paths={'predictions': 'ranking/'})
    frame = _predict(graph, link, None, **options)['predictions']

    assert frame.groupby('ENTITY').size().eq(expected).all()
    if export:
        exported = pd.concat(
            pd.read_parquet(path) for path in tmp_path.glob('ranking/*.parquet'))
        assert exported.groupby('ENTITY').size().eq(expected).all()


@pytest.mark.parametrize('count, message', [
    (None, 'specify the number of predictions'), (501, 'less than or equal to 500')
])
@pytest.mark.parametrize('export', [False, True])
def test_link_prediction_rejects_invalid_count_before_materialization(
        graph, link, count, message, export, tmp_path, monkeypatch):
    from unittest.mock import Mock

    trainer, _, table = link
    _, snapshot, _ = graph
    materialize = Mock(side_effect=AssertionError('materialization began'))
    monkeypatch.setattr('kumo_core.prediction.materialize_prediction_task',
                        materialize)
    before = set(trainer.scratch_dir.glob('runs/*-predict-*'))
    output = (PredictionOutputConfig(
        connector=kumo.FileConnector(root_dir=str(tmp_path)),
        output_paths={'predictions': 'ranking/'}) if export else None)

    with pytest.raises(ValueError, match=message):
        trainer.predict(table, snapshot=snapshot, num_classes_to_return=count,
                        output_config=output)

    materialize.assert_not_called()
    assert set(trainer.scratch_dir.glob('runs/*-predict-*')) == before
    assert not list(tmp_path.iterdir())


def test_link_embeddings_do_not_require_count(graph, link, tmp_path):
    trainer, _, table = link
    _, snapshot, _ = graph
    output = PredictionOutputConfig(
        connector=kumo.FileConnector(root_dir=str(tmp_path)),
        output_paths={'embeddings': 'embeddings/'})

    trainer.predict(table, snapshot=snapshot, output_config=output)

    assert list(tmp_path.glob('embeddings/*.parquet'))


def test_later_export_rejects_extra_fields(graph, classifier, tmp_path):
    _, snapshot, _ = graph
    trainer, _, table = classifier
    result = trainer.predict(table, snapshot=snapshot,
                             binary_classification_threshold=0.5)
    connector = kumo.FileConnector(root_dir=str(tmp_path))

    with pytest.raises(ValueError, match='Trainer.predict'):
        result.export(
            PredictionOutputConfig(connector=connector,
                                   output_paths={'predictions': 'bad/'},
                                   extra_fields=[ANCHOR]))
    assert not (tmp_path / 'bad').exists()

    result.export(
        PredictionOutputConfig(connector=connector,
                               output_paths={'predictions': 'good/'}))
    copied = sorted((tmp_path / 'good').glob('*.parquet'))
    written = [p.read_bytes() for p in copied]
    expected = [Path(f).read_bytes() for f in sorted(result.files)]
    assert written == expected


@pytest.mark.parametrize('num_workers', [1, 2])
def test_multilabel_thresholds_preserve_scores(tmp_path, num_workers):
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=tmp_path / 'scratch')
    try:
        data = tmp_path / 'data'
        data.mkdir()
        pd.DataFrame({
            'ID': range(120),
            'VALUE': np.arange(120) / 120,
            'TAGS': [['a', 'b'], ['b', 'c'], ['a', 'c']] * 33 +
            [['a', 'b']] + [None] * 20,
        }).to_parquet(data / 'users.parquet')
        connector = kumo.FileConnector(root_dir=str(data))
        graph = kumo.Graph({
            'users': kumo.Table.from_source_table(connector['users'],
                                                  primary_key='ID'),
        })
        snapshot = graph.snapshot()
        query = kumo.PredictiveQuery(
            graph, 'PREDICT users.TAGS CLASSIFY TOP 3 FOR EACH users.ID')
        plan = ModelPlan()
        plan.training_job.num_experiments = 1
        plan.training_job.disable_explain = True
        plan.optimization.max_epochs = 1
        plan.optimization.max_steps_per_epoch = 2
        plan.optimization.batch_size = [32]
        plan.model_architecture.channels = [16]
        trainer = kumo.Trainer(plan)
        result = trainer.fit(query.generate_training_table(snapshot),
                             snapshot=snapshot)
        trainer = kumo.Trainer.load(result.bundle_dir)
        table = query.generate_prediction_table(snapshot)
        scores = None
        for threshold in (None, 0.0, 0.25, 1.0):
            output = trainer.predict(
                table, snapshot=snapshot, num_workers=num_workers,
                binary_classification_threshold=threshold).frame
            output = output.sort_values(['ENTITY', 'CLASS']).reset_index(
                drop=True)
            assert len(output) == 20 * 3
            effective = 0.5 if threshold is None else threshold
            np.testing.assert_array_equal(output['PREDICTED'],
                                          output['SCORE'] > effective)
            current = output[['ENTITY', 'CLASS', 'SCORE']]
            if scores is None:
                scores = current
            else:
                pd.testing.assert_frame_equal(scores, current)
        assert not output['PREDICTED'].any()
        for threshold in (-0.1, 1.1, float('nan'), float('inf')):
            with pytest.raises(ValueError, match='between 0 and 1'):
                trainer.predict(table, snapshot=snapshot,
                                binary_classification_threshold=threshold)
        _assert_label_counts(trainer.predict(
            table, snapshot=snapshot, num_workers=num_workers,
            binary_classification_threshold=0.25,
            mlops=kumo.MLOps(str(tmp_path / 'root'))), result)
    finally:
        kumo.configure(scratch_dir=previous)


def _top_scores(frame):
    from kumo_core.score_counts import PROBABILITY_EDGES

    return np.bincount(
        np.searchsorted(PROBABILITY_EDGES,
                        frame.groupby('ENTITY')['SCORE'].max(), 'right'),
        minlength=len(PROBABILITY_EDGES) + 1).tolist()


def _assert_label_counts(run, fitted):
    from kumo_core.score_counts import PROBABILITY_EDGES, Counts

    counts = Counts.from_dict(run.manifest.stats['predictions'].counts)
    output = run.frame
    holdout = Counts.from_dict(fitted.manifest.holdout.counts)
    assert holdout.arrays['top_score'].tolist() == _top_scores(
        fitted.holdout_df())
    assert counts.arrays['top_score'].tolist() == _top_scores(output)
    labels = [name.removesuffix('_PROB') for name in counts.extents
              if name != 'top_score']
    assert dict(zip(labels, counts.arrays['label_positive'].tolist())) == (
        output.groupby('CLASS')['PREDICTED'].sum().to_dict())
    assert counts.arrays['predicted_labels'].tolist() == np.bincount(
        output.groupby('ENTITY')['PREDICTED'].sum(),
        minlength=len(labels) + 1).tolist()
    for label in labels:
        scores = output.loc[output['CLASS'] == label, 'SCORE']
        assert counts.arrays[f'{label}_PROB'].tolist() == np.bincount(
            np.searchsorted(PROBABILITY_EDGES, scores, 'right'),
            minlength=len(PROBABILITY_EDGES) + 1).tolist()


def test_mlops_counts_forecast_horizons_recommended_items_and_top_scores(
        graph, forecast, link, tmp_path):
    from kumo_core.score_counts import Counts
    from kumo_core.score_drift import score_drift

    _, snapshot, _ = graph
    mlops = kumo.MLOps(str(tmp_path / 'root'))

    def counted(fitted, **options):
        trainer, _, table = fitted
        run = trainer.predict(table, snapshot=snapshot, mlops=mlops,
                              **options)
        return (Counts.from_dict(run.manifest.stats['predictions'].counts),
                run.frame)

    counts, frame = counted(forecast)
    edges = np.array(counts.edges['target'])
    frame = frame.sort_values(['ENTITY', 'TIMESTAMP'])
    horizon = frame.groupby('ENTITY').cumcount().to_numpy()
    values = frame['TARGET_PRED'].to_numpy(dtype=np.float64)
    for h in range(3):
        assert counts.arrays[f'TARGET_PRED_h{h + 1}'].tolist() == np.bincount(
            np.searchsorted(edges, values[horizon == h], 'right'),
            minlength=edges.size + 1).tolist()

    counts, frame = counted(link, num_classes_to_return=3)
    per_item = frame['CLASS'].value_counts().reindex(range(20), fill_value=0)
    log2 = np.array(counts.edges['recommendations'])
    assert counts.arrays['item_frequency'].tolist() == np.bincount(
        np.searchsorted(log2, per_item.to_numpy(), 'right'),
        minlength=log2.size + 1).tolist()
    assert (float(counts.arrays['catalogue']),
            float(counts.arrays['items_recommended']),
            float(counts.arrays['recommendations'])) == (
                20, (per_item > 0).sum(), len(frame))
    holdout = Counts.from_dict(link[0].result.manifest.holdout.counts)
    edges = holdout.edges['top_score']
    assert counts.edges['top_score'] == edges
    assert holdout.extents['top_score'] == (edges[0], edges[-1])
    assert holdout.arrays['top_score'].sum() == \
        link[0].result.manifest.holdout.rows
    top = frame.groupby('ENTITY')['SCORE'].max().to_numpy()
    assert counts.arrays['top_score'].tolist() == np.bincount(
        np.searchsorted(edges, top, 'right'), minlength=len(edges) + 1).tolist()
    assert counts.extents['top_score'] == (top.min(), top.max())
    assert score_drift(counts, holdout) is not None


@pytest.mark.parametrize('name', ['classifier', 'link'])
def test_loaded_prediction_explanations_use_current_inputs(graph, name,
                                                           request, tmp_path):
    """Real producers must explain the supplied snapshot, not training seeds."""
    fitted = request.getfixturevalue(name)
    _, previous_snapshot, root = graph
    ents_path = root / 'data/ents.parquet'
    events_path = root / 'data/events.parquet'
    with filesystem.open_file(str(ents_path), 'rb') as source:
        ents = pd.read_parquet(source)
    with filesystem.open_file(str(events_path), 'rb') as source:
        events = pd.read_parquet(source)
    current_ents, current_events = ents.copy(), events.copy()
    current_ents['ID'] += 1000
    current_events['ENT_ID'] += 1000
    data = str(tmp_path / 'data')
    filesystem.makedirs(data, exist_ok=True)
    with filesystem.open_file(f'{data}/ents.parquet', 'wb') as destination:
        current_ents.to_parquet(destination)
    with filesystem.open_file(f'{data}/events.parquet', 'wb') as destination:
        current_events.to_parquet(destination)
    filesystem.copy(str(root / 'data/items.parquet'), f'{data}/items.parquet')
    graph_object = _source_graph(Path(data))
    snapshot = graph_object.snapshot()
    assert snapshot.id != previous_snapshot.id
    trainer = kumo.Trainer.load(fitted[0].result.bundle_dir, accelerator='cpu')
    query = kumo.PredictiveQuery(graph=graph_object, query=fitted[1].query)
    table = query.generate_prediction_table(snapshot)
    result = trainer.predict(
        table, snapshot=snapshot, explanations=True,
        num_classes_to_return=3 if name == 'link' else None)
    xai = result.explanations
    assert xai.availability['samples'].available
    references = xai.samples()
    assert len(references) == len(ents)
    assert xai.coverage.sampled_graphs == len(ents)
    assert xai.coverage.cohort_rows_processed == (
        len(ents) if name == 'classifier' else None)
    assert xai.coverage.checkpoint_path is not None
    assert not Path(xai.coverage.checkpoint_path).is_absolute()
    scored = result.frame
    for reference in references:
        assert reference.original_ids['first'] >= 1000
        sample = xai.sample(reference.artifact_id)
        store = sample.subgraph_store
        assert (store.nodes[store.root_id].pkey ==
                reference.original_ids['first'])
        if name == 'link':
            predicted = scored.loc[
                scored.ENTITY == reference.original_ids['first'], 'CLASS']
            rhs = {store.nodes[node].pkey for node in store.tables['items']
                   if isinstance(store.nodes[node], SubgraphNode)}
            assert set(predicted).issubset(rhs)
    if name == 'classifier':
        assert xai.availability['cohorts'].available
        cohorts = xai.column_analysis().cohorts
        assert cohorts
        assert all(cohort.targets is None for cohort in cohorts)
        # V has no missing values and all roots have historical neighbours.
        cohort = next(cohort for cohort in cohorts
                      if cohort.table_name == 'events' and cohort.col_name == 'V'
                      and cohort.hop == 1)
        assert isinstance(cohort.predictions, list)
        explained_mean = np.dot(cohort.populations, cohort.predictions)
        assert explained_mean == pytest.approx(
            scored['True_PROB'].mean(), abs=0.005)
    else:
        assert xai.availability['cohorts'].reason == 'unsupported'
    reopened = kumo.PredictionExplanations.load(xai.run_root)
    assert reopened.samples() == references
    assert reopened.coverage == xai.coverage
    assert reopened.sample(references[0].artifact_id) == xai.sample(
        references[0].artifact_id)
    assert len(scored) == len(ents) * (3 if name == 'link' else 1)


def test_prediction_producer_failure_retains_successful_outputs(
        graph, classifier, tmp_path, monkeypatch):
    from kumo_core.explain.column_analysis import ColumnAnalysis

    monkeypatch.setenv('KUMO_EXPLAINER_CONTINUE_ON_ERROR', 'true')

    def fail_numerical(*args, **kwargs):
        raise RuntimeError('numerical cohort collection failed')

    monkeypatch.setattr(ColumnAnalysis, 'collect_numerical', fail_numerical)
    trainer, _, table = classifier
    output = PredictionOutputConfig(
        connector=kumo.FileConnector(root_dir=str(tmp_path)),
        output_paths={'predictions': 'predictions/'})
    result = trainer.predict(table, snapshot=graph[1], output_config=output,
                             explanations=True)
    assert len(result.frame) == 50
    assert result.manifest.status == 'succeeded'
    assert pd.read_parquet(tmp_path / 'predictions').shape[0] == 50
    xai = result.explanations
    assert xai.availability['cohorts'].reason == 'failed'
    assert 'numerical cohort collection failed' in xai.availability[
        'cohorts'].generation_failure
    with pytest.raises(kumo.ExplanationUnavailableError):
        xai.column_analysis()
    assert xai.availability['samples'].available
    assert xai.samples()
    reopened = kumo.PredictionExplanations.load(xai.run_root)
    assert reopened.availability == xai.availability
    assert reopened.samples() == xai.samples()
    assert reopened.coverage == xai.coverage


@pytest.mark.parametrize(('cohort_rows', 'sample_rows'), [(12, 5), (5, 12)])
def test_prediction_explanation_row_limits(graph, classifier,
                                           cohort_rows, sample_rows):
    trainer, _, table = classifier
    result = trainer.predict(
        table, snapshot=graph[1], explanations=kumo.PredictionExplanationOptions(
            cohort_rows=cohort_rows, sample_rows=sample_rows))
    assert len(result.frame) == 50
    xai = result.explanations
    assert len(xai.samples()) == sample_rows
    assert xai.coverage.cohort_rows_processed == cohort_rows
    assert xai.coverage.sampled_graphs == sample_rows
    assert xai.coverage.checkpoint_path is not None
    cohort = next(cohort for cohort in xai.column_analysis().cohorts
                  if cohort.table_name == 'events' and cohort.col_name == 'V'
                  and cohort.hop == 1)
    assert isinstance(cohort.predictions, list)
    assert np.dot(cohort.populations, cohort.predictions) == pytest.approx(
        result.frame.head(cohort_rows)['True_PROB'].mean(), abs=0.005)
    reopened = kumo.PredictionExplanations.load(xai.run_root)
    assert reopened.availability == xai.availability
    assert len(reopened.samples()) == sample_rows
    assert reopened.coverage == xai.coverage


def test_null_prediction_feature_keeps_valid_cohorts(graph, classifier, tmp_path):
    """An all-null prediction column is omitted without invalidating peers."""
    _, original_snapshot, root = graph
    data = str(tmp_path / 'null-data')
    filesystem.makedirs(data, exist_ok=True)
    for filename in ['ents.parquet', 'items.parquet', 'events.parquet']:
        filesystem.copy(str(root / 'data' / filename), f'{data}/{filename}')
    # Resolve and retain the numeric schema before making this source all null.
    prediction_graph = _source_graph(Path(data))
    prediction_graph = replace(
        prediction_graph, tables={
            **prediction_graph.tables,
            'events': prediction_graph.tables['events'].with_column(
                'V', stype=Stype.numerical),
        })
    with filesystem.open_file(str(root / 'data/events.parquet'), 'rb') as source:
        events = pd.read_parquet(source)
    events['V'] = pd.Series(np.nan, index=events.index, dtype='float64')
    with filesystem.open_file(f'{data}/events.parquet', 'wb') as destination:
        events.to_parquet(destination)
    snapshot = prediction_graph.snapshot()
    assert snapshot.id != original_snapshot.id
    trainer = kumo.Trainer.load(classifier[0].result.bundle_dir, accelerator='cpu')
    query = kumo.PredictiveQuery(graph=prediction_graph, query=classifier[1].query)
    result = trainer.predict(
        query.generate_prediction_table(snapshot), snapshot=snapshot,
        explanations=kumo.PredictionExplanationOptions(cohort_rows=12,
                                                       sample_rows=3))
    assert len(result.frame) == 50
    xai = result.explanations
    assert xai.availability['cohorts'].available
    assert xai.coverage.cohort_rows_processed == 12
    assert xai.coverage.sampled_graphs == 3
    assert xai.coverage.checkpoint_path is not None
    # Verify replay retained the selected model's actual weights.
    checkpoint = (trainer.scratch_dir / 'runs' / xai.run_id / 'model_bundle'
                  / 'model' / xai.coverage.checkpoint_path)
    with filesystem.open_file(str(checkpoint), 'rb') as staged:
        with filesystem.open_file(
                trainer.result.artifact_paths['selected_checkpoint'], 'rb') as selected:
            assert staged.read() == selected.read()
    cohorts = xai.column_analysis().cohorts
    assert cohorts
    assert all(cohort.cohorts for cohort in cohorts)
    assert not any(cohort.table_name == 'events' and cohort.col_name == 'V'
                   for cohort in cohorts)
    assert any(cohort.table_name == 'events' and cohort.col_name == 'COUNT(*)'
               for cohort in cohorts)
    reopened = kumo.PredictionExplanations.load(xai.run_root)
    assert reopened.coverage == xai.coverage
    assert reopened.column_analysis() == xai.column_analysis()
