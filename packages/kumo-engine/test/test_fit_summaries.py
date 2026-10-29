# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fit's test summaries equal a brute-force pass over its holdout rows."""

import numpy as np
import pandas as pd
import pytest
from kumo_core.score_counts import PROBABILITY_EDGES, Counts
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir


@pytest.fixture(scope='module')
def graph(tmp_path_factory):
    root = tmp_path_factory.mktemp('graph')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    rng = np.random.default_rng(0)
    data = root / 'data'
    data.mkdir()
    pd.DataFrame({'ID': np.arange(300),
                  'GROUP': rng.choice(['a', 'b', 'c'], 300)}).to_parquet(
                      data / 'ents.parquet')
    pd.DataFrame({
        'EID': np.arange(6000),
        'ENT_ID': rng.integers(0, 300, 6000),
        'TS': pd.Timestamp('2024-01-01') +
        pd.to_timedelta(rng.integers(0, 120, 6000), unit='D'),
        'V': rng.normal(size=6000),
    }).to_parquet(data / 'events.parquet')
    connector = kumo.FileConnector(root_dir=str(data))
    graph = kumo.Graph(
        tables={
            'ents': kumo.Table.from_source_table(
                connector['ents.parquet'], primary_key='ID').infer_metadata()
            .with_column('GROUP', stype=Stype.categorical),
            'events': kumo.Table.from_source_table(
                connector['events.parquet'], primary_key='EID',
                time_column='TS').infer_metadata().with_column(
                    'ENT_ID', stype=Stype.ID),
        },
        edges=[Edge(src_table='events', fkey='ENT_ID', dst_table='ents')])
    yield graph, graph.snapshot()
    kumo.configure(scratch_dir=previous)


def _fit(graph, query, threshold=None):
    graph, snapshot = graph
    pquery = kumo.PredictiveQuery(graph=graph, query=query)
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.training_job.refit_trainval = False
    plan.optimization.max_epochs = 1
    plan.optimization.max_steps_per_epoch = 2
    result = kumo.Trainer(model_plan=plan).fit(
        pquery.generate_training_table(snapshot), snapshot=snapshot,
        scoring_defaults=kumo.ScoringDefaults(
            binary_classification_threshold=threshold))
    return result, {s.name: s for s in result._summaries}, result.holdout_df()


def test_binary_test_summaries_count_every_holdout_row(graph):
    result, summaries, holdout = _fit(
        graph, 'PREDICT COUNT(events.*, 0, 7, days) > 0 FOR EACH ents.ID',
        threshold=0.5)

    p = holdout['True_PROB'].to_numpy()
    y = holdout['TARGET'].to_numpy(dtype=bool)
    roc = summaries['test/roc']
    for step, t in enumerate(PROBABILITY_EDGES):
        assert dict(roc.y.points)[step] == pytest.approx(np.mean(p[y] >= t))
        assert dict(roc.x.points)[step] == pytest.approx(np.mean(p[~y] >= t))
    assert summaries['test/confusion/tp'].value == ((p > 0.5) & y).sum()
    assert summaries['test/brier'].value == pytest.approx(np.mean((p - y)**2))
    stored = Counts.from_dict(result.manifest.holdout.counts)
    assert stored.arrays['True_PROB'].sum() == len(holdout)
    assert set(stored.arrays) == {'True_PROB', 'False_PROB'}


def test_regression_test_summaries_count_every_holdout_row(graph):
    result, summaries, holdout = _fit(
        graph, 'PREDICT SUM(events.V, 0, 7, days) FOR EACH ents.ID')

    error = (holdout['TARGET_PRED'] - holdout['TARGET']).to_numpy()
    assert result.metrics['test_mae'] == pytest.approx(np.abs(error).mean())
    assert 'test/mae' not in summaries
    assert summaries['test/bias'].value == pytest.approx(error.mean())
    assert sum(summaries['test/residual/hist'].counts) == len(holdout)
    stored = Counts.from_dict(result.manifest.holdout.counts)
    assert stored.edges['target'] == summaries['test/TARGET_PRED/hist'].edges


def test_multiclass_test_summaries_count_every_holdout_row(graph):
    _, summaries, holdout = _fit(graph, 'PREDICT ents.GROUP FOR EACH ents.ID')

    top = holdout[holdout['PREDICTED']]
    assert summaries['test/accuracy'].value == pytest.approx(
        (top['CLASS'] == top['TARGET']).mean())
    assert sum(map(sum, summaries['test/confusion'].counts)) == len(top)
    assert set(summaries['test/confusion'].labels) <= {'a', 'b', 'c'}


def test_forecast_test_summaries_are_exact_per_horizon(graph):
    _, summaries, holdout = _fit(
        graph, 'PREDICT SUM(events.V, 0, 7, days) FORECAST 3 TIMEFRAMES '
        'FOR EACH ents.ID')

    horizon = holdout.groupby(['ENTITY', 'ANCHOR_TIMESTAMP'])[
        'TIMESTAMP'].rank(method='first').astype(int)
    error = (holdout['TARGET_PRED'] - holdout['TARGET']).abs()
    assert dict(summaries['test/horizon/mae'].points) == pytest.approx(
        error.groupby(horizon).mean().to_dict())
