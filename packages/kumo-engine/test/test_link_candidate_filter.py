# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from kumo_core.pquery import PredictiveQueryPlanError
from kumo_core.prediction_options import prediction_task_type
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

START = pd.Timestamp('2024-01-01')
QUERY = ('PREDICT LIST_DISTINCT(orders.STORE_ID{}, 0, 7, days) '
         'RANK TOP 12 FOR EACH users.USER_ID')
ACTIVE = ' WHERE stores.IS_ACTIVE = true'


@pytest.fixture(scope='module')
def graph(tmp_path_factory):
    # Even stores are active. Stores 50-54 open on day 53, after a day-50
    # anchor but inside its window; stores 55-59 open on day 100.
    root = tmp_path_factory.mktemp('candidate_filter')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    data = root / 'data'
    data.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame({'USER_ID': np.arange(50)}).to_parquet(data / 'users.parquet')
    opened = pd.to_timedelta(np.repeat([0, 53, 100], [50, 5, 5]), unit='D')
    pd.DataFrame({
        'STORE_ID': np.arange(60),
        'IS_ACTIVE': np.arange(60) % 2 == 0,
        'CREATED': START + opened,
    }).to_parquet(data / 'stores.parquet')
    days = pd.to_timedelta(rng.integers(0, 60, 4000), unit='D')
    pd.DataFrame({
        'ORDER_ID': np.arange(4000),
        'USER_ID': rng.integers(0, 50, 4000),
        'STORE_ID': rng.integers(0, 60, 4000),
        'AMOUNT': rng.uniform(0, 10, 4000),
        'TS': START + days,
    }).to_parquet(data / 'orders.parquet')
    connector = kumo.FileConnector(root_dir=str(data))

    def table(name, key, **kwargs):
        return kumo.Table.from_source_table(connector[f'{name}.parquet'],
                                            primary_key=key,
                                            **kwargs).infer_metadata()

    orders = table('orders', 'ORDER_ID', time_column='TS').with_column(
        'USER_ID', stype=Stype.ID).with_column('STORE_ID', stype=Stype.ID)
    graph = kumo.Graph(
        tables={
            'users': table('users', 'USER_ID'),
            'stores': table('stores', 'STORE_ID', time_column='CREATED'),
            'orders': orders,
        }, edges=[
            Edge(src_table='orders', fkey='USER_ID', dst_table='users'),
            Edge(src_table='orders', fkey='STORE_ID', dst_table='stores'),
        ])
    yield graph
    kumo.configure(scratch_dir=previous)


@pytest.fixture(scope='module')
def snapshot(graph):
    return graph.snapshot()


@pytest.mark.parametrize('trained_filter', ['', ACTIVE])
@pytest.mark.parametrize('module', ['context', 'two_tower', 'pairwise'])
def test_prediction_ranks_only_candidates_that_pass_the_filter(
        graph, snapshot, module, trained_filter):
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 2
    plan.optimization.max_steps_per_epoch = 2
    plan.model_architecture.module = module
    trainer = kumo.Trainer(model_plan=plan)
    trained = kumo.PredictiveQuery(graph=graph,
                                   query=QUERY.format(trained_filter))
    trainer.fit(trained.generate_training_table(snapshot), snapshot=snapshot)
    table = kumo.PredictiveQuery(
        graph=graph, query=QUERY.format(ACTIVE)).generate_prediction_table(
            snapshot, prediction_time=START + pd.Timedelta(days=50))

    candidates = set(range(0, 55, 2))
    for count in (12, 60):
        frame = trainer.predict(table, snapshot=snapshot,
                                num_classes_to_return=count).frame

        # Pairwise ranks only sampled stores and pads the rest with nulls.
        assert set(frame['CLASS'].dropna().astype(int)) <= candidates
        if module != 'pairwise':
            assert (frame.groupby('ENTITY').size() == min(
                count, len(candidates))).all()


@pytest.mark.parametrize('trained_filter, predicted_filter', [
    ('', ACTIVE),
    (' WHERE orders.AMOUNT > 5',
     ' WHERE orders.AMOUNT > 5 AND stores.IS_ACTIVE = true'),
])
def test_candidate_conditions_may_change(snapshot, trained_filter,
                                         predicted_filter):
    assert prediction_task_type(
        QUERY.format(trained_filter), QUERY.format(predicted_filter),
        snapshot.graph_config) == 'temporal_link_prediction'


@pytest.mark.parametrize('trained_filter, predicted_filter', [
    ('', ' WHERE orders.AMOUNT > 5'),
    (' WHERE orders.AMOUNT > 5',
     ' WHERE orders.AMOUNT > 5 OR stores.IS_ACTIVE = true'),
])
def test_other_target_conditions_must_match(snapshot, trained_filter,
                                            predicted_filter):
    with pytest.raises(ValueError, match="query's target filter is"):
        prediction_task_type(QUERY.format(trained_filter),
                             QUERY.format(predicted_filter),
                             snapshot.graph_config)


def test_entity_table_condition_is_rejected(snapshot):
    with pytest.raises(PredictiveQueryPlanError, match='Entity table users'):
        prediction_task_type(QUERY.format(''),
                             QUERY.format(' WHERE users.USER_ID = 1'),
                             snapshot.graph_config)


def test_node_task_target_filter_must_match(snapshot):
    query = ('PREDICT COUNT(orders.*{}, 0, 7, days) > 0 '
             'FOR EACH users.USER_ID')
    with pytest.raises(ValueError, match="query's target is"):
        prediction_task_type(query.format(''), query.format(ACTIVE),
                             snapshot.graph_config)
