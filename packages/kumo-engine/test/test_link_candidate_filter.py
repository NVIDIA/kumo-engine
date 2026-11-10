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
def data(tmp_path_factory):
    # Even stores are active. With a day-50 anchor and a 7-day window, store
    # 0 ends at the anchor and store 2 a second later; store 4 has no create
    # time, stores 50-54 open on day 53, store 56 at the window end, store 58
    # a second after it and stores 55, 57 and 59 on day 100.
    root = tmp_path_factory.mktemp('candidate_filter')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    data = root / 'data'
    data.mkdir()
    rng = np.random.default_rng(0)
    pd.DataFrame({'USER_ID': np.arange(50)}).to_parquet(data / 'users.parquet')
    anchor, second = START + pd.Timedelta(days=50), pd.Timedelta(seconds=1)
    created = pd.Series(START, index=range(60))
    created[50:55] = START + pd.Timedelta(days=53)
    created[[55, 57, 59]] = START + pd.Timedelta(days=100)
    created[56] = anchor + pd.Timedelta(days=7)
    created[58] = anchor + pd.Timedelta(days=7) + second
    created[4] = pd.NaT
    ended = pd.Series(pd.NaT, index=range(60), dtype='datetime64[ns]')
    ended[0], ended[2] = anchor, anchor + second
    pd.DataFrame({
        'STORE_ID': np.arange(60),
        'IS_ACTIVE': np.arange(60) % 2 == 0,
        'RATING': np.arange(60) / 10,
        'CREATED': created,
        'ENDED': ended,
    }).to_parquet(data / 'stores.parquet')
    days = pd.to_timedelta(rng.integers(0, 60, 4000), unit='D')
    pd.DataFrame({
        'ORDER_ID': np.arange(4000),
        'USER_ID': rng.integers(0, 50, 4000),
        'STORE_ID': rng.integers(0, 60, 4000),
        'AMOUNT': rng.uniform(0, 10, 4000),
        'TS': START + days,
    }).to_parquet(data / 'orders.parquet')
    yield data
    kumo.configure(scratch_dir=previous)


@pytest.fixture(scope='module')
def graph(data):
    connector = kumo.FileConnector(root_dir=str(data))

    def table(name, key, **kwargs):
        return kumo.Table.from_source_table(connector[f'{name}.parquet'],
                                            primary_key=key,
                                            **kwargs).infer_metadata()

    orders = table('orders', 'ORDER_ID', time_column='TS').with_column(
        'USER_ID', stype=Stype.ID).with_column('STORE_ID', stype=Stype.ID)
    stores = table('stores', 'STORE_ID', time_column='CREATED',
                   end_time_column='ENDED')
    return kumo.Graph(
        tables={
            'users': table('users', 'USER_ID'),
            'stores': stores,
            'orders': orders,
        }, edges=[
            Edge(src_table='orders', fkey='USER_ID', dst_table='users'),
            Edge(src_table='orders', fkey='STORE_ID', dst_table='stores'),
        ])


@pytest.fixture(scope='module')
def snapshot(graph):
    return graph.snapshot()


def _fit(graph, snapshot, query, module='context'):
    plan = ModelPlan()
    plan.training_job.num_experiments = 1
    plan.training_job.disable_explain = True
    plan.optimization.max_epochs = 2
    plan.optimization.max_steps_per_epoch = 2
    plan.model_architecture.module = module
    trainer = kumo.Trainer(model_plan=plan)
    trainer.fit(
        kumo.PredictiveQuery(graph=graph,
                             query=query).generate_training_table(snapshot),
        snapshot=snapshot)
    return trainer


def _predict(trainer, graph, snapshot, query, count, **kwargs):
    table = kumo.PredictiveQuery(graph=graph,
                                 query=query).generate_prediction_table(
                                     snapshot, **kwargs)
    return trainer.predict(table, snapshot=snapshot,
                           num_classes_to_return=count).frame


@pytest.mark.parametrize('trained_filter', ['', ACTIVE])
@pytest.mark.parametrize('module', ['context', 'two_tower'])
def test_prediction_ranks_only_candidates_that_pass_the_filter(
        graph, snapshot, module, trained_filter):
    trainer = _fit(graph, snapshot, QUERY.format(trained_filter), module)

    candidates = (set(range(2, 55, 2)) - {4}) | {56}
    for count in (12, 60):
        frame = _predict(trainer, graph, snapshot, QUERY.format(ACTIVE), count,
                         prediction_time=START + pd.Timedelta(days=50))

        assert set(frame['CLASS'].astype(int)) <= candidates
        assert (frame.groupby('ENTITY').size() == min(count,
                                                      len(candidates))).all()


def test_pairwise_prediction_rejects_the_filter(graph, snapshot):
    trainer = _fit(graph, snapshot, QUERY.format(''), 'pairwise')

    with pytest.raises(ValueError, match='pairwise model ranks the items'):
        _predict(trainer, graph, snapshot, QUERY.format(ACTIVE), 12,
                 prediction_time=START + pd.Timedelta(days=50))


def test_prediction_without_the_trained_filter_ranks_every_valid_store(
        graph, snapshot):
    trainer = _fit(graph, snapshot, QUERY.format(ACTIVE))

    frame = _predict(trainer, graph, snapshot, QUERY.format(''), 60,
                     prediction_time=START + pd.Timedelta(days=50))

    valid = (set(range(1, 55)) - {4}) | {56}
    assert set(frame['CLASS'].astype(int)) == valid
    assert (frame.groupby('ENTITY').size() == len(valid)).all()


@pytest.fixture(scope='module')
def trainer(graph, snapshot):
    return _fit(graph, snapshot, QUERY.format(''))


@pytest.mark.parametrize('condition, message', [
    ('stores.RATING > 100', "no candidates from table 'stores': no row "
     r"passes the target filter's conditions on it \(stores.RATING > 100\)"),
    ('stores.RATING = 5.8', "no candidates from table 'stores'. Check that the "
     "target filter's conditions on it are not overly restrictive. Ensure "
     "some of its rows are created at or before 2024-02-27 00:00:00 and "
     "without an end time or ending after 2024-02-20 00:00:00."),
])
def test_prediction_without_candidates_is_rejected(graph, snapshot, trainer,
                                                   condition, message):
    with pytest.raises(ValueError, match=message):
        _predict(trainer, graph, snapshot, QUERY.format(f' WHERE {condition}'),
                 12, prediction_time=START + pd.Timedelta(days=50))


@pytest.mark.parametrize('trained_filter, predicted_filter', [
    ('', ACTIVE),
    (ACTIVE, ''),
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
