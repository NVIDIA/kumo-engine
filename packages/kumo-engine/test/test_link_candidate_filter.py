# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from kumoapi.graph import Edge
from kumoapi.model_plan import ModelPlan
from kumoapi.typing import Stype

import kumo.engine as kumo
from kumo.engine._scratch import get_scratch_dir

START = pd.Timestamp('2024-01-01')
ANCHOR = START + pd.Timedelta(days=50)
QUERY = ('PREDICT LIST_DISTINCT(purchases.PRODUCT_ID{}, 0, 7, days) '
         'RANK TOP 12 FOR EACH customers.CUSTOMER_ID')
AVAILABLE = ' WHERE products.IS_AVAILABLE = true'
# Even products are available. Odd products 1-29 end on day 40, product 0
# at the anchor and 2 a second later; product 4 has no create time;
# products 50-54 open on day 53, 56 at the window end, 58 a second later
# and 55, 57 and 59 on day 100.
ENDED = set(range(1, 30, 2))
VALID = (set(range(1, 55)) - ENDED - {4}) | {56}
VALID_AVAILABLE = {product for product in VALID if product % 2 == 0}
RATED = {product for product in VALID_AVAILABLE if product > 30}


@pytest.fixture(scope='module')
def graph(tmp_path_factory):
    root = tmp_path_factory.mktemp('candidate_filter')
    previous = get_scratch_dir()
    kumo.configure(scratch_dir=root / 'scratch')
    data = root / 'data'
    data.mkdir()
    rng = np.random.default_rng(0)
    created = pd.Series(START, index=range(60))
    created[50:55] = START + pd.Timedelta(days=53)
    created[[55, 57, 59]] = START + pd.Timedelta(days=100)
    created[56] = ANCHOR + pd.Timedelta(days=7)
    created[58] = created[56] + pd.Timedelta(seconds=1)
    created[4] = pd.NaT
    ended = pd.Series(pd.NaT, index=range(60), dtype='datetime64[ns]')
    ended[list(ENDED)] = START + pd.Timedelta(days=40)
    ended[[0, 2]] = ANCHOR + pd.to_timedelta([0, 1], unit='s')
    customers = pd.DataFrame({'CUSTOMER_ID': range(50)})
    customers.to_parquet(data / 'customers.parquet')
    pd.DataFrame({
        'PRODUCT_ID': range(60),
        'IS_AVAILABLE': np.arange(60) % 2 == 0,
        'RATING': np.arange(60) / 10,
        'CREATED': created,
        'ENDED': ended,
    }).to_parquet(data / 'products.parquet')
    days = pd.to_timedelta(rng.integers(0, 60, 4000), unit='D')
    pd.DataFrame({
        'PURCHASE_ID': range(4000),
        'CUSTOMER_ID': rng.integers(0, 50, 4000),
        'PRODUCT_ID': rng.integers(0, 60, 4000),
        'AMOUNT': rng.uniform(0, 10, 4000),
        'TS': START + days,
    }).to_parquet(data / 'purchases.parquet')
    connector = kumo.FileConnector(root_dir=str(data))

    def table(name, key, **kwargs):
        return kumo.Table.from_source_table(connector[f'{name}.parquet'],
                                            primary_key=key,
                                            **kwargs).infer_metadata()

    purchases = table('purchases', 'PURCHASE_ID', time_column='TS').with_column(
        'CUSTOMER_ID', stype=Stype.ID).with_column('PRODUCT_ID', stype=Stype.ID)
    products = table('products', 'PRODUCT_ID', time_column='CREATED',
                     end_time_column='ENDED')
    yield kumo.Graph(
        tables={
            'customers': table('customers', 'CUSTOMER_ID'),
            'products': products,
            'purchases': purchases,
        }, edges=[
            Edge(src_table='purchases', fkey='CUSTOMER_ID',
                 dst_table='customers'),
            Edge(src_table='purchases', fkey='PRODUCT_ID',
                 dst_table='products'),
        ])
    kumo.configure(scratch_dir=previous)


@pytest.fixture(scope='module')
def predict(graph):
    snapshot, trainers = graph.snapshot(), {}

    def predict(module, trained, predicted, count):
        if (module, trained) not in trainers:
            plan = ModelPlan()
            plan.training_job.num_experiments = 1
            plan.training_job.disable_explain = True
            plan.optimization.max_epochs = 2
            plan.optimization.max_steps_per_epoch = 2
            plan.model_architecture.module = module
            trainer = kumo.Trainer(model_plan=plan)
            query = kumo.PredictiveQuery(graph=graph,
                                         query=QUERY.format(trained))
            trainer.fit(query.generate_training_table(snapshot),
                        snapshot=snapshot)
            trainers[module, trained] = trainer
        query = kumo.PredictiveQuery(graph=graph, query=QUERY.format(predicted))
        table = query.generate_prediction_table(snapshot,
                                                prediction_time=ANCHOR)
        trainer = trainers[module, trained]
        return trainer.predict(table, snapshot=snapshot,
                               num_classes_to_return=count).frame

    return predict


@pytest.mark.parametrize('module, trained, predicted, expected', [
    ('context', '', AVAILABLE, VALID_AVAILABLE),
    ('context', AVAILABLE, AVAILABLE, VALID_AVAILABLE),
    ('two_tower', '', AVAILABLE, VALID_AVAILABLE),
    ('context', '', AVAILABLE + ' AND products.RATING > 3', RATED),
    ('context', AVAILABLE, '', VALID),
    ('context', '', ' WHERE purchases.AMOUNT > 5', 'target filter is'),
    ('context', '', ' WHERE products.RATING > 100', 'No row of table'),
    ('context', '', ' WHERE products.RATING = 5.8', 'valid at the prediction'),
    ('pairwise', '', AVAILABLE, 'pairwise model'),
])
def test_prediction_ranks_valid_passing_candidates(predict, module, trained,
                                                   predicted, expected):
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            predict(module, trained, predicted, 12)
        return
    for count in (12, 60):
        frame = predict(module, trained, predicted, count)
        assert set(frame['CLASS'].astype(int)) <= expected
        assert (frame.groupby('ENTITY').size() == min(count,
                                                      len(expected))).all()
    assert set(frame['CLASS'].astype(int)) == expected


def test_unfiltered_pairwise_prediction_ignores_the_time_rule(predict):
    frame = predict('pairwise', '', '', 12)

    assert set(frame['CLASS'].dropna().astype(int)) & ENDED
