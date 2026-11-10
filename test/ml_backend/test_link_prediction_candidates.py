# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from test.fixtures.dg_indexed_graph import _build_diskgraph_ingest_config
from types import SimpleNamespace

import diskgraph
import pandas as pd
import pyarrow as pa
import pytest
from kumo_core.link_candidates import candidate_node_ids
from kumo_core.pquery import (
    NativeGraph,
    NativeGraphColumn,
    NativeGraphForeignKey,
    NativeGraphTable,
    parse_and_validate_query,
    to_api_graph,
)
from kumo_core.types import TaskType

from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.graph import Graph
from kumo.testing.pquery import get_validated_query
from statedb.utils.errors import PredictionDataMaterializationError

mpt = importlib.import_module(
    'rest.backends.ml_backend.activities.materialize_prediction_task')

QUERY = ('PREDICT LIST_DISTINCT(orders.STORE_ID WHERE {}, 0, 7, days) '
         'RANK TOP 2 FOR EACH users.USER_ID')


def _metadata():
    def column(name, dtype):
        return dict(name=name, stype=None, dtype=dtype, na_values=None,
                    metadata={}, encoder=None, stats={})

    def table(name, key, columns, foreign_keys=(), time_col=None):
        return dict(
            table_name=name, data_source=None, data_version=None,
            pkey=dict(name=key, stype=None, dtype='int', na_values=None,
                      metadata={}, drop_duplicates=False), fkeys=[
                          dict(column(source, 'int'), dst_table=destination,
                               dst_key=None, dst_data_source=None)
                          for source, destination in foreign_keys
                      ], cols=[column(*spec) for spec in columns],
            time_col=time_col, source_name=None, stats={})

    return dict(
        name='LINK_CANDIDATES', data_source_dict={}, tables=[
            table('users', 'USER_ID', []),
            table('stores', 'STORE_ID', [('IS_ACTIVE', 'bool'),
                                         ('RATING', 'float')]),
            table('orders', 'ORDER_ID', [('AMOUNT', 'float')],
                  [('USER_ID', 'users'), ('STORE_ID', 'stores')],
                  column('TS', 'time')),
        ])


def _native_graph():
    def table(name, key, columns, time=None, foreign_keys=()):
        return NativeGraphTable(
            name=name, source_table_name=name,
            columns=tuple(NativeGraphColumn(*spec)
                          for spec in columns), primary_key=key,
            time_column=time, end_time_column=None, foreign_keys=tuple(
                NativeGraphForeignKey(column, destination, column)
                for column, destination in foreign_keys))

    return NativeGraph((
        table('users', 'USER_ID', [('USER_ID', 'int', 'ID')]),
        table('stores', 'STORE_ID', [('STORE_ID', 'int', 'ID'),
                                     ('IS_ACTIVE', 'bool', 'categorical'),
                                     ('RATING', 'float', 'numerical')]),
        table('orders', 'ORDER_ID', [('ORDER_ID', 'int', 'ID'),
                                     ('USER_ID', 'int', 'ID'),
                                     ('STORE_ID', 'int', 'ID'),
                                     ('AMOUNT', 'float', 'numerical'),
                                     ('TS', 'time', 'timestamp')], 'TS',
              [('USER_ID', 'users'), ('STORE_ID', 'stores')]),
    ))


@pytest.fixture(scope='module')
def indexed(tmp_path_factory):
    # Stores 0 and 2 are active; a fifth, active row has no key.
    root = tmp_path_factory.mktemp('link_prediction_candidates')
    orders = list(range(8))
    users = pd.DataFrame({'USER_ID': range(4)})
    users.to_parquet(root / 'users.parquet', index=False)
    pd.DataFrame({
        'STORE_ID': pd.array([0, 1, 2, 3, None], dtype='Int64'),
        'IS_ACTIVE': [True, False, True, False, True],
        'RATING': [0.0, 1.0, 2.0, 3.0, 4.0],
    }).to_parquet(root / 'stores.parquet', index=False)
    pd.DataFrame({
        'ORDER_ID': orders,
        'USER_ID': [order % 4 for order in orders],
        'STORE_ID': [order % 4 for order in orders],
        'AMOUNT': [float(order) for order in orders],
        'TS': pd.date_range('2024-01-01', periods=8),
    }).to_parquet(root / 'orders.parquet', index=False)
    metadata = _metadata()
    diskgraph.ingest(
        json.dumps(_build_diskgraph_ingest_config(str(root), metadata)),
        str(root / 'index'), col_stats=True, cat_count_strategy='hybrid',
        success_marker_content='{}')
    (root / 'metadata.json').write_text(json.dumps(metadata))
    graph = Graph(name='LINK_CANDIDATES')
    graph.connect(ParquetConnector('parquet', str(root)))
    graph.add_from_config(str(root / 'metadata.json'), data_source='parquet')
    return graph, str(root / 'index')


def _hosted(graph, index_path, condition):
    graph.infer_metadata()
    query = get_validated_query(graph, QUERY.format(condition))
    return mpt._link_prediction_candidates(
        SimpleNamespace(validated_predictive_query=query, graph=graph),
        TaskType.TEMPORAL_LINK_PREDICTION, index_path, 'job')


def _listed(ids):
    return None if ids is None else ids.tolist()


def _local(index_path, condition):
    graph = _native_graph()
    parsed = parse_and_validate_query(QUERY.format(condition), graph)
    return candidate_node_ids(parsed.query, to_api_graph(graph),
                              task_type=TaskType(parsed.task_type),
                              index_path=index_path, tt_id='local',
                              runtime=diskgraph)


@pytest.mark.parametrize('condition, stores', [
    ('stores.IS_ACTIVE = true', [0, 2]),
    ('stores.IS_ACTIVE = true AND stores.RATING > 1', [2]),
    ('orders.AMOUNT > 5 OR stores.IS_ACTIVE = true', None),
    ('NOT (orders.AMOUNT > 5 OR stores.IS_ACTIVE = false)', [0, 2]),
])
def test_hosted_and_local_inputs_give_the_same_candidates(
        indexed, condition, stores):
    graph, index_path = indexed
    hosted = _hosted(graph, index_path, condition)
    local = _local(index_path, condition)

    expected = None if stores is None else diskgraph.SamplerEngine(
        index_path).lookup_node_ids('stores', pa.array(stores, pa.int64()))
    assert _listed(hosted) == _listed(local) == expected


def test_hosted_prediction_without_candidates_is_not_retried(indexed):
    graph, index_path = indexed

    with pytest.raises(PredictionDataMaterializationError,
                       match="no candidates from table 'stores'") as error:
        _hosted(graph, index_path, 'stores.RATING > 100')
    assert error.value.non_retryable
