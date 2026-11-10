# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import diskgraph
import pyarrow as pa
import pyarrow.parquet as pq
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

INGESTED_DTYPES = {'bool': 'bool', 'float': 'float64'}
QUERY = ('PREDICT LIST_DISTINCT(orders.STORE_ID WHERE {}, 0, 7, days) '
         'RANK TOP 2 FOR EACH users.USER_ID')


def _table(name, key, columns, time=None, foreign_keys=()):
    return NativeGraphTable(
        name=name, source_table_name=name, columns=tuple(
            NativeGraphColumn(column, dtype, stype)
            for column, dtype, stype in columns), primary_key=key,
        time_column=time, end_time_column=None, foreign_keys=tuple(
            NativeGraphForeignKey(column, table, column)
            for column, table in foreign_keys))


GRAPH = NativeGraph((
    _table('users', 'USER_ID', [('USER_ID', 'int', 'ID')]),
    _table('stores', 'STORE_ID', [('STORE_ID', 'int', 'ID'),
                                  ('IS_ACTIVE', 'bool', 'categorical'),
                                  ('RATING', 'float', 'numerical')]),
    _table('orders', 'ORDER_ID', [('ORDER_ID', 'int', 'ID'),
                                  ('USER_ID', 'int', 'ID'),
                                  ('STORE_ID', 'int', 'ID'),
                                  ('AMOUNT', 'float', 'numerical'),
                                  ('TS', 'time', 'timestamp')], time='TS',
           foreign_keys=[('USER_ID', 'users'), ('STORE_ID', 'stores')]),
))


@pytest.fixture(scope='module')
def index(tmp_path_factory):
    # Stores 0 and 2 are active; a fifth, active row has no key.
    root = tmp_path_factory.mktemp('link_candidates')
    orders = list(range(8))
    tables = {
        'users': {
            'USER_ID': list(range(4))
        },
        'stores': {
            'STORE_ID': [0, 1, 2, 3, None],
            'IS_ACTIVE': [True, False, True, False, True],
            'RATING': [0.0, 1.0, 2.0, 3.0, 4.0],
        },
        'orders': {
            'ORDER_ID': orders,
            'USER_ID': [order % 4 for order in orders],
            'STORE_ID': [order % 4 for order in orders],
            'AMOUNT': [float(order) for order in orders],
            'TS': [1_704_067_200 + 86_400 * order for order in orders],
        },
    }
    config = []
    for table in GRAPH.tables:
        path = str(root / f'{table.name}.parquet')
        pq.write_table(pa.table(tables[table.name]), path)
        config.append({
            'name':
            table.name,
            'primary_key':
            table.primary_key,
            'columns': [{
                'name':
                column.name,
                'ingested_dtype':
                INGESTED_DTYPES.get(column.dtype, 'int64'),
            } for column in table.columns],
            'source': {
                'type': 'parquet',
                'paths': [path]
            },
            'fkeys': [{
                'src_column': key.name,
                'dst_table': key.destination_table
            } for key in table.foreign_keys],
        })
        if table.time_column is not None:
            config[-1]['time_column'] = table.time_column
    diskgraph.ingest(
        config_json=json.dumps({
            'dataset_name': 'link_candidates',
            'num_shards': 1,
            'tables': config
        }), output_dir=str(root / 'index'), col_stats=True)
    return str(root / 'index')


def _candidates(index, condition):
    parsed = parse_and_validate_query(QUERY.format(condition), GRAPH)
    return candidate_node_ids(parsed.query, to_api_graph(GRAPH),
                              task_type=TaskType(parsed.task_type),
                              index_path=index, tt_id='candidates',
                              runtime=diskgraph)


@pytest.mark.parametrize('condition, stores', [
    ('stores.IS_ACTIVE = true', [0, 2]),
    ('stores.IS_ACTIVE = true AND stores.RATING > 1', [2]),
    ('NOT (orders.AMOUNT > 5 OR stores.IS_ACTIVE = false)', [0, 2]),
    ('orders.AMOUNT > 5 OR stores.IS_ACTIVE = true', None),
    ('orders.AMOUNT > 5', None),
])
def test_candidates_are_the_keyed_rows_passing_candidate_conditions(
        index, condition, stores):
    ids = _candidates(index, condition)

    assert (None if ids is None else
            ids.tolist()) == (None if stores is None else
                              diskgraph.SamplerEngine(index).lookup_node_ids(
                                  'stores', pa.array(stores, pa.int64())))


def test_no_passing_candidate_is_rejected(index):
    with pytest.raises(ValueError, match="no candidates from table 'stores'"):
        _candidates(index, 'stores.RATING > 100')
