# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

import pandas as pd
import pytest
import torch
from kumo_core.diskgraph.metadata import DiskgraphSchema
from kumo_core.diskgraph.stores import DiskgraphFeatureStore
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.features.stores.feature import FeatureStoreTableSchema
from kumo_core.features.time import to_datetime
from kumoapi.encoder import Datetime, Encoder, Index, NAStrategy, Numerical
from kumoapi.typing import ColStatType, Stype
from torch_geometric.data import HeteroData

from tests.support.diskgraph_feature_store import make_diskgraph_feature_store


@dataclass
class TestTableSchema:
    stypes: dict[str, Stype] = field(default_factory=dict)
    pkey_name: str | None = None
    fkeys: list[tuple[str, str]] = field(default_factory=list)
    time_col: str | None = None
    end_time_col: str | None = None


def create_graph_data(
    df_dict: dict[str, pd.DataFrame],
    schema_dict: dict[str, TestTableSchema],
    table_encoder_dict: dict[str, dict[str, Encoder]],
) -> tuple[
    DiskgraphFeatureStore,
    DiskgraphSchema,
    dict[str, dict[str, Encoder]],
]:
    """Build graph fixtures from explicit encoders and table data.

    Args:
        df_dict: Synthetic graph-table rows.
        schema_dict: Column and relation metadata for each table.
        table_encoder_dict: Encoders with statistics supplied by the fixture.

    Returns:
        Feature store, graph metadata, and the supplied encoders.

    Raises:
        KeyError: If a table or column has no configured encoder.
    """
    # Convert time columns to `pd.Datetime` ###################################
    for table_name, df in df_dict.items():
        schema = schema_dict[table_name]
        for col_name, stype in schema.stypes.items():
            if stype == Stype.timestamp:
                df[col_name] = to_datetime(df[col_name])
        if schema.time_col is not None:
            df[schema.time_col] = to_datetime(df[schema.time_col])
        if schema.end_time_col is not None:
            df[schema.end_time_col] = to_datetime(df[schema.end_time_col])

    # Create tensor mappers ###################################################
    table_mapper_dict = {
        table_name: {
            column: TensorMapper.from_encoder(table_encoder_dict[table_name][column])
            for column in schema.stypes
        }
        for table_name, schema in schema_dict.items()
    }

    # Create `FeatureStore` ###################################################
    fs_table_schema_dict = {
        table_name:
        FeatureStoreTableSchema(
            mapper_dict=table_mapper_dict[table_name],
            pkey_name=schema.pkey_name,
            time_col_name=schema.time_col,
            end_time_col_name=schema.end_time_col,
        )
        for table_name, schema in schema_dict.items()
    }

    feature_store = make_diskgraph_feature_store(
        df_dict=df_dict,
        table_schema_dict=fs_table_schema_dict,
    )

    # Graph metadata for explanation:
    fkeys = []
    for table_name, schema in schema_dict.items():
        for fkey, dst_table in schema.fkeys:
            fkeys.append((
                f'{table_name}.{fkey}',
                f'{dst_table}.{schema_dict[dst_table].pkey_name}',
            ))

    edge_types = [
        edge_type
        for table_name, schema in schema_dict.items()
        for column_name, destination in schema.fkeys
        for edge_type in (
            (table_name, column_name, destination),
            (destination, f'rev_{column_name}', table_name),
        )
    ]
    graph_schema = DiskgraphSchema(
        table_names=list(schema_dict),
        pkey_dict={
            name: schema.pkey_name
            for name, schema in schema_dict.items()
            if schema.pkey_name is not None
        },
        fk_specs=fkeys,
        raw_edge_types=list(edge_types),
        edge_types=edge_types,
        edge_type_remap={},
        node_types=list(schema_dict),
    )

    return feature_store, graph_schema, table_encoder_dict


@pytest.fixture()
def user_store_graph() -> tuple[
    DiskgraphFeatureStore,
    DiskgraphSchema,
    dict[str, dict[str, Encoder]],
]:
    df_dict: dict[str, pd.DataFrame] = {}
    schema_dict: dict[str, TestTableSchema] = {}

    # USERS table #############################################################
    df_dict['USERS'] = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3],
        'AGE': [20, 30, 40, float('NaN')],
        'GENDER': ['male', 'female', 'female', None],
        'STATUS': ['A', 'B', 'A', 'C'],
    })
    schema_dict['USERS'] = TestTableSchema(
        stypes={
            'AGE': Stype.numerical,
            'GENDER': Stype.categorical,
            'STATUS': Stype.categorical,
        },
        pkey_name='USER_ID',
    )

    # ORDERS table ############################################################
    df_dict['ORDERS'] = pd.DataFrame({
        'ORDER_ID': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'USER_ID': [0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3],
        'STORE_ID': [0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 1, 2],
        'AMOUNT':
        [10, 15, float('NaN'), 20, 25, 30, 10, 25, 20, 10, 15, 15, 20],
        'AMOUNT_CAT':
        [10, 15, float('NaN'), 20, 25, 30, 11, 26, 27, 28, 29, 32, 33],
        'TIME': [
            '2025-01-01',
            '2025-01-02',
            '2025-01-03',
            '2025-01-02',
            '2025-01-03',
            '2025-01-04',
            '2025-01-09',
            '2025-01-02',
            '2025-01-02',
            '2025-01-01',
            '2025-01-02',
            '2025-01-03',
            '2025-01-04',
        ],
    })
    schema_dict['ORDERS'] = TestTableSchema(
        stypes={
            'AMOUNT': Stype.numerical,
            'AMOUNT_CAT': Stype.categorical,
            'TIME': Stype.timestamp,
        },
        pkey_name='ORDER_ID',
        fkeys=[('USER_ID', 'USERS'), ('STORE_ID', 'STORES')],
        time_col='TIME',
    )

    # STORES table ############################################################
    df_dict['STORES'] = pd.DataFrame({
        'STORE_ID': [0, 1, 2],
        'CAT': ['burger', 'burger', 'pizza'],
    })
    schema_dict['STORES'] = TestTableSchema(
        stypes={
            'CAT': Stype.categorical,
        },
        pkey_name='STORE_ID',
    )

    encoders: dict[str, dict[str, Encoder]] = {
        "USERS": {
            "AGE": Numerical(_stats={ColStatType.MEAN: 30.0}),
            "GENDER": Index(
                na_strategy=NAStrategy.SEPARATE,
                _stats={ColStatType.CATEGORY_COUNTS: [["female", 2], ["male", 1]]},
            ),
            "STATUS": Index(
                na_strategy=NAStrategy.SEPARATE,
                _stats={ColStatType.CATEGORY_COUNTS: [["A", 2], ["B", 1], ["C", 1]]},
            ),
        },
        "ORDERS": {
            "AMOUNT": Numerical(_stats={ColStatType.MEAN: 18.75}),
            "AMOUNT_CAT": Index(
                na_strategy=NAStrategy.SEPARATE,
                _stats={
                    ColStatType.CATEGORY_COUNTS: [
                        [value, 1]
                        for value in [10, 11, 15, 20, 25, 26, 27, 28, 29, 30, 32, 33]
                    ]
                },
            ),
            "TIME": Datetime(
                _stats={
                    ColStatType.MIN: "kumo_timestamp_2025-01-01 00:00:00",
                    ColStatType.MAX: "kumo_timestamp_2025-01-09 00:00:00",
                }
            ),
        },
        "STORES": {
            "CAT": Index(
                na_strategy=NAStrategy.SEPARATE,
                _stats={ColStatType.CATEGORY_COUNTS: [["burger", 2], ["pizza", 1]]},
            ),
        },
    }
    return create_graph_data(df_dict, schema_dict, encoders)


@pytest.fixture()
def large_user_store_graph() -> tuple[
    DiskgraphFeatureStore,
    DiskgraphSchema,
    dict[str, dict[str, Encoder]],
]:
    df_dict: dict[str, pd.DataFrame] = {}
    schema_dict: dict[str, TestTableSchema] = {}

    # USERS table #############################################################
    df_dict['USERS'] = pd.DataFrame({
        'USER_ID':
        list(range(100)),
        'AGE': [20, 30, 40, float('NaN')] + [32] * 96,
        'GENDER': ['male', 'female', 'female', None] + ['male'] * 96,
        'STATUS': ['A', 'B', 'A', 'C'] + ['A'] * 96,
    })
    schema_dict['USERS'] = TestTableSchema(
        stypes={
            'AGE': Stype.numerical,
            'GENDER': Stype.categorical,
            'STATUS': Stype.categorical,
        },
        pkey_name='USER_ID',
    )

    # ORDERS table ############################################################
    df_dict['ORDERS'] = pd.DataFrame({
        'ORDER_ID':
        list(range(200)),
        'USER_ID':
        list(range(100)) + [0, 1, 2, 3, 4] * 20,
        'STORE_ID':
        list(range(10)) * 20,
        'AMOUNT':
        [10, 15, float('NaN'), 20, 25, 30, 10, 25, 20, 10] + [15] * 190,
        'AMOUNT_CAT':
        [10, 15, float('NaN'), 20, 25, 30, 11, 26, 27, 28] + [29] * 190,
        'TIME': [
            '2025-01-01',
            '2025-01-02',
            '2025-01-03',
            '2025-01-02',
            '2025-01-03',
            '2025-01-04',
            '2025-01-09',
            '2025-01-02',
            '2025-01-02',
            '2025-01-01',
        ] + ['2025-01-09'] * 190,
    })
    schema_dict['ORDERS'] = TestTableSchema(
        stypes={
            'AMOUNT': Stype.numerical,
            'AMOUNT_CAT': Stype.categorical,
            'TIME': Stype.timestamp,
        },
        pkey_name='ORDER_ID',
        fkeys=[('USER_ID', 'USERS'), ('STORE_ID', 'STORES')],
        time_col='TIME',
    )

    # STORES table ############################################################
    df_dict['STORES'] = pd.DataFrame({
        'STORE_ID': list(range(10)),
        'CAT': ['burger', 'pizza'] * 5,
    })
    schema_dict['STORES'] = TestTableSchema(
        stypes={
            'CAT': Stype.categorical,
        },
        pkey_name='STORE_ID',
    )

    encoders: dict[str, dict[str, Encoder]] = {
        "USERS": {
            "AGE": Numerical(_stats={ColStatType.MEAN: 3162 / 99}),
            "GENDER": Index(
                na_strategy=NAStrategy.SEPARATE,
                _stats={ColStatType.CATEGORY_COUNTS: [["male", 97], ["female", 2]]},
            ),
            "STATUS": Index(
                na_strategy=NAStrategy.SEPARATE,
                _stats={ColStatType.CATEGORY_COUNTS: [["A", 98], ["B", 1], ["C", 1]]},
            ),
        },
        "ORDERS": {
            "AMOUNT": Numerical(_stats={ColStatType.MEAN: 3015 / 199}),
            "AMOUNT_CAT": Index(
                na_strategy=NAStrategy.SEPARATE,
                _stats={
                    ColStatType.CATEGORY_COUNTS: [[29, 190]]
                    + [[value, 1] for value in [10, 11, 15, 20, 25, 26, 27, 28, 30]]
                },
            ),
            "TIME": Datetime(
                _stats={
                    ColStatType.MIN: "kumo_timestamp_2025-01-01 00:00:00",
                    ColStatType.MAX: "kumo_timestamp_2025-01-09 00:00:00",
                }
            ),
        },
        "STORES": {
            "CAT": Index(
                na_strategy=NAStrategy.SEPARATE,
                _stats={ColStatType.CATEGORY_COUNTS: [["burger", 5], ["pizza", 5]]},
            ),
        },
    }
    return create_graph_data(df_dict, schema_dict, encoders)


def user_store_batch(
    feature_store: DiskgraphFeatureStore,
    num_hops: int = 2,
) -> HeteroData:
    """Fixed temporal batch for explanation and edge-exclusion tests.

    These tests consume sampled graphs; they do not test a PyG sampler. Keep
    their input independent of optional native extensions. IDs are ordered
    by user then order time, with disjoint store copies for each root user.
    """
    assert num_hops in (0, 1, 2)
    order_ids = [0, 1, 2, 3, 4, 5, 6, 9, 7, 8, 10, 11, 12]
    order_users = [0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3]
    node_spec = {
        'USERS': ([0, 1, 2, 3], [0, 1, 2, 3], [4, 0, 0]),
        'ORDERS': (order_ids if num_hops else [],
                   order_users if num_hops else [], [0, 13, 0]),
        'STORES': ([0, 1, 1, 2, 0, 0, 1, 2] if num_hops == 2 else [],
                   [0, 0, 1, 1, 1, 3, 3, 3] if num_hops == 2 else [],
                   [0, 0, 8]),
    }
    data = HeteroData()
    for table, (ids, roots, counts) in node_spec.items():
        store = data[table]
        store.n_id = torch.tensor(ids, dtype=torch.long)
        store.num_nodes = len(ids)
        store.batch = torch.tensor(roots, dtype=torch.long)
        store.num_sampled_nodes = counts[:num_hops + 1]
        for attr in feature_store.get_all_tensor_attrs():
            if attr.table_name == table:
                store[attr.attr_name] = feature_store.get_tensor(
                    table, attr.attr_name, store.n_id)

    data['USERS'].input_id = torch.arange(4)
    data['USERS'].seed_time = torch.full(
        (4,), pd.Timestamp('2025-01-09').value, dtype=torch.long)
    data['USERS'].batch_size = 4
    order_local = list(range(13))
    edge_spec = {
        ('ORDERS', 'USER_ID', 'USERS'): (
            [order_local, order_users], [13, 0], 1),
        ('USERS', 'rev_USER_ID', 'ORDERS'): (
            [order_users, order_local], [0, 13], 2),
        ('ORDERS', 'STORE_ID', 'STORES'): ([[], []], [0, 0], 2),
        ('STORES', 'rev_STORE_ID', 'ORDERS'): (
            [[0, 1, 0, 2, 3, 3, 4, 5, 6, 7, 6, 6, 7], order_local],
            [0, 13], 2),
    }
    for edge_type, (edges, counts, first_hop) in edge_spec.items():
        data[edge_type].edge_index = torch.tensor(
            edges if num_hops >= first_hop else [[], []], dtype=torch.long)
        data[edge_type].num_sampled_edges = counts[:num_hops]
    return data
