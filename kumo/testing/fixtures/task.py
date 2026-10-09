# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import pandas as pd
import pytest
from kumo_core.encoders.config import EncodersConfig, TargetEncoderConfig
from kumo_core.encoders.selection import compute_missing_stats_
from kumo_core.encoders.tensor_mappers import TensorMappers
from kumo_core.features.stores.feature import (
    FeatureStore,
    FeatureStoreTableSchema,
    FeatureStoreTrainingTableSchema,
)
from kumo_core.features.stores.graph import GraphStore
from kumo_core.features.stores.in_memory_feature import InMemoryFeatureStore
from kumo_core.training.options import allow_array_targets
from kumo_core.types import ColStatType, TaskType
from kumoapi.encoder import (
    Datetime,
    Encoder,
    Hash,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    Null,
    Numerical,
    NumericalList,
    TextEmbedding,
)
from kumoapi.typing import Stype

from kumo.stats import Stats
from kumo.testing.fixtures.in_memory_graph import InMemoryGraphStore


@dataclass
class TestTableSchema:
    stypes: dict[str, Stype] = field(default_factory=dict)
    pkey_name: str | None = None
    fkeys: list[tuple[str, str]] = field(default_factory=list)
    time_col: str | None = None
    end_time_col: str | None = None


def sample_categorical_encoder() -> Encoder:
    return random.choice([
        Index(na_strategy='separate'),
        Index(na_strategy='most_frequent'),
        Hash(num_components=10),
    ])


def sample_numerical_encoder() -> Encoder:
    return random.choice([
        Numerical(scaler=None, na_strategy='mean'),
        Numerical(scaler='standard', na_strategy='mean'),
        Numerical(scaler='minmax', na_strategy='mean'),
        Numerical(scaler='robust', na_strategy='mean'),
        Numerical(scaler=None, na_strategy='zero'),
        Numerical(scaler='standard', na_strategy='zero'),
        Numerical(scaler='minmax', na_strategy='zero'),
        Numerical(scaler='robust', na_strategy='zero'),
        MinLogNumerical(na_strategy='mean'),
        MinLogNumerical(na_strategy='zero'),
        MaxLogNumerical(na_strategy='mean'),
        MaxLogNumerical(na_strategy='zero'),
    ])


def sample_multicategorical_encoder(sep: str | None = None) -> Encoder:
    na_strategy = random.choice(['zero', 'separate', 'most_frequent'])
    encoder = MultiCategorical(na_strategy=na_strategy)
    value = Stats(value=sep, approx=False) if sep is not None else "|"
    encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] = value
    return encoder


def get_encoder(stype: Stype) -> Encoder:
    if stype == Stype.numerical:
        return sample_numerical_encoder()
    if stype in {Stype.categorical, Stype.ID}:
        return sample_categorical_encoder()
    if stype == Stype.multicategorical:
        return sample_multicategorical_encoder()
    if stype == Stype.timestamp:
        return Datetime()
    if stype == Stype.text:
        return TextEmbedding()
    if stype == Stype.sequence:
        return NumericalList()
    if stype == Stype.image:
        return Null()
    raise NotImplementedError


def get_target_encoder(task_type: TaskType) -> Encoder | None:
    if task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
    }:
        return Index()
    if task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
        return sample_numerical_encoder()
    if task_type.is_multilabel:
        encoder = MultiCategorical(na_strategy='zero')
        if allow_array_targets():
            encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] = None
        else:
            # TODO: Remove this separator as it's not needed for target cols.
            encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR] = '|'
        return encoder
    return None


def create_task_data(
    df_dict: dict[str, pd.DataFrame],
    schema_dict: dict[str, TestTableSchema],
    task_type: TaskType,
    train_table_df: pd.DataFrame,
    encoder_overrides: dict[str, Encoder] | None = None,
    weight_col: str | None = None,
) -> tuple[FeatureStore, GraphStore, EncodersConfig]:

    encoder_overrides = encoder_overrides or {}

    # Convert time columns to `pd.Datetime` ###################################

    for table_name, df in df_dict.items():
        schema = schema_dict[table_name]
        for col_name, stype in schema.stypes.items():
            if stype == Stype.timestamp:
                df[col_name] = pd.to_datetime(df[col_name])
        if schema.time_col is not None:
            df[schema.time_col] = pd.to_datetime(df[schema.time_col])
        if schema.end_time_col is not None:
            df[schema.end_time_col] = pd.to_datetime(df[schema.end_time_col])

    if 'TIME' in train_table_df:
        train_table_df['TIME'] = pd.to_datetime(train_table_df['TIME'])

    # Create Encoders #########################################################

    encoders = EncodersConfig()
    for table_name, schema in schema_dict.items():
        encoders[table_name] = {}
        for col_name, stype in schema.stypes.items():
            encoder = get_encoder(stype)
            compute_missing_stats_(encoder, df_dict[table_name][col_name])
            encoders[table_name][col_name] = encoder_overrides.get(
                f'{table_name}.{col_name}', encoder)
    encoders['TRAIN_TABLE'] = {}

    target_encoder = get_target_encoder(task_type)
    if 'TARGET' in train_table_df:
        assert target_encoder is not None
        compute_missing_stats_(target_encoder, train_table_df['TARGET'])
        encoders.target = TargetEncoderConfig('TARGET', target_encoder)

    # Create `FeatureStore` ###################################################

    mappers = TensorMappers.from_encoders(encoders)

    fs_table_schema_dict = {
        table_name:
        FeatureStoreTableSchema(
            mapper_dict=mappers[table_name],
            pkey_name=schema.pkey_name,
            time_col_name=schema.time_col,
            end_time_col_name=schema.end_time_col,
        )
        for table_name, schema in schema_dict.items()
    }

    # Check if any primary key shares a column name in the training table.
    train_fkeys = []
    for col_name in train_table_df.columns:
        for table_name, schema in schema_dict.items():
            if col_name == schema.pkey_name:
                train_fkeys.append((col_name, table_name))
    assert len(train_fkeys) in {1, 2}

    fs_train_table_schema = FeatureStoreTrainingTableSchema(
        table_name='TRAIN_TABLE',
        fkeys=train_fkeys,
        train_mask_name='TRAIN_MASK',
        val_mask_name='VAL_MASK',
        test_mask_name='TEST_MASK' if 'TEST_MASK' in train_table_df else None,
        mapper_dict=mappers['TRAIN_TABLE'],
        target_mapper=('TARGET', mappers.target.mapper)
        if target_encoder is not None else None,
        time_col_name='TIME' if 'TIME' in train_table_df else None,
        weight_col_name=weight_col,
    )

    df_dict = df_dict | {'TRAIN_TABLE': train_table_df}

    feature_store = InMemoryFeatureStore(
        df_dict=df_dict,
        table_schema_dict=fs_table_schema_dict,
        train_table_schema=fs_train_table_schema,
    )

    # Create `GraphStore` #####################################################

    fkeys = []
    for table_name, schema in schema_dict.items():
        for fkey, dst_table in schema.fkeys:
            fkeys.append((
                f'{table_name}.{fkey}',
                f'{dst_table}.{schema_dict[dst_table].pkey_name}',
            ))

    time_col_dict = {
        table_name: schema.time_col
        for table_name, schema in schema_dict.items()
        if schema.time_col is not None
    }

    graph_store = InMemoryGraphStore(
        df_dict=df_dict,
        fkeys=fkeys,
        time_col_dict=time_col_dict,
    )

    return feature_store, graph_store, encoders


@pytest.fixture()
def user_store_graph(
) -> tuple[dict[str, pd.DataFrame], dict[str, TestTableSchema]]:
    df_dict: dict[str, pd.DataFrame] = {}
    schema_dict: dict[str, TestTableSchema] = {}

    # USERS TABLE #############################################################
    df_dict['USERS'] = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3],
        'AGE': [20, 30, 40, 50],
        'GENDER': ['male', 'female', 'female', 'female'],
    })
    schema_dict['USERS'] = TestTableSchema(
        stypes={
            'AGE': Stype.numerical,
            'GENDER': Stype.categorical,
        },
        pkey_name='USER_ID',
    )

    # ORDERS TABLE ############################################################
    df_dict['ORDERS'] = pd.DataFrame({
        'ORDER_ID': [0, 1, 2, 3, 4, 5],
        'USER_ID': [0, 0, 1, 1, 1, 3],
        'STORE_ID': [0, 1, 0, 1, 2, 2],
        'AMOUNT': [10.0, 15.0, float('NaN'), 20.0, 25.0, 30.0],
        'TIME': ['1990', '1991', '1992', '1991', '1990', '1990'],
    })
    schema_dict['ORDERS'] = TestTableSchema(
        stypes={
            'AMOUNT': Stype.numerical,
            'TIME': Stype.timestamp,
        },
        pkey_name='ORDER_ID',
        fkeys=[('USER_ID', 'USERS'), ('STORE_ID', 'STORES')],
        time_col='TIME',
    )

    # STORES TABLE ############################################################
    df_dict['STORES'] = pd.DataFrame({
        'STORE_ID': [0, 1, 2],
        'CAT': ['burger', 'burger', 'pizza'],
        'MULTI_CAT': ['burger|fries', 'burger|pizza', 'pizza'],
        'TEXT': ['best burger in town', 'italian', 'hot pizza'],
        'EMBEDDING': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    })
    schema_dict['STORES'] = TestTableSchema(
        stypes={
            'CAT': Stype.categorical,
            'MULTI_CAT': Stype.multicategorical,
            'TEXT': Stype.text,
            'EMBEDDING': Stype.sequence,
        },
        pkey_name='STORE_ID',
    )

    return df_dict, schema_dict


def _temporal_task(user_store_graph, task_type) -> tuple[  #
        FeatureStore,
        GraphStore,
        EncodersConfig,
]:
    train_table_df = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        'TIME': ['1993'] * 4 + ['1994'] * 4 + ['1995'] * 4,
        'TRAIN_MASK': [True] * 4 + [False] * 4 + [False] * 4,
        'VAL_MASK': [False] * 4 + [True] * 4 + [False] * 4,
        'TEST_MASK': [False] * 4 + [False] * 4 + [True] * 4,
    })

    if task_type == TaskType.BINARY_CLASSIFICATION:
        train_table_df['TARGET'] = [
            True, True, False, True, False, True, False, True, True, True,
            False, True
        ]

    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        train_table_df['TARGET'] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    elif task_type.is_multilabel:
        if allow_array_targets():
            train_table_df['TARGET'] = [
                ['A'],
                ['A', 'B'],
                ['C'],
                ['B', 'C'],
                ['B'],
                ['B', 'C'],
                ['A'],
                ['C', 'A'],
                ['C'],
                [],
                ['A'],
                ['A', 'C'],
            ]
        else:
            train_table_df['TARGET'] = [
                'A',
                'A|B',
                'C',
                'B|C',
                'B',
                'B|C',
                'A',
                'C|A',
                'C',
                '',
                'A',
                'A|C',
            ]

    elif task_type in {TaskType.REGRESSION, TaskType.FORECASTING}:
        train_table_df['TARGET'] = [
            0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3
        ]

    elif task_type == TaskType.TEMPORAL_LINK_PREDICTION:
        train_table_df['USER_ID'] = [0, 0, 1, 2, 0, 1, 3, 3, 0, 1, 2, 3]
        train_table_df['STORE_ID'] = [0, 1, 2, 1, 2, 1, 0, 1, 0, 1, 2, 2]

    else:
        raise NotImplementedError

    return create_task_data(
        df_dict=user_store_graph[0],
        schema_dict=user_store_graph[1],
        task_type=task_type,
        train_table_df=train_table_df,
    )


@pytest.fixture()
def temporal_task(user_store_graph, task_type) -> tuple[  #
        FeatureStore,
        GraphStore,
        EncodersConfig,
]:
    return _temporal_task(user_store_graph, task_type)


@pytest.fixture()
def get_temporal_task(user_store_graph) -> Callable:
    return partial(_temporal_task, user_store_graph)


@pytest.fixture()
def static_task(user_store_graph, task_type) -> tuple[  #
        FeatureStore,
        GraphStore,
        EncodersConfig,
]:
    df_dict, schema_dict = user_store_graph

    train_table_df = pd.DataFrame({
        'TRAIN_MASK': [True] * 4 + [False] * 4 + [False] * 4,
        'VAL_MASK': [False] * 4 + [True] * 4 + [False] * 4,
        'TEST_MASK': [False] * 4 + [False] * 4 + [True] * 4,
    })

    if task_type == TaskType.STATIC_LINK_PREDICTION:
        train_table_df['USER_ID'] = [0, 0, 1, 2, 0, 1, 3, 3, 0, 1, 2, 3]
        train_table_df['STORE_ID'] = [0, 1, 2, 1, 2, 1, 0, 1, 0, 1, 2, 2]

        if allow_array_targets():
            # Adjust target table to align with pseudo time:
            df_dict['ORDERS']['TIME'] = [1, 1, 1, 2, 1, 3]
        else:
            # Add pseudo-time to training table:
            train_table_df['TIME'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            # Adjust target table to align with pseudo time:
            df_dict['ORDERS']['TIME'] = [1, 2, 0, 6, 3, 12]

    else:
        raise NotImplementedError

    return create_task_data(
        df_dict=df_dict,
        schema_dict=schema_dict,
        task_type=task_type,
        train_table_df=train_table_df,
    )


@pytest.fixture()
def static_order_classification_task(user_store_graph) -> tuple[  #
        FeatureStore,
        GraphStore,
        EncodersConfig,
]:
    """Static order classification task where ORDER.ID is
    the entity and also the fact table.
    """
    df_dict, schema_dict = user_store_graph

    train_table_df = pd.DataFrame({
        'ORDER_ID': [0, 1, 2, 3, 4, 5],
        'TIME': ['1993'] * 6,
        'TARGET': [True, False, True, False, True, False],
        'TRAIN_MASK': [True, True, False, False, False, False],
        'VAL_MASK': [False, False, True, True, False, False],
        'TEST_MASK': [False, False, False, False, True, True],
    })

    return create_task_data(
        df_dict=df_dict,
        schema_dict=schema_dict,
        task_type=TaskType.BINARY_CLASSIFICATION,
        train_table_df=train_table_df,
    )


@pytest.fixture()
def user_item_marketplace_graph(
) -> tuple[dict[str, pd.DataFrame], dict[str, TestTableSchema]]:
    df_dict: dict[str, pd.DataFrame] = {}
    schema_dict: dict[str, TestTableSchema] = {}

    # USERS TABLE #############################################################
    df_dict['USERS'] = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3],
        'AGE': [20, 30, 40, 50],
        'GENDER': ['male', 'female', 'female', 'female'],
    })
    schema_dict['USERS'] = TestTableSchema(
        stypes={
            'AGE': Stype.numerical,
            'GENDER': Stype.categorical,
        },
        pkey_name='USER_ID',
    )

    # ORDERS TABLE ############################################################
    df_dict['ORDERS'] = pd.DataFrame({
        'USER_ID': [0, 0, 1, 1, 1, 3],
        'ITEM_ID': [0, 1, 0, 1, 2, 2],
        'AMOUNT': [10.0, 15.0, float('NaN'), 20.0, 25.0, 30.0],
        'TIME': ['1990', '1991', '1991', '1990', '1991', '1992'],
    })
    schema_dict['ORDERS'] = TestTableSchema(
        stypes={
            'AMOUNT': Stype.numerical,
            'TIME': Stype.timestamp,
        },
        fkeys=[('USER_ID', 'USERS'), ('ITEM_ID', 'ITEMS')],
        time_col='TIME',
    )

    # ITEMS TABLE #############################################################
    df_dict['ITEMS'] = pd.DataFrame({
        'ITEM_ID': [0, 1, 2],
        'CAT': ['burger|fries', 'burger|pizza', 'pizza'],
        'TEXT': ['best burger in town', 'italian', 'hot pizza'],
        'EMBEDDING': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        'TIME': ['1990', '1991', '1992'],
        'END_TIME': ['1992', None, None],
    })
    schema_dict['ITEMS'] = TestTableSchema(
        stypes={
            'CAT': Stype.multicategorical,
            'TEXT': Stype.text,
            'EMBEDDING': Stype.sequence,
        },
        pkey_name='ITEM_ID',
        time_col='TIME',
        end_time_col='END_TIME',
    )

    return df_dict, schema_dict


@pytest.fixture()
def temporal_marketplace_task(user_item_marketplace_graph) -> tuple[  #
        FeatureStore,
        GraphStore,
        EncodersConfig,
]:
    train_table_df = pd.DataFrame({
        'USER_ID': [0, 0, 1, 2, 0, 1, 3, 3, 0, 1, 2, 3],
        'ITEM_ID': [0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 2],
        'TIME': ['1991'] * 4 + ['1992'] * 4 + ['1993'] * 4,
        'TRAIN_MASK': [True] * 4 + [False] * 4 + [False] * 4,
        'VAL_MASK': [False] * 4 + [True] * 4 + [False] * 4,
        'TEST_MASK': [False] * 4 + [False] * 4 + [True] * 4,
    })

    return create_task_data(
        df_dict=user_item_marketplace_graph[0],
        schema_dict=user_item_marketplace_graph[1],
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        train_table_df=train_table_df,
    )


@pytest.fixture()
def forecasting_task() -> tuple[FeatureStore, GraphStore, EncodersConfig]:
    df_dict: dict[str, pd.DataFrame] = {}
    schema_dict: dict[str, TestTableSchema] = {}
    df_dict['USERS'] = pd.DataFrame({
        'USER_ID': [0, 1, 2],
        'AGE': [10, 20, 30],
    })
    schema_dict['USERS'] = TestTableSchema(
        stypes={
            'AGE': Stype.numerical,
        },
        pkey_name='USER_ID',
    )
    df_dict['TRANS'] = pd.DataFrame({
        'USER_ID': [0] * 16 + [1] * 16 + [2] * 16,
        'AMOUNT': [1.0, 2.0, 3.0, 4.0] * 12,
        'DATE': [
            '1990-01-01', '1990-01-03', '1990-01-05', '1990-01-07',
            '1990-01-09', '1990-01-11', '1990-01-13', '1990-01-15',
            '1990-01-17', '1990-01-19', '1990-01-21', '1990-01-23',
            '1990-01-25', '1990-01-27', '1990-01-29', '1990-01-31'
        ] * 3,
        'PRODUCT_ID': [0] * 32 + [1] * 16
    })
    schema_dict['TRANS'] = TestTableSchema(
        stypes={
            'AMOUNT': Stype.numerical,
            'DATE': Stype.timestamp,
        },
        fkeys=[('USER_ID', 'USERS'), ('PRODUCT_ID', 'PRODUCTS')],
        time_col='DATE',
    )
    df_dict['PRODUCTS'] = pd.DataFrame({
        'PRODUCT_ID': [0, 1],
        'BRAND': ['HERMES', 'CHANEL']
    })
    schema_dict['PRODUCTS'] = TestTableSchema(
        stypes={'BRAND': Stype.categorical},
        pkey_name='PRODUCT_ID',
    )

    train_table_df = pd.DataFrame({
        'USER_ID': [0] * 20 + [1] * 20 + [2] * 20,
        'TIME': [
            '1990-01-11', '1990-01-12', '1990-01-13', '1990-01-14',
            '1990-01-15', '1990-01-16', '1990-01-17', '1990-01-18',
            '1990-01-19', '1990-01-20', '1990-01-21', '1990-01-22',
            '1990-01-23', '1990-01-24', '1990-01-25', '1990-01-26',
            '1990-01-27', '1990-01-28', '1990-01-29', '1990-01-30'
        ] * 3,
        # 10/5/5 split:
        'TRAIN_MASK':
        [True, False, True, False, False, False, False, False, False, False] +
        [False] * 10 +
        [False, False, False, False, True, True, True, True, True, False] +
        [False] * 10 +
        [True, True, True, False, False, False, False, True, True, True] +
        [False] * 10,
        'VAL_MASK': [False] * 10 + [True, True, False, True, False] +
        [False] * 15 + [False, True, False, True, True] + [False] * 15 +
        [True, False, True, True, True] + [False] * 5,
        'TEST_MASK': [False] * 15 + [False, True, True, True, False] +
        [False] * 15 + [True, False, True, False, False] + [False] * 15 +
        [True, True, False, True, True],
        'TARGET': [1.0, 2.0, 3.0] * 20,
    })

    return create_task_data(
        df_dict=df_dict,
        schema_dict=schema_dict,
        task_type=TaskType.FORECASTING,
        train_table_df=train_table_df,
    )


@pytest.fixture()
def temporal_hyperedge_task(user_store_graph, task_type) -> tuple[  #
        FeatureStore,
        GraphStore,
        EncodersConfig,
]:
    train_table_df = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        'STORE_ID': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'TIME': ['1993'] * 4 + ['1994'] * 4 + ['1995'] * 4,
        'TRAIN_MASK': [True] * 4 + [False] * 4 + [False] * 4,
        'VAL_MASK': [False] * 4 + [True] * 4 + [False] * 4,
        'TEST_MASK': [False] * 4 + [False] * 4 + [True] * 4,
    })

    if task_type == TaskType.BINARY_CLASSIFICATION:
        train_table_df['TARGET'] = [
            True, True, False, True, False, True, False, True, True, True,
            False, True
        ]

    else:
        raise NotImplementedError

    return create_task_data(
        df_dict=user_store_graph[0],
        schema_dict=user_store_graph[1],
        task_type=task_type,
        train_table_df=train_table_df,
    )


@pytest.fixture()
def temporal_explain_task(task_type: TaskType) -> tuple[  #
        FeatureStore,
        GraphStore,
        EncodersConfig,
]:
    df_dict: dict[str, pd.DataFrame] = {}
    schema_dict: dict[str, TestTableSchema] = {}

    df_dict['USER'] = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3],
        'AGE': [20, 20, 30, 30],  # Numerical
        'NAN': [float('NaN')] * 4,  # Numerical with NaN
        'NULL': [0, 1, 2, 3],  # Numerical with null encoder
        'GENDER': ['male', 'female', 'female', 'female'],  # Categorical
        'TIMESTAMP': ['1988', '1989', '1988', '1987'],  # Timestamp
        'SEQUENCE': [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ],  # Sequence
    })
    schema_dict['USER'] = TestTableSchema(
        stypes={
            'AGE': Stype.numerical,
            'NAN': Stype.numerical,
            'NULL': Stype.numerical,
            'GENDER': Stype.categorical,
            'TIMESTAMP': Stype.timestamp,
            'SEQUENCE': Stype.sequence,
        },
        pkey_name='USER_ID',
        time_col='TIMESTAMP',
    )
    df_dict['ORDER'] = pd.DataFrame({
        'USER_ID': [0, 0, 1, 1, 1, 3],
        'STORE_ID': [0, 1, 0, 1, 0, 1],
        'LOCATION_ID': [0, 0, 1, 1, 1, 2],  # ID
        'AMOUNT': [100.0, 150.0, 200.0,
                   float('NaN'), 250.0, 300.0],  # Numerical with NaN
        'CAT': ['cat_1', 'cat_2', '', '', '', 'cat_1'],  # Categorical
        'SAME_CAT': ['cat_1', 'cat_1', 'cat_1', 'cat_1', 'cat_1',
                     'cat_1'],  # Categorical with single category
        'MULTI_CAT': ['cat_1|cat_2', 'cat_2', '', '', pd.NA,
                      'cat_1|cat_2'],  # Multicategorical
        'SAME_NUM': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'TIMESTAMP': ['1990', '1991', '1991', '1991', '1991',
                      '1993'],  # Timestamp
        'TEXT': ['a', 'b', 'c', 'd', 'e', 'f'],  # Text
    })
    schema_dict['ORDER'] = TestTableSchema(
        stypes={
            'LOCATION_ID': Stype.ID,
            'AMOUNT': Stype.numerical,
            'CAT': Stype.categorical,
            'SAME_CAT': Stype.categorical,
            'MULTI_CAT': Stype.multicategorical,
            'SAME_NUM': Stype.numerical,
            'TIMESTAMP': Stype.timestamp,
            'TEXT': Stype.text,
        },
        fkeys=[('USER_ID', 'USER'), ('STORE_ID', 'STORE')],
        time_col='TIMESTAMP',
    )

    df_dict['STORE'] = pd.DataFrame({
        'STORE_ID': [0, 1],
        'DESC': ['a product', 'another product'],  # Text
        'IMAGE_URL': ['s3://image_1', 's3://image_2'],  # Image URL
    })
    schema_dict['STORE'] = TestTableSchema(
        stypes={
            'DESC': Stype.text,
            'IMAGE_URL': Stype.image,
        },
        pkey_name='STORE_ID',
    )

    if task_type == TaskType.BINARY_CLASSIFICATION:
        train_table_df = pd.DataFrame({
            'USER_ID': [0, 1, 0, 1, 2, 3],
            'TARGET': [True, False, True, False, False, True],
            'TIME': ['1993', '1993', '1994', '1994', '1994', '1994'],
            'TRAIN_MASK': [True] * 2 + [False] * 4,
            'VAL_MASK': [False] * 2 + [True] * 4,
        })
    elif task_type == TaskType.TEMPORAL_LINK_PREDICTION:
        train_table_df = pd.DataFrame({
            'USER_ID': [0, 1, 2, 3, 0, 1, 2, 3],
            'STORE_ID': [0, 1, 0, 1, 0, 1, 0, 1],
            'TIME': ['1993'] * 4 + ['1994'] * 4,
            'TRAIN_MASK': [True] * 4 + [False] * 4,
            'VAL_MASK': [False] * 4 + [True] * 4,
        })
    else:
        raise NotImplementedError

    return create_task_data(
        df_dict=df_dict,
        schema_dict=schema_dict,
        task_type=task_type,
        train_table_df=train_table_df,
        encoder_overrides={'USER.NULL': Null()},
    )
