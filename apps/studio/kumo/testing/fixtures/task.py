# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from kumo_core.diskgraph.metadata import DiskgraphSchema
from kumo_core.encoders.config import EncodersConfig, TargetEncoderConfig
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.features.stores.feature import FeatureStore
from kumo_core.training.options import allow_array_targets
from kumo_core.types import TaskType
from kumoapi.encoder import (
    Datetime,
    Index,
    MultiCategorical,
    NAStrategy,
    Numerical,
    NumericalList,
    TextEmbedding,
)
from kumoapi.typing import ColStatType, Stype
from torch import Tensor


@dataclass
class TestTableSchema:
    stypes: dict[str, Stype] = field(default_factory=dict)
    pkey_name: str | None = None
    fkeys: list[tuple[str, str]] = field(default_factory=list)
    time_col: str | None = None
    end_time_col: str | None = None


def create_task_data(
    df_dict: dict[str, pd.DataFrame],
    schema_dict: dict[str, TestTableSchema],
    encoders: EncodersConfig,
    train_table_df: pd.DataFrame,
    weight_col: str | None = None,
) -> tuple[FeatureStore, DiskgraphSchema, EncodersConfig]:
    """Build model-configuration fixtures with mocked feature-store metadata.

    Args:
        df_dict: Graph-table data used for primary keys and table sizes.
        schema_dict: Graph column and relation metadata.
        encoders: Explicit feature and target encoders, with fixture statistics.
        train_table_df: Training metadata and target values.
        weight_col: Optional training-weight column name.

    Returns:
        Mocked feature metadata, graph schema, and the supplied encoders.

    Raises:
        AssertionError: If the training table has no valid entity keys or its
            target encoder is missing.
        KeyError: If a feature column has no configured encoder.
    """

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

    if "TIME" in train_table_df:
        train_table_df["TIME"] = pd.to_datetime(train_table_df["TIME"])

    encoders["TRAIN_TABLE"] = {}
    if "TARGET" in train_table_df:
        assert encoders.target is not None

    mappers = {
        table_name: {
            column_name: TensorMapper.from_encoder(encoder)
            for column_name, encoder in table_encoders.items()
        }
        for table_name, table_encoders in encoders.items()
    }
    train_fkeys = [
        (column, table_name)
        for column in train_table_df.columns
        for table_name, schema in schema_dict.items()
        if column == schema.pkey_name
    ]
    assert len(train_fkeys) in {1, 2}
    time_columns = {
        name: schema.time_col
        for name, schema in schema_dict.items()
        if schema.time_col is not None
    }
    feature_store = MagicMock(
        spec=FeatureStore,
        table_names=list(schema_dict),
        train_table_name="TRAIN_TABLE",
        entity_table_names=[name for _, name in train_fkeys],
        train_fkey_names=[column for column, _ in train_fkeys],
        time_col_dict=time_columns,
        has_train_table_time_col="TIME" in train_table_df,
        has_train_table_target_col=encoders.target is not None,
        has_train_table_weight_col=weight_col is not None,
        has_train_table_test_mask="TEST_MASK" in train_table_df,
    )
    feature_store.get_target_tensor_mapper.return_value = (
        TensorMapper.from_encoder(encoders.target.encoder)
        if encoders.target is not None
        else None
    )
    feature_store.get_tensor_mapper.side_effect = lambda table, column: mappers[table][
        column
    ]
    feature_store.get_pkey_name.side_effect = lambda table: schema_dict[table].pkey_name
    feature_store.get_time_col_name.side_effect = lambda table: (
        ("TIME" if "TIME" in train_table_df else None)
        if table == "TRAIN_TABLE"
        else time_columns.get(table)
    )
    feature_store.get_end_time_col_name.side_effect = lambda table: (
        schema_dict[table].end_time_col if table in schema_dict else None
    )
    feature_store.get_tensor_size.side_effect = lambda table, *_args, **_kwargs: (
        len(train_table_df) if table == "TRAIN_TABLE" else len(df_dict[table]),
    )
    feature_store.safe_batch_mode.side_effect = lambda *_args, **_kwargs: nullcontext()

    def read_primary_keys(
        query: tuple[str, str, Tensor | slice | None],
    ) -> np.ndarray:
        table_name, attribute, index = query
        if attribute != "pkey":
            raise KeyError(f"This fixture only reads primary keys, got {attribute!r}")
        pkey_name = schema_dict[table_name].pkey_name
        assert pkey_name is not None
        values = df_dict[table_name][pkey_name].to_numpy()
        selected = index.cpu().numpy() if isinstance(index, Tensor) else index
        return values if selected is None else values[selected]

    feature_store.__getitem__.side_effect = read_primary_keys

    # Create `DiskgraphSchema` #####################################################

    fkeys = []
    for table_name, schema in schema_dict.items():
        for fkey, dst_table in schema.fkeys:
            fkeys.append(
                (
                    f"{table_name}.{fkey}",
                    f"{dst_table}.{schema_dict[dst_table].pkey_name}",
                )
            )

    edge_types = [
        edge_type
        for table_name, schema in schema_dict.items()
        for column, destination in schema.fkeys
        for edge_type in (
            (table_name, column, destination),
            (destination, f"rev_{column}", table_name),
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

    return feature_store, graph_schema, encoders


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


def _temporal_task(user_store_graph, task_type) -> tuple[
        FeatureStore,
        DiskgraphSchema,
        EncodersConfig,
]:
    train_table_df = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        'TIME': ['1993'] * 4 + ['1994'] * 4 + ['1995'] * 4,
        'TRAIN_MASK': [True] * 4 + [False] * 4 + [False] * 4,
        'VAL_MASK': [False] * 4 + [True] * 4 + [False] * 4,
        'TEST_MASK': [False] * 4 + [False] * 4 + [True] * 4,
    })

    encoders = EncodersConfig(
        encoders={
            "USERS": {
                "AGE": Numerical(_stats={ColStatType.MEAN: 35.0}),
                "GENDER": Index(
                    _stats={ColStatType.CATEGORY_COUNTS: [["female", 3], ["male", 1]]}
                ),
            },
            "ORDERS": {
                "AMOUNT": Numerical(_stats={ColStatType.MEAN: 20.0}),
                "TIME": Datetime(
                    _stats={
                        ColStatType.MIN: "kumo_timestamp_1990-01-01 00:00:00",
                        ColStatType.MAX: "kumo_timestamp_1992-01-01 00:00:00",
                    }
                ),
            },
            "STORES": {
                "CAT": Index(
                    _stats={ColStatType.CATEGORY_COUNTS: [["burger", 2], ["pizza", 1]]}
                ),
                "MULTI_CAT": MultiCategorical(
                    _stats={
                        ColStatType.MULTI_CATEGORIES_SEPARATOR: "|",
                        ColStatType.MULTI_CATEGORY_COUNTS: [
                            ["burger", 2],
                            ["pizza", 2],
                            ["fries", 1],
                        ],
                    }
                ),
                "TEXT": TextEmbedding(),
                "EMBEDDING": NumericalList(
                    _stats={
                        ColStatType.SEQUENCE_MIN_LENGTH: 2,
                        ColStatType.SEQUENCE_MAX_LENGTH: 2,
                    }
                ),
            },
        }
    )

    if task_type == TaskType.BINARY_CLASSIFICATION:
        encoders.target = TargetEncoderConfig(
            "TARGET",
            Index(_stats={ColStatType.CATEGORY_COUNTS: [[True, 8], [False, 4]]}),
        )
        train_table_df['TARGET'] = [
            True, True, False, True, False, True, False, True, True, True,
            False, True
        ]

    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        encoders.target = TargetEncoderConfig(
            "TARGET",
            Index(_stats={ColStatType.CATEGORY_COUNTS: [[0, 4], [1, 4], [2, 4]]}),
        )
        train_table_df['TARGET'] = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]

    elif task_type.is_multilabel:
        encoders.target = TargetEncoderConfig(
            "TARGET",
            MultiCategorical(
                na_strategy=NAStrategy.ZERO,
                _stats={
                    ColStatType.MULTI_CATEGORIES_SEPARATOR: None
                    if allow_array_targets()
                    else "|",
                    ColStatType.MULTI_CATEGORY_COUNTS: [["A", 6], ["C", 6], ["B", 4]],
                },
            ),
        )
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
        encoders.target = TargetEncoderConfig(
            "TARGET", Numerical(_stats={ColStatType.MEAN: 0.2})
        )
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
        encoders=encoders,
        train_table_df=train_table_df,
    )


@pytest.fixture()
def temporal_task(user_store_graph, task_type) -> tuple[
        FeatureStore,
        DiskgraphSchema,
        EncodersConfig,
]:
    return _temporal_task(user_store_graph, task_type)
