# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os.path as osp
from typing import Any, Callable

import pandas as pd
import pytest
from kumoapi.encoder import Null, Numerical, TextEmbedding
from kumoapi.typing import Dtype, Stype

from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.graph import Graph
from kumo.task.task import Task
from kumo.testing.fixtures.task import (
    forecasting_task,
    static_task,
    temporal_explain_task,
    temporal_task,
    user_store_graph,
)


@pytest.fixture(autouse=True)
def disable_logging():
    # In ML tests, we usually don't care about heaps of logs from other modules
    logging.getLogger('kumo.data').setLevel(logging.ERROR)
    logging.getLogger('kumo.pquery').setLevel(logging.ERROR)
    logging.getLogger('kumo.stats').setLevel(logging.ERROR)
    logging.getLogger('kumo.graph').setLevel(logging.ERROR)
    logging.getLogger('kumo.connector').setLevel(logging.ERROR)


@pytest.fixture()
def create_explainer_task() -> Callable:
    return _create_explainer_task


def _create_explainer_task(
    tmp_path: str,
    task_name: str,
    single_table: bool = False,
    link_pred: bool = False,
) -> Any:
    user_df = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3],
        'AGE': [20, 20, 30, 30],  # Numerical
        'NAN': [float('NaN')] * 4,  # Numerical with NaN
        'NULL': [0, 1, 2, 3],  # Numerical with null encoder
        'GENDER': ['male', 'female', 'female', 'female'],  # Categorical
        'TIMESTAMP':
        pd.to_datetime(['1988', '1989', '1988', '1987']),  # Timestamp
        'SEQUENCE1': [  # Array-based sequence:
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ],
        'SEQUENCE2': [  # String-based sequence:
            '[0.1, 0.2, 0.3, 0.4]',
            '[0.1, 0.2, 0.3, 0.4]',
            '[0.1, 0.2, 0.3, 0.4]',
            '[0.1, 0.2, 0.3, 0.4]',
        ],
    })
    user_df.to_parquet(osp.join(tmp_path, 'USER.parquet'))
    order_df = pd.DataFrame({
        'USER_ID': [0, 0, 1, 1, 1, 3],
        'STORE_ID': [0, 1, 0, 1, 0, 1],
        'ORDER_ID': [0, 1, 2, 3, 4, 5],  # Unique IDs
        'LOCATION_ID': [0, 0, 1, 1, 1, 2],  # ID
        'AMOUNT': [100.0, 150.0, 200.0,
                   float('NaN'), 250.0, 300.0],  # Numerical with NaN
        'CAT': ['cat_1', 'cat_2', '', '', '', 'cat_1'],  # Categorical
        'SAME_CAT': ['cat_1', 'cat_1', 'cat_1', 'cat_1', 'cat_1',
                     'cat_1'],  # Categorical with single category
        'MULTI_CAT': ['cat_1|cat_2', 'cat_2', '', '', pd.NA,
                      'cat_1|cat_2'],  # Multicategorical
        'SAME_NUM': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'TIMESTAMP':
        pd.to_datetime(['1990', '1991', '1991', '1991', '1991',
                        '1993']),  # Timestamp
        'TEXT': ['a', 'b', 'c', 'd', 'e', 'f'],  # Text
    })
    order_df.to_parquet(osp.join(tmp_path, 'ORDER.parquet'))
    store_df = pd.DataFrame({
        'STORE_ID': [0, 1],
        'DESC': ['a product', 'another product'],  # Text
        'IMAGE_URL': ['s3://image_1', 's3://image_2'],  # Image URL
    })
    store_df.to_parquet(osp.join(tmp_path, 'STORE.parquet'))

    graph = Graph(name=task_name)
    graph.connect(ParquetConnector(name='parquet', root_dir=tmp_path))
    graph.add('USER', pkey='USER_ID', time_col='TIMESTAMP')
    if not single_table:
        graph.add('ORDER', pkey='ORDER_ID', fkeys=['USER_ID', 'STORE_ID'],
                  time_col='TIMESTAMP')
        graph.add('STORE', pkey='STORE_ID')
    graph.infer_metadata()

    # Override and check stypes ###############################################
    graph['USER', 'AGE'].stype = Stype.numerical  # type: ignore
    graph['USER', 'NAN'].stype = Stype.numerical  # type: ignore
    graph['USER', 'NULL'].stype = Stype.numerical  # type: ignore
    graph['USER', 'SEQUENCE2'].stype = Stype.sequence  # type: ignore
    assert graph['USER', 'GENDER'].stype == Stype.categorical  # type: ignore
    assert graph['USER', 'TIMESTAMP'].stype == Stype.timestamp  # type: ignore

    if not single_table:
        assert graph['ORDER', 'AMOUNT'].stype == Stype.numerical  # type: ignore
        assert graph['ORDER', 'CAT'].stype == Stype.categorical  # type: ignore
        assert graph['ORDER',
                     'SAME_CAT'].stype == Stype.categorical  # type: ignore
        graph['ORDER',
              'MULTI_CAT'].stype = Stype.multicategorical  # type: ignore
        graph['ORDER',
              'MULTI_CAT'].multicategorical_delimiter = '|'  # type: ignore
        graph['ORDER', 'SAME_NUM'].stype = Stype.numerical  # type: ignore
        graph['ORDER', 'ORDER_ID'].stype = Stype.ID  # type: ignore
        graph['ORDER', 'LOCATION_ID'].stype = Stype.ID  # type: ignore
        graph['ORDER', 'TEXT'].stype = Stype.text  # type: ignore
        assert graph['ORDER',
                     'TIMESTAMP'].stype == Stype.timestamp  # type: ignore
        graph['STORE', 'DESC'].stype = Stype.text  # type: ignore
        graph['STORE', 'IMAGE_URL'].stype = Stype.image  # type: ignore

    graph.infer_links()
    graph.infer_encoders()

    # Override encoders #######################################################
    graph['USER', 'NAN'].encoder = Numerical()  # type: ignore
    graph['USER', 'NULL'].encoder = Null()  # type: ignore
    if not single_table:
        graph['STORE', 'DESC'].encoder = TextEmbedding()  # type: ignore

    if not link_pred:
        train_table_df = pd.DataFrame({
            'USER_ID': [0, 1, 0, 1, 2, 3],  # Two dummy values for training.
            'TARGET': [True, False, True, False, False, True],
            'TIMESTAMP':
            pd.to_datetime(['1993', '1993', '1994', '1994', '1994', '1994']),
            'arange': [0, 1, 2, 3, 4, 5],
        })
        task_type = 'binary_classification'
        target_col = dict(name='TARGET', dtype=Dtype.bool)
        fkeys = [dict(name='USER_ID', dst_table='USER')]
        split_ratio = [0.3, 0.7]
    else:
        train_table_df = pd.DataFrame({
            # Dummy values for training.
            'USER_ID': [0, 1, 2, 3, 0, 1, 2, 3],
            'STORE_ID': [0, 1, 0, 1, 0, 1, 0, 1],
            'TIMESTAMP':
            pd.to_datetime(['1993'] * 4 + ['1994'] * 4),
            'arange': [0, 1, 2, 3, 4, 5, 6, 7],
        })
        task_type = 'temporal_link_prediction'
        target_col = None
        fkeys = [
            dict(name='USER_ID', dst_table='USER'),
            dict(name='STORE_ID', dst_table='STORE'),
        ]
        split_ratio = [0.5, 0.5]

    train_table_df.to_parquet(osp.join(tmp_path, 'TRAIN_TABLE.parquet'))

    task = Task(
        graph,
        task_type=task_type,
        train_table_name='TRAIN_TABLE',
        target_col=target_col,  # type: ignore
        fkeys=fkeys,  # type: ignore
        time_col='TIMESTAMP',
        task_name=task_name,
    ).fixed_split(split_ratio)

    if link_pred:
        task.target_to_entity_paths = [
            [('parquet.ORDER', 'USER_ID', 'parquet.USER')],
            [('parquet.ORDER', 'STORE_ID', 'parquet.STORE')],
        ]
    if not single_table:
        return task, user_df, order_df, store_df
    return task, user_df


__all__ = [
    'forecasting_task',
    'user_store_graph',
    'temporal_task',
    'temporal_explain_task',
    'static_task',
]
