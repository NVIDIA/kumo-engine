# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from kumo_core.encoders.tensor_mapper import (
    CategoricalTensorMapper,
    NumericalTensorMapper,
    TimestampTensorMapper,
)
from kumo_core.features.stores.feature import FeatureStoreAttr
from kumo_core.features.stores.in_memory_feature import InMemoryFeatureStore
from torch_frame import TensorFrame


def test_in_memory_feature_engine():
    user_df = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3],
        'AGE': [20, 30, 40, 50],
        'GENDER': ['male', 'female', 'female', 'female'],
    })
    order_df = pd.DataFrame({
        'USER_ID': [0, 0, 1, 1, 1, 3],
        'AMOUNT': [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
        'TIME': ['1990', '1991', '1992', '1991', '1990', '1990'],
    })
    train_table_df = pd.DataFrame({
        'USER_ID': [0, 1],
        'TARGET': [True, False],
        'TIME': ['1993', '1993'],
        'TRAIN_MASK': [True, True],
        'VAL_MASK': [True, True],
        'TEST_MASK': [True, True],
    })

    feature_store = InMemoryFeatureStore(
        df_dict={
            'USERS': user_df,
            'ORDERS': order_df,
            'TRAIN_TABLE': train_table_df,
        },
        table_schema_dict={
            'USERS':
            dict(
                mapper_dict={
                    'AGE': NumericalTensorMapper(),
                    'GENDER': CategoricalTensorMapper([['female', 3],
                                                       ['male', 1]]),
                },
                pkey_name='USER_ID',
            ),
            'ORDERS':
            dict(
                mapper_dict={
                    'AMOUNT': NumericalTensorMapper(),
                    'TIME': TimestampTensorMapper(),
                },
                time_col_name='TIME',
            ),
        },
        train_table_schema=dict(
            table_name='TRAIN_TABLE',
            fkeys=[('USER_ID', 'USERS')],
            train_mask_name='TRAIN_MASK',
            val_mask_name='VAL_MASK',
            test_mask_name='TEST_MASK',
            target_mapper=('TARGET',
                           CategoricalTensorMapper([[False, 1], [True, 1]])),
            time_col_name='TIME',
        ),
    )

    attrs = feature_store.get_all_tensor_attrs()
    assert len(attrs) == 8
    assert FeatureStoreAttr('USERS', 'tf') in attrs
    assert FeatureStoreAttr('ORDERS', 'tf') in attrs
    assert FeatureStoreAttr('ORDERS', 'time') in attrs
    assert FeatureStoreAttr('TRAIN_TABLE', 'tf') in attrs
    assert FeatureStoreAttr('TRAIN_TABLE', 'time') in attrs
    assert FeatureStoreAttr('TRAIN_TABLE', 'y') in attrs

    assert feature_store.has_train_table_time_col

    tf = feature_store['USERS', 'tf', 0:4]
    assert isinstance(tf, TensorFrame)
    assert tf.num_rows == 4
    assert tf.feat_dict['numerical'].size() == (4, 1)
    assert tf.feat_dict['categorical'].size() == (4, 1)

    time = feature_store['ORDERS', 'time', 0:6]
    assert time.size() == (6, )

    target = feature_store['TRAIN_TABLE', 'y', :]
    assert target.size() == (2, 1)
