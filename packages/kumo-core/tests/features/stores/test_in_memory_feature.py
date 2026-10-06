# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
import math
from collections import defaultdict
from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from kumoapi.encoder import Index, Numerical
from kumoapi.typing import Stype
from torch import Tensor
from torch_frame import TensorFrame

from kumo_core.encoders.config import EncodersConfig, TargetEncoderConfig
from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper, NumericalTensorMapper
from kumo_core.features.stores.feature import FeatureStoreAttr, FeatureStoreTableSchema, FeatureStoreTrainingTableSchema
from kumo_core.features.stores.in_memory_feature import InMemoryFeatureStore
from kumo_core.types.features import FeatureStoreData


@pytest.fixture
def df_dict() -> dict[str, pd.DataFrame]:
    return {
        'USER':
        pd.DataFrame({
            'USER_ID': [0, 10, 20, 30],
            'AGE': [20, 30, 40, 50],
            'GENDER': ['male', 'female', 'female', 'female'],
            'TIME': ['1990', '1990', '1991', '1991'],
            'END_TIME': ['1998', '1998', '1999', ''],
        }),
        'ARTICLE':
        pd.DataFrame({'ARTICLE_ID': [0, 10]}),
        'TRAIN_TABLE':
        pd.DataFrame({
            'ENTITY': [0, 10, 20, 30] * 3,
            'TRAIN_MASK': [True] * 4 + [False] * 4 + [False] * 4,
            'VAL_MASK': [False] * 4 + [True] * 4 + [False] * 4,
            'TEST_MASK': [False] * 4 + [False] * 4 + [True] * 4,
            'AGE': [20, 30, 40, 50] * 3,
            'TARGET': [True, False] * 5 + [False] * 2,
            'TIME': ['1993'] * 4 + ['1994'] * 4 + ['1995'] * 4,
            'WEIGHT': [0.1, 0.2, 0.3, 0.4] * 3,
        }),
    }


def assert_feature_store(feature_store: InMemoryFeatureStore) -> None:
    assert str(feature_store) == 'InMemoryFeatureStore(num_tables=2)'

    assert feature_store.table_names == ['USER', 'ARTICLE']
    assert feature_store.train_table_name == 'TRAIN_TABLE'
    assert feature_store.entity_table_names == ['USER']
    assert feature_store.train_fkey_names == ['ENTITY']
    assert feature_store.time_col_dict == {'USER': 'TIME'}
    assert feature_store.has_train_table_time_col
    assert feature_store.has_train_table_test_mask
    assert feature_store.has_train_table_target_col
    assert feature_store.has_train_table_weight_col
    assert feature_store.get_pkey_name('USER') == 'USER_ID'
    assert feature_store.get_pkey_name('ARTICLE') == 'ARTICLE_ID'
    with pytest.raises(KeyError):
        assert feature_store.get_pkey_name('TRAIN_TABLE')
    assert feature_store.get_time_col_name('USER') == 'TIME'
    assert feature_store.get_time_col_name('TRAIN_TABLE') == 'TIME'
    assert feature_store.get_end_time_col_name('USER') == 'END_TIME'
    assert feature_store.get_end_time_col_name('TRAIN_TABLE') is None

    assert feature_store.get_tensor_size('USER') == (4, )
    assert feature_store.get_tensor_size('ARTICLE') == (2, )
    assert feature_store.get_tensor_size('TRAIN_TABLE') == (12, )

    assert isinstance(feature_store.get_tensor_mapper('USER', 'AGE'),
                      NumericalTensorMapper)
    assert isinstance(feature_store.get_tensor_mapper('TRAIN_TABLE', 'AGE'),
                      NumericalTensorMapper)
    assert isinstance(feature_store.get_target_tensor_mapper(),
                      CategoricalTensorMapper)

    attrs = feature_store.get_all_tensor_attrs()
    assert len(attrs) == 10
    assert attrs[0] == FeatureStoreAttr('USER', 'tf')
    assert attrs[1] == FeatureStoreAttr('ARTICLE', 'tf')
    assert attrs[2] == FeatureStoreAttr('USER', 'pkey')
    assert attrs[3] == FeatureStoreAttr('ARTICLE', 'pkey')
    assert attrs[4] == FeatureStoreAttr('USER', 'time')
    assert attrs[5] == FeatureStoreAttr('TRAIN_TABLE', 'tf')
    assert attrs[6] == FeatureStoreAttr('TRAIN_TABLE', 'fkey')
    assert attrs[7] == FeatureStoreAttr('TRAIN_TABLE', 'time')
    assert attrs[8] == FeatureStoreAttr('TRAIN_TABLE', 'y')
    assert attrs[9] == FeatureStoreAttr('TRAIN_TABLE', 'weight')

    with pytest.raises(KeyError, match="is not fully specified"):
        feature_store.get_tensor('TRAIN_TABLE', 'y')
    with pytest.raises(KeyError, match="does not exist"):
        feature_store['USER', 'y', 0:4]
    with pytest.raises(KeyError, match="does not exist"):
        feature_store['TRAIN_TABLE', 'x', :]
    with pytest.raises(KeyError, match="does not exist"):
        feature_store['ITEM', 'x', 0:4]
    with pytest.raises(KeyError, match="only fetch a mini-batch"):
        feature_store['USER', 'tf', :]

    tf = feature_store['USER', 'tf', 0:4]
    assert isinstance(tf, TensorFrame)
    assert tf.num_rows == 4
    assert tf.num_cols == 2
    assert tf.col_names_dict == {
        'numerical': ['AGE'],
        'categorical': ['GENDER'],
    }
    assert tf.feat_dict['numerical'].allclose(
        torch.tensor([[20.0], [30.0], [40.0], [50.0]]))
    # 0 and 1 can be flipped as it's a binary label tensor:
    assert tf.feat_dict['categorical'].equal(torch.tensor(
        [[0], [1], [1], [1]])) ^ tf.feat_dict['categorical'].equal(
            torch.tensor([[1], [0], [0], [0]]))
    assert isinstance(feature_store['USER', 'time', 0:4], Tensor)
    assert isinstance(feature_store['USER', 'end_time', 0:4], Tensor)
    assert isinstance(feature_store['USER', 'pkey', 0:4], np.ndarray)
    assert feature_store['USER', 'pkey', 0:4].tolist() == [0, 10, 20, 30]
    node_id = feature_store.get_node_id('USER', np.array([20, 10]))
    assert node_id.equal(torch.tensor([2, 1]))

    tf = feature_store['ARTICLE', 'tf', 0:2]
    assert isinstance(tf, TensorFrame)
    assert tf.num_rows == 2
    assert tf.num_cols == 0
    assert tf.col_names_dict == {}
    assert tf.feat_dict == {}

    tf = feature_store['TRAIN_TABLE', 'tf', :]
    assert isinstance(tf, TensorFrame)
    assert tf.num_rows == 12
    assert tf.num_cols == 1
    assert tf.col_names_dict == {'numerical': ['AGE']}
    assert tf.feat_dict['numerical'].allclose(
        torch.tensor([[20.0], [30.0], [40.0], [50.0]] * 3))
    assert feature_store['TRAIN_TABLE', 'fkey', :].equal(
        torch.tensor([[0], [1], [2], [3]] * 3))
    assert feature_store['TRAIN_TABLE', 'train_mask', :].equal(
        torch.tensor([True] * 4 + [False] * 4 + [False] * 4))
    assert feature_store['TRAIN_TABLE', 'val_mask', :].equal(
        torch.tensor([False] * 4 + [True] * 4 + [False] * 4))
    assert feature_store['TRAIN_TABLE', 'test_mask', :].equal(
        torch.tensor([False] * 4 + [False] * 4 + [True] * 4))
    assert feature_store['TRAIN_TABLE', 'y', :].equal(  #
        torch.tensor([[True], [False]] * 5 + [[False]] * 2))
    assert isinstance(feature_store['TRAIN_TABLE', 'time', :], Tensor)
    assert feature_store['TRAIN_TABLE', 'weight', :].allclose(
        torch.tensor([0.1, 0.2, 0.3, 0.4] * 3))

    df = feature_store['USER', 'df', 2:4]
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame({
            'AGE': [40, 50],
            'GENDER': ['female', 'female'],
            'TIME': [pd.Timestamp('1991'),
                     pd.Timestamp('1991')],
            'END_TIME': [pd.Timestamp('1999'), pd.NaT],
        }), check_dtype=False)

    df = feature_store['ARTICLE', 'df', 0:2]
    assert df.shape == (2, 0)

    train_mask, val_mask = feature_store.multi_get_tensor([
        FeatureStoreAttr('TRAIN_TABLE', 'train_mask', None),
        FeatureStoreAttr('TRAIN_TABLE', 'val_mask', None),
    ])
    assert feature_store['TRAIN_TABLE', 'train_mask', :].equal(train_mask)
    assert feature_store['TRAIN_TABLE', 'val_mask', :].equal(val_mask)

    pred_table = pd.DataFrame({
        'ENTITY': [0, 1, 2, 3] * 3,
        'AGE': [120, 130, 140, 150] * 3,
        'TIME': ['1996'] * 12,
        'WEIGHT': [0.1, 0.2, 0.3, 0.4] * 3,
    })
    feature_store.set_train_table(pred_table)
    tf = feature_store['TRAIN_TABLE', 'tf', :]
    assert isinstance(tf, TensorFrame)
    assert tf.num_rows == 12
    assert tf.num_cols == 1
    assert tf.col_names_dict == {'numerical': ['AGE']}
    assert tf.feat_dict['numerical'].allclose(
        torch.tensor([[120.0], [130.0], [140.0], [150.0]] * 3))


def create_feature_store(
    df_dict: dict[str, pd.DataFrame],
    pmm: bool = True,
) -> InMemoryFeatureStore:
    table_schema_dict: dict[str, FeatureStoreTableSchema | dict[str, Any]] = {}
    table_schema_dict['USER'] = FeatureStoreTableSchema(
        mapper_dict={
            'AGE': NumericalTensorMapper(),
            'GENDER': CategoricalTensorMapper([['male', 1], ['female', 1]]),
        },
        pkey_name='USER_ID',
        time_col_name='TIME',
        end_time_col_name='END_TIME',
    )
    table_schema_dict['ARTICLE'] = FeatureStoreTableSchema(
        mapper_dict={},
        pkey_name='ARTICLE_ID',
    )
    train_table_schema = FeatureStoreTrainingTableSchema(
        table_name='TRAIN_TABLE',
        fkeys=[('ENTITY', 'USER')],
        train_mask_name='TRAIN_MASK',
        val_mask_name='VAL_MASK',
        test_mask_name='TEST_MASK',
        mapper_dict={'AGE': NumericalTensorMapper()},
        target_mapper=('TARGET',
                       CategoricalTensorMapper([[False, 1], [True, 1]])),
        time_col_name='TIME',
        weight_col_name='WEIGHT',
    )
    return InMemoryFeatureStore(
        df_dict=df_dict,
        table_schema_dict=table_schema_dict,
        train_table_schema=train_table_schema,
        pmm=pmm,
    )


@pytest.mark.parametrize('pmm', [True, False])
def test_in_memory_feature_store(
    df_dict: dict[str, pd.DataFrame],
    pmm: bool,
) -> None:
    feature_store = create_feature_store(df_dict, pmm)
    assert_feature_store(feature_store)


@pytest.mark.parametrize('pmm', [True, False])
def test_from_encoders(df_dict: dict[str, pd.DataFrame], pmm: bool) -> None:
    feature_store = InMemoryFeatureStore.from_encoders(
        df_dict=df_dict,
        encoders=EncodersConfig(
            encoders={
                'USER': {
                    'AGE': Numerical(),
                    'GENDER': Index(),
                },
                'ARTICLE': {},
                'TRAIN_TABLE': {
                    'AGE': Numerical()
                },
            },
            target=TargetEncoderConfig(
                col_name='TARGET',
                encoder=Index(),
            ),
        ),
        pkey_name_dict={
            'USER': 'USER_ID',
            'ARTICLE': 'ARTICLE_ID',
        },
        time_col_dict={
            'TRAIN_TABLE': 'TIME',
            'USER': 'TIME',
        },
        end_time_col_dict={
            'USER': 'END_TIME',
        },
        train_table_name='TRAIN_TABLE',
        train_fkeys=[('ENTITY', 'USER')],
        train_mask_names=('TRAIN_MASK', 'VAL_MASK', 'TEST_MASK'),
        train_weight_col_name='WEIGHT',
        pmm=pmm,
    )
    assert_feature_store(feature_store)


def patch_get_tensor(fs: InMemoryFeatureStore) -> AbstractContextManager:
    return patch.object(
        fs,
        '_get_tensor',
        side_effect=fs._get_tensor,
    )


@pytest.mark.parametrize('safe', [False, True])
def test_batch_mode(safe: bool, df_dict: dict[str, pd.DataFrame]) -> None:
    fs = create_feature_store(df_dict)
    fn = fs.safe_batch_mode if safe else fs.batch_mode
    assert fs._batch_size is None

    num_rows = len(df_dict['USER'])

    # Test fetching a tensor in chunks
    with patch_get_tensor(fs) as mocked_method:
        expected = fs['USER', 'time', :]
    assert mocked_method.call_count == 1

    with patch_get_tensor(fs) as mocked_method, fn(3):
        out = fs['USER', 'time', :]
    assert mocked_method.call_count == math.ceil(num_rows / 3)
    assert torch.equal(out, expected)

    # Test fetching a tensor frame in chunks
    with patch_get_tensor(fs) as mocked_method:
        expected = fs['USER', 'tf', torch.arange(num_rows)]
    assert mocked_method.call_count == 1

    with patch_get_tensor(fs) as mocked_method, fn(3):
        # 1st chunk: [0, 1, 2]
        # 2nd chunk: [3]
        out = fs['USER', 'tf', torch.arange(num_rows)]
    assert mocked_method.call_count == math.ceil(num_rows / 3)
    assert out == expected

    # Test fetching a ndarray in chunks
    with patch_get_tensor(fs) as mocked_method:
        expected = fs['USER', 'pkey', :]
    assert mocked_method.call_count == 1

    with patch_get_tensor(fs) as mocked_method, fn(3):
        out = fs['USER', 'pkey', :]
    assert mocked_method.call_count == math.ceil(num_rows / 3)
    assert np.array_equal(out, expected)

    for s in [
            slice(None, None, None),  # [:]
            slice(0, num_rows, None),  # [0:num_rows]
            slice(None, num_rows, None)  # [:num_rows]
    ]:
        with patch_get_tensor(fs) as mocked_method:
            expected = fs['USER', 'time', s]
        assert mocked_method.call_count == 1

        with patch_get_tensor(fs) as mocked_method, fn(3):
            out = fs['USER', 'time', s]
        assert mocked_method.call_count == math.ceil(num_rows / 3)
        assert torch.equal(out, expected)


def test_safe_batch_mode(df_dict: dict[str, pd.DataFrame]) -> None:
    fs = create_feature_store(df_dict)

    def side_effect(attr: FeatureStoreAttr) -> FeatureStoreData | None:
        if fs._batch_size == 4:
            raise RuntimeError("Something unexpected happened!")
        return np.array([0, 10, 20, 30])[attr.index]

    assert fs._batch_size is None
    assert fs._min_batch_size is None

    with patch.object(
            InMemoryFeatureStore,
            '_get_tensor',
            side_effect=side_effect,
    ):
        with fs.safe_batch_mode(batch_size=4, min_batch_size=2):
            assert fs._batch_size == 4
            assert fs._min_batch_size == 2
            out = fs['USER', 'pkey', :]
            assert fs._batch_size == 2
            assert fs._min_batch_size == 2

    assert fs._batch_size is None
    assert fs._min_batch_size is None

    assert np.array_equal(out, fs['USER', 'pkey', :])


def test_sanitize() -> None:
    df_dict = {}
    df_dict['USER'] = pd.DataFrame({
        'USER_ID':
        pd.Series([0, 1, 1, None, 2], dtype='Int64'),
    })
    df_dict['FACT'] = pd.DataFrame({
        'ID':
        pd.Series([0, 1, 2, 3, None], dtype='Int64'),
        'USER_ID':
        pd.Series([0, 1, None, 2, 3], dtype='Int64'),
        'TIME':
        pd.Series(['1990', None, '1991', None, '1992'],
                  dtype='datetime64[ns]'),
    })

    df_dict = InMemoryFeatureStore.sanitize(
        df_dict=df_dict,
        pkey_name_dict={
            'USER': 'USER_ID',
            'FACT': 'ID',
        },
        time_col_dict={'FACT': 'TIME'},
    )

    pd.testing.assert_frame_equal(
        df_dict['USER'].reset_index(drop=True),
        pd.DataFrame({
            'USER_ID': pd.Series([0, 1, 2], dtype='Int64'),
        }),
    )

    pd.testing.assert_frame_equal(
        df_dict['FACT'].reset_index(drop=True),
        pd.DataFrame({
            'ID':
            pd.Series([0, 2], dtype='Int64'),
            'USER_ID':
            pd.Series([0, None], dtype='Int64'),
            'TIME':
            pd.Series(['1990', '1991'], dtype='datetime64[ns]'),
        }),
    )


def test_all_dtypes() -> None:
    df_dict = {}
    df_dict['test'] = pd.DataFrame({
        'bool': [True, False, True],
        'int': [1, 2, 3],
        'byte':
        np.array([1, 2, 3], dtype='int8'),
        'int16':
        np.array([1, 2, 3], dtype='int16'),
        'int32':
        np.array([1, 2, 3], dtype='int32'),
        'int64':
        np.array([1, 2, 3], dtype='int64'),
        'float': [1.1, 2.2, 3.3],
        'float32':
        np.array([1.1, 2.2, 3.3], dtype='float32'),
        'float64':
        np.array([1.1, 2.2, 3.3], dtype='float64'),
        'string': ['foo', 'bar', 'baz'],
        'binary': [b'abc', b'def', b'ghi'],
        'image': ['foo', 'bar', 'baz'],
        'date':
        pd.to_datetime([
            '2023-01-01',
            '2023-01-02',
            '2023-01-03',
        ]).date,
        'time': [
            datetime.time(12, 0),
            datetime.time(13, 30),
            datetime.time(14, 45),
        ],
        'timedelta': [
            pd.Timedelta(days=1),
            pd.Timedelta(hours=2),
            None,
        ],
        'floatlist': [[1.1, 2.2], [3.3, 4.4], [5.5]],
        'intlist': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        'stringlist': [['a', 'b'], ['c'], ['d', 'e', 'f']],
        'unsupported': [3 + 4j, 3 + 4j, 3 + 4j],
    })

    stypes: dict[str, dict[str, Stype]] = defaultdict(dict)
    stypes['test']['bool'] = Stype.categorical
    stypes['test']['int'] = Stype.numerical
    stypes['test']['byte'] = Stype.categorical
    stypes['test']['int16'] = Stype.numerical
    stypes['test']['int32'] = Stype.categorical
    stypes['test']['int64'] = Stype.numerical
    stypes['test']['float'] = Stype.numerical
    stypes['test']['float32'] = Stype.categorical
    stypes['test']['float64'] = Stype.numerical
    stypes['test']['string'] = Stype.multicategorical
    stypes['test']['binary'] = Stype.categorical
    stypes['test']['image'] = Stype.image
    stypes['test']['date'] = Stype.timestamp
    stypes['test']['time'] = Stype.timestamp
    stypes['test']['timedelta'] = Stype.numerical
    stypes['test']['floatlist'] = Stype.multicategorical
    stypes['test']['intlist'] = Stype.sequence
    stypes['test']['stringlist'] = Stype.multicategorical
    stypes['test']['unsupported'] = Stype.unsupported

    InMemoryFeatureStore.from_encoders(
        df_dict,
        encoders=EncodersConfig.from_stypes(df_dict, stypes),
        pkey_name_dict={},
        time_col_dict={},
        end_time_col_dict={},
    )
