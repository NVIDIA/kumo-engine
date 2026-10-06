# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import decimal
import os

import pandas as pd
import pytest
from kumo_core.types import TaskType
from kumoapi.typing import Dtype

from io_utils import fsutil
from kumo.pquery.utils import validate_weight_column
from kumo.validation import ValidationException


@pytest.fixture
def weight_test_data(tempdir):
    # Create a parquet table with multiple columns for testing
    parquet_dir = os.path.join(tempdir.name, 'weight_test')
    fsutil.makedirs(parquet_dir, exist_ok=True)
    parquet_path = os.path.join(parquet_dir, 'weight_test.parquet')
    df = pd.DataFrame({
        'weight_float': [1.0, 2.0, 3.0],
        'weight_int': [1, 2, 3],
        'weight_list': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        'weight_str': ['1.0', '2.0', '3.0'],
        'weight_decimal': [
            decimal.Decimal('1.0'),
            decimal.Decimal('2.0'),
            decimal.Decimal('3.0')
        ],
    })
    df.to_parquet(parquet_path)
    return parquet_dir


@pytest.mark.parametrize(
    "weight_column, task_type, is_array_target, expected_name, expected_dtype",
    [
        ('weight_decimal', TaskType.REGRESSION, False, 'weight_decimal',
         Dtype.float),
        ('weight_float', TaskType.REGRESSION, False, 'weight_float',
         Dtype.float),
        ('weight_int', TaskType.BINARY_CLASSIFICATION, False, 'weight_int',
         Dtype.float),
        ('weight_list', TaskType.MULTILABEL_RANKING, True, 'weight_list',
         Dtype.floatlist),
        ('weight_float', TaskType.MULTILABEL_RANKING, False, 'weight_float',
         Dtype.float),
        ('weight_list', TaskType.MULTILABEL_CLASSIFICATION, True, 'weight_list',
         Dtype.floatlist),
        ('weight_list', TaskType.STATIC_LINK_PREDICTION, True, 'weight_list',
         Dtype.floatlist),
    ])
def test_validate_weight_column_success(weight_test_data, weight_column,
                                        task_type, is_array_target,
                                        expected_name, expected_dtype):
    weight_col = validate_weight_column(weight_column, weight_test_data,
                                        task_type, is_array_target)
    assert weight_col.name == expected_name
    assert weight_col.dtype == expected_dtype


@pytest.mark.parametrize(
    "weight_column, task_type, is_array_target, error_pattern",
    [('nonexistent_col', TaskType.REGRESSION, False,
      "Weight column 'nonexistent_col' not found in training table"),
     ('weight_str', TaskType.REGRESSION, False,
      "Weight column 'weight_str' must be a numerical type"),
     ('weight_float', TaskType.MULTILABEL_RANKING, True,
      "Weight column 'weight_float' must be an array type"),
     ("weight_list", TaskType.LINK_PREDICTION, False,
      "Weight column 'weight_list' must be a numerical type")])
def test_validate_weight_column_errors(weight_test_data, weight_column,
                                       task_type, is_array_target,
                                       error_pattern):
    with pytest.raises(ValidationException, match=error_pattern):
        validate_weight_column(weight_column, weight_test_data, task_type,
                               is_array_target)


def test_get_train_table_schema_routes_s3_through_arrow_cloud_filesystem(
        monkeypatch):
    # The S3 / GCS branch builds a ParquetDataset from the credentialed choke
    # point with a scheme-stripped path.
    from kumo.pquery import utils as pquery_utils

    seen = {}
    sentinel_fs = object()

    def fake_acf(path):
        seen['fs_path'] = path
        return sentinel_fs

    class _FakeDataset:
        def __init__(self, path_or_paths, filesystem=None):
            seen['path'] = path_or_paths
            seen['filesystem'] = filesystem
            self.schema = 'schema-sentinel'

    monkeypatch.setattr(
        pquery_utils.fsutil,
        'listdir',
        lambda p, suffix=None: ['s3://bucket/tt/part-0.parquet'],
    )
    monkeypatch.setattr(pquery_utils.fsutil, 'isdir', lambda _p: False)
    monkeypatch.setattr(pquery_utils, 'arrow_cloud_filesystem', fake_acf)
    monkeypatch.setattr(pquery_utils.pq, 'ParquetDataset', _FakeDataset)

    schema = pquery_utils.get_train_table_schema('s3://bucket/tt')

    assert schema == 'schema-sentinel'
    assert seen['fs_path'] == 's3://bucket/tt/part-0.parquet'
    assert seen['path'] == 'bucket/tt/part-0.parquet'
    assert seen['filesystem'] is sentinel_fs


def test_get_train_table_schema_accepts_single_s3_parquet(monkeypatch):
    from kumo.pquery import utils as pquery_utils

    seen = {}
    sentinel_fs = object()

    def fail_listdir(*_args, **_kwargs):
        raise AssertionError('single parquet files must not be listed')

    class _FakeDataset:
        def __init__(self, path_or_paths, filesystem=None):
            seen['path'] = path_or_paths
            seen['filesystem'] = filesystem
            self.schema = 'schema-sentinel'

    monkeypatch.setattr(pquery_utils.fsutil, 'listdir', fail_listdir)
    monkeypatch.setattr(pquery_utils.fsutil, 'isdir', lambda _p: False)
    monkeypatch.setattr(pquery_utils, 'arrow_cloud_filesystem',
                        lambda _p: sentinel_fs)
    monkeypatch.setattr(pquery_utils.pq, 'ParquetDataset', _FakeDataset)

    schema = pquery_utils.get_train_table_schema(
        's3://bucket/immutable/tt.parquet')

    assert schema == 'schema-sentinel'
    assert seen['path'] == 'bucket/immutable/tt.parquet'
    assert seen['filesystem'] is sentinel_fs


def test_get_train_table_schema_accepts_single_local_parquet(tmp_path):
    parquet_path = tmp_path / 'tt.parquet'
    pd.DataFrame({'weight': [1.0]}).to_parquet(parquet_path)

    from kumo.pquery import utils as pquery_utils

    schema = pquery_utils.get_train_table_schema(str(parquet_path))

    assert schema.names == ['weight']


def test_get_train_table_schema_ignores_nested_bundle_directories(tmp_path):
    from kumo.pquery import utils as pquery_utils

    seed_dir = tmp_path / '_test_seeds'
    seed_dir.mkdir()
    pd.DataFrame({'seed_only': [1]}).to_parquet(seed_dir / 'part-00000.parquet')
    pd.DataFrame({'weight': [1.0]}).to_parquet(tmp_path / 'tt.parquet')

    schema = pquery_utils.get_train_table_schema(str(tmp_path))

    assert schema.names == ['weight']
