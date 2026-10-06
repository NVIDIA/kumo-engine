# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import pathlib

import pandas as pd
import pytest

from kumo.validation import ValidationException
from rest.utils.table import validate_csv_file, validate_parquet_file


def test_validate_parquet_file(tmp_path: pathlib.Path):
    temp_path = tmp_path.resolve().as_posix()
    curr_dir = osp.dirname(osp.realpath(__file__))
    # A Python code file shouldn't be a valid parquet file.
    with pytest.raises(ValidationException):
        validate_parquet_file(osp.join(curr_dir, "test_table.py"))
    # Parquet file but with unsupported data types
    data = [{'col1': 1, 'col2': 'a', 'col3': {'a': 'f'}}]

    df = pd.DataFrame.from_records(data)
    invalid_dtypes_path = osp.join(temp_path, "invalid_dtypes.parquet")
    df.to_parquet(invalid_dtypes_path)
    with pytest.raises(ValidationException):
        validate_parquet_file(invalid_dtypes_path)

    # Parquet file with valid data types
    valid_dtypes_path = osp.join(temp_path, "valid_dtypes.parquet")
    df = pd.DataFrame.from_records([
        {
            'col1': 1,
            'col2': 'a'
        },
        {
            'col1': 2,
            'col2': 'b'
        },
    ])
    df.to_parquet(valid_dtypes_path)
    # Validation should not raise an error:
    validate_parquet_file(valid_dtypes_path)

    # Empty parquet file should throw an error:
    empty_parquet_path = osp.join(temp_path, "empty.parquet")
    df = pd.DataFrame()
    df.to_parquet(empty_parquet_path)
    with pytest.raises(ValidationException):
        validate_parquet_file(empty_parquet_path)


_MLFORMED_DATA = '''"this","is",123,"a","normal","line"
"line","with "an" internal","quote"
"short line","with
an
"internal quote", 1 comma and
linebreaks"
un "quot" ed,text,with,1,2,3,numbers
"quoted","number","series","1,2,3"
"invalid \xAB utf-8"
'''


def test_validate_csv_file(tmp_path: pathlib.Path):
    temp_path = tmp_path.resolve().as_posix()
    # CSV without a valid separator should fail:
    invalid_sep_path = osp.join(temp_path, "invalid_sep.csv")
    with open(invalid_sep_path, "w") as f:
        f.write(_MLFORMED_DATA)
    with pytest.raises(ValidationException):
        validate_csv_file(invalid_sep_path)

    header_only_path = osp.join(temp_path, "header_only.csv")
    with open(header_only_path, "w") as f:
        f.write("col1,col2\n")
    with pytest.raises(ValidationException):
        validate_csv_file(header_only_path)


def test_validate_single_column_csv_file(tmp_path: pathlib.Path):
    temp_path = tmp_path.resolve().as_posix()
    # CSV without a valid separator should fail:
    single_column_path = osp.join(temp_path, "single_column.csv")
    with open(single_column_path, "w") as f:
        f.write("pkey\n1\n2\n3\n4")

    validate_csv_file(single_column_path)
