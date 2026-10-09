# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import uuid
from typing import Callable, Generator, Tuple

import pandas as pd
import pytest

from kumo.testing.rest import writable_snowflake_config
from kumo.testing.snowflake import SnowflakeTableConnector


@pytest.fixture()
def writable_snowflake_table(
        snowflake_credentials
) -> Generator[SnowflakeTableConnector, None, None]:
    name = f'TEST_PREDICTION_{str(uuid.uuid1()).upper().replace("-", "")}'
    connector = writable_snowflake_config('snowflake').as_connector('xva19026')
    try:
        yield SnowflakeTableConnector(name, connector)
    finally:
        connector.drop_table_if_exists(name)


@pytest.fixture()
def synthetic_data_for_weighted_tt() -> Callable:
    """
    Return a function that returns a tuple of two DataFrames, one for the
    original data and one for the weighted data.

    The function takes an optional issue parameter that can be
    one of the following:
    - 'na_weight': missing weight values
    - 'missing_col': missing entity column in the weighted data
    - 'mismatched_dtype': mismatched entity column dtype in the weighted data
    - 'all_negative_weight': all weights are negative
    - 'some_negative_weight': some weights are negative
    - 'array_target_weight_has_negative_values': array target has negative values
    - 'array_target_weight_no_issues': array target with array weight has no issues
    - None: no issue
    """
    def _synthetic_data_for_weighted_tt(
            issue: str | None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assert issue in [
            'na_weight', 'missing_col', 'mismatched_dtype',
            'all_negative_weight', 'some_negative_weight', None,
            'array_target_weight_has_negative_values',
            'array_target_weight_no_issues'
        ]
        orig_df = pd.DataFrame({
            'TARGET': [1, 2, 3],
            'TIMESTAMP': [
                datetime.datetime(2022, 6, 30, 0, 0, 0),
                datetime.datetime(2022, 6, 30, 0, 0, 0),
                datetime.datetime(2022, 6, 30, 0, 0, 0)
            ],
            'ENTITY': [1, 2, 3]
        })
        weight_df = pd.DataFrame({
            'TARGET': [1, 2, 3],
            'TIMESTAMP': [
                datetime.datetime(2022, 6, 30, 0, 0, 0),
                datetime.datetime(2022, 6, 30, 0, 0, 0),
                datetime.datetime(2022, 6, 30, 0, 0, 0)
            ],
            'ENTITY': [1, 2, 3],
            'WEIGHT': [1, 2, 3]
        })
        if issue == 'na_weight':
            weight_df['WEIGHT'] = [1, None, 3]
        if issue == 'missing_col':
            weight_df = weight_df.drop(columns=['ENTITY'])
        if issue == 'mismatched_dtype':
            weight_df['ENTITY'] = weight_df['ENTITY'].astype(str)
        if issue == 'all_negative_weight':
            weight_df['WEIGHT'] = [-1, -2, -3]
        if issue == 'some_negative_weight':
            weight_df['WEIGHT'] = [1, -2, 3]
        if issue in [
                'array_target_weight_has_negative_values',
                'array_target_weight_no_issues'
        ]:
            orig_df['TARGET'] = weight_df['TARGET'] = [[1, 2, 3], [4, 5, 6],
                                                       [7, 8, 9]]
            weight_df['WEIGHT'] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            if issue == 'array_target_weight_has_negative_values':
                weight_df['WEIGHT'] = [[1, 2, -3], [4, 5, 6], [7, 8, 9]]
        return orig_df, weight_df

    return _synthetic_data_for_weighted_tt
