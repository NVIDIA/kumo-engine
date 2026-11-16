# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from io_utils.pyarrow import (
    DecimalOutOfBoundsError,
    fix_types_for_pandas,
    safe_to_pandas,
)


def test_safe_to_pandas():
    # Create sample data with various timestamp scenarios
    data = {
        'valid_timestamp': [
            pd.Timestamp('2025-03-24 12:00:00'),
            pd.Timestamp('2025-03-25 13:00:00')
        ],
        'out_of_bounds_timestamp':
        ['2263-01-01 00:00:00', '9999-12-31 23:59:59'],
        'non_timestamp': ['sample text', 'another text']
    }

    df = pd.DataFrame(data)
    # Cast into timestamp column with out of bounds timestamps
    table = pa.Table.from_pandas(df)
    table = table.set_column(
        1, 'out_of_bounds_timestamp',
        pc.strptime(table.column("out_of_bounds_timestamp"),
                    format='%Y-%m-%d %H:%M:%S', unit='us'))

    # Convert the PyArrow Table back to pandas DataFrame using the function
    result_df = safe_to_pandas(table)

    # Check that the DataFrame is not None
    assert result_df is not None

    # Verify that the valid timestamps remain unchanged
    assert result_df.loc[0, 'valid_timestamp'] == pd.Timestamp(
        '2025-03-24 12:00:00')
    assert result_df.loc[1, 'valid_timestamp'] == pd.Timestamp(
        '2025-03-25 13:00:00')

    # Verify that out-of-bounds timestamps are converted to NaT
    assert pd.isna(result_df.loc[0, 'out_of_bounds_timestamp'])
    assert pd.isna(result_df.loc[1, 'out_of_bounds_timestamp'])

    # Verify that non-timestamp columns remain unchanged
    assert result_df.loc[0, 'non_timestamp'] == 'sample text'
    assert result_df.loc[1, 'non_timestamp'] == 'another text'


def test_fix_types_for_pandas_reports_decimal_int64_overflow_safely():
    table = pa.table({
        'wide_decimal':
        pa.array([Decimal('9223372036854775808')],
                 type=pa.decimal128(38, 0))
    })

    with pytest.raises(
            DecimalOutOfBoundsError,
            match="wide_decimal.*outside the signed 64-bit integer range"):
        fix_types_for_pandas(table)
