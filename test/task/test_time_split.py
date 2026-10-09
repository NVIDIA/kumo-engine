# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import pathlib
from typing import Any

import pandas as pd
import pytest
from kumo_core.train.mapping import NODE_INDEX_COLUMN
from kumoapi.typing import Dtype, TimeUnit

from io_utils import fsutil
from kumo.config.task import Mask
from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.column import ForeignKey, TimeColumn
from kumo.task.splits.parsing import get_split
from kumo.task.splits.time_split import TimeSplit
from kumo.task.training_table import TrainingTable
from kumo.testing.data import financial_parquet_connector


def _create_training_table() -> TrainingTable:
    r"""Creates a TrainingTable object based on the LOAN table in the
    financial dataset."""
    connector = financial_parquet_connector()
    fkey = ForeignKey(name='STATUS', dst_table='parquet.random_table',
                      dtype=Dtype.string)
    time_col = TimeColumn(name='DATE', dtype=Dtype.date)
    return TrainingTable(
        table_name="LOAN_TRAIN_SCHEMA",
        connector=connector,
        task_name='dummy_task',
        time_col=time_col,
        fkeys=[fkey],
    )


def _create_medium_training_table(tmp_path: pathlib.Path) -> TrainingTable:
    """Creates a medium TrainingTable object with fixed test data."""
    # Create test dataframe
    df = pd.DataFrame({
        'ENTITY': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'TIMESTAMP': [
            '1970-02-12', '1970-02-09', '1970-02-06', '1970-02-03',
            '1970-01-31', '1970-01-28', '1970-01-25', '1970-01-22',
            '1970-01-19', '1970-01-16', '1970-01-13', '1970-01-10',
            '1970-01-07', '1970-01-04', '1970-01-01'
        ],
        'TARGET': [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0],
        'DUMMY': [0] * 15  # Required for foreign key
    })
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df[NODE_INDEX_COLUMN] = range(df.shape[0])

    # Create parquet connector and persist data
    root_dir = osp.join(tmp_path.resolve().as_posix(), 'parquet')
    fsutil.makedirs(root_dir)
    connector = ParquetConnector(
        name='parquet',
        root_dir=root_dir,
    )
    df.to_parquet(osp.join(root_dir, 'MEDIUM_TABLE.parquet'))

    # Create training table
    fkey = ForeignKey(name='DUMMY', dst_table='parquet.random_table',
                      dtype=Dtype.string)
    time_col = TimeColumn(name='TIMESTAMP', dtype=Dtype.date)

    return TrainingTable(
        table_name='MEDIUM_TABLE',
        connector=connector,
        task_name='dummy_task',
        time_col=time_col,
        fkeys=[fkey],
    )


def _create_small_training_table(tmp_path: pathlib.Path) -> TrainingTable:
    """Creates a small TrainingTable object with fixed test data."""
    # Create test dataframe
    df = pd.DataFrame({
        'ENTITY': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'TIMESTAMP': ['1970-01-12', '1970-01-09', '1970-01-06'] * 5,
        'TARGET': [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0],
        'DUMMY': [0] * 15  # Required for foreign key
    })
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df[NODE_INDEX_COLUMN] = range(df.shape[0])

    # Create parquet connector and persist data
    root_dir = osp.join(tmp_path.resolve().as_posix(), 'parquet')
    fsutil.makedirs(root_dir)
    connector = ParquetConnector(
        name='parquet',
        root_dir=root_dir,
    )
    df.to_parquet(osp.join(root_dir, 'SMALL_TABLE.parquet'))

    # Create training table
    fkey = ForeignKey(name='DUMMY', dst_table='parquet.random_table',
                      dtype=Dtype.string)
    time_col = TimeColumn(name='TIMESTAMP', dtype=Dtype.date)

    return TrainingTable(
        table_name='SMALL_TABLE',
        connector=connector,
        task_name='dummy_task',
        time_col=time_col,
        fkeys=[fkey],
    )


def to_timestamp(boundary: Any) -> pd.Timestamp:
    ts = pd.Timestamp(boundary)
    assert isinstance(ts, pd.Timestamp)
    return ts


def to_timedelta(timeframe_size: int) -> pd.Timedelta:
    td = pd.Timedelta(days=timeframe_size)
    assert isinstance(td, pd.Timedelta)
    return td


def test_quotation_equivalence():
    base_split = TimeSplit(
        timestamp_boundaries=[
            ('2022-01-01', '2022-01-02'),
            ('2022-01-03', '2022-01-04'),
            ('2022-01-05', '2022-01-06'),
        ], )

    single_quote_split = get_split("TimeSplit("
                                   "timestamp_boundaries=["
                                   "('2022-01-01', '2022-01-02'), "
                                   "('2022-01-03', '2022-01-04'), "
                                   "('2022-01-05', '2022-01-06')"
                                   "])")
    double_quote_split = get_split('TimeSplit('
                                   'timestamp_boundaries=['
                                   '("2022-01-01", "2022-01-02"), '
                                   '("2022-01-03", "2022-01-04"), '
                                   '("2022-01-05", "2022-01-06")'
                                   '])')
    assert single_quote_split == double_quote_split
    assert base_split == single_quote_split
    assert base_split == double_quote_split


def test_list_tuple_equivalence():
    base_split = TimeSplit(timestamp_boundaries=[
        ('2022-01-01', '2022-01-02'),
        ('2022-01-03', '2022-01-04'),
        ('2022-01-05', '2022-01-06'),
    ])

    tuple_split = get_split("TimeSplit("
                            "timestamp_boundaries=["
                            "('2022-01-01', '2022-01-02'), "
                            "('2022-01-03', '2022-01-04'), "
                            "('2022-01-05', '2022-01-06')"
                            "])")
    list_split = get_split('TimeSplit('
                           'timestamp_boundaries=['
                           '["2022-01-01", "2022-01-02"], '
                           '["2022-01-03", "2022-01-04"], '
                           '["2022-01-05", "2022-01-06"]'
                           '])')
    assert tuple_split == list_split
    assert base_split == tuple_split
    assert base_split == list_split


def test_argument_exclusivity():
    # Make sure the right errors are raised
    with pytest.raises(ValueError, match="Must specify exactly one of"):
        TimeSplit()

    with pytest.raises(ValueError, match="Cannot specify multiple"):
        TimeSplit(
            timestamp_boundaries=[('2022-01-01', '2022-01-02')],
            fractions=[0.5, 0.3, 0.2],
        )

    with pytest.raises(ValueError, match="Cannot specify multiple"):
        TimeSplit(
            timestamp_boundaries=[('2022-01-01', '2022-01-02')],
            indices=[(-4, -2), (-2, 0)],
        )

    with pytest.raises(ValueError, match="Cannot specify multiple"):
        TimeSplit(
            fractions=[0.5, 0.5],
            indices=[(-4, -2), (-2, 0)],
        )

    # Make sure the allowed values are accepted
    TimeSplit(fractions=[0.1, 0.2, 0.7], )
    TimeSplit(
        indices=[(-366, -301), (-301, -201), (-201, -100)],
        unit="days",
        timeframe_size=7,
        anchor_start_offset=0,
    )
    TimeSplit(
        timestamp_boundaries=[
            ('2022-01-01', '2022-01-02'),
            ('2022-01-03', '2022-01-04'),
            ('2022-01-05', '2022-01-06'),
        ], )


def test_offset_units():
    with pytest.raises(
            ValueError,
            match="'years' is not a valid TimeUnit",
    ):
        TimeSplit(
            indices=[(-365, -300), (-300, -200), (-200, -100)],
            unit="years",  # type: ignore
        )

    with pytest.raises(
            ValueError,
            match="'day' is not a valid TimeUnit",
    ):
        TimeSplit(
            indices=[(-365, -301), (-300, -201), (-200, -101)],
            unit="day",  # type: ignore
        )

    # Make sure the allowed values are accepted
    TimeSplit(
        indices=[(-365, -301), (-300, -201), (-200, -101)],
        unit="days",
        timeframe_size=7,
        anchor_start_offset=0,
    )
    TimeSplit(
        indices=[(-365, -301), (-300, -201), (-200, -101)],
        unit="hours",
        timeframe_size=7,
        anchor_start_offset=0,
    )
    TimeSplit(
        indices=[(-365, -301), (-300, -201), (-200, -101)],
        unit="minutes",
        timeframe_size=7,
        anchor_start_offset=0,
    )
    TimeSplit(
        indices=[(-365, -301), (-300, -201), (-200, -101)],
        unit="months",
        timeframe_size=7,
        anchor_start_offset=0,
    )


def test_offset_ordering():
    # Make sure the right errors are raised
    with pytest.raises(
            ValueError,
            match=("The start .* of the \\(train\\) "
                   "split must be strictly smaller than the end .*"),
    ):
        TimeSplit(
            indices=[
                (-300, -365),
                (-250, -225),
                (-200, -100),
            ],
            unit="days",
            timeframe_size=7,
            anchor_start_offset=0,
        )

    with pytest.raises(
            ValueError,
            match=("The start .* of the \\(val\\) split must be "
                   "strictly smaller than the end .*"),
    ):
        TimeSplit(
            indices=[
                (-365, -300),
                (-225, -250),
                (-200, -100),
            ],
            unit="days",
            timeframe_size=7,
            anchor_start_offset=0,
        )

    with pytest.raises(
            ValueError,
            match=("The start .* of the \\(test\\) split must be "
                   "strictly smaller than the end .*"),
    ):
        TimeSplit(
            indices=[
                (-365, -300),
                (-250, -225),
                (-100, -200),
            ],
            unit="days",
            timeframe_size=7,
            anchor_start_offset=0,
        )

    # Make sure the allowed values are accepted
    TimeSplit(
        indices=[
            (-365, -300),
            (-250, -225),
            (-200, -100),
        ],
        unit="days",
        timeframe_size=7,
        anchor_start_offset=0,
    )


def test_boundary_ordering():
    with pytest.raises(
            ValueError,
            match=("The start .* of the \\(train\\) split must be "
                   "strictly smaller than the end .*"),
    ):
        TimeSplit(
            timestamp_boundaries=[
                ("2023-01-02", "2023-01-01"),
                ("2023-02-01", "2023-02-15"),
                ("2023-03-01", "2023-03-15"),
            ], )

    with pytest.raises(
            ValueError,
            match=("The start .* of the \\(val\\) split must be "
                   "strictly smaller than the end .*"),
    ):
        TimeSplit(
            timestamp_boundaries=[
                ("2023-01-01", "2023-01-15"),
                ("2023-02-15", "2023-02-01"),
                ("2023-03-01", "2023-03-15"),
            ], )

    with pytest.raises(
            ValueError,
            match=("The start .* of the \\(test\\) split must be "
                   "strictly smaller than the end .*"),
    ):
        TimeSplit(
            timestamp_boundaries=[
                ("2023-01-01", "2023-01-15"),
                ("2023-02-01", "2023-02-15"),
                ("2023-03-15", "2023-03-01"),
            ], )

    TimeSplit(
        timestamp_boundaries=[
            ("2023-01-01", "2023-01-15"),
            ("2023-02-01", "2023-02-15"),
            ("2023-03-01", "2023-03-15"),
        ], )


def test_fractions():
    """Test fraction-based splits validation."""
    # Valid cases
    TimeSplit(fractions=[0.6, 0.4])  # Two splits summing to 1
    TimeSplit(fractions=[0.7, 0.2, 0.1])  # Three splits summing to 1
    TimeSplit(fractions=[0.333, 0.333, 0.334])  # Close to equal thirds

    # Invalid number of splits
    with pytest.raises(ValueError, match="At least two splits are required"):
        TimeSplit(fractions=[1.0])

    with pytest.raises(ValueError,
                       match="No more than three splits is supported"):
        TimeSplit(fractions=[0.25, 0.25, 0.25, 0.25])

    # Invalid fraction values
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        TimeSplit(fractions=[-0.1, 1.1])

    with pytest.raises(ValueError, match="must be between 0 and 1"):
        TimeSplit(fractions=[1.5, 0.5])

    # Fractions not summing to 1
    with pytest.raises(ValueError, match="Fractions must sum to 1.0, got 0.8"):
        TimeSplit(fractions=[0.5, 0.3])

    with pytest.raises(ValueError, match="Fractions must sum to 1.0, got 1.2"):
        TimeSplit(fractions=[0.6, 0.4, 0.2])

    # Edge cases
    TimeSplit(fractions=[0.999999, 0.000001])  # Very uneven split
    TimeSplit(fractions=[0.333333, 0.333333, 0.333334])  # Precise thirds

    # Floating point precision edge cases
    TimeSplit(fractions=[0.7, 0.3 + 1e-7])  # Small floating point difference

    with pytest.raises(ValueError, match="Fractions must sum to 1.0"):
        TimeSplit(fractions=[0.7, 0.3 + 1e-5])  # Too large difference


def test_overlap():
    with pytest.raises(
            ValueError,
            match=("The end .* of the \\(train\\) split must be "
                   "strictly smaller than the start .* of the next "
                   "\\(val\\) split."),
    ):
        TimeSplit(indices=[(-6, -4), (-5, -3), (-2, -1)])

    with pytest.raises(
            ValueError,
            match=("The end .* of the \\(val\\) split must be "
                   "strictly smaller than the start .* of the next "
                   "\\(test\\) split."),
    ):
        TimeSplit(
            indices=[(-6, -5), (-4, -2), (-3, -1)],
            unit="days",
            timeframe_size=7,
            anchor_start_offset=0,
        )

    TimeSplit(
        indices=[(-7, -6), (-5, -4), (-3, -1)],
        unit="days",
        timeframe_size=7,
        anchor_start_offset=0,
    )

    TimeSplit(
        indices=[(-7, -5), (-4, -3), (-2, -1)],
        unit="days",
        timeframe_size=7,
        anchor_start_offset=0,
        strict_split_boundaries=False,
    )


def test_split_length():
    # Test object variants
    with pytest.raises(ValueError, match="At least two splits are required"):
        TimeSplit(indices=[(-2, -1)])

    with pytest.raises(ValueError,
                       match="No more than three splits is supported"):
        TimeSplit(indices=[(-8, -7), (-6, -5), (-4, -3), (-2, -1)])

    with pytest.raises(ValueError,
                       match="Data must be a list of tuples with two elements"):
        TimeSplit(indices=[(-4), (-2), (-1)])  # type: ignore

    with pytest.raises(ValueError,
                       match="Data must be a list of tuples with two elements"):
        TimeSplit(
            indices=[(), (-3), (-1)],  # type: ignore
            unit="days",
            timeframe_size=7,
            anchor_start_offset=0,
        )

    with pytest.raises(ValueError,
                       match="Data must be a list of tuples with two elements"):
        TimeSplit(
            indices=[(-8, -7, -6), (-5, -3), (-2, -1)],  # type: ignore
            unit="days",
            timeframe_size=7,
            anchor_start_offset=0,
        )

    # Test string variants
    with pytest.raises(ValueError, match="At least two splits are required"):
        get_split(
            "TimeSplit(indices=[(-2, -1)], unit='days', timeframe_size=7, "
            "anchor_start_offset=0)")

    with pytest.raises(ValueError,
                       match="No more than three splits is supported"):
        get_split("TimeSplit(indices=[(-8, -7), (-6, -5), (-4, -3), (-2, -1)], "
                  "unit='days', timeframe_size=7, anchor_start_offset=0)")

    with pytest.raises(ValueError,
                       match="Data must be a list of tuples with two elements"):
        get_split("TimeSplit(indices=[(-4), (-2), (-1)], unit='days', "
                  "timeframe_size=7, anchor_start_offset=0)")  # type: ignore

    with pytest.raises(ValueError,
                       match="Data must be a list of tuples with two elements"):
        get_split("TimeSplit(indices=[(), (-3), (-1)], unit='days', "
                  "timeframe_size=7, anchor_start_offset=0)")  # type: ignore

    with pytest.raises(ValueError,
                       match="Data must be a list of tuples with two elements"):
        get_split("TimeSplit(indices=[(-8, -7, -6), (-5, -3), (-2, -1)], "
                  "unit='days', timeframe_size=7, anchor_start_offset=0)"
                  )  # type: ignore


def test_indices_negative():
    """Test that indices must be negative."""
    # Valid cases
    TimeSplit(indices=[(-4, -3), (-2, -1)])
    TimeSplit(indices=[(-6, -5), (-4, -3), (-2, -1)], unit="days",
              timeframe_size=7, anchor_start_offset=0)

    # Invalid cases
    with pytest.raises(ValueError, match="Indices must be non-positive"):
        TimeSplit(indices=[(0, 1), (2, 3)], unit="days", timeframe_size=7,
                  anchor_start_offset=0)

    with pytest.raises(ValueError, match="Indices must be non-positive"):
        TimeSplit(indices=[(-2, -1), (0, 1)], unit="days", timeframe_size=7,
                  anchor_start_offset=0)

    with pytest.raises(ValueError, match="Indices must be non-positive"):
        TimeSplit(indices=[(-4, 1), (-3, -2), (-1, 0)], unit="days",
                  timeframe_size=7, anchor_start_offset=0)


def test_time_unit_support():
    """Test TimeUnit support for unit and timeframe_unit parameters."""
    # Test unit parameter with TimeUnit
    TimeSplit(indices=[(-4, -3), (-2, -1)], unit=TimeUnit.DAYS,
              timeframe_size=7, anchor_start_offset=0)
    TimeSplit(indices=[(-4, -3), (-2, -1)], unit=TimeUnit.MONTHS,
              timeframe_size=7, anchor_start_offset=0)
    TimeSplit(indices=[(-4, -3), (-2, -1)], unit=TimeUnit.HOURS,
              timeframe_size=7, anchor_start_offset=0)
    TimeSplit(indices=[(-4, -3), (-2, -1)], unit=TimeUnit.MINUTES,
              timeframe_size=7, anchor_start_offset=0)

    # Test timeframe_unit parameter with TimeUnit
    TimeSplit(indices=[(-4, -3), (-2, -1)], unit=TimeUnit.DAYS,
              anchor_start_offset=0, timeframe_size=1)
    TimeSplit(
        indices=[(-4, -3), (-2, -1)],
        unit="days",  # Test string with TimeUnit
        anchor_start_offset=0,
        timeframe_size=1)
    TimeSplit(indices=[(-4, -3), (-2, -1)], unit=TimeUnit.DAYS,
              anchor_start_offset=0, timeframe_size=1)

    # Test invalid TimeUnit values
    with pytest.raises(ValueError, match="'years' is not a valid TimeUnit"):
        TimeSplit(
            indices=[(-4, -3), (-2, -1)],
            unit="years",  # type: ignore
            timeframe_size=7,
            anchor_start_offset=0,
        )

    # Test case sensitivity
    TimeSplit(indices=[(-4, -3), (-2, -1)], unit="DAYS", timeframe_size=7,
              anchor_start_offset=0)  # Upper case
    TimeSplit(indices=[(-4, -3), (-2, -1)], unit="Days", timeframe_size=7,
              anchor_start_offset=0)  # Mixed case


def get_time_values_from_mask(table: TrainingTable, mask: Mask) -> pd.Series:
    """Get the timestamp values from a table's time column for the rows=
    specified by the mask.

    Args:
        table (TrainingTable): The table containing the time column.
        mask (Mask): The mask specifying which rows to get timestamps for.

    Returns:
        pd.Series: The timestamp values from the table's time column for the
        masked rows.
    """
    time_col_df = table.time_col.pandas_indexed()
    assert isinstance(time_col_df, pd.DataFrame)
    time_col = time_col_df[table.time_col.name]
    assert isinstance(time_col, pd.Series)
    mask_idx = mask.index_tensor.tolist()
    assert isinstance(mask_idx, list)
    rows = time_col[mask_idx]
    assert isinstance(rows, pd.Series)
    return rows


def test_timeframe_params():
    """Test validation of timeframe parameters consistency."""
    # Valid cases - all None or all set
    TimeSplit(
        indices=[(-4, -3), (-2, -1)],
        unit=None,
        timeframe_size=None,
        anchor_start_offset=None,
    )

    TimeSplit(
        indices=[(-4, -3), (-2, -1)],
        unit="days",
        timeframe_size=7,
        anchor_start_offset=0,
    )

    # Invalid cases - mixed None and non-None
    with pytest.raises(
            ValueError,
            match=("anchor_start_offset, timeframe_size must all be "
                   "None or all be not None")):
        TimeSplit(
            indices=[(-4, -3), (-2, -1)],
            unit="days",
            timeframe_size=None,
            anchor_start_offset=0,
        )


def test_boundary_value_validation():
    """Test validation of boundary value types and formats."""

    # Valid cases
    TimeSplit(timestamp_boundaries=[('2022-01-01',
                                     '2022-02-01'), ('2022-02-01',
                                                     '2022-03-01')])
    TimeSplit(
        offset_boundaries=[('-INF', -180), (-180, -90), (-90, 0)],
        unit=TimeUnit.DAYS,
        anchor_start_offset=0,
        timeframe_size=90,
    )
    TimeSplit(
        offset_boundaries=[(-365, -180), (-180, -90), (-90, 0)],
        unit=TimeUnit.DAYS,
        timeframe_size=90,
        anchor_start_offset=0,
    )

    # Invalid cases - mixing types
    with pytest.raises(
            ValueError,
            match=("When using timestamp strings, all boundaries must "
                   "be strings.")):
        TimeSplit(
            offset_boundaries=[
                ('2022-01-01', -90),  # type: ignore
                (-90, 0)
            ],
            unit=TimeUnit.DAYS,
        )

    # Invalid timestamp format
    with pytest.raises(ValueError, match="Invalid timestamp format"):
        TimeSplit(
            timestamp_boundaries=[('invalid-date', '2022-01-01'),
                                  ('2022-01-01', '2022-02-01')],
            unit=TimeUnit.DAYS,
        )

    # Invalid -INF usage
    with pytest.raises(
            ValueError,
            match="When using integer offsets, all boundaries must be integers"
    ):
        TimeSplit(offset_boundaries=[(-180, -90), ('-INF', 0)],
                  unit=TimeUnit.DAYS)


@pytest.mark.parametrize('split_config', [
    {
        'timestamp_boundaries': [
            ('2022-01-01', '2022-01-02'),
            ('2022-01-03', '2022-01-04'),
        ]
    },
    {
        'offset_boundaries': [(-180, -120), (-120, -60)],
        'unit': 'days',
        'timeframe_size': 7,
        'anchor_start_offset': 0,
    },
    {
        'indices': [(-2, -1), (-1, 0)],
        'unit': 'days',
        'timeframe_size': 7,
        'anchor_start_offset': 0,
    },
    {
        'fractions': [0.6, 0.4],
        'unit': 'days',
        'timeframe_size': 7,
        'anchor_start_offset': 0,
    },
])
def test_get_boundaries_initial(split_config):
    """Test that get_boundaries initially returns None for all split types."""
    time_split = TimeSplit(**split_config)
    assert time_split.get_boundaries() is None
