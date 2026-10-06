# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from kumo_core.sources import arrow_parquet
from kumo_core.sources.object_store import iter_column_dtypes
from kumo_core.types.conversions import (
    to_pyarrow_type,
    unsupported_pyarrow_composite,
)

from io_utils import logging
from kumo.config import BaseColumnConfig, TableMetadataConfig
from kumo.connector.connector import UnreadableColumnError
from kumo.validation import ValidationException
from service_framework.env_var import EnvVar

# Protect against large number of very small files.
S3_READ_TIMEOUT = 600  # 10 minutes
_MAX_READ_SAMPLE_TIME = 10  # 10 seconds
PARQUET_PARTITION_SIZE = EnvVar(
    name='PARQUET_PARTITION_SIZE', type=int,
    description='Maximum allowed partition size '
    'in bytes, while reading parquet files into driver memory.'
    'Default is 4GiB, which is 4x the recommended max size '
    'of a partition.', default_value=4 * 1024 * 1024 * 1024)  # 4GiB

logger = logging.getLogger(__name__)


def add_metadata_cols_from_parquet_dataset(
        metadata: TableMetadataConfig,
        dataset: pq.ParquetDataset) -> TableMetadataConfig:
    metadata.cols.extend(
        BaseColumnConfig(name, dtype=dtype)
        for name, dtype in iter_column_dtypes(dataset.schema))
    return metadata


def warn_on_large_parquet_partition(
    file_paths: List[str],
    file_sizes: List[int],
):
    """Use this when reading parquet files into memory. This
    raises an early error message when we are sure ther will be OOM.
    Even with this check the driver might OOM if too many large
    files are read or uncompressed data is too large."""
    assert len(file_paths) == len(file_sizes)
    size_limit = PARQUET_PARTITION_SIZE.parse_value()
    for file_size, file_path in zip(file_sizes, file_paths):
        if file_size > size_limit:
            logger.error(
                "Bad parquet partitioning detected. "
                f"File '{file_path}' is larger than the maximum allowed size "
                f"of {size_limit/1024*1024} MB. "
                "Might cause OOM errors. ")


def validate_parquet_schema(schema: pa.Schema):
    """This function validates the schema of a pyarrow dataset/table to ensure
    that the dtypes are valid for the Kumo platform.

    For invalid dtypes, the function will raise an error.
    """
    unsupported_columns = [
        f"  - column '{name}' with data type '{data_type}'"
        for name, data_type in arrow_parquet.unsupported_columns(schema)
    ]
    if len(unsupported_columns) > 0:
        raise ValidationException(
            title="Unsupported Parquet Data Types",
            message=("One or more columns in this parquet table contains data "
                     "types that are not supported by Kumo:\n\n" +
                     "\n".join(unsupported_columns) + "\n\n" +
                     arrow_parquet.UNSUPPORTED_ADVICE))


def clean_parquet_table(
    table: pa.Table,
    col_types: Optional[Dict[str, Any]] = None,
    sample: bool = False,
):
    """
        Clean the parquet table that can be loaded into pandas dataframe. It
        transforms the decimal columns to float columns.

        Args:
            table (pyarrow.Table): The parquet table to be cleaned.
            col_types (Dict[str, Any], optional): The column types expected by
                pandas. (default: :obj:`None`)
            sample (bool): Whether we are sampling the table or not.
                (default: :obj:`False`)
        """
    if not sample:
        validate_parquet_schema(table.schema)

    for idx, col_name in enumerate(table.column_names):
        col = table.column(col_name)

        if pa.types.is_decimal(col.type):
            # `col.type.scale` is number of digits right of the decimal
            # point. `col.type.precision` is the number of digits left
            # of it. More information at:
            # github.com/apache/parquet-format/blob/master/LogicalTypes.md
            if col.type.scale == 0:
                # NOTE (matthias, vid) Parquet may contain ID columns
                # encoded as pa.Decimal(128). In this case, unsafe casting
                # to `int64` is a good workaround as long as casting to
                # string is not yet supported:
                # (1) The cast will be safe most of the time safe anyway;
                # (2) in case of overflows it is unlikely that there
                # will occur any collisions.
                # TODO `pa.Decimal128` to `pa.string()` cast not yet
                # supported.
                # NOTE when precision >= 18, the behavior is different in
                # kumo/data/utils.py, as the behavior here is legacy.
                expected_type = pa.int64()  # Potentially unsafe cast!
            else:
                expected_type = pa.float64()

            if col_types is not None and col_name in col_types:
                # pyarrow Decimal -> string not yet supported, see above
                # Avoid casting Decimal128 to string even if provided
                if not (pa.types.is_decimal(col.type) and to_pyarrow_type(
                        col_types[col_name]) == pa.string()):
                    expected_type = to_pyarrow_type(col_types[col_name])
                else:
                    logger.warning(
                        f'Column {col_name} with data type {col.type} '
                        f'has dtype set to {col_types[col_name]}. '
                        f'Cast from {col.type} to {col_types[col_name]} '
                        f'is not yet supported, the column will be cast '
                        f'into {expected_type} instead.')
            col = col.cast(expected_type, safe=False)
            table = table.set_column(idx, col_name, col)
        # NOTE add other necessary transform such as datetime.
        elif pa.types.is_time(col.type):
            # Pandas-facing reads normalize TIME to Kumo's timestamp
            # representation. PyArrow requires TIME -> INT64 -> TIMESTAMP.
            int64_col = col.cast(pa.int64(), safe=False)
            timestamp_col = int64_col.cast(pa.timestamp('us'), safe=False)
            table = table.set_column(idx, col_name, timestamp_col)
        else:
            if unsupported_pyarrow_composite(col.type):
                raise UnreadableColumnError(
                    f"Column {col_name} is a "
                    "composite type not supported by Kumo. Please "
                    "remove the column or modify the type in the"
                    " source data", unreadable_column=col_name)
    return table


# Number of sub-units in a single nanosecond, keyed by pyarrow temporal unit.
_UNIT_PER_NS = {'s': 1_000_000_000, 'ms': 1_000_000, 'us': 1_000, 'ns': 1}


def _clamp_temporal_column_to_ns(col: pa.ChunkedArray, col_name: str):
    """Clamp an out-of-range temporal column so it can be losslessly coerced
    to nanosecond resolution.

    Kumo represents all ``date``/``time``/``timedelta`` columns as pandas
    ``datetime64[ns]`` / ``timedelta64[ns]`` (see
    ``kumo_core.types``).
    The pandas nanosecond range is only ~1677-2262, while parquet temporal
    columns can hold ``s``/``ms``/``us`` values far outside it. Converting such
    a column to nanoseconds with ``safe=False`` does *not* error: it silently
    integer-wraps the value (e.g. a year-3000 ``us`` timestamp becomes
    ~1830). That is silent data corruption.

    This helper detects values outside the nanosecond-representable range and
    clamps them to the corresponding ``pd.Timestamp.min``/``pd.Timestamp.max``
    boundary (matching the legacy Spark casting convention in
    ``kumo.data.utils``), emitting a ``warning`` that names the column and the
    number of clamped values. After clamping, conversion to nanoseconds is
    lossless and cannot wrap.

    Columns already within range are returned unchanged, so the common
    in-range case (lossless ``us``/``s`` -> ``ns``) is preserved exactly.

    Args:
        col: A timestamp or duration column (already unit-typed by pyarrow).
        col_name: The column name, used for the warning message.

    Returns:
        The (possibly clamped) column, with the same pyarrow type.
    """
    unit = col.type.unit
    # ``ns`` columns are already representable; nothing to clamp.
    div = _UNIT_PER_NS.get(unit)
    if div is None or div == 1:
        return col

    # Express the nanosecond range in the column's own unit, rounding the
    # bounds *inward* so that the subsequent cast back to nanoseconds stays
    # strictly within range (``pd.Timestamp.min`` carries sub-microsecond
    # nanoseconds, so a naive truncation would itself overflow).
    lo = -(-pd.Timestamp.min.value // div)  # ceil division
    hi = pd.Timestamp.max.value // div  # floor division

    ints = col.cast(pa.int64())
    below = pc.less(ints, pa.scalar(lo, pa.int64()))  # type: ignore
    above = pc.greater(ints, pa.scalar(hi, pa.int64()))  # type: ignore
    num_clamped = pc.sum(pc.cast(pc.or_(below, above),
                                 pa.int64())).as_py()  # type: ignore
    if not num_clamped:
        return col

    logger.warning(
        f"Column '{col_name}' contains {num_clamped} temporal value(s) "
        f"outside the representable nanosecond range "
        f"({pd.Timestamp.min} to {pd.Timestamp.max}); these values were "
        f"clamped to the nearest in-range boundary to avoid silent integer "
        f"overflow. The original out-of-range values are not preserved.")
    clamped = pc.if_else(below, pa.scalar(lo, pa.int64()), ints)  # type: ignore
    clamped = pc.if_else(above, pa.scalar(hi, pa.int64()),
                         clamped)  # type: ignore
    return clamped.cast(col.type)


def to_pandas_ns_safe(table: pa.Table, **kwargs) -> "pd.DataFrame":
    """Convert a pyarrow table to pandas at Kumo's ``datetime64[ns]`` contract
    without silently wrapping out-of-range temporal values.

    This is the single conversion boundary shared by the parquet-backed
    connectors. It clamps
    any out-of-nanosecond-range timestamp / duration column (see
    ``_clamp_temporal_column_to_ns``) and then converts to pandas with
    ``coerce_temporal_nanoseconds=True`` so every temporal column lands at
    nanosecond resolution.

    ``safe=False`` is retained for the conversion because it is still required
    for non-temporal reasons (e.g. reading ``Decimal128`` id columns cast to
    ``int64``); the temporal-overflow hazard it would otherwise enable is
    removed by the pre-clamp above.

    Any caller-provided ``to_pandas`` keyword arguments (``types_mapper``,
    ``ignore_metadata``, ...) are forwarded unchanged.
    """
    for idx, col_name in enumerate(table.column_names):
        col = table.column(col_name)
        if pa.types.is_timestamp(col.type) or pa.types.is_duration(col.type):
            clamped = _clamp_temporal_column_to_ns(col, col_name)
            if clamped is not col:
                table = table.set_column(idx, col_name, clamped)

    kwargs.setdefault('coerce_temporal_nanoseconds', True)
    return table.to_pandas(safe=False, **kwargs)
