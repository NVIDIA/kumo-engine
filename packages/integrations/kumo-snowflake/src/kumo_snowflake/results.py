# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

import pandas as pd
import pyarrow as pa
from kumo_snowflake.source import microsecond_precision_kwargs
from pyarrow import ArrowInvalid

logger = logging.getLogger(__name__)


def restore_timestamp_precision(table: pa.Table) -> pa.Table:
    """Use nanosecond pandas timestamps only where the Arrow cast is lossless."""
    for index, field in enumerate(table.schema):
        if (not pa.types.is_timestamp(field.type) or field.type.unit == 'ns'):
            continue

        nanosecond_type = pa.timestamp('ns', tz=field.type.tz)
        try:
            column = table.column(index).cast(nanosecond_type, safe=True)
        except ArrowInvalid:
            continue

        nanosecond_field = pa.field(
            field.name,
            nanosecond_type,
            nullable=field.nullable,
            metadata=field.metadata,
        )
        table = table.set_column(index, nanosecond_field, column)

    return table


def dataframe(cursor: Any) -> pd.DataFrame:
    if cursor.rowcount != 0:
        try:
            table = cursor.fetch_arrow_all(
                **microsecond_precision_kwargs(cursor.fetch_arrow_all))
            return restore_timestamp_precision(table).to_pandas(
                timestamp_as_object=True, date_as_object=True)
        except (ArrowInvalid, SystemError) as error:
            logger.error('Failed to decode Snowflake dataframe (error_type=%s)',
                         type(error).__name__)
    return pd.DataFrame(columns=list(pd.DataFrame(cursor.description)['name']))


def sample(connection: Any, source: str, count: int, *,
           timeout: int | None = None) -> pd.DataFrame:
    from kumo_core.custom_sql import sample_select
    from kumo_snowflake.source import read_cursor
    with read_cursor(connection, sample_select(source, count),
                     timeout=timeout) as cursor:
        return dataframe(cursor)
