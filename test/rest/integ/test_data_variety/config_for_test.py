# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from kumoapi.typing import Stype


class ConnectorType(Enum):
    file = 'file'
    snowflake = 'snowflake'
    bigquery = 'bigquery'


class FailStage(Enum):
    LIST_SOURCE = 'LIST_SOURCE'
    VALIDATE = 'VALIDATE'
    REFRESH = 'REFRESH'
    MATERIALIZE = 'MATERIALIZE'


# SNOWFLAKE
# https://app.snowflake.com/lfwgwbp/zxa66432/#/data/databases/KUMO/schemas/DATA_TYPE_TEST/table/ALL_DATA_TYPES
snow_table_name = 'snow_table'
snow_pkey_cols = ['ID']
snow_default_stypes = {
    'ID': Stype.ID,
    'NUM_NUMBER': Stype.numerical,
    'NUM_INT': Stype.numerical,
    'NUM_SMALLINT': Stype.numerical,
    'NUM_BIGINT': Stype.numerical,
    'NUM_DECIMAL': Stype.numerical,
    'NUM_NUMERIC': Stype.numerical,
    'NUM_FLOAT': Stype.numerical,
    'NUM_DOUBLE': Stype.numerical,
    'NUM_REAL': Stype.numerical,
    'STR_VARCHAR': Stype.categorical,
    'STR_CHAR': Stype.categorical,
    'STR_CHARACTER': Stype.categorical,
    'STR_string': Stype.categorical,
    'STR_TEXT': Stype.categorical,
    'BOOL_VALUE': Stype.categorical,
    'DATE_VALUE': Stype.timestamp,
    'TIME_VALUE': Stype.timestamp,
    'TIMESTAMP_VALUE': Stype.timestamp,
    'TIMESTAMPTZ_VALUE': Stype.timestamp,
    'TIMESTAMPLTZ_VALUE': Stype.timestamp,
    'TIMESTAMPNTZ_VALUE': Stype.timestamp,
    'VARIANT_VALUE': Stype.categorical,
    'OBJECT_VALUE': Stype.unsupported,
    'ARRAY_VALUE': Stype.categorical,
    'GEO_GEOGRAPHY': Stype.unsupported,
    'BIN_BINARY': Stype.categorical,
    'BIN_VARBINARY': Stype.categorical
}
snow_time_cols = [
    'DATE_VALUE', 'TIME_VALUE', 'TIMESTAMP_VALUE', 'TIMESTAMPLTZ_VALUE',
    'TIMESTAMPNTZ_VALUE', 'TIMESTAMPTZ_VALUE'
]
snow_time_cols_na_count = [200, 0, 0, 0, 0, 0]
snowflake_args = {
    "connector_type": ConnectorType.snowflake,
    "source_name": "ALL_DATA_TYPES",
    "table_name": snow_table_name,
    "pkey": snow_pkey_cols,
    "default_stypes": snow_default_stypes,
    "time_cols": snow_time_cols,
    "time_cols_na_count": snow_time_cols_na_count,
    "fail_stage": None,
    "fail_message": None,
}

# Parquet data generated for this integration fixture.
parquet_default_stypes = {
    'id': Stype.ID,
    'bool_col': Stype.categorical,
    'int8_col': Stype.numerical,
    'int16_col': Stype.numerical,
    'int32_col': Stype.numerical,
    'int64_col': Stype.numerical,
    'float32_col': Stype.numerical,
    'float64_col': Stype.numerical,
    'string_col': Stype.categorical,
    'binary_col': Stype.categorical,
    'date_col': Stype.timestamp,
    'timestamp_col': Stype.timestamp,
    'decimal_col': Stype.numerical,
    'list_col': Stype.multicategorical,
    'map_col': Stype.categorical
}
parquet_args = {
    "connector_type": ConnectorType.file,
    "source_name": "passing_dummy",
    "table_name": "parquet_passing",
    "pkey": ["id"],
    "default_stypes": parquet_default_stypes,
    "time_cols": ["timestamp_col", "time_col", "date_col"],
    "time_cols_na_count": [600, 600, 600],
    "fail_stage": None,
    "fail_message": None,
}

# Parquet metadata with unsigned columns. Sample rows cannot be represented by
# the supported API types, so the endpoint returns metadata and omits rows.
unsupported_parquet_args = {
    "connector_type": ConnectorType.file,
    "source_name": "failing_dummy",
    "table_name": "parquet_failing",
    "pkey": ["id"],
    "default_stypes": {
        "id": Stype.ID,
        "uint8_col": Stype.unsupported,
        "uint16_col": Stype.unsupported,
        "uint32_col": Stype.numerical,
        "uint64_col": Stype.unsupported,
    },
    "time_cols": None,
    "time_cols_na_count": None,
    "expect_metadata_only": True,
    "fail_stage": None,
    "fail_message": None,
}

# Parquet data with TIME column. Spark could not load TIME(MICROS); DiskGraph
# genuinely supports time64[us], so the refresh now SUCCEEDS (previously this
# row asserted a REFRESH-stage failure). Kept as a success case to preserve
# TIME(MICROS) ingest coverage.
default_stypes = {'id': Stype.ID, 'time_col': Stype.timestamp}
parquet_time_col_args = {
    "connector_type": ConnectorType.file,
    "source_name": "failing_time_col",
    "table_name": "parquet_time_col",
    "pkey": ["id"],
    "default_stypes": default_stypes,
    "time_cols": ['time_col'],
    "time_cols_na_count": None,
    "fail_stage": None,
    "fail_message": None,
}

# Passing BigQuery data.
bigquery_args = {
    "connector_type": ConnectorType.bigquery,
    "source_name": "all_supported_data_types",
    "table_name": "bigquery_passing",
    "pkey": ["id"],
    "default_stypes": {
        'id': Stype.ID,
        'int64_col': Stype.numerical,
        'float64_col': Stype.numerical,
        'numeric_col': Stype.numerical,
        'string_col': Stype.categorical,
        'bytes_col': Stype.categorical,
        'bool_col': Stype.categorical,
        'date_col': Stype.timestamp,
        'datetime_col': Stype.timestamp,
        'timestamp_col': Stype.timestamp,
        'array_int64_col': Stype.sequence
    },
    "time_cols": ["timestamp_col", "datetime_col", "date_col"],
    "time_cols_na_count": [500, 500, 500],
    "fail_stage": None,
    "fail_message": None,
}
