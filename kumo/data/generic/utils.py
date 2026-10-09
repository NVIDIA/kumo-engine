# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic dataframe helpers.

DG-Rust writes parquet from the ingest path and PyArrow handles reads
where needed.
"""

from io_utils import logging

logger = logging.getLogger(__name__)


def get_fully_qualified_snowflake_udf_name(udf_func_name: str) -> str:
    """Qualify a Snowflake UDF name with SNOWFLAKE_DATABASE / SNOWFLAKE_SCHEMA."""
    from service_framework.deploy_env import (
        SNOWFLAKE_DATABASE,
        SNOWFLAKE_SCHEMA,
    )
    if SNOWFLAKE_DATABASE and SNOWFLAKE_SCHEMA:
        logger.debug(f'Qualifying UDF function name {udf_func_name} with '
                     f'database {SNOWFLAKE_DATABASE} and schema '
                     f'{SNOWFLAKE_SCHEMA}')
        return f'{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.{udf_func_name}'
    return udf_func_name
