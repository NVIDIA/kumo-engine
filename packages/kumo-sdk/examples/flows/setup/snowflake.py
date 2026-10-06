# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Connecting data on Snowflake. Running this example requires an
administrator-managed Snowflake identity assigned to the project's team.
Set KUMO_SNOWFLAKE_IDENTITY_ID and KUMO_PROJECT to those resource IDs.

This script covers all setup and initialization CUJs on the Snowflake data
warehouse. CUJs that are not supported are explicitly marked below.
"""
import os

import pandas as pd
from kumoapi.source_table import SourceColumn
from kumoapi.typing import Dtype, Stype

import kumoai_studio as kumo

API_URL = os.getenv("KUMO_API_URL", "http://localhost:10002")
API_KEY = os.environ["KUMO_API_KEY"]
kumo.init(url=API_URL, api_key=API_KEY, project=os.environ["KUMO_PROJECT"])

# 1. Create a schema-specific connector from an authorized identity.
connector = kumo.SnowflakeConnector(
    name="snowflake_connector",
    identity=os.environ["KUMO_SNOWFLAKE_IDENTITY_ID"],
    warehouse="WH_XS",
    database="KUMO",
    schema_name="CUSTOMER_LTV",
)

# 2. The same service-account identity can back another schema connector
# without copying credentials.
_ = kumo.SnowflakeConnector(
    name="snowflake_public_connector",
    identity=os.environ["KUMO_SNOWFLAKE_IDENTITY_ID"],
    warehouse="WH_XS",
    database="KUMO",
    schema_name="PUBLIC",
)

# 3. Load a Snowflake connector by name (e.g. from the UI).
connector = kumo.SnowflakeConnector.get_by_name("snowflake_connector")

# Invalid names raise appropriate errors:
try:
    _ = kumo.SnowflakeConnector.get_by_name("invalid_name")
except ValueError as e:
    assert "does not exist" in str(e)
else:
    raise RuntimeError("Exception not raised.")

# 4. You can list all tables behind a Snowflake connector:
table_names = connector.table_names()
assert {'customer', 'stock', 'transaction'}.issubset(set(table_names))

# 5. You can read the schema information from a raw table in a given connector:
# 5a) You can fetch column names and data types:
customer = connector['customer']
columns = customer.columns
assert columns == [
    SourceColumn(name='CustomerID', stype=Stype.numerical, dtype=Dtype.float,
                 is_primary=False)
]
# 5b) Error handling if table name is wrong:
try:
    nonexistent = connector['nonexistent']
except ValueError as e:
    assert "does not exist" in str(e)
else:
    raise RuntimeError("Exception not raised.")

# 6. Read a limited # of samples from a raw table in a given connector:
customer = connector['customer']
head_rows = customer.head(num_rows=10)
assert isinstance(head_rows, pd.DataFrame)
assert len(head_rows) == 10
