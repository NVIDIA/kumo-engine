# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Preprocessing data on Snowflake. Please note that running this example
script requires setting:
    * SNOWFLAKE_USER
    * SNOWFLAKE_PASSWORD
    * SNOWFLAKE_ACCOUNT
    * KUMO_SNOWFLAKE_IDENTITY_ID
    * KUMO_PROJECT
environment variables. This script can be slightly modified for SPCS native
apps, by passing the relevant credentials into `kumo.init`.

This script covers all setup and initialization CUJs on the Snowflake data
warehouse. CUJs that are not supported are explicitly marked below.
"""
import os

import snowflake.connector

import kumoai_studio as kumo

API_URL = os.getenv("KUMO_API_URL", "http://localhost:10002")
API_KEY = os.environ["KUMO_API_KEY"]
kumo.init(url=API_URL, api_key=API_KEY, project=os.environ["KUMO_PROJECT"])

# 1. Initialize both the Kumo Snowflake connector and a Snowflake connector to
# the data warehouse:
connector = kumo.SnowflakeConnector(
    name="snowflake_connector_dataprep",
    identity=os.environ["KUMO_SNOWFLAKE_IDENTITY_ID"],
    warehouse="WH_XS",
    database="KUMO",
    schema_name="HM_SAMPLED",
)
session = snowflake.connector.connect(
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    warehouse="WH_XS",
    database="KUMO",
    schema="HM_SAMPLED",
)

# 2. Minor data preprocessing case (a composite PK), particularly for
# development:

# Without leaving the worksheet, allow (lightweight) data processing, even for
# large-scale data. Support running it without downloading the data (i.e.
# either in Kumo platform or in customer data platform).

# a) Create an entity table where primary key is the concatenation of
# customer_id and sales_channel_id.
sql_statement = """
    CREATE OR REPLACE TABLE entities AS
    SELECT CONCAT(customer_id, sales_channel_id) AS pk, *
    FROM customers
    CROSS JOIN (SELECT DISTINCT sales_channel_id FROM transactions)
"""
session.cursor().execute(sql_statement)
print(f"Created table 'entities' with statement {sql_statement}")

# b) Create a processed transactions table where foreign key is the
# concatenation of customer_id and sales_channel_id.
sql_statement = """
    CREATE OR REPLACE TABLE transactions_processed AS
    SELECT *, CONCAT(customer_id, sales_channel_id) AS fk
    FROM transactions
"""
session.cursor().execute(sql_statement)
print(f"Created table 'transactions_processed' with statement {sql_statement}")

# c) Create Kumo tables from these source tables:
entities = kumo.Table.from_source_table(
    source_table=connector["ENTITIES"],
    primary_key="PK",
)
transactions = kumo.Table.from_source_table(
    source_table=connector["TRANSACTIONS_PROCESSED"],
    time_column="T_DAT",
)

# d) View properties of the created Kumo tables:
print(f"Entities Kumo table:\n{entities.metadata}")
print(f"Transactions Kumo table:\n{transactions.metadata}")

# Not supported: SDK can load and work with tables created in the UI. This
# is currently deprioritized to avoid confusing workflows between the SDK
# and the UI.
