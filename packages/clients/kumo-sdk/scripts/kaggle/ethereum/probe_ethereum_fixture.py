# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe the ethereum-fraud Kaggle fixture staged on S3 via Snowflake.

Reads the fixture manifest and infers each table's parquet schema, so the
LocalBackend run can be written against the real column names and types.
"""
from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.ETHEREUM_FRAUD_30GB'
TABLES = ['addresses', 'labeled_address_ids', 'transactions']

session = Session.builder.config('connection_name', 'tlb').create()

session.sql('CREATE FILE FORMAT IF NOT EXISTS HYU.KAGGLE_FIXTURES.JSON_FF '
            'TYPE = JSON').collect()
session.sql('CREATE FILE FORMAT IF NOT EXISTS HYU.KAGGLE_FIXTURES.PARQUET_FF '
            'TYPE = PARQUET').collect()

print('--- manifest.json ---')
rows = session.sql(f"SELECT $1 FROM '@{STAGE}/_metadata/manifest.json' "
                   '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.JSON_FF)').collect()
for row in rows:
    print(row[0])

for table in TABLES:
    print(f'\n--- {table} schema ---')
    inferred = session.sql(
        'SELECT COLUMN_NAME, TYPE, NULLABLE FROM TABLE(INFER_SCHEMA('
        f"LOCATION => '@{STAGE}/{table}/', "
        "FILE_FORMAT => 'HYU.KAGGLE_FIXTURES.PARQUET_FF'))").collect()
    for row in inferred:
        print(f'  {row[0]:<28} {row[1]:<20} nullable={row[2]}')
