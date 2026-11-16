# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe the IoT-23 Kaggle fixture staged on S3, reached through Snowflake.

Row 2 of the fixture matrix. Local AWS credentials cannot see
``s3://kumo-public-datasets/``, but the Snowflake account has a storage
integration (``PUBLIC_DATASET``) whose allowed location is the whole bucket, so
an external stage is the only way in from this box.

This creates that stage, then reads the fixture manifest and infers each
table's parquet schema, so everything downstream is written against the real
column names and types rather than against the matrix's description of them.
"""
from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.IOT23_10GB'
URL = 's3://kumo-public-datasets/kaggle-fixtures/iot23/10gb/'
TABLES = ['hosts', 'network_flows']

session = Session.builder.config('connection_name', 'tlb').create()

session.sql('CREATE SCHEMA IF NOT EXISTS HYU.KAGGLE_FIXTURES').collect()
session.sql('CREATE FILE FORMAT IF NOT EXISTS HYU.KAGGLE_FIXTURES.JSON_FF '
            'TYPE = JSON').collect()
session.sql('CREATE FILE FORMAT IF NOT EXISTS HYU.KAGGLE_FIXTURES.PARQUET_FF '
            'TYPE = PARQUET').collect()
session.sql(f"CREATE STAGE IF NOT EXISTS {STAGE} URL = '{URL}' "
            'STORAGE_INTEGRATION = PUBLIC_DATASET').collect()

print('--- files ---')
for row in session.sql(f'LIST @{STAGE}').collect():
    print(f'  {row[0]:<80} {row[1]:>14,}')

print('\n--- manifest.json ---')
for row in session.sql(
        f"SELECT $1 FROM '@{STAGE}/_metadata/manifest.json' "
        '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.JSON_FF)').collect():
    print(row[0])

for table in TABLES:
    print(f'\n--- {table} schema ---')
    inferred = session.sql(
        'SELECT COLUMN_NAME, TYPE, NULLABLE FROM TABLE(INFER_SCHEMA('
        f"LOCATION => '@{STAGE}/{table}/', "
        "FILE_FORMAT => 'HYU.KAGGLE_FIXTURES.PARQUET_FF'))").collect()
    for row in inferred:
        print(f'  {row[0]:<28} {row[1]:<20} nullable={row[2]}')
