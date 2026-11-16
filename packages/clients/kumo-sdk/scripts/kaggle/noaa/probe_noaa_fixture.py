# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe the NOAA GHCN-Daily Kaggle fixture staged on S3, through Snowflake.

The fixture behind matrix rows 5 (multilabel classification), 8 (multilabel
ranking) and 11 (forecasting) -- one S3 root, one pair of tables, three
predictive queries.

Local AWS credentials cannot see ``s3://kumo-public-datasets/``, but the
Snowflake account carries a storage integration (``PUBLIC_DATASET``) whose
allowed location is the whole bucket, so an external stage is the only way in
from this box. This creates that stage, lists it, reads the fixture manifest and
infers each table's parquet schema, so everything downstream is written against
the real column names and types rather than against the matrix's description of
them -- which has been wrong in at least one respect on all four previous rows.
"""
from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.NOAA_GHCN_10GB'
URL = 's3://kumo-public-datasets/kaggle-fixtures/noaa-ghcn/10gb/'
TABLES = ['observations', 'stations']

session = Session.builder.config('connection_name', 'tlb').create()
session.sql('USE WAREHOUSE KUMO_L').collect()

session.sql('CREATE SCHEMA IF NOT EXISTS HYU.KAGGLE_FIXTURES').collect()
session.sql('CREATE FILE FORMAT IF NOT EXISTS HYU.KAGGLE_FIXTURES.JSON_FF '
            'TYPE = JSON').collect()
session.sql('CREATE FILE FORMAT IF NOT EXISTS HYU.KAGGLE_FIXTURES.PARQUET_FF '
            'TYPE = PARQUET').collect()
session.sql(f"CREATE STAGE IF NOT EXISTS {STAGE} URL = '{URL}' "
            'STORAGE_INTEGRATION = PUBLIC_DATASET').collect()

print('--- files ---')
total = 0
rows = session.sql(f'LIST @{STAGE}').collect()
for row in rows:
    total += int(row[1])
    print(f'  {row[0]:<90} {int(row[1]):>16,}')
print(f'  {"TOTAL":<90} {total:>16,} over {len(rows)} files')

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
        print(f'  {row[0]:<28} {row[1]:<24} nullable={row[2]}')
