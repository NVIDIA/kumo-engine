# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe the NYC FHV (for-hire vehicle) Kaggle fixture staged on S3.

The fixture behind matrix rows 9 (regression, ``trips.trip_seconds``),
12 (forecasting, daily pickup demand per zone) and 16 (temporal link
prediction, next drop-off zones per zone) -- one S3 root, three tables
(``providers``, ``trips``, ``zones``), three predictive queries.

Local AWS credentials cannot see ``s3://kumo-public-datasets/``; the Snowflake
account carries the ``PUBLIC_DATASET`` storage integration whose allowed
location is the whole bucket, so an external stage is the only way in from this
box. This creates the stage, lists it, reads the fixture manifest and infers
each table's parquet schema, so everything downstream is written against real
column names and types rather than against the matrix's description of them --
which has been wrong on four of the five previous fixtures.
"""
from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.NYC_FHV_20GB'
URL = 's3://kumo-public-datasets/kaggle-fixtures/nyc-fhv/20gb/'
TABLES = ['providers', 'trips', 'zones']

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
per_table: dict[str, tuple[int, int]] = {}
rows = session.sql(f'LIST @{STAGE}').collect()
for row in rows:
    name, size = row[0], int(row[1])
    total += size
    key = name.split('/')[1] if '/' in name else name
    count, acc = per_table.get(key, (0, 0))
    per_table[key] = (count + 1, acc + size)
    if len(rows) <= 40:
        print(f'  {name:<90} {size:>16,}')
print(f'  {"TOTAL":<90} {total:>16,} over {len(rows)} files')
print('\n--- per prefix ---')
for key, (count, acc) in sorted(per_table.items()):
    print(f'  {key:<40} {count:>6} files {acc:>18,} B')

print('\n--- manifest.json ---')
try:
    for row in session.sql(
            f"SELECT $1 FROM '@{STAGE}/_metadata/manifest.json' "
            '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.JSON_FF)').collect():
        print(row[0])
except Exception as exc:  # noqa: BLE001
    print(f'  no manifest: {exc}')

for table in TABLES:
    print(f'\n--- {table} schema ---')
    inferred = session.sql(
        'SELECT COLUMN_NAME, TYPE, NULLABLE FROM TABLE(INFER_SCHEMA('
        f"LOCATION => '@{STAGE}/{table}/', "
        "FILE_FORMAT => 'HYU.KAGGLE_FIXTURES.PARQUET_FF'))").collect()
    for row in inferred:
        print(f'  {row[0]:<28} {row[1]:<24} nullable={row[2]}')
