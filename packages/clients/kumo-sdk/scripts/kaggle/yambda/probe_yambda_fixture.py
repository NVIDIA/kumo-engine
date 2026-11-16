# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe the Yambda-5B Kaggle fixture staged on S3 -- STEP ZERO.

The fixture behind matrix rows 6 (multilabel classification), 7 (multilabel
ranking), 10 (regression), 13 (static link prediction) and 15 (temporal link
prediction) -- one S3 root, three tables (``interactions``, ``users``,
``items``), five predictive queries.

The matrix has *unresolved placeholders* for this fixture: the stored-size
column reads ``YAMBDA_STORED_SIZE`` and the row count ``YAMBDA_TOTAL_ROWS``,
literally, where every other row carries real numbers.  That is a strong hint
the fixture was templated but never built.  This script settles the question
before any modelling work is planned.

Local AWS credentials cannot see ``s3://kumo-public-datasets/``; the Snowflake
account carries the ``PUBLIC_DATASET`` storage integration whose allowed
location is the whole bucket, so an external stage is the only way in.
"""
from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.YAMBDA_100GB'
URL = 's3://kumo-public-datasets/kaggle-fixtures/yambda/100gb/'
TABLES = ['interactions', 'users', 'items']

session = Session.builder.config('connection_name', 'tlb').create()
session.sql('USE WAREHOUSE KUMO_L').collect()

session.sql('CREATE SCHEMA IF NOT EXISTS HYU.KAGGLE_FIXTURES').collect()
session.sql('CREATE FILE FORMAT IF NOT EXISTS HYU.KAGGLE_FIXTURES.JSON_FF '
            'TYPE = JSON').collect()
session.sql('CREATE FILE FORMAT IF NOT EXISTS HYU.KAGGLE_FIXTURES.PARQUET_FF '
            'TYPE = PARQUET').collect()
session.sql(f"CREATE STAGE IF NOT EXISTS {STAGE} URL = '{URL}' "
            'STORAGE_INTEGRATION = PUBLIC_DATASET').collect()

print(f'--- LIST @{STAGE}  ({URL}) ---')
total = 0
per_table: dict[str, tuple[int, int]] = {}
rows = session.sql(f'LIST @{STAGE}').collect()
for row in rows:
    name, size = row[0], int(row[1])
    total += size
    key = name.split('/')[1] if '/' in name else name
    count, acc = per_table.get(key, (0, 0))
    per_table[key] = (count + 1, acc + size)
    if len(rows) <= 60:
        print(f'  {name:<90} {size:>16,}')
print(f'  {"TOTAL":<90} {total:>16,} over {len(rows)} files')
print('\n--- per prefix ---')
for key, (count, acc) in sorted(per_table.items()):
    print(f'  {key:<40} {count:>6} files {acc:>18,} B')
if not rows:
    print('  (EMPTY PREFIX -- nothing staged at this location)')

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
    try:
        inferred = session.sql(
            'SELECT COLUMN_NAME, TYPE, NULLABLE FROM TABLE(INFER_SCHEMA('
            f"LOCATION => '@{STAGE}/{table}/', "
            "FILE_FORMAT => 'HYU.KAGGLE_FIXTURES.PARQUET_FF'))").collect()
        if not inferred:
            print('  (no files -- table absent)')
        for row in inferred:
            print(f'  {row[0]:<28} {row[1]:<24} nullable={row[2]}')
    except Exception as exc:  # noqa: BLE001
        print(f'  INFER_SCHEMA failed: {exc}')

# Widen the search: is the fixture anywhere else under kaggle-fixtures/?
print('\n--- sibling prefixes under kaggle-fixtures/ ---')
ROOT_STAGE = 'HYU.KAGGLE_FIXTURES.ALL_FIXTURES'
session.sql(f"CREATE STAGE IF NOT EXISTS {ROOT_STAGE} "
            "URL = 's3://kumo-public-datasets/kaggle-fixtures/' "
            'STORAGE_INTEGRATION = PUBLIC_DATASET').collect()
prefixes: dict[str, tuple[int, int]] = {}
for row in session.sql(f'LIST @{ROOT_STAGE}').collect():
    name, size = row[0], int(row[1])
    parts = name.split('/')
    key = '/'.join(parts[1:3]) if len(parts) > 2 else name
    count, acc = prefixes.get(key, (0, 0))
    prefixes[key] = (count + 1, acc + size)
for key, (count, acc) in sorted(prefixes.items()):
    print(f'  {key:<40} {count:>6} files {acc:>18,} B')

print('\n--- anything named yambda anywhere in the bucket ---')
BUCKET_STAGE = 'HYU.KAGGLE_FIXTURES.BUCKET_ROOT'
session.sql(f"CREATE STAGE IF NOT EXISTS {BUCKET_STAGE} "
            "URL = 's3://kumo-public-datasets/' "
            'STORAGE_INTEGRATION = PUBLIC_DATASET').collect()
try:
    hits = session.sql(
        f"LIST @{BUCKET_STAGE} PATTERN = '.*[yY][aA][mM][bB][dD][aA].*'"
    ).collect()
    if not hits:
        print('  (no object anywhere in the bucket matches /yambda/i)')
    for row in hits[:60]:
        print(f'  {row[0]:<100} {int(row[1]):>16,}')
    if len(hits) > 60:
        print(f'  ... and {len(hits) - 60} more')
except Exception as exc:  # noqa: BLE001
    print(f'  bucket-wide LIST failed: {exc}')
