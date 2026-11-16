# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cheap reads straight off the stage, before anything is materialized.

Two things have to be right before the one expensive pass over 150 part files
(``build_nyc_tables.py --raw-only``) is worth starting:

* **The timestamp unit.** All four datetime columns arrive as
  ``NUMBER(38, 0)``. NOAA's were microseconds; row 2 read microseconds as
  nanoseconds and silently produced the year 48,355,176. A wrong guess here is
  not caught by anything downstream, so it is checked against a known-good
  interval -- 2019-2022 -- before the cast is written into the raw copy.
* **The part-file shape.** Row 1's parts were each a full-history 10% sample,
  row 2's were one capture each, NOAA's were globally date-sorted. Which of the
  three this is decides whether a prefix slice is a usable tier (it never has
  been) and whether the tier predicate has to be a hash.

Both are answered from a handful of part files, not from the 745 million rows.
"""
from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.NYC_FHV_20GB'
FF = '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.PARQUET_FF)'

session = Session.builder.config('connection_name', 'tlb').create()
session.sql('USE WAREHOUSE KUMO_L').collect()


def show(label: str, sql: str) -> None:
    print(f'\n--- {label} ---', flush=True)
    for row in session.sql(sql).collect():
        print('  ' + '  '.join('NULL' if v is None else str(v) for v in row),
              flush=True)


show(
    'zones (all 264)', f"""
    SELECT MIN($1:zone_id::NUMBER), MAX($1:zone_id::NUMBER),
           COUNT(*), COUNT(DISTINCT $1:zone_id::NUMBER)
    FROM '@{STAGE}/zones/' {FF}
""")

show(
    'providers (all 4)', f"""
    SELECT $1:provider_id::STRING FROM '@{STAGE}/providers/' {FF}
""")

show('file names', f'LIST @{STAGE}/trips/')

# One part file, raw integers, so the unit can be read off the magnitude.
show(
    'trips part-00000, first 3 rows raw', f"""
    SELECT $1:trip_id::STRING, $1:request_datetime::NUMBER,
           $1:pickup_datetime::NUMBER, $1:dropoff_datetime::NUMBER,
           $1:trip_seconds::NUMBER, $1:pickup_zone_id::NUMBER,
           $1:dropoff_zone_id::NUMBER, $1:provider_id::STRING
    FROM '@{STAGE}/trips/part-00000.parquet' {FF} LIMIT 3
""")

# 1e9 s ~ 2001, 1e12 ms ~ 2001, 1e15 us ~ 2001, 1e18 ns ~ 2001. A 2019-2022
# fixture is ~1.55e9 s, so the exponent alone settles it.
show(
    'trips part-00000, pickup_datetime magnitude', f"""
    SELECT MIN($1:pickup_datetime::NUMBER), MAX($1:pickup_datetime::NUMBER),
           TO_TIMESTAMP_NTZ(MIN($1:pickup_datetime::NUMBER) / 1000000),
           TO_TIMESTAMP_NTZ(MAX($1:pickup_datetime::NUMBER) / 1000000)
    FROM '@{STAGE}/trips/part-00000.parquet' {FF}
""")

# Are parts contiguous in time (a prefix slice is January 2019) or uniform
# samples of the whole span? Five spread-out parts answer it.
for part in ('part-00000', 'part-00037', 'part-00075', 'part-00112',
             'part-00149'):
    show(
        f'trips {part} shape', f"""
        SELECT COUNT(*),
               COUNT(DISTINCT $1:trip_id::STRING),
               TO_TIMESTAMP_NTZ(MIN($1:pickup_datetime::NUMBER) / 1000000),
               TO_TIMESTAMP_NTZ(MAX($1:pickup_datetime::NUMBER) / 1000000),
               COUNT(DISTINCT $1:pickup_zone_id::NUMBER),
               COUNT(DISTINCT $1:provider_id::STRING)
        FROM '@{STAGE}/trips/{part}.parquet' {FF}
    """)
