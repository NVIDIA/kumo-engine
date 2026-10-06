# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Peek at raw Yambda rows off the stage, cheaply.

``INFER_SCHEMA`` says ``event_time`` is ``NUMBER(38, 0)`` -- an integer, not a
timestamp -- so its unit and epoch have to be read off real values before the
graph can carry a ``Create date``.  Same for ``played_ratio_pct`` (integer
percent? basis points?) and ``event_type`` (which categories exist at all, and
how many, since rows 6/7 rank over exactly those).

One parquet part is 1.14 GB and holds ~50M rows, so everything here is scoped to
a single file with a ``LIMIT`` or a single-file aggregate.
"""
from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.YAMBDA_100GB'
PART = f'@{STAGE}/interactions/part-00000.parquet'
FF = 'HYU.KAGGLE_FIXTURES.PARQUET_FF'

session = Session.builder.config('connection_name', 'tlb').create()
session.sql('USE WAREHOUSE KUMO_L').collect()


def show(title: str, sql: str, limit: int = 40) -> None:
    print(f'\n--- {title} ---')
    print(f'    {" ".join(sql.split())[:300]}')
    try:
        rows = session.sql(sql).collect()
    except Exception as exc:  # noqa: BLE001
        print(f'    FAILED: {exc}')
        return
    if not rows:
        print('    (no rows)')
    for row in rows[:limit]:
        print('   ', ' | '.join(f'{c}={row[c]!r}' for c in row.as_dict()))


show(
    '10 raw rows', f"""
    SELECT $1:interaction_id::NUMBER AS interaction_id,
           $1:user_id::NUMBER        AS user_id,
           $1:item_id::NUMBER        AS item_id,
           $1:event_time::NUMBER     AS event_time,
           $1:event_type::STRING     AS event_type,
           $1:is_organic::NUMBER     AS is_organic,
           $1:played_ratio_pct::NUMBER AS played_ratio_pct,
           $1:track_length_seconds::NUMBER AS track_length_seconds
    FROM '{PART}' (FILE_FORMAT => {FF}) LIMIT 10""")

show(
    'event_time range and candidate epochs (one part)', f"""
    SELECT MIN($1:event_time::NUMBER) AS et_min,
           MAX($1:event_time::NUMBER) AS et_max,
           TO_VARCHAR(TO_TIMESTAMP_NTZ(MIN($1:event_time::NUMBER)))       AS as_seconds_min,
           TO_VARCHAR(TO_TIMESTAMP_NTZ(MAX($1:event_time::NUMBER)))       AS as_seconds_max,
           TO_VARCHAR(TO_TIMESTAMP_NTZ(MIN($1:event_time::NUMBER), 3))    AS as_millis_min,
           TO_VARCHAR(TO_TIMESTAMP_NTZ(MAX($1:event_time::NUMBER), 3))    AS as_millis_max,
           COUNT(*) AS n
    FROM '{PART}' (FILE_FORMAT => {FF})""")

show(
    'event_type distribution (one part)', f"""
    SELECT $1:event_type::STRING AS event_type, COUNT(*) AS n,
           COUNT_IF($1:is_organic::NUMBER = 1) AS organic,
           MIN($1:played_ratio_pct::NUMBER) AS prp_min,
           MAX($1:played_ratio_pct::NUMBER) AS prp_max,
           COUNT_IF($1:played_ratio_pct IS NULL) AS prp_null
    FROM '{PART}' (FILE_FORMAT => {FF}) GROUP BY 1 ORDER BY 2 DESC""")

show(
    'played_ratio_pct distribution (one part)', f"""
    SELECT COUNT(*) AS n,
           COUNT_IF($1:played_ratio_pct IS NULL) AS nulls,
           MIN($1:played_ratio_pct::NUMBER) AS p_min,
           APPROX_PERCENTILE($1:played_ratio_pct::NUMBER, 0.01) AS p01,
           APPROX_PERCENTILE($1:played_ratio_pct::NUMBER, 0.25) AS p25,
           APPROX_PERCENTILE($1:played_ratio_pct::NUMBER, 0.50) AS p50,
           APPROX_PERCENTILE($1:played_ratio_pct::NUMBER, 0.75) AS p75,
           APPROX_PERCENTILE($1:played_ratio_pct::NUMBER, 0.99) AS p99,
           MAX($1:played_ratio_pct::NUMBER) AS p_max,
           AVG($1:played_ratio_pct::NUMBER) AS p_avg,
           STDDEV($1:played_ratio_pct::NUMBER) AS p_std
    FROM '{PART}' (FILE_FORMAT => {FF})""")

show(
    'track_length_seconds and is_organic (one part)', f"""
    SELECT COUNT_IF($1:track_length_seconds IS NULL) AS tls_null,
           MIN($1:track_length_seconds::NUMBER) AS tls_min,
           APPROX_PERCENTILE($1:track_length_seconds::NUMBER, 0.5) AS tls_p50,
           MAX($1:track_length_seconds::NUMBER) AS tls_max,
           COUNT_IF($1:is_organic::NUMBER = 1) AS organic,
           COUNT_IF($1:is_organic::NUMBER = 0) AS inorganic,
           COUNT_IF($1:is_organic IS NULL) AS org_null
    FROM '{PART}' (FILE_FORMAT => {FF})""")

show(
    'is one part one user range? (partitioning shape)', f"""
    SELECT MIN($1:user_id::NUMBER) AS u_min, MAX($1:user_id::NUMBER) AS u_max,
           COUNT(DISTINCT $1:user_id::NUMBER) AS u_distinct,
           MIN($1:interaction_id::NUMBER) AS i_min,
           MAX($1:interaction_id::NUMBER) AS i_max,
           COUNT(*) AS n
    FROM '{PART}' (FILE_FORMAT => {FF})""")

show(
    'users table sample', f"""
    SELECT $1:user_id::NUMBER AS user_id
    FROM '@{STAGE}/users/part-00000.parquet' (FILE_FORMAT => {FF}) LIMIT 5""")
show(
    'items table sample', f"""
    SELECT $1:item_id::NUMBER AS item_id
    FROM '@{STAGE}/items/part-00000.parquet' (FILE_FORMAT => {FF}) LIMIT 5""")
show(
    'users/items id ranges', f"""
    SELECT (SELECT MIN($1:user_id::NUMBER)
            FROM '@{STAGE}/users/part-00000.parquet'
            (FILE_FORMAT => {FF})) AS u_min,
           (SELECT MAX($1:user_id::NUMBER)
            FROM '@{STAGE}/users/part-00000.parquet'
            (FILE_FORMAT => {FF})) AS u_max,
           (SELECT MIN($1:item_id::NUMBER)
            FROM '@{STAGE}/items/part-00000.parquet'
            (FILE_FORMAT => {FF})) AS i_min,
           (SELECT MAX($1:item_id::NUMBER)
            FROM '@{STAGE}/items/part-00000.parquet'
            (FILE_FORMAT => {FF})) AS i_max""")
