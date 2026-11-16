# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Materializes the NYC FHV fixture for matrix rows 9, 12 and 16.

One fixture, three predictive queries::

    row  9  PREDICT trips.trip_seconds
            FOR EACH trips.trip_id                          (regression)
    row 12  PREDICT COUNT(trips.*, 0, 1, days)
            FOR EACH zones.zone_id                          (forecasting)
    row 16  PREDICT LIST_DISTINCT(trips.dropoff_zone_id, 0, 7, days)
            RANK TOP 20 FOR EACH zones.zone_id              (temporal LP)

Two tier families, because the entity differs
---------------------------------------------

The three rows do **not** share a tiering rule, and pretending they do would
corrupt one of them.

* **Row 9's entity is ``trips`` itself** -- a 745-million-row fact table, and the
  target ``trip_seconds`` is a static column on that same row. Thinning trips
  removes entities but leaves every surviving entity's label untouched, so a
  uniform hash sample of ``trip_id`` is a complete, correct instance of the
  published task on fewer trips. That is rows 2-4's rule.

* **Rows 12 and 16's entity is ``zones``** -- 264 rows -- and both targets are
  aggregates over the trips attached to a zone. Thinning trips **changes the
  label**: a 1% sample divides every zone's daily ``COUNT`` by 100 and deletes
  most of the distinct drop-off zones the ``LIST_DISTINCT`` target is asking
  for. That is NOAA's rule (rows 5/8/11), and there the answer was to sample the
  *entity* table instead -- unavailable here, because 264 zones is already the
  smallest the entity set gets and the graph contract wants all of them.

  So the only axis left is **time**: a tier is every trip inside a contiguous
  window from the fixture's start. Each zone's daily counts and destination sets
  are then exactly right inside the window, and the tier ladder buys anchors
  rather than entities -- which is the right axis anyway, since the training
  table for a dimension-table entity is entities x anchors (NOAA S6).

Both families read the same ``TRIPS_RAW`` and share one ``ZONES_RAW`` /
``PROVIDERS_RAW``.

Why the raw copy holds every column
-----------------------------------

Row 4 S2 measured a two-column re-read of an external parquet stage costing as
much as the original twenty-one-column pass (726 s against 600-1300 s): the
``$1:<field>`` variant decode dominates, not the projection width. So the one
pass over the 150 part files materializes **all 25** published columns plus a
precomputed sample bucket and a day index, and every tier after it is a scan of
a native Snowflake table.

The holdout, which row 9 needs and rows 12 and 16 do not
--------------------------------------------------------

Rows 1-4 all needed ``--holdout-pct``: their targets were static *columns*, so
Kumo compiles batch prediction to ``WHERE target IS NULL`` and a fully labeled
fixture leaves nothing to score (row 1 S1). Row 9's ``trips.trip_seconds`` is a
static column, so it needs one; rows 12 and 16 are temporal aggregates and NOAA
S2 measured that those get no prediction-mode filter at all.

The holdout is applied **only** to the tier table the graph reads, and the truth
is never carried alongside it -- scoring joins back to ``TRIPS_RAW`` by
``trip_id``. Carrying a ``trip_seconds_true`` column into the graph table would
hand the model the answer.
"""
import argparse
import time
from typing import Dict, Optional, TypedDict

from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.NYC_FHV_20GB'
FF = '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.PARQUET_FF)'
SCHEMA = 'HYU.NYC_FHV'

#: All 745,287,023 trips, typed, plus the tier predicates. One pass over 150
#: part files builds it and no later step touches the stage again.
RAW = f'{SCHEMA}.TRIPS_RAW'
ZONES_RAW = f'{SCHEMA}.ZONES_RAW'
PROVIDERS_RAW = f'{SCHEMA}.PROVIDERS_RAW'

#: The fixture's first pickup, measured off ``part-00000``. ``day_index`` counts
#: whole days from here, so a window tier is ``day_index < n``.
EPOCH = '2019-02-01'


class _Scale(TypedDict):
    r"""One fixture tier."""
    #: Window length from ``EPOCH``; ``None`` means the whole span.
    days: Optional[int]
    #: Parts per 10,000 of ``MOD(ABS(HASH(trip_id)), 10000)``; 10000 means
    #: every trip.
    permille: int


#: The two families are named apart on purpose: an ``r9`` tier is a uniform
#: sample of the whole 2019-2022 span and is **wrong** for rows 12 and 16, and a
#: ``z`` tier is a contiguous window and answers a narrower question for row 9.
SCALES: Dict[str, _Scale] = {
    # Row 9: uniform hash sample of trips across the whole span.
    'r9tiny': _Scale(days=None, permille=2),
    'r9smoke': _Scale(days=None, permille=10),
    'r9small': _Scale(days=None, permille=50),
    'r9med': _Scale(days=None, permille=200),
    'r9big': _Scale(days=None, permille=1000),
    # Rows 12 and 16: every trip inside a contiguous window.
    'z30': _Scale(days=30, permille=10000),
    'z90': _Scale(days=90, permille=10000),
    'z365': _Scale(days=365, permille=10000),
    'zfull': _Scale(days=None, permille=10000),
}


def run(session: Session, label: str, sql: str) -> None:
    start = time.monotonic()
    print(f'[{label}] running...', flush=True)
    session.sql(sql).collect()
    print(f'[{label}] done in {time.monotonic() - start:.1f}s', flush=True)


def table_exists(session: Session, name: str) -> bool:
    database, schema, table = name.split('.')
    rows = session.sql(
        f"SHOW TABLES LIKE '{table}' IN SCHEMA {database}.{schema}").collect()
    return len(rows) > 0


def build_raw(session: Session) -> None:
    r"""The single pass over the staged parquet.

    All four datetime columns arrive as a signed integer count of
    **microseconds** since the epoch -- measured, not assumed: ``part-00000``'s
    first ``pickup_datetime`` is ``1548979286000000``, which is 2019-02-01, and
    no other unit puts a 2019-2022 taxi fixture anywhere near the right century
    (row 2 S7 read microseconds as nanoseconds and got the year 48,355,176).
    They are stored as real ``TIMESTAMP_NTZ`` so neither the tier builds nor the
    graph has to know the unit.

    ``trip_bucket`` and ``day_index`` are the two tier predicates precomputed
    once over 745 million rows instead of re-derived on every tier build.
    """
    run(
        session, f'staged parquet -> {RAW}', f"""
        CREATE OR REPLACE TABLE {RAW} AS
        SELECT $1:trip_id::STRING                  AS trip_id,
               $1:provider_id::STRING              AS provider_id,
               $1:dispatching_base_num::STRING     AS dispatching_base_num,
               $1:originating_base_num::STRING     AS originating_base_num,
               TO_TIMESTAMP_NTZ($1:request_datetime::NUMBER / 1000000)
                                                   AS request_datetime,
               TO_TIMESTAMP_NTZ($1:on_scene_datetime::NUMBER / 1000000)
                                                   AS on_scene_datetime,
               TO_TIMESTAMP_NTZ($1:pickup_datetime::NUMBER / 1000000)
                                                   AS pickup_datetime,
               TO_TIMESTAMP_NTZ($1:dropoff_datetime::NUMBER / 1000000)
                                                   AS dropoff_datetime,
               $1:pickup_zone_id::NUMBER(38,0)     AS pickup_zone_id,
               $1:dropoff_zone_id::NUMBER(38,0)    AS dropoff_zone_id,
               $1:trip_miles::FLOAT                AS trip_miles,
               $1:trip_seconds::NUMBER(38,0)       AS trip_seconds,
               $1:base_passenger_fare::FLOAT       AS base_passenger_fare,
               $1:tolls::FLOAT                     AS tolls,
               $1:bcf::FLOAT                       AS bcf,
               $1:sales_tax::FLOAT                 AS sales_tax,
               $1:congestion_surcharge::FLOAT      AS congestion_surcharge,
               $1:airport_fee::FLOAT               AS airport_fee,
               $1:tips::FLOAT                      AS tips,
               $1:driver_pay::FLOAT                AS driver_pay,
               $1:shared_request_flag::STRING      AS shared_request_flag,
               $1:shared_match_flag::STRING        AS shared_match_flag,
               $1:access_a_ride_flag::STRING       AS access_a_ride_flag,
               $1:wav_request_flag::STRING         AS wav_request_flag,
               $1:wav_match_flag::STRING           AS wav_match_flag,
               MOD(ABS(HASH($1:trip_id::STRING)), 10000)  AS trip_bucket,
               DATEDIFF('day', '{EPOCH}'::DATE,
                        TO_TIMESTAMP_NTZ(
                            $1:pickup_datetime::NUMBER / 1000000)::DATE)
                                                   AS day_index
        FROM '@{STAGE}/trips/' {FF}
    """)

    run(
        session, f'staged parquet -> {ZONES_RAW}', f"""
        CREATE OR REPLACE TABLE {ZONES_RAW} AS
        SELECT $1:zone_id::NUMBER(38,0) AS zone_id FROM '@{STAGE}/zones/' {FF}
    """)

    run(
        session, f'staged parquet -> {PROVIDERS_RAW}', f"""
        CREATE OR REPLACE TABLE {PROVIDERS_RAW} AS
        SELECT $1:provider_id::STRING AS provider_id
        FROM '@{STAGE}/providers/' {FF}
    """)

    for name in (RAW, ZONES_RAW, PROVIDERS_RAW):
        count = session.sql(f'SELECT COUNT(*) FROM {name}').collect()[0][0]
        print(f'{name}: {count:,} rows', flush=True)


TRIP_COLUMNS = [
    'trip_id',
    'provider_id',
    'dispatching_base_num',
    'originating_base_num',
    'request_datetime',
    'on_scene_datetime',
    'pickup_datetime',
    'dropoff_datetime',
    'pickup_zone_id',
    'dropoff_zone_id',
    'trip_miles',
    'trip_seconds',
    'base_passenger_fare',
    'tolls',
    'bcf',
    'sales_tax',
    'congestion_surcharge',
    'airport_fee',
    'tips',
    'driver_pay',
    'shared_request_flag',
    'shared_match_flag',
    'access_a_ride_flag',
    'wav_request_flag',
    'wav_match_flag',
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', choices=list(SCALES), default='r9tiny')
    parser.add_argument('--warehouse', default='KUMO_L')
    parser.add_argument('--rebuild-raw', action='store_true')
    parser.add_argument('--raw-only', action='store_true')
    parser.add_argument(
        '--holdout-pct', type=int, default=0,
        help='blank trips.trip_seconds on this percentage of trips, chosen by '
        'a second independent hash of trip_id. Row 9 only: a static '
        'column target compiles batch prediction to WHERE target IS NULL, '
        'so a fully labeled table has nothing to predict on (row 1 S1). '
        'The truth is NOT carried in the tier table -- scoring joins back '
        'to TRIPS_RAW.')
    args = parser.parse_args()

    suffix = args.scale.upper()
    if args.holdout_pct:
        suffix = f'{suffix}_H{args.holdout_pct}'
    trips_table = f'{SCHEMA}.TRIPS_{suffix}'

    session = Session.builder.config('connection_name', 'tlb').create()
    # The connection profile names no warehouse, so the default is KUMO_XS.
    # `USE WAREHOUSE` on the session is the only thing that moves work.
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()
    session.sql(f'CREATE SCHEMA IF NOT EXISTS {SCHEMA}').collect()
    print('warehouse:',
          session.sql('SELECT CURRENT_WAREHOUSE()').collect()[0][0],
          flush=True)

    if args.rebuild_raw or not table_exists(session, RAW):
        build_raw(session)
    if args.raw_only:
        print('TOTAL raw only', flush=True)
        return

    spec = SCALES[args.scale]
    where = []
    if spec['permille'] < 10000:
        where.append(f"trip_bucket < {spec['permille']}")
    if spec['days'] is not None:
        where.append(f"day_index >= 0 AND day_index < {spec['days']}")
    predicate = ' AND '.join(where) if where else 'TRUE'

    # The holdout hashes a *salted* key, which is not a stylistic choice: row 2's
    # first build nested two thresholds on the same hash, and
    # `MOD(h, 10000) < 5` implies `MOD(h, 100) < 5`, so a `< 20` holdout on the
    # same `h` selected **every** sampled row and the slice had zero labels.
    # `|| '/holdout'` decorrelates the two.
    if args.holdout_pct:
        target = (f"CASE WHEN MOD(ABS(HASH(trip_id || '/holdout')), 100) < "
                  f'{args.holdout_pct} THEN NULL ELSE trip_seconds END')
    else:
        target = 'trip_seconds'
    projection = ',\n               '.join(
        f'{target} AS trip_seconds' if c == 'trip_seconds' else c
        for c in TRIP_COLUMNS)

    run(
        session, f'trips -> {trips_table}', f"""
        CREATE OR REPLACE TABLE {trips_table} AS
        SELECT {projection}
        FROM {RAW} WHERE {predicate}
    """)

    stats = session.sql(f"""
        SELECT COUNT(*), COUNT(DISTINCT trip_id),
               MIN(pickup_datetime), MAX(pickup_datetime),
               COUNT(DISTINCT pickup_zone_id), COUNT(DISTINCT dropoff_zone_id),
               COUNT(DISTINCT provider_id),
               COUNT_IF(trip_seconds IS NULL),
               COUNT_IF(pickup_zone_id IS NULL),
               COUNT_IF(dropoff_zone_id IS NULL),
               COUNT(DISTINCT pickup_datetime::DATE)
        FROM {trips_table}
    """).collect()[0]
    print(f'\n{trips_table}: {stats[0]:,} rows, {stats[1]:,} distinct trip_id',
          flush=True)
    print(
        f'  pickup span         {stats[2]} -> {stats[3]} '
        f'({stats[10]:,} distinct days)', flush=True)
    print(f'  pickup zones        {stats[4]}', flush=True)
    print(f'  dropoff zones       {stats[5]}', flush=True)
    print(f'  providers           {stats[6]}', flush=True)
    print(
        f'  trip_seconds NULL   {stats[7]:,} '
        f'({stats[7] / max(stats[0], 1):.2%})', flush=True)
    print(f'  pickup_zone NULL    {stats[8]:,}', flush=True)
    print(f'  dropoff_zone NULL   {stats[9]:,}', flush=True)

    if stats[0] == 0:
        raise SystemExit('degenerate slice: no trips selected')
    if stats[0] != stats[1]:
        print('  WARNING: trip_id is not unique in this slice', flush=True)

    # The holdout side, and the guard rows 2 and 4 both wished they had had.
    # A slice with nothing labeled cannot train and one with nothing blanked
    # cannot be scored, and either is worth knowing now rather than forty
    # minutes into an AutoML run.
    #
    # The second half of the check is row 4's `'-'` lesson transposed: after
    # blanking, `trip_seconds IS NULL` has to mean "held out" and nothing else.
    # If the *source* already carries nulls, those rows join the prediction set
    # with no truth behind them and quietly contaminate the score.
    if args.holdout_pct:
        source_nulls = session.sql(f"""
            SELECT COUNT_IF(trip_seconds IS NULL) FROM {RAW}
            WHERE {predicate}
        """).collect()[0][0]
        blanked = stats[7]
        print(
            f'\nholdout {args.holdout_pct}%: {blanked:,} blanked, '
            f'{stats[0] - blanked:,} labeled', flush=True)
        print(f'  trip_seconds already NULL in the source: {source_nulls:,}',
              flush=True)
        if blanked == 0 or blanked == stats[0]:
            raise SystemExit('degenerate holdout: nothing labeled or nothing '
                             'held out')
        if source_nulls:
            print(
                '  WARNING: source nulls are indistinguishable from held-out '
                'rows in the graph table; they have no truth to score '
                'against', flush=True)

        # Truth for the held-out rows only. Row 2 deliberately stopped taking a
        # full copy: at this row count a whole second table is tens of GB for
        # two columns' worth of use.
        truth_table = f'{trips_table}_TRUTH'
        run(
            session, f'held-out truth -> {truth_table}', f"""
            CREATE OR REPLACE TABLE {truth_table} AS
            SELECT trip_id, trip_seconds, pickup_zone_id, dropoff_zone_id,
                   trip_miles, pickup_datetime, dropoff_datetime
            FROM {RAW}
            WHERE {predicate}
              AND MOD(ABS(HASH(trip_id || '/holdout')), 100)
                  < {args.holdout_pct}
        """)
        truth_stats = session.sql(f"""
            SELECT COUNT(*), AVG(trip_seconds),
                   APPROX_PERCENTILE(trip_seconds, 0.5),
                   MIN(trip_seconds), MAX(trip_seconds),
                   STDDEV(trip_seconds)
            FROM {truth_table}
        """).collect()[0]
        print(
            f'{truth_table}: {truth_stats[0]:,} rows; trip_seconds '
            f'mean={truth_stats[1]:,.1f} median={truth_stats[2]:,.0f} '
            f'min={truth_stats[3]:,} max={truth_stats[4]:,} '
            f'sd={truth_stats[5]:,.1f}', flush=True)

    # Trips per zone-day, which is what row 12's target actually is and what
    # decides whether a window tier has any signal in it at all.
    per = session.sql(f"""
        SELECT MIN(n), APPROX_PERCENTILE(n, 0.5), AVG(n), MAX(n), COUNT(*)
        FROM (SELECT pickup_zone_id, pickup_datetime::DATE AS d, COUNT(*) AS n
              FROM {trips_table} GROUP BY 1, 2)
    """).collect()[0]
    print(
        f'\ntrips per (pickup zone, day): min={per[0]:,} '
        f'median={per[1]:,.0f} mean={per[2]:,.1f} max={per[3]:,} '
        f'over {per[4]:,} non-empty zone-days', flush=True)

    print('TOTAL', flush=True)


if __name__ == '__main__':
    main()
