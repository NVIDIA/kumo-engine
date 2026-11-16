# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Every matrix claim about the NYC FHV fixture, checked against the data.

Four of the five previous fixtures deviated from the matrix in at least one way,
so nothing here is inherited. Run against ``HYU.NYC_FHV.TRIPS_RAW`` -- the one
typed copy ``build_nyc_tables.py --raw-only`` materializes -- so every section is
a scan of a native table rather than another pass over the stage.

Sections, and what each is for:

``--keys``     ``trip_id`` uniqueness, and whether the two zone foreign keys are
               clean in both directions. The graph has **two** edges into
               ``zones`` and rows 12 and 16 traverse exactly one of them
               (``probe_nyc_query.py``), so an orphan or a null on the wrong
               side is a silent loss of entities.
``--time``     the span and density of ``pickup_datetime``, and whether the four
               datetime columns are ordered the way their names imply. Density
               is what decides the tier ladder: rows 12 and 16 are tiered by
               *time window*, so trips-per-day is the exchange rate between a
               tier and its cost.
``--columns``  null rate and cardinality per column, because row 4 lost a run to
               a column that was non-null in 0.31% of rows: a 10,000-row
               inference sample came back entirely null, which yields no
               statistics, which is a hard error rather than a default.
``--parts``    already answered from the stage in ``probe_nyc_raw.py`` -- the
               parts are globally sorted by ``pickup_datetime``, so a prefix
               slice is February 2019 -- and re-checked here at full scale.
"""
import argparse
import time

from snowflake.snowpark import Session

RAW = 'HYU.NYC_FHV.TRIPS_RAW'
ZONES = 'HYU.NYC_FHV.ZONES_RAW'
PROVIDERS = 'HYU.NYC_FHV.PROVIDERS_RAW'

COLUMNS = [
    'TRIP_ID',
    'PROVIDER_ID',
    'DISPATCHING_BASE_NUM',
    'ORIGINATING_BASE_NUM',
    'REQUEST_DATETIME',
    'ON_SCENE_DATETIME',
    'PICKUP_DATETIME',
    'DROPOFF_DATETIME',
    'PICKUP_ZONE_ID',
    'DROPOFF_ZONE_ID',
    'TRIP_MILES',
    'TRIP_SECONDS',
    'BASE_PASSENGER_FARE',
    'TOLLS',
    'BCF',
    'SALES_TAX',
    'CONGESTION_SURCHARGE',
    'AIRPORT_FEE',
    'TIPS',
    'DRIVER_PAY',
    'SHARED_REQUEST_FLAG',
    'SHARED_MATCH_FLAG',
    'ACCESS_A_RIDE_FLAG',
    'WAV_REQUEST_FLAG',
    'WAV_MATCH_FLAG',
]


def show(session: Session, label: str, sql: str) -> None:
    start = time.monotonic()
    print(f'\n--- {label} ---', flush=True)
    rows = session.sql(sql).collect()
    for row in rows:
        print('  ' + '  '.join('NULL' if v is None else str(v) for v in row),
              flush=True)
    print(f'  ({time.monotonic() - start:.1f}s, {len(rows)} rows)', flush=True)


def keys(session: Session) -> None:
    show(
        session, 'trip_id uniqueness (matrix: key trips.trip_id)', f"""
        SELECT COUNT(*) AS rows_, COUNT(trip_id) AS non_null,
               COUNT(DISTINCT trip_id) AS distinct_, COUNT(*) -
               COUNT(DISTINCT trip_id) AS duplicates
        FROM {RAW}
    """)

    # Both directions, both edges. An orphan means a trip pointing at a zone the
    # graph does not hold; a zone with no trips is an entity that can never have
    # a non-empty target, which is what made three quarters of NOAA's entity set
    # dead weight (NOAA S4).
    show(
        session, 'pickup_zone_id -> zones.zone_id', f"""
        SELECT COUNT(DISTINCT t.pickup_zone_id) AS referenced,
               COUNT_IF(t.pickup_zone_id IS NULL) AS null_fkey,
               (SELECT COUNT(*) FROM {ZONES}) AS zones_,
               (SELECT COUNT(*) FROM {ZONES} z WHERE NOT EXISTS
                    (SELECT 1 FROM {RAW} r WHERE r.pickup_zone_id = z.zone_id))
                   AS zones_with_no_pickup
        FROM {RAW} t
    """)
    show(
        session, 'dropoff_zone_id -> zones.zone_id', f"""
        SELECT COUNT(DISTINCT t.dropoff_zone_id) AS referenced,
               COUNT_IF(t.dropoff_zone_id IS NULL) AS null_fkey,
               (SELECT COUNT(*) FROM {ZONES} z WHERE NOT EXISTS
                    (SELECT 1 FROM {RAW} r WHERE r.dropoff_zone_id = z.zone_id))
                   AS zones_with_no_dropoff
        FROM {RAW} t
    """)
    show(
        session, 'orphan zone references (fkey not in zones)', f"""
        SELECT
          (SELECT COUNT(*) FROM {RAW} t LEFT JOIN {ZONES} z
             ON t.pickup_zone_id = z.zone_id
           WHERE t.pickup_zone_id IS NOT NULL AND z.zone_id IS NULL)
              AS orphan_pickup,
          (SELECT COUNT(*) FROM {RAW} t LEFT JOIN {ZONES} z
             ON t.dropoff_zone_id = z.zone_id
           WHERE t.dropoff_zone_id IS NOT NULL AND z.zone_id IS NULL)
              AS orphan_dropoff
    """)
    show(
        session, 'provider_id -> providers.provider_id', f"""
        SELECT COUNT(DISTINCT provider_id) AS referenced,
               COUNT_IF(provider_id IS NULL) AS null_fkey,
               (SELECT COUNT(*) FROM {PROVIDERS}) AS providers_
        FROM {RAW}
    """)
    show(
        session, 'trips per provider', f"""
        SELECT provider_id, COUNT(*) AS n,
               MIN(pickup_datetime), MAX(pickup_datetime)
        FROM {RAW} GROUP BY 1 ORDER BY 2 DESC
    """)


def times(session: Session) -> None:
    show(
        session, 'pickup_datetime span and density', f"""
        SELECT MIN(pickup_datetime), MAX(pickup_datetime),
               COUNT(DISTINCT pickup_datetime::DATE) AS distinct_days,
               DATEDIFF('day', MIN(pickup_datetime), MAX(pickup_datetime)) + 1
                   AS calendar_days,
               COUNT_IF(pickup_datetime IS NULL) AS null_pickup
        FROM {RAW}
    """)
    show(
        session, 'trips per month', f"""
        SELECT DATE_TRUNC('month', pickup_datetime)::DATE AS month_,
               COUNT(*) AS trips, COUNT(DISTINCT pickup_zone_id) AS zones_,
               COUNT(DISTINCT provider_id) AS providers_
        FROM {RAW} GROUP BY 1 ORDER BY 1
    """)
    # Do the four datetime columns stand in the order their names imply, and
    # how often is each null? `on_scene_datetime` is the one that will not be.
    show(
        session, 'datetime column ordering and nulls', f"""
        SELECT COUNT_IF(request_datetime IS NULL)     AS null_request,
               COUNT_IF(on_scene_datetime IS NULL)    AS null_on_scene,
               COUNT_IF(dropoff_datetime IS NULL)     AS null_dropoff,
               COUNT_IF(request_datetime > pickup_datetime)  AS request_after,
               COUNT_IF(dropoff_datetime < pickup_datetime)  AS dropoff_before,
               MIN(request_datetime), MAX(dropoff_datetime)
        FROM {RAW}
    """)


def columns(session: Session) -> None:
    print(f'\n{"column":<24} {"non-null":>16} {"share":>9} {"distinct":>14}',
          flush=True)
    for name in COLUMNS:
        row = session.sql(f"""
            SELECT COUNT({name}), COUNT(*), APPROX_COUNT_DISTINCT({name})
            FROM {RAW}
        """).collect()[0]
        print(
            f'{name:<24} {row[0]:>16,} {row[0] / row[1]:>8.2%} '
            f'{row[2]:>14,}', flush=True)


def parts(session: Session) -> None:
    # Confirms at full scale what `probe_nyc_raw.py` established from five part
    # files: the fixture is globally time-sorted, so a prefix slice is a month
    # rather than a sample, and every tier predicate has to be a hash or an
    # explicit window.
    show(
        session, 'trip_id prefix vs pickup month (parts are time-sorted)', f"""
        SELECT LEFT(trip_id, 7) AS trip_id_prefix, COUNT(*) AS n,
               MIN(pickup_datetime), MAX(pickup_datetime)
        FROM {RAW} GROUP BY 1 ORDER BY 1 LIMIT 8
    """)
    show(
        session, 'hash bucket uniformity (the tier predicate)', f"""
        SELECT MOD(ABS(HASH(trip_id)), 10) AS bucket, COUNT(*) AS n,
               MIN(pickup_datetime), MAX(pickup_datetime)
        FROM {RAW} GROUP BY 1 ORDER BY 1
    """)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ('keys', 'time', 'columns', 'parts'):
        parser.add_argument(f'--{name}', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--warehouse', default='KUMO_L')
    args = parser.parse_args()

    session = Session.builder.config('connection_name', 'tlb').create()
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()

    if args.all or args.keys:
        keys(session)
    if args.all or args.time:
        times(session)
    if args.all or args.columns:
        columns(session)
    if args.all or args.parts:
        parts(session)


if __name__ == '__main__':
    main()
