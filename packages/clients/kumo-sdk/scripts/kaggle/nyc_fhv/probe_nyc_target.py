# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What the three NYC FHV targets are worth with **no model at all**.

Every fixture in this series so far has had a structural channel that accounted
for most of its headline score -- row 2's flow-duration proxy, row 4's
single-label capture sessions, NOAA's 94%-recall persistence baseline. Quoting a
model's MAE or MAP without the corresponding no-model number has been
misleading on all five. This computes the number first, in SQL, over the whole
745-million-row fixture rather than over a tier.

``--leak``       Row 9 only, and it is the important one. ``trip_seconds``
                 looks like it is *defined* as ``dropoff_datetime -
                 pickup_datetime``. ``kumo_core/pquery.py`` excludes the target
                 column from the entity table's own features but nothing else,
                 so if the identity holds, the published query on the published
                 table is a subtraction with a GNN wrapped around it. This
                 measures exactly how often it holds and how big the residual
                 is when it does not.
``--duration``   Row 9's baselines: the spread of ``trip_seconds``, and how much
                 of it a ``GROUP BY (pickup_zone, dropoff_zone)`` median
                 recovers. That group-by uses precisely the graph structure the
                 GNN is being paid to learn, so it is the number the model has
                 to beat.
``--demand``     Row 12's baseline: how far a zone's daily pickup count actually
                 moves day to day, which is what makes an MAE on a count
                 readable. Also the weekly cycle, because a one-day persistence
                 baseline on a series with a strong weekly cycle is a weak
                 baseline and a seven-day-lag one is the honest floor.
``--destinations`` Row 16's baseline: how much of a zone's next-7-day
                 destination set is just its previous 7 days, by recall and
                 Jaccard -- NOAA S4's measurement transposed to zone pairs.
"""
import argparse

from snowflake.snowpark import Session

RAW = 'HYU.NYC_FHV.TRIPS_RAW'
ZONES = 'HYU.NYC_FHV.ZONES_RAW'

#: An anchor deep inside the fixture rather than at its end: 2022-06-01 is after
#: the COVID trough and before the last month, so both the before and after
#: windows are full.
ANCHOR = '2022-06-01'


def show(session: Session, label: str, sql: str) -> None:
    print(f'\n--- {label} ---', flush=True)
    for row in session.sql(sql).collect():
        print('  ' + '  '.join('NULL' if v is None else str(v) for v in row),
              flush=True)


def leak(session: Session) -> None:
    # If `elapsed` and `trip_seconds` agree, row 9 is not a prediction problem
    # when `dropoff_datetime` is a feature. Reported as an exact-match rate and
    # as the distribution of the residual, because "almost always exact" and
    # "always within a minute" are different findings.
    show(
        session, 'trip_seconds vs dropoff - pickup, over all 745M trips', f"""
        SELECT COUNT(*) AS rows_,
               COUNT_IF(diff = 0) AS exact,
               COUNT_IF(ABS(diff) <= 1) AS within_1s,
               COUNT_IF(ABS(diff) <= 60) AS within_60s,
               AVG(ABS(diff)) AS mean_abs_diff,
               APPROX_PERCENTILE(ABS(diff), 0.5) AS median_abs_diff,
               APPROX_PERCENTILE(ABS(diff), 0.99) AS p99_abs_diff,
               MAX(ABS(diff)) AS max_abs_diff
        FROM (SELECT DATEDIFF('second', pickup_datetime, dropoff_datetime)
                     - trip_seconds AS diff
              FROM {RAW})
    """)
    # The softer proxies. Fare is set from time and distance, so it is not an
    # identity but it is close enough that keeping it is worth stating.
    show(
        session, 'correlation of trip_seconds with the other columns', f"""
        SELECT CORR(trip_seconds, trip_miles)           AS r_miles,
               CORR(trip_seconds, base_passenger_fare)  AS r_fare,
               CORR(trip_seconds, driver_pay)           AS r_driver_pay,
               CORR(trip_seconds, tips)                 AS r_tips,
               CORR(trip_seconds,
                    DATEDIFF('second', pickup_datetime, dropoff_datetime))
                                                        AS r_elapsed
        FROM {RAW}
    """)


def duration(session: Session) -> None:
    show(
        session, 'trip_seconds distribution', f"""
        SELECT COUNT(*), COUNT_IF(trip_seconds IS NULL) AS null_,
               AVG(trip_seconds), STDDEV(trip_seconds),
               MIN(trip_seconds),
               APPROX_PERCENTILE(trip_seconds, 0.01),
               APPROX_PERCENTILE(trip_seconds, 0.5),
               APPROX_PERCENTILE(trip_seconds, 0.99),
               MAX(trip_seconds),
               COUNT_IF(trip_seconds = 0) AS zero_,
               COUNT_IF(trip_seconds > 14400) AS over_4h
        FROM {RAW}
    """)
    # The baseline the model has to beat: a group-by over the two zone keys,
    # which is exactly the two-hop structure the graph provides.
    show(
        session, 'MAE of the (pickup,dropoff) median, held out by hash', f"""
        WITH labeled AS (
            SELECT pickup_zone_id, dropoff_zone_id, trip_seconds
            FROM {RAW} WHERE MOD(ABS(HASH(trip_id || '/holdout')), 100) >= 20
        ), held AS (
            SELECT pickup_zone_id, dropoff_zone_id, trip_seconds
            FROM {RAW} WHERE MOD(ABS(HASH(trip_id || '/holdout')), 100) < 20
        ), pair AS (
            SELECT pickup_zone_id, dropoff_zone_id,
                   MEDIAN(trip_seconds) AS m, COUNT(*) AS n
            FROM labeled GROUP BY 1, 2
        ), overall AS (SELECT AVG(trip_seconds) AS mu FROM labeled)
        SELECT COUNT(*) AS scored,
               AVG(ABS(h.trip_seconds - o.mu))                  AS mae_constant,
               AVG(ABS(h.trip_seconds - COALESCE(p.m, o.mu)))   AS mae_zone_pair,
               SQRT(AVG(POWER(h.trip_seconds - COALESCE(p.m, o.mu), 2)))
                                                                AS rmse_zone_pair,
               COUNT_IF(p.m IS NULL)                            AS unseen_pairs
        FROM held h CROSS JOIN overall o
        LEFT JOIN pair p ON p.pickup_zone_id = h.pickup_zone_id
                        AND p.dropoff_zone_id = h.dropoff_zone_id
    """)


def demand(session: Session) -> None:
    show(
        session, 'daily pickup count per zone: spread and movement', f"""
        WITH daily AS (
            SELECT pickup_zone_id AS z, pickup_datetime::DATE AS d,
                   COUNT(*) AS n
            FROM {RAW} GROUP BY 1, 2
        ), lagged AS (
            SELECT z, d, n,
                   LAG(n) OVER (PARTITION BY z ORDER BY d) AS n_prev,
                   LAG(n, 7) OVER (PARTITION BY z ORDER BY d) AS n_week
            FROM daily
        )
        SELECT COUNT(*) AS zone_days, AVG(n) AS mean_n, STDDEV(n) AS sd_n,
               APPROX_PERCENTILE(n, 0.5) AS median_n, MAX(n) AS max_n,
               AVG(ABS(n - n_prev))  AS mae_persistence_1d,
               AVG(ABS(n - n_week))  AS mae_persistence_7d
        FROM lagged WHERE n_week IS NOT NULL
    """)
    # A one-day-lag baseline is weak on a series with a weekly cycle, so the
    # cycle is measured rather than assumed.
    show(
        session, 'trips by day of week', f"""
        SELECT DAYNAME(pickup_datetime) AS dow, COUNT(*) AS n
        FROM {RAW} GROUP BY 1 ORDER BY 2 DESC
    """)
    show(
        session, 'zones with no pickups on a given day (empty targets)', f"""
        WITH grid AS (
            SELECT z.zone_id, d.d
            FROM {ZONES} z CROSS JOIN
                 (SELECT DISTINCT pickup_datetime::DATE AS d FROM {RAW}
                  WHERE pickup_datetime >= '{ANCHOR}'
                    AND pickup_datetime < DATEADD(day, 28, '{ANCHOR}'::DATE)) d
        )
        SELECT COUNT(*) AS zone_days,
               COUNT_IF(t.n IS NULL OR t.n = 0) AS empty_,
               AVG(COALESCE(t.n, 0)) AS mean_n
        FROM grid g
        LEFT JOIN (SELECT pickup_zone_id, pickup_datetime::DATE AS d,
                          COUNT(*) AS n
                   FROM {RAW} GROUP BY 1, 2) t
               ON t.pickup_zone_id = g.zone_id AND t.d = g.d
    """)


def destinations(session: Session) -> None:
    # NOAA S4's stability measurement, transposed: how much of a zone's next
    # seven days of destinations is just its previous seven days.
    show(
        session, f'destination-set stability across {ANCHOR}', f"""
        WITH before_ AS (
            SELECT pickup_zone_id AS z,
                   ARRAY_AGG(DISTINCT dropoff_zone_id) AS s
            FROM {RAW}
            WHERE pickup_datetime > DATEADD(day, -7, '{ANCHOR}'::DATE)
              AND pickup_datetime <= '{ANCHOR}'
            GROUP BY 1
        ), after_ AS (
            SELECT pickup_zone_id AS z,
                   ARRAY_AGG(DISTINCT dropoff_zone_id) AS s
            FROM {RAW}
            WHERE pickup_datetime > '{ANCHOR}'
              AND pickup_datetime <= DATEADD(day, 7, '{ANCHOR}'::DATE)
            GROUP BY 1
        )
        SELECT COUNT(*) AS zones_,
               AVG(ARRAY_SIZE(a.s)) AS mean_destinations_after,
               MIN(ARRAY_SIZE(a.s)), MAX(ARRAY_SIZE(a.s)),
               AVG(ARRAY_SIZE(ARRAY_INTERSECTION(a.s, b.s))
                   / ARRAY_SIZE(a.s)) AS mean_recall_of_previous_week,
               AVG(ARRAY_SIZE(ARRAY_INTERSECTION(a.s, b.s))
                   / ARRAY_SIZE(ARRAY_DISTINCT(ARRAY_CAT(a.s, b.s))))
                   AS mean_jaccard,
               COUNT_IF(ARRAY_SIZE(ARRAY_INTERSECTION(a.s, b.s))
                        = ARRAY_SIZE(a.s)) AS fully_covered
        FROM after_ a JOIN before_ b USING (z)
    """)
    # `RANK TOP 20` against a destination set this size is the question of
    # whether the cut-off binds at all -- on NOAA `TOP 25` never bound, because
    # a station reported a median of three elements.
    show(
        session, 'destinations per zone in a 7-day window (does TOP 20 bind?)',
        f"""
        SELECT APPROX_PERCENTILE(n, 0.05), APPROX_PERCENTILE(n, 0.5),
               APPROX_PERCENTILE(n, 0.95), MAX(n), COUNT_IF(n > 20) AS over_20,
               COUNT(*) AS zones_
        FROM (SELECT pickup_zone_id, COUNT(DISTINCT dropoff_zone_id) AS n
              FROM {RAW}
              WHERE pickup_datetime > '{ANCHOR}'
                AND pickup_datetime <= DATEADD(day, 7, '{ANCHOR}'::DATE)
              GROUP BY 1)
    """)
    # The popularity floor: if a handful of zones absorb most drop-offs, a
    # single global ranking is already a strong ranker and the model has to
    # beat *that*, not chance.
    show(
        session, 'drop-off concentration (the popularity floor)', f"""
        SELECT dropoff_zone_id, COUNT(*) AS n,
               RATIO_TO_REPORT(COUNT(*)) OVER () AS share
        FROM {RAW} GROUP BY 1 ORDER BY 2 DESC LIMIT 12
    """)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ('leak', 'duration', 'demand', 'destinations'):
        parser.add_argument(f'--{name}', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--warehouse', default='KUMO_L')
    args = parser.parse_args()

    session = Session.builder.config('connection_name', 'tlb').create()
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()

    if args.all or args.leak:
        leak(session)
    if args.all or args.duration:
        duration(session)
    if args.all or args.demand:
        demand(session)
    if args.all or args.destinations:
        destinations(session)


if __name__ == '__main__':
    main()
