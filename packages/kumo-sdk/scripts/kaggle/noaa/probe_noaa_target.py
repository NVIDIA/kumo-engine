# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What rows 5, 8 and 11 are actually asking, measured on the fixture.

Rows 2, 3 and 4 each turned out to have a structural answer that most of the
headline score was really measuring -- "which capture is this?" in rows 2 and 4,
"which class is in this time window?" in row 3. The equivalent question here has
a different shape, because the target is an aggregate over a *future* window
rather than a static column, and because the entity is a physical instrument
whose behaviour barely changes from month to month.

So the questions this answers are:

``--stability``
    How much of the row 5/8 target is simply last month repeated? A weather
    station that recorded TMAX, TMIN and PRCP in April records them in May. If
    the Jaccard overlap between consecutive windows is near 1, the task is
    "reproduce this station's instrument list", the persistence baseline is the
    number that matters, and any model score has to be read against it.
``--coverage``
    How many stations are reporting at each point in time, and over what span.
    NOAA runs from 1763 to 2019 and the station population grows by four orders
    of magnitude across it, so a split cut by row quantile puts almost every
    modern station on one side. This is the row 3 S3 / row 4 S5 check adapted:
    the thing correlated with time here is not the label but *which entities
    exist at all*.
``--tmax``
    Row 11's target. GHCN's missing sentinel is ``-9999`` and it survives into
    ``value``; a per-week mean over a station that reported one sentinel is
    ruined. This measures how much of the fixture that is, and how far a
    one-week-ahead TMAX mean actually moves -- which is the only thing that
    makes an MAE in tenths of a degree interpretable.
``--labels``
    Label cardinality inside a target window, which decides whether
    ``TOP 25`` is a binding constraint or decoration.
"""
import argparse

from snowflake.snowpark import Session

SCHEMA = 'HYU.NOAA'
RAW = f'{SCHEMA}.OBSERVATIONS_RAW'

#: The anchor ``run_noaa_local.py`` scores at: the last observation in the
#: fixture (2019-06-03) minus one 30-day target window.
ANCHOR = '2019-05-04'
WINDOW_DAYS = 30


def show(session: Session, title: str, sql: str) -> None:
    print(f'\n--- {title} ---', flush=True)
    for row in session.sql(sql).collect():
        print(
            '  ' + '  '.join(f'{v:,}' if isinstance(v, int) else str(v)
                             for v in row), flush=True)


def stability(session: Session, anchor: str, window: int) -> None:
    r"""Jaccard overlap between the target window and the window before it.

    This is the no-model ceiling for rows 5 and 8 stated as a property of the
    data rather than as a score. ``run_noaa_local.py`` computes the persistence
    *baseline* from real predictions; this computes the quantity that baseline
    is exploiting, over every station at once, so the two can be checked against
    each other.
    """
    show(
        session, f'element-set overlap across the {window}-day boundary at '
        f'{anchor}', f"""
        WITH after AS (
            SELECT station_id, ARRAY_AGG(DISTINCT element) AS e
            FROM {RAW}
            WHERE event_time > '{anchor}'
              AND event_time <= DATEADD(day, {window}, '{anchor}')
            GROUP BY 1),
        before AS (
            SELECT station_id, ARRAY_AGG(DISTINCT element) AS e
            FROM {RAW}
            WHERE event_time <= '{anchor}'
              AND event_time > DATEADD(day, -{window}, '{anchor}')
            GROUP BY 1),
        joined AS (
            SELECT a.station_id,
                   ARRAY_SIZE(a.e) AS n_after,
                   ARRAY_SIZE(ARRAY_INTERSECTION(a.e, b.e)) AS n_shared,
                   ARRAY_SIZE(ARRAY_DISTINCT(ARRAY_CAT(a.e, b.e))) AS n_union
            FROM after a LEFT JOIN before b USING (station_id))
        SELECT COUNT(*)                                        AS stations,
               COUNT_IF(n_shared IS NULL)                      AS no_prior,
               AVG(n_after)                                    AS mean_labels,
               AVG(COALESCE(n_shared, 0) / n_after)            AS mean_recall,
               AVG(COALESCE(n_shared, 0) / NULLIF(n_union, 0)) AS mean_jaccard,
               COUNT_IF(n_shared = n_union)                    AS identical
        FROM joined
    """)


def coverage(session: Session) -> None:
    r"""Active stations and observation density over time.

    Printed by decade and then, because the split cuts inside the last few
    percent of rows, by year over the final stretch.
    """
    show(
        session, 'active stations per decade', f"""
        SELECT FLOOR(YEAR(event_time) / 10) * 10 AS decade,
               COUNT(DISTINCT station_id) AS stations, COUNT(*) AS rows_
        FROM {RAW} GROUP BY 1 ORDER BY 1
    """)
    show(
        session, 'active stations per year, 2010 onwards', f"""
        SELECT YEAR(event_time) AS yr, COUNT(DISTINCT station_id) AS stations,
               COUNT(*) AS rows_, COUNT(DISTINCT element) AS elements
        FROM {RAW} WHERE event_time >= '2010-01-01' GROUP BY 1 ORDER BY 1
    """)
    # A station that stopped reporting in 1900 is still an entity of the task
    # and still gets a prediction; only its subgraph is empty near the anchor.
    show(
        session, 'stations by last year they reported', f"""
        SELECT FLOOR(yr / 10) * 10 AS last_decade, COUNT(*) AS stations
        FROM (SELECT station_id, YEAR(MAX(event_time)) AS yr
              FROM {RAW} GROUP BY 1)
        GROUP BY 1 ORDER BY 1
    """)


def tmax(session: Session, anchor: str) -> None:
    r"""Row 11's target: contamination first, then how far it moves.

    ``-9999`` is GHCN's missing sentinel and it is present in ``value`` rather
    than encoded as SQL ``NULL``, so it enters ``AVG(value)`` at face value. A
    single sentinel inside a seven-day window drags that week's mean by about
    -1,400 tenths of a degree, which is 140 C. Anything quoted as an MAE has to
    be quoted knowing how often that happens.
    """
    show(
        session, 'TMAX value sanity over the whole fixture', f"""
        SELECT COUNT(*)                                  AS n,
               COUNT_IF(value <= -9990)                  AS sentinel_low,
               COUNT_IF(value < -700 AND value > -9990)  AS implausible_low,
               COUNT_IF(value > 600)                     AS implausible_high,
               MIN(value), MAX(value), AVG(value), STDDEV(value)
        FROM {RAW} WHERE element = 'TMAX'
    """)
    show(
        session, 'quality_flag on TMAX rows', f"""
        SELECT COALESCE(quality_flag, '(null)') AS flag, COUNT(*) AS n
        FROM {RAW} WHERE element = 'TMAX' GROUP BY 1 ORDER BY 2 DESC LIMIT 15
    """)
    # Week-on-week movement is what a forecast has to beat. If the persistence
    # error is small the task is easy and the MAE should be read against it, not
    # against zero.
    show(
        session, f'week-on-week TMAX mean movement around {anchor}', f"""
        WITH w AS (
            SELECT station_id,
                   AVG(IFF(event_time > '{anchor}', value, NULL)) AS after,
                   AVG(IFF(event_time <= '{anchor}', value, NULL)) AS before
            FROM {RAW}
            WHERE element = 'TMAX'
              AND event_time > DATEADD(day, -7, '{anchor}')
              AND event_time <= DATEADD(day, 7, '{anchor}')
              AND value BETWEEN -700 AND 600
            GROUP BY 1)
        SELECT COUNT(*) AS stations,
               COUNT_IF(before IS NOT NULL AND after IS NOT NULL) AS both,
               AVG(ABS(after - before))    AS mean_abs_change,
               APPROX_PERCENTILE(ABS(after - before), 0.5) AS median_abs_change,
               STDDEV(after)               AS sd_of_target
        FROM w
    """)


def labels(session: Session, anchor: str, window: int) -> None:
    r"""How many distinct elements a station reports inside one target window.

    ``CLASSIFY TOP 25`` / ``RANK TOP 25`` cap the label space at 25 of the
    fixture's 137 elements. If the 99th percentile of labels-per-window is well
    under 25 then the cap never binds on any individual entity and its only
    effect is on the label *vocabulary*.
    """
    show(
        session, f'labels per station in the {window} days after {anchor}',
        f"""
        SELECT COUNT(*) AS stations, MIN(k), APPROX_PERCENTILE(k, 0.5),
               APPROX_PERCENTILE(k, 0.9), APPROX_PERCENTILE(k, 0.99), MAX(k),
               AVG(k)
        FROM (SELECT station_id, COUNT(DISTINCT element) AS k
              FROM {RAW}
              WHERE event_time > '{anchor}'
                AND event_time <= DATEADD(day, {window}, '{anchor}')
              GROUP BY 1)
    """)
    show(
        session, 'element frequency inside that window (top 30)', f"""
        SELECT element, COUNT(DISTINCT station_id) AS stations, COUNT(*) AS n
        FROM {RAW}
        WHERE event_time > '{anchor}'
          AND event_time <= DATEADD(day, {window}, '{anchor}')
        GROUP BY 1 ORDER BY 2 DESC LIMIT 30
    """)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--warehouse', default='KUMO_L')
    parser.add_argument('--anchor', default=ANCHOR)
    parser.add_argument('--window-days', type=int, default=WINDOW_DAYS)
    for name in ('stability', 'coverage', 'tmax', 'labels'):
        parser.add_argument(f'--{name}', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    session = Session.builder.config('connection_name', 'tlb').create()
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()

    checks = [
        ('stability', lambda s: stability(s, args.anchor, args.window_days)),
        ('coverage', coverage),
        ('tmax', lambda s: tmax(s, args.anchor)),
        ('labels', lambda s: labels(s, args.anchor, args.window_days)),
    ]
    selected = [(n, f) for n, f in checks if args.all or getattr(args, n)]
    for name, check in (selected or checks):
        print(f'\n{"=" * 70}\n{name}\n{"=" * 70}', flush=True)
        check(session)


if __name__ == '__main__':
    main()
