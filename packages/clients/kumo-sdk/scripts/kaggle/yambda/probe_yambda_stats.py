# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What the five Yambda targets are worth with no model at all.

Every previous block in this series was decided before AutoML started, by one
or two numbers that a single ``GROUP BY`` produces:

* NYC row 16 lost to a ``GROUP BY`` because its median entity reached **215 of
  264** candidates -- the task was saturated and ``RANK TOP 20`` was arithmetic,
  not ranking.
* BCCC row 14 was the informative one because its median entity reached **1 of
  407,244**.
* NOAA rows 5/8 were decided by a *stability* measurement: last month's element
  set recovered **94%** of next month's, so the model's whole job was already
  done by persistence.

This runs the same instruments on Yambda, over ``INTERACTIONS_RAW`` (the
four-part pool ``build_yambda_tables.py`` materialises), before any tier is
trained:

``--claims``      the matrix's own numbers, and the fixture's shape
``--saturation``  rows 13/15: truth-set size against the candidate universe,
                  and the popularity concentration a global ranking would exploit
``--stability``   rows 6/7 and 15: how much of the next window's target set the
                  *previous* window already contains -- the no-model ceiling
``--target``      row 10: the distribution of ``played_ratio_pct`` and the
                  structure of its nulls
"""
import argparse

from snowflake.snowpark import Session

SCHEMA = 'HYU.YAMBDA'
RAW = f'{SCHEMA}.INTERACTIONS_RAW'


def show(session: Session, title: str, sql: str, limit: int = 40) -> None:
    print(f'\n--- {title} ---', flush=True)
    try:
        rows = session.sql(sql).collect()
    except Exception as exc:  # noqa: BLE001
        print(f'    FAILED: {exc}')
        return
    if not rows:
        print('    (no rows)')
    for row in rows[:limit]:
        print('   ', '  '.join(f'{c}={row[c]!r}' for c in row.as_dict()))


def claims(session: Session) -> None:
    show(
        session, 'the pool, against the manifest', f"""
        SELECT COUNT(*) AS interactions,
               COUNT(DISTINCT user_id) AS users,
               COUNT(DISTINCT item_id) AS items,
               COUNT(DISTINCT interaction_id) AS keys,
               MIN(event_time) AS first_event, MAX(event_time) AS last_event,
               DATEDIFF('day', MIN(event_time), MAX(event_time)) AS days,
               COUNT(*) / COUNT(DISTINCT user_id) AS per_user
        FROM {RAW}""")
    show(
        session, 'event types over the pool', f"""
        SELECT event_type, COUNT(*) AS n,
               COUNT(*) / (SELECT COUNT(*) FROM {RAW}) AS share,
               COUNT(DISTINCT user_id) AS users,
               COUNT(DISTINCT item_id) AS items,
               COUNT_IF(is_organic = 1) AS organic,
               COUNT_IF(played_ratio_pct IS NULL) AS prp_null
        FROM {RAW} GROUP BY 1 ORDER BY 2 DESC""")
    show(
        session, 'is played_ratio_pct null exactly on non-listen events?', f"""
        SELECT COUNT_IF(played_ratio_pct IS NULL) AS prp_null,
               COUNT_IF(event_type <> 'listen') AS not_listen,
               COUNT_IF(played_ratio_pct IS NULL AND event_type = 'listen')
                   AS listen_but_null,
               COUNT_IF(played_ratio_pct IS NOT NULL
                        AND event_type <> 'listen') AS nonlisten_but_valued
        FROM {RAW}""")
    show(
        session, 'monthly volume (is the span uniform?)', f"""
        SELECT DATE_TRUNC('month', event_time) AS month, COUNT(*) AS n,
               COUNT(DISTINCT user_id) AS users
        FROM {RAW} GROUP BY 1 ORDER BY 1""", limit=80)


def saturation(session: Session, window: int) -> None:
    r"""Rows 13/15: how hard is the ranking problem, before any model."""
    show(
        session, 'distinct items per user, over the whole pool', f"""
        SELECT COUNT(*) AS users, MIN(n) AS mn,
               APPROX_PERCENTILE(n, 0.05) AS p05,
               APPROX_PERCENTILE(n, 0.25) AS p25,
               APPROX_PERCENTILE(n, 0.50) AS median,
               APPROX_PERCENTILE(n, 0.75) AS p75,
               APPROX_PERCENTILE(n, 0.95) AS p95,
               MAX(n) AS mx, AVG(n) AS mean
        FROM (SELECT user_id, COUNT(DISTINCT item_id) AS n
              FROM {RAW} GROUP BY 1)""")
    show(
        session, f'distinct items per user in one {window}-day window '
        '(row 15 truth-set size)', f"""
        WITH bounds AS (SELECT MAX(event_time) AS last_event FROM {RAW})
        SELECT COUNT(*) AS active_users, MIN(n) AS mn,
               APPROX_PERCENTILE(n, 0.25) AS p25,
               APPROX_PERCENTILE(n, 0.50) AS median,
               APPROX_PERCENTILE(n, 0.75) AS p75,
               APPROX_PERCENTILE(n, 0.95) AS p95,
               MAX(n) AS mx, AVG(n) AS mean
        FROM (SELECT user_id, COUNT(DISTINCT item_id) AS n
              FROM {RAW}, bounds
              WHERE event_time > DATEADD(day, -{window}, last_event)
              GROUP BY 1)""")
    show(
        session, 'users per item, and the popularity concentration a global '
        'ranking would exploit', f"""
        WITH per_item AS (SELECT item_id, COUNT(DISTINCT user_id) AS users
                          FROM {RAW} GROUP BY 1),
             ranked AS (SELECT item_id, users,
                               ROW_NUMBER() OVER (ORDER BY users DESC) AS rk
                        FROM per_item),
             total AS (SELECT SUM(users) AS s FROM per_item)
        SELECT (SELECT COUNT(*) FROM per_item) AS distinct_items,
               (SELECT APPROX_PERCENTILE(users, 0.5) FROM per_item) AS med_users_per_item,
               (SELECT MAX(users) FROM per_item) AS max_users_per_item,
               (SELECT SUM(users) FROM ranked WHERE rk <= 10) / (SELECT s FROM total) AS top10_share,
               (SELECT SUM(users) FROM ranked WHERE rk <= 100) / (SELECT s FROM total) AS top100_share,
               (SELECT SUM(users) FROM ranked WHERE rk <= 1000)
                   / (SELECT s FROM total) AS top1000_share,
               (SELECT COUNT(*) FROM per_item WHERE users = 1) AS items_with_one_user"""
    )


def stability(session: Session, window: int) -> None:
    r"""The no-model ceiling: how much of the next window is in the last one.

    NOAA's version of this decided that block before AutoML started.  Two
    targets are measured -- rows 6/7's ``event_type`` sets and rows 13/15's
    ``item_id`` sets -- because they should behave completely differently: a
    user's *event-type* mix is nearly constant, while re-listening to the same
    *track* in the next 30 days is a real prediction.
    """
    for column, label in (('event_type', 'rows 6/7'), ('item_id', 'row 15')):
        show(
            session, f'{label}: {column} set stability across a {window}-day '
            'boundary', f"""
            WITH bounds AS (SELECT MAX(event_time) AS last_event FROM {RAW}),
                 anchor AS (SELECT DATEADD(day, -{window}, last_event) AS a,
                                   last_event FROM bounds),
                 nxt AS (SELECT user_id, ARRAY_AGG(DISTINCT {column}) AS s
                         FROM {RAW}, anchor
                         WHERE event_time > a AND event_time <= last_event
                         GROUP BY 1),
                 prv AS (SELECT user_id, ARRAY_AGG(DISTINCT {column}) AS s
                         FROM {RAW}, anchor
                         WHERE event_time > DATEADD(day, -{window}, a)
                           AND event_time <= a
                         GROUP BY 1),
                 joined AS (
                     SELECT n.user_id, ARRAY_SIZE(n.s) AS n_after,
                            ARRAY_SIZE(ARRAY_INTERSECTION(n.s, p.s)) AS shared,
                            ARRAY_SIZE(ARRAY_DISTINCT(ARRAY_CAT(n.s, p.s)))
                                AS n_union
                     FROM nxt n LEFT JOIN prv p USING (user_id))
            SELECT COUNT(*) AS active_users,
                   COUNT_IF(shared IS NULL) AS no_history,
                   AVG(n_after) AS mean_target_size,
                   AVG(COALESCE(shared, 0) / NULLIF(n_after, 0)) AS mean_recall,
                   AVG(COALESCE(shared, 0) / NULLIF(n_union, 0)) AS mean_jaccard,
                   COUNT_IF(shared = n_union) AS identical_sets
            FROM joined""")


def target(session: Session) -> None:
    r"""Row 10: what ``played_ratio_pct`` looks like and where its nulls are."""
    show(
        session, 'played_ratio_pct distribution over the pool', f"""
        SELECT COUNT(*) AS n, COUNT_IF(played_ratio_pct IS NULL) AS nulls,
               MIN(played_ratio_pct) AS mn,
               APPROX_PERCENTILE(played_ratio_pct, 0.01) AS p01,
               APPROX_PERCENTILE(played_ratio_pct, 0.25) AS p25,
               APPROX_PERCENTILE(played_ratio_pct, 0.50) AS median,
               APPROX_PERCENTILE(played_ratio_pct, 0.75) AS p75,
               APPROX_PERCENTILE(played_ratio_pct, 0.99) AS p99,
               MAX(played_ratio_pct) AS mx, AVG(played_ratio_pct) AS mean,
               STDDEV(played_ratio_pct) AS sd
        FROM {RAW}""")
    show(
        session, 'the target is bimodal -- how bimodal', f"""
        SELECT COUNT_IF(played_ratio_pct = 0) AS exactly_0,
               COUNT_IF(played_ratio_pct BETWEEN 1 AND 9) AS one_to_9,
               COUNT_IF(played_ratio_pct BETWEEN 10 AND 89) AS ten_to_89,
               COUNT_IF(played_ratio_pct BETWEEN 90 AND 99) AS ninety_to_99,
               COUNT_IF(played_ratio_pct = 100) AS exactly_100,
               COUNT_IF(played_ratio_pct > 100) AS over_100
        FROM {RAW} WHERE played_ratio_pct IS NOT NULL""")
    show(
        session, 'how much of the variance a per-item / per-user median '
        'already explains (the structural baselines)', f"""
        WITH labeled AS (SELECT * FROM {RAW} WHERE played_ratio_pct IS NOT NULL),
             g AS (SELECT AVG(played_ratio_pct) AS m FROM labeled),
             im AS (SELECT item_id, MEDIAN(played_ratio_pct) AS v
                    FROM labeled GROUP BY 1),
             um AS (SELECT user_id, MEDIAN(played_ratio_pct) AS v
                    FROM labeled GROUP BY 1)
        SELECT AVG(ABS(l.played_ratio_pct - g.m)) AS mae_constant,
               AVG(ABS(l.played_ratio_pct - im.v)) AS mae_item_median,
               AVG(ABS(l.played_ratio_pct - um.v)) AS mae_user_median,
               1 - SUM(POWER(l.played_ratio_pct - im.v, 2))
                   / SUM(POWER(l.played_ratio_pct - g.m, 2)) AS r2_item_median,
               1 - SUM(POWER(l.played_ratio_pct - um.v, 2))
                   / SUM(POWER(l.played_ratio_pct - g.m, 2)) AS r2_user_median
        FROM labeled l
        JOIN im ON im.item_id = l.item_id
        JOIN um ON um.user_id = l.user_id
        CROSS JOIN g""")
    show(
        session, 'is track_length_seconds a feature or a leak?', f"""
        SELECT CORR(played_ratio_pct, track_length_seconds) AS corr_len,
               CORR(played_ratio_pct, is_organic) AS corr_organic
        FROM {RAW} WHERE played_ratio_pct IS NOT NULL""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--claims', action='store_true')
    parser.add_argument('--saturation', action='store_true')
    parser.add_argument('--stability', action='store_true')
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--window-days', type=int, default=30)
    parser.add_argument('--warehouse', default='KUMO_L')
    args = parser.parse_args()

    session = Session.builder.config('connection_name', 'tlb').create()
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()

    if args.all or args.claims:
        claims(session)
    if args.all or args.saturation:
        saturation(session, args.window_days)
    if args.all or args.stability:
        stability(session, args.window_days)
    if args.all or args.target:
        target(session)


if __name__ == '__main__':
    main()
