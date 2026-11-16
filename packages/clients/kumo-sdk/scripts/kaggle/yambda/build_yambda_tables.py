# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialise Yambda-5B tier tables in Snowflake for matrix rows 6/7/10/13/15.

The fixture is **4,796,250,510 rows / 101.7 GiB** across three tables, which is
5.5x NOAA's row count and 6.4x NYC's stored bytes -- the largest in the matrix
by both measures.  Nothing on a 15 GB box reaches it, so every run here is a
tier, and the tier definition is the most consequential decision in the block.

**One tier family, cut on the user.**  Four of the five rows (6, 7, 13, 15) have
``users.user_id`` as the entity and an *aggregate* over ``interactions`` as the
target, so -- exactly as NOAA S6 and BCCC row 14 S6 established -- thinning a
user's interactions **changes that user's label**.  The predicate is therefore

    MOD(ABS(HASH(user_id)), 1000000) < ppm

so a user is wholly in or wholly out and the tiers nest.  Row 10's entity is the
fact table, where a row sample would have been correct, but it is run on the
same user tiers anyway: the point of row 10 is a *graph* model of a listening
event, and a uniform row sample would leave every user with 1/n of their history
as context.  Stated so the choice is visible rather than inherited.

**Two derived table sets, because two rows need holdouts:**

* ``--holdout-pct`` (row 10): blanks ``played_ratio_pct`` on a salted hash of
  ``interaction_id`` and writes the truth to a side table.  Row 10's target is a
  *static column*, so its compiled prediction plan carries an
  ``is_null`` filter and returns **only** blanked rows -- verified in
  ``probe_yambda_query.py`` before any data existed.  The salt matters: NYC row 2
  nested two thresholds on one hash and ``MOD(h, 10000) < 5`` implies
  ``MOD(h, 100) < 5``, so the holdout swallowed the whole slice.
* ``--lp-holdout`` (row 13): a two-sided holdout over distinct ``(user, item)``
  pairs.  Row 13 is a *static* aggregate whose target is "every item this user
  ever touched", which is exactly what the graph contains -- scoring against it
  without a holdout measures memorisation.  BCCC row 14 S6's correction is
  carried: every user gives up at least one item **and keeps at least one**, and
  **every held-out item keeps at least one other user**, or half the ground truth
  is unreachable by construction.

Everything reads the stage exactly once, into ``INTERACTIONS_RAW``.  Row 4 S2
measured a narrow re-read of a variant-typed parquet stage costing as much as a
wide one -- the ``$1:<field>`` decode dominates, not the projection -- so the
raw pass takes every column on the first go.
"""
import argparse
import time
from typing import Any, List, Tuple

from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.YAMBDA_100GB'
FF = 'HYU.KAGGLE_FIXTURES.PARQUET_FF'
SCHEMA = 'HYU.YAMBDA'
RAW = f'{SCHEMA}.INTERACTIONS_RAW'
USERS_RAW = f'{SCHEMA}.USERS_RAW'
ITEMS_RAW = f'{SCHEMA}.ITEMS_RAW'

BUCKETS = 10_000

#: **The raw pool is four of the ninety-six published parts.**
#:
#: ``probe_yambda_raw.py`` measured the fixture's physical layout:
#: ``part-00000`` holds exactly 50,000,000 rows, ``interaction_id`` 0 ..
#: 49,999,999, and ``user_id`` 1 .. 10,423 -- **contiguous, sorted, complete**.
#: So each part is a contiguous block of ~10,420 whole users, and reading four
#: parts spread across the range costs a **4.6 GB** scan instead of 109 GB for
#: the same ~41,700 users a hash predicate over the whole stage would have had
#: to scan everything to find.  Every previous block in this series paid the
#: full-stage scan because its fixture was not laid out this way; this one is,
#: and the layout was measured rather than assumed.
#:
#: The parts are spread rather than taken as a prefix so that any ordering in
#: ``user_id`` (activity, signup date) cannot be mistaken for the fixture.  Both
#: the pool's mean interactions-per-user and its event-type mix are checked
#: against the manifest's global figures after the build.
RAW_PARTS = ['part-00000', 'part-00024', 'part-00048', 'part-00072']

#: parts per 10,000 of the raw pool's users, BCCC row 14 S6's scheme.  The pool
#: is ~41,700 users and 4,785,859,887 / 1,000,000 = **4,786 interactions per
#: user**, so a tier's interaction count is ~4,786 x its user count -- two
#: orders of magnitude denser per entity than any previous fixture, and the
#: reason the user counts here look so much smaller than NOAA's or BCCC's.
SCALES = {
    'y_tiny': 50,  # ~209 users, ~1.0M interactions
    'y_smoke': 250,  # ~1,042 users, ~5.0M
    'y_small': 1_000,  # ~4,170 users, ~20M
    'y_part': 3_000,  # ~12,510 users, ~60M
    'y_pool': 10_000,  # the whole raw pool, ~41,700 users, ~200M
}

#: ``event_time`` arrives as ``NUMBER(38,0)`` with no epoch: the values run
#: 50,000 .. 130,000,000,000 and are all multiples of 25,000.
#: ``--time-divisor`` converts to seconds before ``TO_TIMESTAMP_NTZ``.
#:
#: **1000 (milliseconds) is inferred, not published.**  Within one user's
#: consecutive rows the gaps are 550,000 / 700,000 / 750,000 / 800,000 /
#: 1,150,000 units against ``track_length_seconds x played_ratio_pct`` of
#: 107 / 145 / 160 / 230 seconds actually played.  Under milliseconds every gap
#: comfortably exceeds the audio it must contain; under any finer unit
#: (25,000/s, microseconds) several gaps are *shorter than the track the row
#: says was played to 100%*, which is impossible.  That bounds the unit from
#: below at milliseconds.  ``event_time_raw`` is carried on the raw table so the
#: cast can be redone without re-reading the stage.
DEFAULT_TIME_DIVISOR = 1000


def run(session: Session, label: str, sql: str) -> Any:
    start = time.time()
    print(f'[{label}] ...', flush=True)
    result = session.sql(sql).collect()
    print(f'[{label}] done in {time.time() - start:.1f}s', flush=True)
    return result


def table_exists(session: Session, fqn: str) -> bool:
    try:
        session.sql(f'SELECT 1 FROM {fqn} LIMIT 1').collect()
        return True
    except Exception:  # noqa: BLE001
        return False


def build_raw(session: Session, divisor: int) -> None:
    r"""Read the selected parquet parts into typed native tables.

    ``user_bucket`` is precomputed so every tier build is an integer compare
    rather than a re-hash, ``event_time`` is cast to a real ``TIMESTAMP_NTZ``
    and ``event_time_raw`` is carried beside it so the unit can be revisited
    without re-reading the stage.

    **Boundary users are dropped.**  A part is a contiguous run of 50,000,000
    rows, so the first and last ``user_id`` in each part may have the rest of
    their history in the neighbouring part.  A truncated user is a *corrupted
    label* for rows 6, 7, 13 and 15, whose targets are aggregates over the whole
    of a user's interactions -- the same reason the tiers cut on the user in the
    first place.  Eight users out of ~41,700 is a rounding error to lose and a
    real defect to keep.
    """
    files = ' OR '.join(f"METADATA$FILENAME LIKE '%{part}%'"
                        for part in RAW_PARTS)
    run(
        session, 'users raw', f"""
        CREATE OR REPLACE TABLE {USERS_RAW} AS
        SELECT $1:user_id::NUMBER(38,0) AS user_id
        FROM '@{STAGE}/users/' (FILE_FORMAT => {FF})""")
    run(
        session, 'items raw', f"""
        CREATE OR REPLACE TABLE {ITEMS_RAW} AS
        SELECT $1:item_id::NUMBER(38,0) AS item_id
        FROM '@{STAGE}/items/' (FILE_FORMAT => {FF})""")
    run(
        session, f'interactions raw ({len(RAW_PARTS)} of 96 parts)', f"""
        CREATE OR REPLACE TABLE {SCHEMA}.__RAW_STAGED AS
        SELECT $1:interaction_id::NUMBER(38,0)       AS interaction_id,
               $1:user_id::NUMBER(38,0)              AS user_id,
               $1:item_id::NUMBER(38,0)              AS item_id,
               $1:event_time::NUMBER(38,0)           AS event_time_raw,
               $1:event_type::STRING                 AS event_type,
               $1:is_organic::NUMBER(38,0)           AS is_organic,
               $1:played_ratio_pct::NUMBER(38,0)     AS played_ratio_pct,
               $1:track_length_seconds::NUMBER(38,0) AS track_length_seconds,
               METADATA$FILENAME                     AS part
        FROM '@{STAGE}/interactions/' (FILE_FORMAT => {FF})
        WHERE {files}""")
    print('  parts read, and their user ranges:')
    boundary: List[int] = []
    for r in session.sql(f"""
            SELECT part, COUNT(*), MIN(user_id), MAX(user_id),
                   COUNT(DISTINCT user_id)
            FROM {SCHEMA}.__RAW_STAGED GROUP BY 1 ORDER BY 1""").collect():
        print(f'    {r[0].split("/")[-1]:<24} {r[1]:>12,} rows  '
              f'users {r[2]:,} .. {r[3]:,}  ({r[4]:,} distinct)')
        boundary.extend([int(r[2]), int(r[3])])
    excluded = ', '.join(str(u) for u in sorted(set(boundary)))
    print(f'  dropping {len(set(boundary))} boundary users: {excluded}')

    run(
        session, 'raw table', f"""
        CREATE OR REPLACE TABLE {RAW} AS
        SELECT interaction_id, user_id, item_id,
               TO_TIMESTAMP_NTZ(event_time_raw / {divisor}) AS event_time,
               event_time_raw, event_type, is_organic, played_ratio_pct,
               track_length_seconds,
               MOD(ABS(HASH(user_id)), {BUCKETS}) AS user_bucket
        FROM {SCHEMA}.__RAW_STAGED
        WHERE user_id NOT IN ({excluded})""")
    session.sql(f'DROP TABLE IF EXISTS {SCHEMA}.__RAW_STAGED').collect()

    row = session.sql(f"""
        SELECT COUNT(*), COUNT(DISTINCT user_id), COUNT(DISTINCT item_id),
               MIN(event_time), MAX(event_time),
               COUNT(DISTINCT interaction_id),
               DATEDIFF('day', MIN(event_time), MAX(event_time))
        FROM {RAW}""").collect()[0]
    print(f'  raw interactions        {row[0]:>16,}')
    print(f'  raw distinct users      {row[1]:>16,}')
    print(f'  raw distinct items      {row[2]:>16,}  of 9,390,623 published')
    print(f'  raw distinct keys       {row[5]:>16,}'
          f'   {"UNIQUE" if row[5] == row[0] else "*** NOT UNIQUE ***"}')
    print(f'  interactions per user   {row[0] / max(row[1], 1):>16,.0f}  '
          f'against 4,786 for the whole fixture')
    print(f'  span                    {row[3]}  ->  {row[4]}  '
          f'({row[6]:,} days at divisor {divisor})')


def tier_tables(session: Session, scale: str,
                bucket: int) -> Tuple[str, str, str]:
    r"""``USERS_<T>``, ``ITEMS_<T>`` and ``INTERACTIONS_<T>`` for one tier.

    The entity set is built **first** and the fact rows are filtered to it, so
    no interaction can point at a user the graph does not hold.  ``items`` is
    then narrowed to the items the tier's interactions actually reach: carrying
    all 9,390,623 published items would make every candidate outside the tier an
    isolated, featureless node that no model can rank and no baseline can
    propose -- BCCC row 14 S6's unreachable-truth failure, in the other
    direction.
    """
    tag = scale.upper()
    users = f'{SCHEMA}.USERS_{tag}'
    items = f'{SCHEMA}.ITEMS_{tag}'
    facts = f'{SCHEMA}.INTERACTIONS_{tag}'

    run(
        session, f'{scale} interactions', f"""
        CREATE OR REPLACE TABLE {facts} AS
        SELECT interaction_id, user_id, item_id, event_time, event_type,
               is_organic, played_ratio_pct, track_length_seconds
        FROM {RAW} WHERE user_bucket < {bucket}""")
    run(
        session, f'{scale} users', f"""
        CREATE OR REPLACE TABLE {users} AS
        SELECT DISTINCT user_id FROM {facts}""")
    run(
        session, f'{scale} items', f"""
        CREATE OR REPLACE TABLE {items} AS
        SELECT DISTINCT item_id FROM {facts}""")
    return users, items, facts


def describe(session: Session, users: str, items: str, facts: str) -> None:
    row = session.sql(f"""
        SELECT (SELECT COUNT(*) FROM {users}),
               (SELECT COUNT(*) FROM {items}),
               COUNT(*), COUNT(DISTINCT user_id), COUNT(DISTINCT item_id),
               MIN(event_time), MAX(event_time),
               DATEDIFF('day', MIN(event_time), MAX(event_time)),
               COUNT_IF(user_id IS NULL), COUNT_IF(item_id IS NULL),
               COUNT_IF(event_time IS NULL), COUNT_IF(played_ratio_pct IS NULL),
               COUNT(DISTINCT interaction_id)
        FROM {facts}""").collect()[0]
    print(f'\n  users in tier           {row[0]:>16,}')
    print(f'  items in tier           {row[1]:>16,}')
    print(f'  interactions            {row[2]:>16,}')
    print(f'  distinct users seen     {row[3]:>16,}')
    print(f'  distinct items seen     {row[4]:>16,}')
    print(f'  distinct keys           {row[12]:>16,}'
          f'   {"UNIQUE" if row[12] == row[2] else "*** NOT UNIQUE ***"}')
    print(f'  span                    {row[5]} -> {row[6]}  ({row[7]} days)')
    print(f'  null user / item / time {row[8]:,} / {row[9]:,} / {row[10]:,}')
    print(f'  null played_ratio_pct   {row[11]:>16,}  '
          f'({100.0 * row[11] / max(row[2], 1):.2f}%)')
    if row[2] == 0:
        raise SystemExit('degenerate slice: no interactions selected')

    print('\n  event_type over the tier')
    for r in session.sql(f"""
            SELECT event_type, COUNT(*) AS n,
                   COUNT(DISTINCT user_id) AS users,
                   COUNT_IF(played_ratio_pct IS NULL) AS prp_null
            FROM {facts} GROUP BY 1 ORDER BY 2 DESC""").collect():
        print(f'    {r[0]:<16} {r[1]:>14,} rows  {r[2]:>10,} users  '
              f'{r[3]:>14,} null prp')

    print('\n  interactions per user   min/p25/median/p75/p95/max/mean')
    r = session.sql(f"""
        SELECT MIN(n), APPROX_PERCENTILE(n,0.25), APPROX_PERCENTILE(n,0.5),
               APPROX_PERCENTILE(n,0.75), APPROX_PERCENTILE(n,0.95),
               MAX(n), AVG(n)
        FROM (SELECT user_id, COUNT(*) AS n FROM {facts} GROUP BY 1)
        """).collect()[0]
    print('    ' + '  '.join(f'{v:,.0f}' for v in r))

    print('\n  distinct items per user min/p25/median/p75/p95/max/mean')
    r = session.sql(f"""
        SELECT MIN(n), APPROX_PERCENTILE(n,0.25), APPROX_PERCENTILE(n,0.5),
               APPROX_PERCENTILE(n,0.75), APPROX_PERCENTILE(n,0.95),
               MAX(n), AVG(n)
        FROM (SELECT user_id, COUNT(DISTINCT item_id) AS n FROM {facts}
              GROUP BY 1)""").collect()[0]
    print('    ' + '  '.join(f'{v:,.0f}' for v in r))

    print('\n  distinct event_types per user  min/median/max/mean')
    r = session.sql(f"""
        SELECT MIN(n), APPROX_PERCENTILE(n,0.5), MAX(n), AVG(n)
        FROM (SELECT user_id, COUNT(DISTINCT event_type) AS n FROM {facts}
              GROUP BY 1)""").collect()[0]
    print('    ' + '  '.join(f'{v:,.2f}' for v in r))


def build_holdout(session: Session, facts: str, pct: int) -> Tuple[str, str]:
    r"""Row 10's static-column holdout: blank the target, keep truth aside.

    The hash is salted (``interaction_id || '/holdout'``) so that no tier
    predicate can imply the holdout predicate.  Truth goes to a side table, never
    a second column on the graph table -- carrying ``played_ratio_pct_true``
    would hand the model the answer.
    """
    held = f'{facts}_H{pct}'
    truth = f'{held}_TRUTH'
    predicate = (f"MOD(ABS(HASH(interaction_id::STRING || '/holdout')), 100) "
                 f'< {pct}')
    run(
        session, f'holdout {pct}%', f"""
        CREATE OR REPLACE TABLE {held} AS
        SELECT interaction_id, user_id, item_id, event_time, event_type,
               is_organic,
               CASE WHEN {predicate} THEN NULL ELSE played_ratio_pct END
                 AS played_ratio_pct,
               track_length_seconds
        FROM {facts}""")
    run(
        session, f'holdout {pct}% truth', f"""
        CREATE OR REPLACE TABLE {truth} AS
        SELECT interaction_id, user_id, item_id, event_time, event_type,
               is_organic, played_ratio_pct, track_length_seconds
        FROM {facts}
        WHERE {predicate} AND played_ratio_pct IS NOT NULL""")
    row = session.sql(f"""
        SELECT (SELECT COUNT(*) FROM {facts}),
               (SELECT COUNT_IF(played_ratio_pct IS NULL) FROM {facts}),
               (SELECT COUNT_IF(played_ratio_pct IS NULL) FROM {held}),
               (SELECT COUNT(*) FROM {truth})""").collect()[0]
    print(f'\n  rows                    {row[0]:>16,}')
    print(f'  source nulls (no truth) {row[1]:>16,}  '
          f'<- these join the prediction set with nothing to score against')
    print(f'  nulls after blanking    {row[2]:>16,}')
    print(f'  scoreable truth rows    {row[3]:>16,}')
    if row[3] == 0 or row[2] == row[0]:
        raise SystemExit(
            'degenerate holdout: nothing labeled or nothing held out')
    return held, truth


def build_lp_holdout(session: Session, facts: str, users: str, items: str,
                     pct: int) -> Tuple[str, str, str]:
    r"""Row 13's static-LP holdout over distinct ``(user, item)`` pairs.

    Two invariants, both from BCCC row 14 S6, the second of which was that
    block's mid-run correction:

    * every user **gives up at least one** item and **keeps at least one** --
      otherwise a user either has no truth or no context;
    * every held-out item **keeps at least one other user** -- otherwise the
      item is an isolated, featureless node and its truth is unreachable by
      construction.

    Both are enforced by ranking, not by a uniform hash: a uniform sample of
    pairs cannot express "at least one on each side".
    """
    lp = f'{facts}_LP'
    truth = f'{lp}_TRUTH'
    lp_items = f'{items}_LP'

    # Rank each user's distinct items by a salted hash, and each item's distinct
    # users the same way.  A pair is held out when it is in the user's top
    # `pct`% AND is neither the user's only item nor the item's only user.
    run(
        session, 'lp pair holdout', f"""
        CREATE OR REPLACE TABLE {SCHEMA}.__LP_PAIRS AS
        WITH pairs AS (
            SELECT user_id, item_id, COUNT(*) AS n
            FROM {facts} GROUP BY 1, 2),
        ranked AS (
            SELECT p.*,
                   COUNT(*) OVER (PARTITION BY user_id) AS user_degree,
                   COUNT(*) OVER (PARTITION BY item_id) AS item_degree,
                   ROW_NUMBER() OVER (
                       PARTITION BY user_id
                       ORDER BY HASH(user_id, item_id, 42)) AS user_rank,
                   ROW_NUMBER() OVER (
                       PARTITION BY item_id
                       ORDER BY HASH(user_id, item_id, 42)) AS item_rank
            FROM pairs p)
        SELECT user_id, item_id, n, user_degree, item_degree,
               (user_degree > 1 AND item_degree > 1
                AND user_rank <= GREATEST(1, FLOOR(user_degree * {pct} / 100))
                AND user_rank < user_degree
                AND item_rank < item_degree) AS is_held
        FROM ranked""")
    run(
        session, 'lp graph table', f"""
        CREATE OR REPLACE TABLE {lp} AS
        SELECT f.interaction_id, f.user_id, f.item_id, f.event_time,
               f.event_type, f.is_organic, f.played_ratio_pct,
               f.track_length_seconds
        FROM {facts} f
        JOIN {SCHEMA}.__LP_PAIRS p
          ON p.user_id = f.user_id AND p.item_id = f.item_id
        WHERE NOT p.is_held""")
    run(
        session, 'lp truth', f"""
        CREATE OR REPLACE TABLE {truth} AS
        SELECT user_id, item_id, n FROM {SCHEMA}.__LP_PAIRS WHERE is_held""")
    run(
        session, 'lp items', f"""
        CREATE OR REPLACE TABLE {lp_items} AS
        SELECT DISTINCT item_id FROM {lp}""")

    row = session.sql(f"""
        SELECT (SELECT COUNT(*) FROM {SCHEMA}.__LP_PAIRS),
               (SELECT COUNT(*) FROM {truth}),
               (SELECT COUNT(DISTINCT user_id) FROM {truth}),
               (SELECT COUNT(*) FROM {facts}),
               (SELECT COUNT(*) FROM {lp}),
               (SELECT COUNT(*) FROM {lp_items}),
               (SELECT COUNT(*) FROM {items})""").collect()[0]
    print(f'\n  distinct (user,item)    {row[0]:>16,}')
    print(f'  held-out pairs          {row[1]:>16,}  '
          f'({100.0 * row[1] / max(row[0], 1):.1f}%)')
    print(f'  users with truth        {row[2]:>16,}')
    print(f'  interactions before/after {row[3]:>14,} / {row[4]:,}')
    print(f'  items before/after      {row[6]:>16,} / {row[5]:,}')
    orphan = session.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT item_id FROM {truth}
            MINUS SELECT item_id FROM {lp_items})""").collect()[0][0]
    print(f'  held-out items unreachable in the graph: {orphan:,}  '
          f'({100.0 * orphan / max(row[1], 1):.2f}% of truth)  '
          f'{"OK" if orphan == 0 else "*** BCCC S6 FAILURE ***"}')
    if row[1] == 0:
        raise SystemExit('degenerate lp holdout: nothing held out')
    return lp, truth, lp_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', default='y_smoke', choices=list(SCALES))
    parser.add_argument('--warehouse', default='KUMO_L')
    parser.add_argument('--rebuild-raw', action='store_true')
    parser.add_argument('--raw-only', action='store_true')
    parser.add_argument('--time-divisor', type=int,
                        default=DEFAULT_TIME_DIVISOR)
    parser.add_argument('--holdout-pct', type=int, default=0,
                        help='row 10: blank this %% of played_ratio_pct')
    parser.add_argument('--lp-holdout-pct', type=int, default=0,
                        help='row 13: hold out this %% of (user,item) pairs')
    args = parser.parse_args()

    session = Session.builder.config('connection_name', 'tlb').create()
    # The connection profile names no warehouse, so the default is KUMO_XS;
    # USE WAREHOUSE on the session is the only thing that moves work.
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()
    session.sql(f'CREATE SCHEMA IF NOT EXISTS {SCHEMA}').collect()
    print('warehouse:',
          session.sql('SELECT CURRENT_WAREHOUSE()').collect()[0][0])

    if args.rebuild_raw or not table_exists(session, RAW):
        build_raw(session, args.time_divisor)
    else:
        print(f'{RAW} exists; use --rebuild-raw to re-read the stage')
    if args.raw_only:
        print('\nTOTAL raw only')
        return

    bucket = SCALES[args.scale]
    print(f'\n=== tier {args.scale} '
          f'({bucket} / {BUCKETS} of the raw pool\'s users) ===')
    users, items, facts = tier_tables(session, args.scale, bucket)
    describe(session, users, items, facts)

    if args.holdout_pct:
        print(f'\n=== row 10 holdout ({args.holdout_pct}%) ===')
        build_holdout(session, facts, args.holdout_pct)
    if args.lp_holdout_pct:
        print(f'\n=== row 13 link holdout ({args.lp_holdout_pct}%) ===')
        build_lp_holdout(session, facts, users, items, args.lp_holdout_pct)

    print('\nTOTAL')


if __name__ == '__main__':
    main()
