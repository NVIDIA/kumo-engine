# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Materializes the NOAA GHCN-Daily fixture for matrix rows 5, 8 and 11.

One fixture, three predictive queries::

    row  5  PREDICT LIST_DISTINCT(observations.element) CLASSIFY TOP 25
            FOR EACH stations.station_id
    row  8  ... the same, RANK TOP 25
    row 11  PREDICT AVG(observations.value WHERE element = 'TMAX', 0, 7, days)
            FOR EACH stations.station_id

so this script builds *one* set of tables and all three rows read them. Nothing
about the tables is query-specific; the only thing the three rows disagree about
is what is asked of them.

Why the tiers are cut on ``station_id``
---------------------------------------

Rows 2, 3 and 4 sampled the *fact* table, because the entity was the fact table
and a row was an entity. Here the entity is ``stations``, a dimension table of
113,942 rows, and the target is an aggregate over each station's own
observations. Thinning a station's observations would therefore **change its
label**: a station that reports ``WT01`` on four days in fifty years loses the
class entirely under a 1% row sample, and the multilabel target is exactly the
set of elements it reports.

So a tier is a hash-sampled subset of *stations* carried with **all** of their
observations::

    MOD(ABS(HASH(station_id)), 10000) < permille10

Every tier is then a complete, correct instance of the published task on a
random subset of stations, rather than a corrupted instance of the whole one.
It also nests -- ``TINY`` ⊂ ``SMOKE`` ⊂ ``SMALL`` ⊂ ``PART`` -- so a station
scored at one tier is scored against the same ground truth at the next.

The part files make hashing mandatory rather than merely tidy, and for a
different reason than the obvious one. They are **globally sorted by date**:
``part-00000`` is 1763-01-01 to 1884-12-15 over 1,698 stations, and
``part-00579`` is 2019-05-23 to 2019-06-03 over 26,415. So a prefix slice is the
eighteenth century, with almost no stations and none of the modern instruments.
This is a third pattern -- row 1's parts were each a full-history sample, row
2's were one capture each.

Why the raw copy holds every column
-----------------------------------

Row 4 S2 measured a two-column re-read of an external parquet stage costing as
much as the original twenty-one-column pass (726 s against 600-1300 s): the
``$1:<field>`` variant decode dominates, not the projection width. So the one
pass over 580 part files materializes **all eight** published columns plus a
precomputed sample bucket, and every tier after it is a scan of a native
Snowflake table.

No holdout, deliberately
------------------------

Rows 1-4 all needed ``--holdout-pct``: their targets were static *columns*, so
Kumo compiled batch prediction to ``WHERE target IS NULL`` and the fully-labeled
fixture left nothing to score (row 1 S1). None of rows 5, 8 or 11 has a static
column to blank -- the target is an aggregate over a related table -- so there
is nothing to hold out, and whether batch prediction can select rows without one
is a question this fixture answers rather than assumes. See ``FINDINGS.md`` S2.

Ground truth for scoring is therefore recomputed in SQL from the tier tables
themselves -- ``STATION_ELEMENTS_<TIER>``, built here, and the windowed truth
``run_noaa_local.py`` derives at the scoring anchor -- rather than carved out of
them.
"""
import argparse
import time

from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.NOAA_GHCN_10GB'
FF = '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.PARQUET_FF)'
SCHEMA = 'HYU.NOAA'

#: All 2,895,739,748 observations, typed, plus the sample bucket. ~25 GB
#: uncompressed at the source; one pass over 580 part files builds it and no
#: later step touches the stage again.
RAW = f'{SCHEMA}.OBSERVATIONS_RAW'

#: All 113,942 stations. One column wide -- the published table has nothing but
#: the key, exactly like ``hosts`` in rows 2-4 -- so this costs one part file.
STATIONS_RAW = f'{SCHEMA}.STATIONS_RAW'

#: Sample width in parts per 10,000 of ``station_id``, against 113,942 stations
#: and 2,895,739,748 observations (~25.4K observations per station on average,
#: heavily skewed: a 1763 European station and a 2020 automatic gauge differ
#: three orders of magnitude).
#:
#: What actually binds is neither of the obvious candidates. Ingest is trivial
#: -- DiskGraph took 4.8 s and 109 MB for ``tiny``'s 4.86M observations, which
#: extrapolates to ~47 min and ~65 GB for all 2.9 billion. And the entity table
#: is tiny: even ``full`` is 113,942 stations, fewer than row 2's ``smoke`` tier
#: had flows. The constraint is the *training table*, which for a temporal query
#: is the cross product of entities and 30-day anchors over the fixture's whole
#: 256-year span: 231 x 1,818 = 419,958 rows at ``tiny``, and ~355 million at
#: ``full``. Worse, the span grows with the tier (``tiny`` starts in 1870,
#: ``small`` in 1839, ``full`` in 1763), so cost grows faster than station
#: count. See ``FINDINGS.md`` S6.
#:
#: Station counts below are measured, not nominal.
SCALES = {
    'tiny': 20,  # 231 stations, 4,862,394 observations, 61 elements
    'smoke': 100,  # 1,150 stations, 27,158,157 observations, 81 elements
    'small': 500,  # 5,642 stations, 141,129,745 observations, 100 elements
    'part': 2000,  # ~22,800 stations
    'half': 5000,  # ~57,000 stations
    'full': 10000,  # 113,942 stations, 2,895,739,748 observations, 137
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

    ``observation_date`` arrives as a signed integer count of **microseconds**
    since the epoch, and it is negative for most of the nineteenth century and
    all of the eighteenth: the first row of ``part-00000`` is
    ``-6532272000000000``, which is 1763-01-01. It is stored here as a real
    ``TIMESTAMP_NTZ`` so that neither the tier builds nor the graph has to know
    the unit, and so that a wrong guess fails at the cast rather than silently
    producing the year 48,355,176 (row 2 S7 hit exactly that reading
    microseconds as nanoseconds).

    ``station_bucket`` is the tier predicate precomputed once over 2.9 billion
    rows instead of re-hashed on every tier build.
    """
    run(
        session, f'staged parquet -> {RAW}', f"""
        CREATE OR REPLACE TABLE {RAW} AS
        SELECT $1:station_id::STRING             AS station_id,
               TO_TIMESTAMP_NTZ(
                   $1:observation_date::NUMBER / 1000000) AS event_time,
               $1:element::STRING                AS element,
               $1:value::NUMBER(38,0)            AS value,
               $1:measurement_flag::STRING       AS measurement_flag,
               $1:quality_flag::STRING           AS quality_flag,
               $1:source_flag::STRING            AS source_flag,
               $1:observation_time::STRING       AS observation_time,
               MOD(ABS(HASH($1:station_id::STRING)), 10000)
                                                 AS station_bucket
        FROM '@{STAGE}/observations/' {FF}
    """)

    run(
        session, f'staged parquet -> {STATIONS_RAW}', f"""
        CREATE OR REPLACE TABLE {STATIONS_RAW} AS
        SELECT $1:station_id::STRING AS station_id,
               MOD(ABS(HASH($1:station_id::STRING)), 10000) AS station_bucket
        FROM '@{STAGE}/stations/' {FF}
    """)

    for name in (RAW, STATIONS_RAW):
        count = session.sql(f'SELECT COUNT(*) FROM {name}').collect()[0][0]
        print(f'{name}: {count:,} rows', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', choices=list(SCALES), default='smoke')
    parser.add_argument('--warehouse', default='KUMO_L')
    parser.add_argument(
        '--rebuild-raw', action='store_true',
        help='re-read the 580 staged part files even if '
        'OBSERVATIONS_RAW already exists')
    parser.add_argument('--raw-only', action='store_true',
                        help='build the raw copies and stop')
    args = parser.parse_args()

    suffix = args.scale.upper()
    obs_table = f'{SCHEMA}.OBSERVATIONS_{suffix}'
    stations_table = f'{SCHEMA}.STATIONS_{suffix}'
    truth_table = f'{SCHEMA}.STATION_ELEMENTS_{suffix}'

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
        return

    bucket = SCALES[args.scale]

    # The station set first: it is the entity table, and the observations are
    # filtered to it rather than the other way round, so that no observation can
    # point at a station the graph does not hold. `station_bucket` is stored on
    # both sides, so both filters are an integer compare rather than a hash of
    # 2.9 billion strings.
    run(
        session, f'stations -> {stations_table}', f"""
        CREATE OR REPLACE TABLE {stations_table} AS
        SELECT station_id FROM {STATIONS_RAW} WHERE station_bucket < {bucket}
    """)

    # Every observation of every sampled station, with none of its history
    # thinned -- see the module docstring for why that is not optional.
    run(
        session, f'observations -> {obs_table}', f"""
        CREATE OR REPLACE TABLE {obs_table} AS
        SELECT station_id, event_time, element, value, measurement_flag,
               quality_flag, source_flag, observation_time
        FROM {RAW} WHERE station_bucket < {bucket}
    """)

    # The row 5/8 target, recomputed rather than held out: the set of distinct
    # elements each station ever reports. This is the label the multilabel
    # queries ask for, materialized so scoring does not have to re-derive it
    # from a 2.9-billion-row table on every run.
    run(
        session, f'station element sets -> {truth_table}', f"""
        CREATE OR REPLACE TABLE {truth_table} AS
        SELECT s.station_id,
               COUNT(o.element)          AS n_observations,
               COUNT(DISTINCT o.element) AS n_elements,
               ARRAY_AGG(DISTINCT o.element) AS elements
        FROM {stations_table} s
        LEFT JOIN {obs_table} o USING (station_id)
        GROUP BY 1
    """)

    for name in (stations_table, obs_table, truth_table):
        count = session.sql(f'SELECT COUNT(*) FROM {name}').collect()[0][0]
        print(f'{name}: {count:,} rows', flush=True)

    # Which elements survived the sample, and how many stations report each.
    # For a multilabel target this is the class list, and `CLASSIFY TOP 25`
    # keeps the 25 most frequent -- so the row that matters is the 25th, and
    # whether there is anything below it.
    print(
        f'\n{"element":<10} {"observations":>16} {"stations":>10} '
        f'{"station share":>14}', flush=True)
    n_stations = session.sql(
        f'SELECT COUNT(*) FROM {stations_table}').collect()[0][0]
    rows = session.sql(f"""
        SELECT element, COUNT(*) AS n, COUNT(DISTINCT station_id) AS s
        FROM {obs_table} GROUP BY 1 ORDER BY 3 DESC, 2 DESC
    """).collect()
    for rank, (element, n, s) in enumerate(rows, start=1):
        marker = '  <- 25th' if rank == 25 else ''
        print(
            f'{element:<10} {n:>16,} {s:>10,} '
            f'{s / n_stations:>13.4%}{marker}', flush=True)
    print(
        f'{"TOTAL":<10} {sum(r[1] for r in rows):>16,} over {len(rows)} '
        'distinct elements', flush=True)

    # Observations per station, which is what decides ingest cost and how much
    # history each entity's subgraph actually has.
    stats = session.sql(f"""
        SELECT MIN(n_observations), APPROX_PERCENTILE(n_observations, 0.5),
               AVG(n_observations), MAX(n_observations),
               MIN(n_elements), APPROX_PERCENTILE(n_elements, 0.5),
               MAX(n_elements), COUNT_IF(n_observations = 0)
        FROM {truth_table}
    """).collect()[0]
    print(
        f'\nobservations per station: min={stats[0]:,} '
        f'median={stats[1]:,.0f} mean={stats[2]:,.0f} max={stats[3]:,}',
        flush=True)
    print(
        f'elements per station:     min={stats[4]} median={stats[5]:.0f} '
        f'max={stats[6]}', flush=True)
    print(f'stations with no observations at all: {stats[7]:,}', flush=True)

    if stats[7] == n_stations:
        raise SystemExit('degenerate slice: no station has any observation')


if __name__ == '__main__':
    main()
