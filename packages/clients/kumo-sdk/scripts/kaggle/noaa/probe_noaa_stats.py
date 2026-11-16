# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Checks every claim the fixture matrix makes about NOAA GHCN-Daily.

All four previous fixtures deviated from the matrix in at least one way -- row 1
had a non-unique primary key and an unusable join table, row 3's key collided
0.13% of the time, rows 1 and 2 disagreed about whether a part file is a sample
or a slice -- so nothing here is inherited and nothing is assumed.

Everything runs against ``HYU.NOAA.OBSERVATIONS_RAW``, the single typed copy
``build_noaa_tables.py --raw-only`` materializes, rather than against the
external stage. Row 2 S7 measured a stage scan at 600-1300 s on ``KUMO_L`` and
row 4 S2 measured that narrowing the projection does not help, so the stage is
read exactly once, ever.

What is checked
---------------

``--counts``
    Row counts against the matrix's 2,895,853,690 and the manifest; key
    uniqueness on both tables; referential integrity of
    ``observations.station_id -> stations.station_id`` in *both* directions.
``--time``
    The span and unit of ``observation_date``, its density by decade, and how
    many observations fall before the epoch. This fixture reaches back to 1763,
    which no previous one did, and it is what makes the temporal split
    interesting (row 3 S3, row 4 S5).
``--elements``
    Element cardinality and distribution, by observation and by station, and
    what ``CLASSIFY TOP 25`` actually keeps.
``--values``
    The published-unit claim: TMAX in tenths of a degree Celsius, and the range
    of every other head element, which decides whether row 11's target is on a
    sane scale.
``--columns``
    Null rates of the five non-key columns. Row 4 lost a run to ``SERVICE``, a
    column non-null in 0.011% of rows: per-table inference draws a 10,000-row
    sample, an all-null sample yields no statistics, and an unresolvable column
    is a hard error rather than a default.
``--parts``
    Whether a part file is a contiguous slice or a uniform sample. Row 1's parts
    each spanned the whole history; row 2's were one capture each. The answer
    decides whether a tier can be a prefix.
"""
import argparse

from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.NOAA_GHCN_10GB'
FF = '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.PARQUET_FF)'
SCHEMA = 'HYU.NOAA'
RAW = f'{SCHEMA}.OBSERVATIONS_RAW'
STATIONS = f'{SCHEMA}.STATIONS_RAW'

#: What the matrix and the fixture's own ``_metadata/manifest.json`` claim.
MATRIX_ROWS = 2_895_853_690
MANIFEST_OBSERVATIONS = 2_895_739_748
MANIFEST_STATIONS = 113_942


def show(session: Session, title: str, sql: str) -> list:
    print(f'\n--- {title} ---', flush=True)
    rows = session.sql(sql).collect()
    for row in rows:
        print(
            '  ' + '  '.join(f'{v:,}' if isinstance(v, int) else str(v)
                             for v in row), flush=True)
    return rows


def counts(session: Session) -> None:
    r"""Row counts and key integrity.

    The matrix figure has been the **sum over all tables** on every previous row
    (383,756,326 across three ethereum tables, 583,341,095 across two iot23
    ones, 48,590,026 across two BCCC ones), so it is checked as a sum here
    rather than as an observation count.

    Referential integrity is checked both ways because the two directions fail
    differently. An observation pointing at an absent station is a dangling
    foreign key, which the graph either drops or rejects; a station with no
    observations is a legal entity with an *empty* multilabel target, which is a
    much subtler problem -- it trains the model to predict nothing.
    """
    n_obs, n_obs_stations, n_null = session.sql(
        f'SELECT COUNT(*), COUNT(DISTINCT station_id), '
        f'COUNT_IF(station_id IS NULL) FROM {RAW}').collect()[0]
    n_st, n_st_distinct, n_st_null = session.sql(
        f'SELECT COUNT(*), COUNT(DISTINCT station_id), '
        f'COUNT_IF(station_id IS NULL) FROM {STATIONS}').collect()[0]

    print(f'observations                 {n_obs:>16,}  '
          f'(manifest {MANIFEST_OBSERVATIONS:,})')
    print(f'stations                     {n_st:>16,}  '
          f'(manifest {MANIFEST_STATIONS:,})')
    print(f'sum                          {n_obs + n_st:>16,}  '
          f'(matrix   {MATRIX_ROWS:,})')
    print(f'  matrix figure is the sum:  {n_obs + n_st == MATRIX_ROWS}')
    print(f'stations.station_id distinct {n_st_distinct:>16,}  '
          f'null {n_st_null:,}')
    print(f'  primary key is unique:     {n_st_distinct == n_st}')
    print(f'observations reference       {n_obs_stations:>16,} '
          f'distinct stations, null {n_null:,}')

    orphans = session.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT station_id FROM {RAW}
        ) o LEFT JOIN {STATIONS} s USING (station_id)
        WHERE s.station_id IS NULL
    """).collect()[0][0]
    empty = session.sql(f"""
        SELECT COUNT(*) FROM {STATIONS} s
        WHERE NOT EXISTS (SELECT 1 FROM {RAW} o
                          WHERE o.station_id = s.station_id)
    """).collect()[0][0]
    print(f'observed stations absent from stations: {orphans:,}')
    print(f'stations with zero observations:        {empty:,}')

    # (station_id, event_time, element) is the natural key of a GHCN daily
    # record. Whether it is unique decides whether an observation could be used
    # as an entity at all, and whether the multilabel target is well defined.
    dupes = session.sql(f"""
        SELECT COUNT(*) FROM (
            SELECT station_id, event_time, element, COUNT(*) AS n
            FROM {RAW} GROUP BY 1, 2, 3 HAVING n > 1)
    """).collect()[0][0]
    print(f'duplicate (station, date, element) triples: {dupes:,}')


def time_span(session: Session) -> None:
    r"""The event-time column: unit, span and density.

    ``observation_date`` is a signed count of **microseconds** since the epoch
    and is negative for everything before 1970, which is most of the fixture's
    span. That is new: rows 1-4 all sat entirely after the epoch.

    Density by decade is the number that governs the temporal split. The split
    is cut by row *quantile*, so if 90% of the rows are after 1970 then a 20%
    test tail is a few years wide while the training stage carries two
    centuries -- the same mechanism that made row 4's validation stage 2.7 hours
    wide.
    """
    low, high, nulls = session.sql(
        f'SELECT MIN(event_time), MAX(event_time), '
        f'COUNT_IF(event_time IS NULL) FROM {RAW}').collect()[0]
    print(f'event_time span: {low} -> {high}, null {nulls:,}')

    show(
        session, 'observations by decade', f"""
        SELECT FLOOR(YEAR(event_time) / 10) * 10 AS decade, COUNT(*) AS n,
               COUNT(DISTINCT station_id) AS stations
        FROM {RAW} GROUP BY 1 ORDER BY 1
    """)

    show(
        session, 'row quantiles of event_time (where a split would cut)', f"""
        SELECT APPROX_PERCENTILE(DATE_PART(EPOCH_SECOND, event_time), 0.01),
               APPROX_PERCENTILE(DATE_PART(EPOCH_SECOND, event_time), 0.50),
               APPROX_PERCENTILE(DATE_PART(EPOCH_SECOND, event_time), 0.80),
               APPROX_PERCENTILE(DATE_PART(EPOCH_SECOND, event_time), 0.90),
               APPROX_PERCENTILE(DATE_PART(EPOCH_SECOND, event_time), 0.99)
        FROM {RAW}
    """)


def elements(session: Session) -> None:
    r"""The multilabel class list, by observation and by station.

    Rows 5 and 8 predict the *set* of elements a station reports, so the class
    frequency that matters is the **station** count, not the observation count:
    ``TMAX`` occurring a billion times at 40,000 stations is one label on 40,000
    entities. ``CLASSIFY TOP 25`` keeps the 25 most frequent classes, so the
    line to read is the 25th and whatever falls off below it.
    """
    n_stations = session.sql(
        f'SELECT COUNT(*) FROM {STATIONS}').collect()[0][0]
    n_elements = session.sql(
        f'SELECT COUNT(DISTINCT element) FROM {RAW}').collect()[0][0]
    print(f'{n_elements} distinct elements over {n_stations:,} stations')

    print(f'\n{"rank":>4} {"element":<10} {"observations":>16} '
          f'{"stations":>10} {"station share":>14}')
    rows = session.sql(f"""
        SELECT element, COUNT(*) AS n, COUNT(DISTINCT station_id) AS s
        FROM {RAW} GROUP BY 1 ORDER BY 3 DESC, 2 DESC
    """).collect()
    for rank, (element, n, s) in enumerate(rows, start=1):
        marker = '  <- CLASSIFY TOP 25 cut' if rank == 25 else ''
        print(
            f'{rank:>4} {element:<10} {n:>16,} {s:>10,} '
            f'{s / n_stations:>13.4%}{marker}', flush=True)

    show(
        session, 'elements per station', f"""
        SELECT MIN(k), APPROX_PERCENTILE(k, 0.25), APPROX_PERCENTILE(k, 0.5),
               APPROX_PERCENTILE(k, 0.75), MAX(k), AVG(k)
        FROM (SELECT station_id, COUNT(DISTINCT element) AS k
              FROM {RAW} GROUP BY 1)
    """)

    show(
        session, 'observations per station', f"""
        SELECT MIN(n), APPROX_PERCENTILE(n, 0.25), APPROX_PERCENTILE(n, 0.5),
               APPROX_PERCENTILE(n, 0.75), APPROX_PERCENTILE(n, 0.99), MAX(n)
        FROM (SELECT station_id, COUNT(*) AS n FROM {RAW} GROUP BY 1)
    """)


def values(session: Session) -> None:
    r"""The matrix's unit claim, checked against the data.

    "NOAA values use each element's published unit, including tenths of degrees
    for TMAX" -- so TMAX should run to roughly +-600 (-60 C to +60 C) and
    have a mean around 150, not 15. Row 11's target is ``AVG(value WHERE element
    = 'TMAX')``, so getting this wrong makes the regression's error scale
    meaningless rather than merely odd.
    """
    show(
        session, 'value distribution for the head elements', f"""
        SELECT element, COUNT(*) AS n, MIN(value), APPROX_PERCENTILE(value, .5),
               AVG(value), MAX(value), COUNT_IF(value IS NULL)
        FROM {RAW}
        WHERE element IN ('TMAX', 'TMIN', 'PRCP', 'SNOW', 'SNWD', 'TAVG')
        GROUP BY 1 ORDER BY 2 DESC
    """)

    show(
        session, 'TMAX per-station daily mean (row 11 target scale)', f"""
        SELECT MIN(m), APPROX_PERCENTILE(m, 0.05), APPROX_PERCENTILE(m, 0.5),
               APPROX_PERCENTILE(m, 0.95), MAX(m), AVG(m), STDDEV(m)
        FROM (SELECT station_id, AVG(value) AS m FROM {RAW}
              WHERE element = 'TMAX' GROUP BY 1)
    """)


def columns(session: Session) -> None:
    r"""Null rates, because an all-null inference sample is a hard failure."""
    show(
        session, 'null rate and cardinality per column', f"""
        SELECT 'measurement_flag', COUNT_IF(measurement_flag IS NOT NULL),
               COUNT(DISTINCT measurement_flag) FROM {RAW}
        UNION ALL SELECT 'quality_flag', COUNT_IF(quality_flag IS NOT NULL),
               COUNT(DISTINCT quality_flag) FROM {RAW}
        UNION ALL SELECT 'source_flag', COUNT_IF(source_flag IS NOT NULL),
               COUNT(DISTINCT source_flag) FROM {RAW}
        UNION ALL SELECT 'observation_time',
               COUNT_IF(observation_time IS NOT NULL),
               COUNT(DISTINCT observation_time) FROM {RAW}
        UNION ALL SELECT 'value', COUNT_IF(value IS NOT NULL),
               COUNT(DISTINCT value) FROM {RAW}
    """)


def parts(session: Session) -> None:
    r"""Is a part file a contiguous slice or a uniform sample?

    Row 1's parts each spanned the whole ten-year history, so one part was a
    representative 10% sample. Row 2's were one capture each, so a prefix held
    a single malware family and no benign traffic at all. The answer decides
    whether a tier may be a prefix of the fixture or has to be hash-sampled, and
    it has differed between fixtures, so it is measured rather than assumed.

    This is the one probe that reads the stage, and it reads five named part
    files rather than the directory, so it costs five files instead of 580.
    """
    for part in ('00000', '00100', '00289', '00450', '00579'):
        rows = session.sql(f"""
            SELECT COUNT(*),
                   COUNT(DISTINCT $1:station_id::STRING),
                   MIN($1:station_id::STRING), MAX($1:station_id::STRING),
                   TO_TIMESTAMP_NTZ(
                       MIN($1:observation_date::NUMBER) / 1000000),
                   TO_TIMESTAMP_NTZ(
                       MAX($1:observation_date::NUMBER) / 1000000)
            FROM '@{STAGE}/observations/part-{part}.parquet' {FF}
        """).collect()[0]
        print(
            f'part-{part}: {rows[0]:>10,} rows  {rows[1]:>7,} stations  '
            f'{rows[2]}..{rows[3]}  {rows[4]} -> {rows[5]}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--warehouse', default='KUMO_L')
    for name in ('counts', 'time', 'elements', 'values', 'columns', 'parts'):
        parser.add_argument(f'--{name}', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    session = Session.builder.config('connection_name', 'tlb').create()
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()

    checks = [('counts', counts), ('time', time_span), ('elements', elements),
              ('values', values), ('columns', columns), ('parts', parts)]
    selected = [(n, f) for n, f in checks
                if args.all or getattr(args, n.replace('-', '_'))]
    if not selected:
        selected = checks
    for name, check in selected:
        print(f'\n{"=" * 70}\n{name}\n{"=" * 70}', flush=True)
        check(session)


if __name__ == '__main__':
    main()
