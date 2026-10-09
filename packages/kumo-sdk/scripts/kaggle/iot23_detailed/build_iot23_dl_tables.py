# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Materializes the IoT-23 fixture for the *multiclass* row of the matrix.

Row 4 -- ``kaggle_iot23_flow_detailed_label_10gb`` -- is row 2's fixture with
the target swapped: ``detailed_label`` (the attack family) instead of
``is_malicious``. Same S3 root, same stage, same two tables, same graph
contract. So this script is deliberately a near-copy of
``scripts/kaggle/iot23/build_iot23_tables.py``, and the differences are the
interesting part.

What is reused
--------------

``HYU.IOT23.NETWORK_FLOWS_RAW`` already holds all 325,309,946 flows, typed, from
row 2's one pass over the staged parquet. It is **not** rebuilt here.

What has to be built
--------------------

``NETWORK_FLOWS_RAW`` does not carry ``detailed_label`` at all: row 2 dropped it
in the ``SELECT`` that created the table, because for a binary target it is a
leak. Row 4 therefore needs one new read of the stage -- but only two columns
wide, ``flow_id`` and ``detailed_label``, into ``DETAILED_LABELS_RAW``. Every
tier after that is a join of two native Snowflake tables.

The leakage inverts
-------------------

Row 2 dropped ``label``, ``detailed_label``, ``scenario_id``, ``uid`` and ``ts``
before they ever reached ``NETWORK_FLOWS_RAW``. Three of those five are still
dropped for the same reasons (``scenario_id`` names the capture, ``uid`` is
noise, ``ts`` duplicates ``event_time``), and ``detailed_label`` is now the
target. The two that change sides:

``is_malicious``
    Row 2's *target* is row 4's *leak*, and an exact one:
    ``is_malicious`` is true iff ``detailed_label <> '-'``. Knowing it collapses
    16 classes to 2 for free. Excluded from every tier table here.
``label``
    ``'Malicious'``/``'Benign'`` -- the same collapse as a string. Never made it
    into ``NETWORK_FLOWS_RAW``, so nothing to do, but it is a leak for this row
    as much as for row 2.

``scenario_id`` deserves its own note. It is still dropped, and it is still
embedded in ``flow_id`` (``'<scenario_id>:<uid>'``), and row 2 S3 measured that
hosts are 99.87% capture-private -- so the graph reconstructs the capture even
without the column. That is *worse* here than it was there: a capture is one
malware sample, and the classes of this task **are** malware families. See
``probe_iot23_dl_leakage.py``, which measures the multiclass form of it.

Sampling
--------

The sample and holdout predicates are byte-identical to row 2's, on purpose.
That makes ``NETWORK_FLOWS_DL_SMALL`` exactly the same 3,254,288 flows as
``NETWORK_FLOWS_SMALL``, with exactly the same rows held out -- so any
difference in the results is the target and nothing else. That is the whole
value of running this row.

Scales are nested deterministic samples of ``flow_id``::

    MOD(ABS(HASH(flow_id)), 10000) < permille10

Uniform over the whole fixture rather than a prefix, because a prefix here is
one capture: the parts are scenario-aligned, so ``part-00003`` is 1M flows of
exactly one malware family and nothing else (row 2 S6).

Holdout
-------

Fourth confirmed instance of the static-target problem (row 1 S1 predicted rows
2, 3, 4 and 10; this is row 4). ``PREDICT network_flows.detailed_label``
compiles to ``WHERE detailed_label IS NULL``, and the fixture labels all 325M
flows, so batch prediction as published selects nothing. ``--holdout-pct``
blanks a deterministic slice; the true labels survive in ``*_TRUTH``.

Note that ``'-'`` -- benign -- is a real class value, not a missing one, so
blanking has to write SQL ``NULL`` and the two must not be confused: after the
holdout, ``NULL`` means "held out" and ``'-'`` means "benign".
"""
import argparse
import time

from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.IOT23_10GB'
FF = '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.PARQUET_FF)'
SCHEMA = 'HYU.IOT23'

#: Row 2's typed fact table, reused as-is. 325,309,946 rows, no
#: ``detailed_label`` column.
RAW = f'{SCHEMA}.NETWORK_FLOWS_RAW'

#: The one thing row 2 did not leave behind: the target, keyed by ``flow_id``.
#: Two columns rather than twenty-one, so the stage read is column-pruned.
LABELS = f'{SCHEMA}.DETAILED_LABELS_RAW'

#: Suffix that keeps row 4's tier tables away from row 2's. Row 2's
#: ``NETWORK_FLOWS_SMALL`` is unusable here -- it dropped the target -- and
#: clobbering it would invalidate row 2's numbers.
PREFIX = 'NETWORK_FLOWS_DL'

#: Sample width in parts per 10,000 of `flow_id`, against 325,309,946 flows.
#: Identical to row 2's, so the tiers hold the same rows.
SCALES = {
    'smoke': 5,  # ~163K flows
    'small': 100,  # ~3.3M flows
    'part': 500,  # ~16.3M flows
    'half': 5000,  # ~163M flows
    'full': 10000,  # 325,309,946 flows
}

#: Deterministic in the key itself, and salted apart from each other. The two
#: predicates must hash *different* strings: nesting two thresholds on one hash
#: makes them perfectly dependent, which is how row 2's first build produced a
#: smoke slice with zero labeled rows.
#: ``{col}`` is the key expression: bare where only one table is in scope, and
#: qualified inside the join, where ``flow_id`` exists on both sides and a bare
#: reference would not resolve.
SAMPLED = 'MOD(ABS(HASH({col})), 10000) < {permille10}'
HELD_OUT = "MOD(ABS(HASH({col} || '/holdout')), 100) < {pct}"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', choices=list(SCALES), default='smoke')
    parser.add_argument('--warehouse', default='KUMO_L')
    parser.add_argument(
        '--holdout-pct', type=int, default=20,
        help='percent of flows whose detailed_label is blanked '
        'so batch prediction has rows to score')
    parser.add_argument(
        '--rebuild-labels', action='store_true',
        help='re-read the staged parquet even if '
        'DETAILED_LABELS_RAW already exists')
    parser.add_argument('--labels-only', action='store_true',
                        help='build DETAILED_LABELS_RAW and stop')
    args = parser.parse_args()

    suffix = args.scale.upper()
    flows_table = f'{SCHEMA}.{PREFIX}_{suffix}'
    hosts_table = f'{SCHEMA}.HOSTS_DL_{suffix}'
    truth_table = f'{flows_table}_TRUTH'

    session = Session.builder.config('connection_name', 'tlb').create()
    # The size has to be set on the session: `USE WAREHOUSE` is the only thing
    # that moves work, and the connection profile names none, so the default is
    # KUMO_XS.
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()
    print('warehouse:',
          session.sql('SELECT CURRENT_WAREHOUSE()').collect()[0][0],
          flush=True)

    if not table_exists(session, RAW):
        raise SystemExit(
            f'{RAW} is missing. It is row 2\'s table and costs a full pass '
            'over the stage to rebuild; see scripts/kaggle/iot23/.')

    if args.rebuild_labels or not table_exists(session, LABELS):
        # The only stage read this row needs. Two columns out of twenty-two, so
        # Snowflake prunes the parquet rather than materializing the variant --
        # the point of the exercise is that this is much cheaper than the
        # 21-column pass that built NETWORK_FLOWS_RAW.
        run(
            session, f'staged parquet (2 cols) -> {LABELS}', f"""
            CREATE OR REPLACE TABLE {LABELS} AS
            SELECT $1:flow_id::STRING        AS flow_id,
                   $1:detailed_label::STRING AS detailed_label
            FROM '@{STAGE}/network_flows/' {FF}
        """)
        count, distinct, nulls = session.sql(
            f"""SELECT COUNT(*), COUNT(DISTINCT detailed_label),
                       COUNT_IF(detailed_label IS NULL) FROM {LABELS}"""
        ).collect()[0]
        print(
            f'{LABELS}: {count:,} rows, {distinct} distinct labels, '
            f'{nulls:,} null', flush=True)

    if args.labels_only:
        return

    sampled = SAMPLED.format(col='flow_id', permille10=SCALES[args.scale])
    held_out = HELD_OUT.format(col='flow_id', pct=args.holdout_pct)

    # `IS_MALICIOUS` is excluded rather than merely unused: it is `detailed_label
    # <> '-'` exactly, so leaving it in the table is one careless column list
    # away from a perfect leak. `SERVICE`, `LOCAL_SOURCE` and `LOCAL_DESTINATION`
    # are kept here and dropped in the run script, exactly as row 2 does -- they
    # are all-but-empty rather than leaky, and worth keeping visible.
    # Both sides are filtered to the tier *before* the join rather than after
    # it, which is both the cheap way to do it -- the join runs on the sample,
    # not on 325M x 325M -- and the only unambiguous way: `flow_id` exists on
    # both sides, so a bare `flow_id` inside the sample or holdout predicate
    # would not resolve.
    run(
        session, f'flows ({args.holdout_pct}% held out) -> {flows_table}', f"""
        CREATE OR REPLACE TABLE {flows_table} AS
        SELECT f.* EXCLUDE (is_malicious),
               IFF({HELD_OUT.format(col='d.flow_id', pct=args.holdout_pct)},
                   NULL, d.detailed_label) AS detailed_label
        FROM (SELECT * FROM {RAW} WHERE {sampled}) f
        JOIN (SELECT * FROM {LABELS} WHERE {sampled}) d
          ON f.flow_id = d.flow_id
    """)

    run(
        session, f'held-out labels -> {truth_table}', f"""
        CREATE OR REPLACE TABLE {truth_table} AS
        SELECT flow_id, detailed_label FROM {LABELS}
        WHERE {sampled} AND {held_out}
    """)

    # The published `hosts` table is 258M rows of nothing but `host_id`: exactly
    # the set of endpoint IPs, with no attributes to contribute. So the host node
    # set is derived from the flows in scope rather than read back from the
    # stage. That guarantees no foreign key points at a missing row and no
    # isolated node is ingested -- an unreferenced, featureless node cannot
    # affect a GNN, it can only cost an ingest.
    run(
        session, f'endpoint IPs -> {hosts_table}', f"""
        CREATE OR REPLACE TABLE {hosts_table} AS
        SELECT DISTINCT ip AS host_id FROM (
            SELECT source_ip AS ip FROM {flows_table}
            UNION ALL
            SELECT destination_ip FROM {flows_table}
        )
        WHERE ip IS NOT NULL
    """)

    for name in (flows_table, hosts_table, truth_table):
        count = session.sql(f'SELECT COUNT(*) FROM {name}').collect()[0][0]
        print(f'{name}: {count:,} rows', flush=True)

    # Which classes actually survived the sample is the number to read first.
    # The full fixture has 16 classes but the tail runs down to 2 rows in 325M,
    # so a uniform sample of a percent or two is a *different task* -- 8 or 9
    # classes, not 16 -- and comparing tiers compares tasks (row 3 S1).
    print(f'\n{"class":<34} {"labeled":>12} {"held out":>10}', flush=True)
    rows = session.sql(f"""
        SELECT COALESCE(t.detailed_label, f.detailed_label) AS cls,
               COUNT_IF(f.detailed_label IS NOT NULL) AS labeled,
               COUNT_IF(f.detailed_label IS NULL) AS held_out
        FROM {flows_table} f
        LEFT JOIN {truth_table} t USING (flow_id)
        GROUP BY 1 ORDER BY 2 + 3 DESC
    """).collect()
    total_labeled = total_held = 0
    for cls, labeled, held in rows:
        total_labeled += labeled
        total_held += held
        print(f'{str(cls):<34} {labeled:>12,} {held:>10,}', flush=True)
    print(
        f'{"TOTAL":<34} {total_labeled:>12,} {total_held:>10,} '
        f'over {len(rows)} classes', flush=True)

    # A slice with nothing labeled, or nothing held out, is unusable -- the
    # first cannot train and the second cannot be scored -- so say which, rather
    # than dividing by zero several minutes later.
    if not total_labeled or not total_held:
        raise SystemExit(
            f'degenerate slice: labeled={total_labeled:,} '
            f'held_out={total_held:,}; the sample and holdout predicates are '
            'not independent')


if __name__ == '__main__':
    main()
