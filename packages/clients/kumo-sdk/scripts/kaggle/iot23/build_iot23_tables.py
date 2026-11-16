# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Materializes the IoT-23 fixture as Snowflake tables.

The local backend's source area reads Snowflake *tables*, not stages, so the
staged parquet has to land in a schema before the SDK can see it. This builds
one scale at a time into ``HYU.IOT23``.

Shape of the build
------------------

Everything derives from one typed copy of the fact table, ``NETWORK_FLOWS_RAW``
(all 325,309,946 rows). Building it costs one pass over the staged parquet;
every scale after that is a subset of a native Snowflake table, which is far
cheaper than re-parsing 12 GB of variant-typed parquet per scale. ``--rebuild-
raw`` forces that first pass again.

Scales are nested deterministic samples of ``flow_id``::

    MOD(ABS(HASH(flow_id)), 10000) < permille10

so ``smoke`` is a subset of ``small`` is a subset of ``part``, and a given
``flow_id`` lands in the same tier on every rebuild. The sample is uniform over
the whole fixture rather than a prefix, because a prefix here means one capture:
the parts are scenario-aligned (``probe_iot23_stats.py --full``), so
``part-00003`` is 1M flows of exactly one malware capture and nothing else. A
prefix would train on one attack and call it the fixture.

The trade is that a uniform row sample thins each host's neighbourhood, which
matters for a graph model. It is survivable here only because the degree
distribution is extreme -- the scanning hosts have millions of flows each -- so
a 1% sample still leaves them with thousands. Say so when quoting slice numbers
against full-scale ones.

Columns
-------

Dropped, and why:

``label``
    The string form of the target. ``'Malicious'`` iff ``is_malicious``, with no
    exceptions in 325M rows. Pure leak.
``detailed_label``
    The attack family (``PartOfAHorizontalPortScan``, ``C&C``, ...), which is
    ``'-'`` exactly when the flow is benign. Also the target of matrix row 4.
``scenario_id``
    Names the IoT-23 capture. A capture is one malware sample run against one
    device, so scenario is close to deterministic for the label; as a
    categorical feature it is the answer, not a hint.
``uid``
    Zeek's per-connection id, unique per row. Noise with a high-cardinality
    encoder attached.
``ts``
    ``event_time`` in float seconds -- the same clock, and the time column is
    built from ``event_time`` instead.

``flow_id`` stays as the primary key. Note that it is literally
``'<scenario_id>:<uid>'``, so the dropped scenario survives inside the key; keys
are ID-typed and not featurized, but it is worth knowing.

Holdout
-------

``--holdout-pct`` is what makes the fixture scoreable at all. For a *static*
column target Kumo compiles the prediction table to ``WHERE target IS NULL`` --
train on the labeled rows, score the unlabeled ones -- and the fixture labels
every flow, so batch prediction over it as published selects nothing. Blanking a
deterministic slice of ``is_malicious`` restores the intended shape. The blanked
rows keep their true label in ``*_TRUTH``, so the scores can be checked against
labels the model never saw.

Unlike the ethereum build, ``*_TRUTH`` holds only the held-out
``(flow_id, is_malicious)`` pairs rather than a full copy of the table: at these
row counts a full copy is another 10 GB for two columns' worth of use.
"""
import argparse
import time

from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.IOT23_10GB'
FF = '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.PARQUET_FF)'
SCHEMA = 'HYU.IOT23'
RAW = f'{SCHEMA}.NETWORK_FLOWS_RAW'

#: Sample width in parts per 10,000 of `flow_id`, against 325,309,946 flows.
#: `full` is the whole fact table; the rest nest inside each other.
SCALES = {
    'smoke': 5,  # ~163K flows
    'small': 100,  # ~3.3M flows
    'part': 500,  # ~16.3M flows
    'half': 5000,  # ~163M flows
    'full': 10000,  # 325,309,946 flows
}

#: Read out of the staged parquet's variant column and cast to real types.
#: `event_time` is *microseconds* since epoch -- `ts` is the same instant in
#: float seconds, which is how the unit was settled -- so it needs unit 6.
#: Reading it as nanoseconds silently yields the year 48,355,176 and then fails
#: in the arrow decoder rather than at the cast.
FLOW_COLUMNS = """
    $1:flow_id::STRING                   AS flow_id,
    $1:is_malicious::BOOLEAN             AS is_malicious,
    $1:source_ip::STRING                 AS source_ip,
    $1:destination_ip::STRING            AS destination_ip,
    $1:source_port::NUMBER(38,0)         AS source_port,
    $1:destination_port::NUMBER(38,0)    AS destination_port,
    $1:protocol::STRING                  AS protocol,
    $1:service::STRING                   AS service,
    $1:duration::FLOAT                   AS duration,
    $1:source_bytes::NUMBER(38,0)        AS source_bytes,
    $1:destination_bytes::NUMBER(38,0)   AS destination_bytes,
    $1:connection_state::STRING          AS connection_state,
    $1:local_source::STRING              AS local_source,
    $1:local_destination::STRING         AS local_destination,
    $1:missed_bytes::NUMBER(38,0)        AS missed_bytes,
    $1:history::STRING                   AS history,
    $1:source_packets::NUMBER(38,0)      AS source_packets,
    $1:source_ip_bytes::NUMBER(38,0)     AS source_ip_bytes,
    $1:destination_packets::NUMBER(38,0) AS destination_packets,
    $1:destination_ip_bytes::NUMBER(38,0) AS destination_ip_bytes,
    TO_TIMESTAMP_NTZ($1:event_time::NUMBER(38,0), 6) AS event_time
"""

#: Deterministic in the key itself, so the same rows are sampled and the same
#: rows are held out on every rebuild and at every scale.
#:
#: The two predicates must hash *different* strings. Nesting two thresholds on
#: one hash makes them perfectly dependent, not independent: taking the sample
#: as ``MOD(h, 10000) < 5`` means every sampled row also has ``MOD(h, 100) < 5``,
#: so a ``< 20`` holdout on the same ``h`` selects all of them. The first build
#: did exactly that and produced a smoke slice with zero labeled rows -- which
#: presents as a divide-by-zero in the summary, several minutes after the
#: mistake. Salting the holdout hash decorrelates them.
SAMPLED = 'MOD(ABS(HASH(flow_id)), 10000) < {permille10}'
HELD_OUT = "MOD(ABS(HASH(flow_id || '/holdout')), 100) < {pct}"


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
        help='percent of flows whose is_malicious is blanked '
        'so batch prediction has rows to score')
    parser.add_argument(
        '--rebuild-raw', action='store_true',
        help='re-read the staged parquet even if '
        'NETWORK_FLOWS_RAW already exists')
    args = parser.parse_args()

    suffix = args.scale.upper()
    flows_table = f'{SCHEMA}.NETWORK_FLOWS_{suffix}'
    hosts_table = f'{SCHEMA}.HOSTS_{suffix}'
    truth_table = f'{flows_table}_TRUTH'

    session = Session.builder.config('connection_name', 'tlb').create()
    # The size has to be set on the session: `USE WAREHOUSE` is the only thing
    # that moves work, and the connection profile names none, so the default is
    # KUMO_XS.
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()
    session.sql(f'CREATE SCHEMA IF NOT EXISTS {SCHEMA}').collect()
    print('warehouse:',
          session.sql('SELECT CURRENT_WAREHOUSE()').collect()[0][0],
          flush=True)

    if args.rebuild_raw or not table_exists(session, RAW):
        run(
            session, f'staged parquet -> {RAW}', f"""
            CREATE OR REPLACE TABLE {RAW} AS
            SELECT {FLOW_COLUMNS}
            FROM '@{STAGE}/network_flows/' {FF}
        """)

    sampled = SAMPLED.format(permille10=SCALES[args.scale])
    held_out = HELD_OUT.format(pct=args.holdout_pct)

    run(
        session, f'flows ({args.holdout_pct}% held out) -> {flows_table}', f"""
        CREATE OR REPLACE TABLE {flows_table} AS
        SELECT * REPLACE (IFF({held_out}, NULL, is_malicious) AS is_malicious)
        FROM {RAW}
        WHERE {sampled}
    """)

    run(
        session, f'held-out labels -> {truth_table}', f"""
        CREATE OR REPLACE TABLE {truth_table} AS
        SELECT flow_id, is_malicious FROM {RAW}
        WHERE {sampled} AND {held_out}
    """)

    # The published `hosts` table is 258M rows of nothing but `host_id`: it is
    # exactly the set of endpoint IPs, with no attributes to contribute. So the
    # host node set is derived from the flows in scope instead of read back from
    # the stage. That guarantees no foreign key points at a missing row and no
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
    malicious, labeled, held_out_rows = session.sql(
        f"""SELECT COUNT_IF(is_malicious) AS malicious,
                   COUNT_IF(is_malicious IS NOT NULL) AS labeled,
                   COUNT_IF(is_malicious IS NULL) AS held_out
            FROM {flows_table}""").collect()[0]
    # A slice with nothing labeled, or nothing held out, is unusable -- the
    # first cannot train and the second cannot be scored -- so say which,
    # rather than dividing by zero and reporting a traceback for it.
    rate = f'{100 * malicious / labeled:.2f}%' if labeled else 'n/a'
    print(
        f'{flows_table}: {malicious:,} malicious of {labeled:,} labeled '
        f'({rate}), {held_out_rows:,} held out for scoring', flush=True)
    if not labeled or not held_out_rows:
        raise SystemExit(
            f'degenerate slice: labeled={labeled:,} held_out={held_out_rows:,};'
            ' the sample and holdout predicates are not independent')


if __name__ == '__main__':
    main()
