# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""What row 14's target is worth with no model at all, over the whole fixture.

Every previous block in this series found that the number which made the
model's number readable was a no-model baseline, and NYC row 16 found something
stronger: one SQL query, run before any training, said the task was degenerate.
Its median entity reached 215 of 264 possible destinations, so ``RANK TOP 20``
against a 215-element truth set was not a ranking problem and the model duly
lost to a ``GROUP BY``.

This asks the same question of BCCC, on all 48,120,094 rows of
``HYU.BCCC.NETWORK_FLOWS_RAW`` (row 3's fact table; nothing here re-reads the
stage), plus the structural questions row 14 raises that row 3's probe did not:

``--claims``       the matrix's row counts and the identity of ``hosts``
``--shape``        the source/destination cardinalities and degree distributions
                   -- the saturation measurement
``--popularity``   how concentrated destination traffic is, i.e. what a global
                   ranking is worth
``--structure``    reciprocity, subnet locality and capture privacy: the
                   no-model channels a GNN could be recovering instead of
                   learning anything about hosts
``--tiers``        what each tier of ``build_bccc_lp_tables.py`` costs and
                   contains, so a tier can be chosen before it is built
"""
import argparse
import time
from typing import Any, List

from snowflake.snowpark import Session

RAW = 'HYU.BCCC.NETWORK_FLOWS_RAW'
TOTAL_ROWS = 48_120_094


def q(session: Session, label: str, sql: str, limit: int = 40) -> List[Any]:
    start = time.monotonic()
    rows = session.sql(sql).collect()
    print(f'\n--- {label}  ({time.monotonic() - start:.1f}s)', flush=True)
    for row in rows[:limit]:
        print('   ', row, flush=True)
    return rows


def claims(session: Session) -> None:
    r"""The matrix's arithmetic, checked the same way rows 1-5 checked theirs.

    The matrix's ``48,590,026`` is the **sum** of the two tables, as it was in
    every previous row, and ``hosts`` is exactly the union of the two endpoint
    columns rather than an independently published node list.
    """
    q(
        session, 'fact table', f"""
        SELECT COUNT(*) AS flows,
               COUNT(DISTINCT flow_id) AS distinct_keys,
               COUNT(DISTINCT src_ip) AS src_hosts,
               COUNT(DISTINCT dst_ip) AS dst_hosts,
               COUNT_IF(src_ip IS NULL) AS null_src,
               COUNT_IF(dst_ip IS NULL) AS null_dst,
               MIN(event_time) AS first_flow, MAX(event_time) AS last_flow
        FROM {RAW}
    """)
    q(
        session, 'hosts is the union of the endpoints', f"""
        SELECT COUNT(*) AS union_of_endpoints FROM (
            SELECT src_ip AS ip FROM {RAW}
            UNION
            SELECT dst_ip FROM {RAW})
    """)
    q(session, 'distinct (src, dst) pairs -- the edge set row 14 predicts',
      f'SELECT COUNT(*) FROM (SELECT DISTINCT src_ip, dst_ip FROM {RAW})')


def shape(session: Session) -> None:
    r"""The saturation measurement: truth-set size against candidate-set size.

    NYC row 16's degeneracy was visible here and nowhere else. The number to
    read is the median destination count against the 407,244-node candidate
    set, and against the published ``RANK TOP 100``.
    """
    q(
        session, 'destinations per source host', f"""
        WITH d AS (SELECT src_ip, COUNT(DISTINCT dst_ip) AS n, COUNT(*) AS f
                   FROM {RAW} GROUP BY 1)
        SELECT COUNT(*) AS source_hosts, AVG(n) AS mean_dst, MIN(n),
               APPROX_PERCENTILE(n, 0.05) AS p05,
               APPROX_PERCENTILE(n, 0.25) AS p25,
               APPROX_PERCENTILE(n, 0.50) AS median,
               APPROX_PERCENTILE(n, 0.75) AS p75,
               APPROX_PERCENTILE(n, 0.95) AS p95,
               APPROX_PERCENTILE(n, 0.99) AS p99, MAX(n) AS max_dst,
               COUNT_IF(n = 1) AS degree_1,
               COUNT_IF(n >= 2) AS degree_2plus,
               COUNT_IF(n > 100) AS over_top_k,
               AVG(f) AS mean_flows, APPROX_PERCENTILE(f, 0.5) AS median_flows,
               MAX(f) AS max_flows
        FROM d
    """)
    q(
        session, 'source hosts per destination', f"""
        WITH d AS (SELECT dst_ip, COUNT(DISTINCT src_ip) AS n FROM {RAW}
                   GROUP BY 1)
        SELECT COUNT(*) AS destination_hosts, AVG(n) AS mean_src,
               COUNT_IF(n = 1) AS reached_by_one_host,
               APPROX_PERCENTILE(n, 0.5) AS median, MAX(n) AS max_src
        FROM d
    """)


def popularity(session: Session) -> None:
    r"""What a single global ranking of destinations is worth.

    ``top100_pairs / total_pairs`` is the recall a popularity ranker achieves
    at k=100 if every host were scored against the same list -- the floor row 14
    has to clear.
    """
    q(
        session, 'the twelve busiest destinations', f"""
        WITH d AS (SELECT dst_ip, COUNT(*) AS flows,
                          COUNT(DISTINCT src_ip) AS srcs
                   FROM {RAW} GROUP BY 1),
             r AS (SELECT dst_ip, flows, srcs,
                          ROW_NUMBER() OVER (ORDER BY srcs DESC) AS rk FROM d)
        SELECT rk, dst_ip, srcs, flows, flows / {TOTAL_ROWS}.0 AS flow_share
        FROM r WHERE rk <= 12 ORDER BY rk
    """)
    q(
        session, 'pair coverage of the global top-k', f"""
        WITH d AS (SELECT dst_ip, COUNT(DISTINCT src_ip) AS srcs FROM {RAW}
                   GROUP BY 1),
             r AS (SELECT dst_ip, srcs,
                          ROW_NUMBER() OVER (ORDER BY srcs DESC) AS rk FROM d)
        SELECT SUM(IFF(rk <= 1, srcs, 0)) AS top1,
               SUM(IFF(rk <= 10, srcs, 0)) AS top10,
               SUM(IFF(rk <= 100, srcs, 0)) AS top100,
               SUM(IFF(rk <= 1000, srcs, 0)) AS top1000,
               SUM(srcs) AS total_pairs
        FROM r
    """)


def structure(session: Session) -> None:
    r"""The no-model channels a GNN could be recovering instead of host
    behaviour.

    Row 3 S9 left one question explicitly open: the fixture's 30 capture
    sessions are single-label and, since both endpoints are edges into
    ``hosts``, the graph may reconstruct the capture even with ``capture_id``
    dropped. For row 14 the analogous question is sharper -- if a host talks to
    exactly one destination and both belong to one capture, then "predict the
    destination" is "predict the capture", and a subnet prefix answers it.
    """
    q(
        session, 'reciprocity: how often the reverse pair also exists', f"""
        WITH p AS (SELECT DISTINCT src_ip, dst_ip FROM {RAW})
        SELECT COUNT(*) AS pairs,
               COUNT_IF(EXISTS (SELECT 1 FROM p r
                                WHERE r.src_ip = p.dst_ip
                                  AND r.dst_ip = p.src_ip)) AS reciprocated
        FROM p
    """)
    q(
        session, 'subnet locality: share of pairs inside one /24 and one /16',
        f"""
        WITH p AS (SELECT DISTINCT src_ip, dst_ip FROM {RAW})
        SELECT COUNT(*) AS pairs,
               COUNT_IF(REGEXP_SUBSTR(src_ip, '^[0-9]+\\\\.[0-9]+\\\\.[0-9]+')
                        = REGEXP_SUBSTR(dst_ip,
                                        '^[0-9]+\\\\.[0-9]+\\\\.[0-9]+'))
                   AS same_24,
               COUNT_IF(REGEXP_SUBSTR(src_ip, '^[0-9]+\\\\.[0-9]+')
                        = REGEXP_SUBSTR(dst_ip, '^[0-9]+\\\\.[0-9]+'))
                   AS same_16
        FROM p
    """)
    q(
        session, 'capture privacy of a source host '
        '(capture_id is the prefix of flow_id)', f"""
        WITH h AS (SELECT src_ip,
                          COUNT(DISTINCT SPLIT_PART(flow_id, ':', 1)) AS caps
                   FROM {RAW} GROUP BY 1)
        SELECT COUNT(*) AS source_hosts, AVG(caps) AS mean_captures,
               COUNT_IF(caps = 1) AS single_capture_hosts,
               COUNT_IF(caps = 1) / COUNT(*) AS share, MAX(caps) AS max
        FROM h
    """)
    q(
        session, 'how much of a host\'s destination set one capture explains',
        f"""
        WITH e AS (SELECT DISTINCT src_ip, dst_ip,
                          SPLIT_PART(flow_id, ':', 1) AS capture FROM {RAW})
        SELECT COUNT(*) AS pairs,
               COUNT(DISTINCT capture) AS captures,
               COUNT(DISTINCT src_ip) AS srcs
        FROM e
    """)


def tiers(session: Session) -> None:
    r"""What each tier contains, before it is built.

    A tier is a hash sample of *source hosts* carried with all of their flows,
    so its flow count is not proportional to its sample width -- flows per
    source host span 1 to 2,396,254.
    """
    q(
        session, 'tier sizes', f"""
        WITH b AS (SELECT MOD(ABS(HASH(src_ip)), 10000) AS h, src_ip, dst_ip
                   FROM {RAW})
        SELECT n.n AS permille10,
               COUNT_IF(h < n.n) AS flows,
               COUNT(DISTINCT IFF(h < n.n, src_ip, NULL)) AS source_hosts,
               COUNT(DISTINCT IFF(h < n.n, dst_ip, NULL)) AS destination_hosts
        FROM b, (SELECT 25 AS n UNION ALL SELECT 100 UNION ALL SELECT 500
                 UNION ALL SELECT 2000 UNION ALL SELECT 10000) n
        GROUP BY n.n ORDER BY n.n
    """)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ('claims', 'shape', 'popularity', 'structure', 'tiers'):
        parser.add_argument(f'--{name}', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--warehouse', default='KUMO_L')
    args = parser.parse_args()

    session = Session.builder.config('connection_name', 'tlb').create()
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()
    print('warehouse:',
          session.sql('SELECT CURRENT_WAREHOUSE()').collect()[0][0],
          flush=True)

    selected = [
        name
        for name in ('claims', 'shape', 'popularity', 'structure', 'tiers')
        if args.all or getattr(args, name)
    ]
    if not selected:
        selected = ['claims', 'shape']
    for name in selected:
        print(f'\n{"=" * 74}\n{name}\n{"=" * 74}', flush=True)
        globals()[name](session)


if __name__ == '__main__':
    main()
