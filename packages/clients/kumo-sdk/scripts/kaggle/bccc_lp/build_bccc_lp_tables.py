# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Materializes matrix row 14's tier tables from row 3's BCCC fact table.

Row 14 is ``PREDICT LIST_DISTINCT(network_flows.dst_ip) / RANK TOP 100 /
FOR EACH hosts.host_id`` -- static link prediction over the source->destination
graph. It reads the **same** ``HYU.BCCC.NETWORK_FLOWS_RAW`` that row 3 built
(48,120,094 deduplicated rows, 1,479 s on ``KUMO_L``; see
``../bccc/build_bccc_tables.py``), and nothing here re-reads the stage.

What differs from row 3, and why
-------------------------------

Row 3's tiers are built for a *static column* target -- ``PREDICT
network_flows.label FOR EACH network_flows.flow_id`` -- so they hash-sample the
**fact** table and blank 20% of ``label``. Neither is right for row 14, and the
tables are given a distinct ``_LP_`` infix rather than being reused.

**The tier is a sample of source hosts, not of flows.** Row 14's entity is
``hosts.host_id`` and its target is the *set of destinations* a host reaches, so
thinning a host's flows changes its label -- NOAA S5's rule, and NYC S9's.
Measured on the full fact table, the median source host reaches exactly **one**
destination, so a 1% flow sample would delete most of the target outright. The
tier predicate is therefore

    MOD(ABS(HASH(src_ip)), 10000) < permille10

which keeps a hash-sampled subset of source hosts **with all of their flows**,
making every tier a complete, correct instance of the published task on fewer
entities. Measured tier sizes, from one pass over the fact table:

=============  ===========  ==========  ============  ============
tier           permille10   src hosts   flows         distinct dst
=============  ===========  ==========  ============  ============
``lp_tiny``             25         978        34,845           451
``lp_smoke``           100       3,927       363,990         4,670
``lp_small``           500      19,362     1,931,972        11,857
``lp_part``           2000      78,167     8,507,450        22,854
``lp_full``          10000     390,140    48,120,094        39,343
=============  ===========  ==========  ============  ============

**The holdout is over (src, dst) pairs, not over rows.** Row 14 needs no
holdout to *produce* predictions -- an aggregate target gets no prediction-mode
filter, which NOAA S2 measured and ``probe_bccc_lp_query.py`` re-confirms from
the compiled plan (``filters=[null]``). But a static link-prediction query has
no future to be scored against: the target is "every destination this host ever
reaches", which is exactly what the graph already contains, so scoring the model
against it measures memorisation and nothing else. The backend's own
``RandomLinkSplit([0.8, 0.1, 0.1])`` does hold links out internally, and its
``test_map@k`` is a real number -- but it is not reproducible from outside the
run and it cannot be compared against a no-model baseline.

So a deterministic slice of the *distinct pairs* is removed from the graph
entirely and kept in ``*_TRUTH``:

* only hosts with **two or more** distinct destinations are eligible, so no host
  is left with an empty neighbourhood;
* every eligible host contributes **at least one** held-out destination and
  **keeps at least one**, i.e. it is a per-entity leave-*n*-out split rather
  than a uniform 20% of all pairs. Uniform sampling would leave most hosts
  unscoreable, since 57% of source hosts have degree 1 and another large share
  have degree 2;
* **all** flows for a held-out pair are removed, not a fraction of them.
  Removing some would leave the edge in the graph and make its "prediction"
  a lookup;
* the same invariant is imposed on the **destination** side, and it is not
  cosmetic. A first version of this build constrained only the source, and at
  ``lp_smoke`` **50.2% of the held-out destinations then appeared nowhere else
  in the graph** -- an isolated, featureless node that no model can rank and no
  baseline can propose, so half the ground truth was unreachable by
  construction. A destination is therefore eligible only if at least two source
  hosts reach it, and at most ``degree - 1`` of its pairs are ever held out.

Both invariants are checked after the fact and printed; the "unreachable"
count is expected to be zero and is reported rather than assumed.

``hosts`` is still the union of *all* endpoints in the tier's scope, computed
**before** the holdout is applied, so a held-out destination is still a
candidate the model can rank. Otherwise the answer would not be in the
candidate set at all.

Columns
-------

All 323 columns are materialized on the first pass, renamed. Two reasons, both
carried forward:

* row 4 wasted 726 s re-reading the stage for two columns it had dropped, so
  every tier here carries everything the run might want (``event_time``,
  ``label``, both ports, the protocol) even though the default run declares
  none of them;
* **142 of the 323 column names contain a ``ColStatType`` token**, which
  ``kumo_core/pquery.py::_column_stats`` matches by substring against the whole
  statistics key -- so ``PAYLOAD_BYTES_MEAN``'s parsed "mean" is really its
  standard deviation, and ``MAX_HEADER_BYTES`` parses four invented statistics
  out of a categorical column's payload. Row 3 S6 is the full account. The
  workaround is the same one: rename in the tier table, via row 3's
  ``stat_safe_name``, imported rather than copied so the two rows cannot drift.
  **This is a deviation from the fixture's published column names.**

``event_time`` is materialized but is *not* declared as the graph's time column
by default, and that is not a preference either: with a time column on
``network_flows`` the published static query is rejected outright --
*"Time range required for LIST_DISTINCT with time column"*. See
``probe_bccc_lp_query.py`` and ``FINDINGS.md``.
"""
import argparse
import os
import sys
import time

from snowflake.snowpark import Session

# Row 3's build is imported rather than copied: the 315-name feature lists, the
# `RAW` column order and the stat-token rename all have to stay identical to
# the table they read, and a second copy of a 315-element list is a bug waiting
# to happen. Row 3's files are not modified.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bccc'))
# yapf and isort disagree about this one statement -- isort explodes it one
# name per line, yapf packs the names back onto a single continuation line
# because of the trailing `noqa`, and each undoes the other on every run. isort
# owns it.
# yapf: disable
from build_bccc_tables import (  # noqa: E402
    RAW,
    RAW_COLUMNS,
    SCHEMA,
    stat_safe_name,
)

# yapf: enable

#: Sample width in parts per 10,000 of `src_ip`, against 390,140 source hosts.
#: Measured flow counts are in the module docstring; they are *not* proportional
#: to the sample width, because flows per source host span 1 to 2,396,254.
SCALES = {
    'lp_tiny': 25,  # ~978 hosts, ~35K flows
    'lp_smoke': 100,  # ~3,927 hosts, ~364K flows
    'lp_small': 500,  # ~19,362 hosts, ~1.93M flows
    'lp_part': 2000,  # ~78,167 hosts, ~8.51M flows
    'lp_full': 10000,  # 390,140 hosts, 48.12M flows
}

#: The tier predicate. Hashes `src_ip`, so a host is either wholly in a tier or
#: wholly out of it, and the tiers nest.
SAMPLED = 'MOD(ABS(HASH(src_ip)), 10000) < {permille10}'


def tier_select() -> str:
    r"""The tier ``SELECT`` list: every RAW column, renamed.

    Unlike row 3's ``tier_select`` nothing is blanked here -- row 14's holdout
    removes whole rows rather than nulling a column, because its target is the
    edge itself.
    """
    parts = []
    for name in RAW_COLUMNS:
        safe = stat_safe_name(name)
        parts.append(f'{name} AS {safe}' if safe != name else name)
    return ',\n    '.join(parts)


def run(session: Session, label: str, sql: str) -> None:
    start = time.monotonic()
    print(f'[{label}] running...', flush=True)
    session.sql(sql).collect()
    print(f'[{label}] done in {time.monotonic() - start:.1f}s', flush=True)


def one(session: Session, sql: str):
    return session.sql(sql).collect()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', choices=list(SCALES), default='lp_smoke')
    parser.add_argument('--warehouse', default='KUMO_L')
    parser.add_argument(
        '--holdout-pct', type=int, default=20,
        help='percent of each eligible host\'s distinct '
        'destinations to remove from the graph and keep '
        'in *_TRUTH; at least one and never all of them')
    parser.add_argument(
        '--no-holdout', action='store_true',
        help='build the tier with every edge present. The run '
        'then has nothing to be scored against except '
        'the edges it trained on')
    args = parser.parse_args()

    suffix = args.scale.upper()
    flows_table = f'{SCHEMA}.NETWORK_FLOWS_{suffix}'
    hosts_table = f'{SCHEMA}.HOSTS_{suffix}'
    truth_table = f'{flows_table}_TRUTH'
    sampled = SAMPLED.format(permille10=SCALES[args.scale])

    session = Session.builder.config('connection_name', 'tlb').create()
    # `USE WAREHOUSE` is the only thing that moves work; the connection profile
    # names none, so the default is KUMO_XS.
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()
    session.sql(f'CREATE SCHEMA IF NOT EXISTS {SCHEMA}').collect()
    print('warehouse:',
          one(session, 'SELECT CURRENT_WAREHOUSE()')[0], flush=True)
    print(f'source   : {RAW} (row 3\'s fact table, not rebuilt)', flush=True)

    # `hosts` first, and from the *pre-holdout* scope. A held-out destination
    # has to stay a candidate the model can rank, or the answer is not in the
    # candidate set and every metric is zero by construction.
    run(
        session, f'endpoint IPs -> {hosts_table}', f"""
        CREATE OR REPLACE TABLE {hosts_table} AS
        SELECT DISTINCT ip AS host_id FROM (
            SELECT src_ip AS ip FROM {RAW} WHERE {sampled}
            UNION ALL
            SELECT dst_ip FROM {RAW} WHERE {sampled}
        )
        WHERE ip IS NOT NULL
    """)

    if args.no_holdout:
        held_predicate = 'FALSE'
        run(
            session, f'held-out pairs -> {truth_table} (empty)', f"""
            CREATE OR REPLACE TABLE {truth_table} AS
            SELECT src_ip, dst_ip, 0 AS flows, 0 AS deg
            FROM {RAW} WHERE FALSE
        """)
    else:
        # A per-entity leave-n-out split over distinct pairs. `rk` orders a
        # host's destinations by a salted hash of the pair, so the choice is
        # deterministic in the data and independent of the tier predicate --
        # row 2's first build nested two thresholds on one hash and produced a
        # slice with zero labels.
        #
        # `LEAST(GREATEST(1, FLOOR(deg * pct / 100)), deg - 1)` is what
        # guarantees the two invariants: at least one destination held out, and
        # at least one left behind. Hosts with `deg = 1` are excluded outright.
        run(
            session, f'held-out pairs -> {truth_table}', f"""
            CREATE OR REPLACE TABLE {truth_table} AS
            WITH pairs AS (
                SELECT src_ip, dst_ip, COUNT(*) AS flows
                FROM {RAW} WHERE {sampled}
                GROUP BY 1, 2
            ), degrees AS (
                SELECT src_ip, dst_ip, flows,
                       COUNT(*) OVER (PARTITION BY src_ip) AS deg,
                       COUNT(*) OVER (PARTITION BY dst_ip) AS dst_deg,
                       ROW_NUMBER() OVER (
                           PARTITION BY src_ip
                           ORDER BY ABS(HASH(src_ip || '>' || dst_ip
                                             || '/holdout'))) AS rk
                FROM pairs
            ), src_side AS (
                -- Every eligible source host gives up at least one destination
                -- and keeps at least one.
                SELECT src_ip, dst_ip, flows, deg, dst_deg,
                       ROW_NUMBER() OVER (
                           PARTITION BY dst_ip
                           ORDER BY ABS(HASH(dst_ip || '<' || src_ip
                                             || '/holdout'))) AS dst_rk
                FROM degrees
                WHERE deg >= 2 AND dst_deg >= 2
                  AND rk <= LEAST(GREATEST(1, FLOOR(deg * {args.holdout_pct}
                                                    / 100.0)), deg - 1)
            )
            -- ...and every destination keeps at least one source, so it is
            -- still a node with edges and is rankable at all.
            SELECT src_ip, dst_ip, flows, deg
            FROM src_side
            WHERE dst_rk <= dst_deg - 1
        """)
        held_predicate = f"""EXISTS (
            SELECT 1 FROM {truth_table} h
            WHERE h.src_ip = f.src_ip AND h.dst_ip = f.dst_ip)"""

    run(
        session, f'flows -> {flows_table}', f"""
        CREATE OR REPLACE TABLE {flows_table} AS
        SELECT {tier_select()}
        FROM {RAW} f
        WHERE {sampled} AND NOT {held_predicate}
    """)

    for name in (flows_table, hosts_table, truth_table):
        count = one(session, f'SELECT COUNT(*) FROM {name}')[0]
        print(f'{name}: {count:,} rows', flush=True)

    # What the tier actually is, as a link-prediction instance. Every number
    # here is one the metrics have to be read against, and none of them is
    # visible in any run artefact.
    shape = one(
        session, f"""
        SELECT COUNT(*) AS flows,
               COUNT(DISTINCT src_ip) AS srcs,
               COUNT(DISTINCT dst_ip) AS dsts
        FROM {flows_table}
    """)
    n_hosts = one(session, f'SELECT COUNT(*) FROM {hosts_table}')[0]
    print(
        f'\ngraph after the holdout: {shape[0]:,} flows, '
        f'{shape[1]:,} source hosts, {shape[2]:,} destination hosts, '
        f'{n_hosts:,} candidate nodes', flush=True)

    degrees = one(
        session, f"""
        WITH d AS (SELECT src_ip, COUNT(DISTINCT dst_ip) AS n
                   FROM {flows_table} GROUP BY 1)
        SELECT AVG(n), MIN(n),
               APPROX_PERCENTILE(n, 0.5), APPROX_PERCENTILE(n, 0.95), MAX(n)
        FROM d
    """)
    print(
        f'destinations per host in the graph: mean {float(degrees[0]):.2f} '
        f'min {degrees[1]} median {degrees[2]:.0f} p95 {degrees[3]:.0f} '
        f'max {degrees[4]}', flush=True)

    if args.no_holdout:
        print(
            '\nno holdout: nothing to score against that the model did not '
            'train on', flush=True)
        return

    scoreable = one(
        session, f"""
        SELECT COUNT(DISTINCT src_ip) AS hosts, COUNT(*) AS pairs,
               AVG(deg) AS mean_deg
        FROM {truth_table}
    """)
    print(
        f'\nheld out: {scoreable[1]:,} pairs over {scoreable[0]:,} hosts '
        f'({100.0 * scoreable[0] / max(shape[1], 1):.1f}% of source hosts '
        f'are scoreable)', flush=True)

    # The one way this holdout can be silently degenerate: a held-out
    # destination that appears nowhere else in the graph is an isolated node
    # with no edges and no features, and no model can rank it. Reported rather
    # than repaired, because repairing it would bias the holdout toward
    # popular destinations.
    orphaned = one(
        session, f"""
        SELECT COUNT(*) FROM (
            SELECT DISTINCT dst_ip FROM {truth_table}
            EXCEPT
            SELECT DISTINCT dst_ip FROM {flows_table})
    """)[0]
    n_truth_dst = one(
        session, f"""
        SELECT COUNT(*) FROM (SELECT DISTINCT dst_ip FROM {truth_table})
    """)[0]
    print(
        f'held-out destinations unreachable anywhere in the graph: '
        f'{orphaned:,} of {n_truth_dst:,} distinct '
        f'({100.0 * orphaned / max(n_truth_dst, 1):.1f}%) -- these are '
        'isolated candidate nodes and are not rankable by any model',
        flush=True)

    if not scoreable[1]:
        raise SystemExit(
            'no held-out pairs: every source host in this tier has exactly '
            'one destination, so there is nothing to score')


if __name__ == '__main__':
    main()
