# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Fixture-shape probe for IoT-23: every matrix claim, checked.

Row 1 taught that the matrix describes the fixture approximately -- ethereum's
declared primary key had duplicates and its declared join table was unusable --
so nothing here is taken on trust. The questions that decide how the fixture can
be sliced and modelled:

* is ``flow_id`` actually unique, and is ``host_id``?
* what is ``event_time``: seconds, millis, nanos? Which years does it span?
* class balance of ``is_malicious``, and how it moves across part files;
* which columns leak the target. ``label`` and ``detailed_label`` are the
  obvious ones, but ``scenario_id`` names the IoT-23 capture, and a capture is
  one malware sample, so it may be a perfect predictor as well;
* are the parts contiguous in time, or is each a full-history sample? This
  decides whether a slice is a coherent graph or a random shred of one.
* how many of the 258M ``hosts`` rows a slice actually touches.

``--full`` runs the whole-fixture scans (all 23 parts, ~12 GB read); without it
the scans are restricted to one part so the script stays a couple of minutes.
"""
import argparse
import time

from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.IOT23_10GB'
FF = '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.PARQUET_FF)'
ONE_PART = 'part-00003.parquet'  # 50 MB: small, but not one of the tiny ones


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--warehouse', default='KUMO_L')
    parser.add_argument(
        '--cardinality', action='store_true',
        help='distinct counts per column of '
        'HYU.IOT23.NETWORK_FLOWS_RAW, which decide the '
        'inferred stypes')
    args = parser.parse_args()

    session = Session.builder.config('connection_name', 'tlb').create()
    session.sql(f'USE WAREHOUSE {args.warehouse}').collect()

    flows = (f"'@{STAGE}/network_flows/' {FF}"
             if args.full else f"'@{STAGE}/network_flows/{ONE_PART}' {FF}")

    def show(title: str, sql: str) -> None:
        print(f'\n--- {title} ---', flush=True)
        start = time.monotonic()
        try:
            rows = session.sql(sql).collect()
        except Exception as error:  # keep the later probes running
            print(f'  ERROR {type(error).__name__}: {str(error)[:300]}',
                  flush=True)
            return
        for row in rows[:40]:
            print(' ', row, flush=True)
        if len(rows) > 40:
            print(f'  ... {len(rows) - 40} more', flush=True)
        print(f'  ({time.monotonic() - start:.1f}s)', flush=True)

    if args.cardinality:
        # Stype inference is a function of distinct counts, so these numbers
        # decide the model. `kumo_core.stype_inference`: a string column with
        # more than MAX_CATEGORIES (4000) distinct values is *text*, which
        # pulls in a text encoder; an integer column is categorical only if it
        # is under 4000 distinct *and* under UNIQUE_INT_THRESHOLD (5%) unique.
        # Run against the whole fact table rather than a slice: a column can
        # cross the threshold with scale and silently change the graph.
        show(
            'NETWORK_FLOWS_RAW: distinct values per column', """
            SELECT COUNT(*) AS n_rows,
                   COUNT(DISTINCT protocol) AS protocol,
                   COUNT(DISTINCT service) AS service,
                   COUNT(DISTINCT connection_state) AS connection_state,
                   COUNT(DISTINCT history) AS history,
                   COUNT(DISTINCT local_source) AS local_source,
                   COUNT(DISTINCT local_destination) AS local_destination,
                   COUNT(DISTINCT source_port) AS source_port,
                   COUNT(DISTINCT destination_port) AS destination_port,
                   COUNT(DISTINCT source_ip) AS source_ip,
                   COUNT(DISTINCT destination_ip) AS destination_ip
            FROM HYU.IOT23.NETWORK_FLOWS_RAW
            """)
        return

    show('network_flows: one raw row', f'SELECT $1 FROM {flows} LIMIT 1')

    show('hosts: one raw row',
         f"SELECT $1 FROM '@{STAGE}/hosts/' {FF} LIMIT 3")

    # A dimension table's key has to be unique for the graph to use it, and
    # ethereum's was not. `APPROX_COUNT_DISTINCT` cannot answer this -- its
    # error is ~1%, which at 258M rows is wider than any duplicate count worth
    # finding -- so the exact distinct count is taken despite the sort.
    if args.full:
        show(
            'hosts: key uniqueness', f"""
            SELECT COUNT(*) AS n_rows,
                   COUNT(DISTINCT $1:host_id::STRING) AS distinct_host_ids,
                   COUNT_IF($1:host_id IS NULL) AS null_host_id
            FROM '@{STAGE}/hosts/' {FF}
            """)

    # `flow_id` reads as "<scenario_id>:<uid>", so it should be unique even if
    # `uid` collides across captures. Both counts are exact here.
    show(
        'network_flows: key uniqueness + null keys', f"""
        SELECT COUNT(*) AS n_rows,
               COUNT(DISTINCT $1:flow_id::STRING) AS flow_ids,
               COUNT_IF($1:flow_id IS NULL) AS null_flow_id,
               COUNT(DISTINCT $1:uid::STRING) AS uids
        FROM {flows}
        """)

    # `event_time` is an integer of unknown unit; `ts` is a float epoch second.
    # 1525879831015811 against ts=1.525879831015811e9 says microseconds, which
    # matters because reading it as nanoseconds lands in the year 48,355,176 --
    # and Snowflake will happily do that and then fail in the arrow decoder.
    show(
        'network_flows: time columns', f"""
        SELECT MIN($1:ts::FLOAT) AS min_ts, MAX($1:ts::FLOAT) AS max_ts,
               MIN($1:event_time::NUMBER(38,0)) AS min_event_time,
               MAX($1:event_time::NUMBER(38,0)) AS max_event_time,
               TO_TIMESTAMP_NTZ(MIN($1:event_time::NUMBER(38,0)), 6) AS from_us,
               TO_TIMESTAMP_NTZ(MAX($1:event_time::NUMBER(38,0)), 6) AS to_us,
               COUNT_IF($1:event_time IS NULL) AS null_event_time
        FROM {flows}
        """)

    show(
        'network_flows: class balance', f"""
        SELECT COUNT(*) AS n_rows,
               COUNT_IF($1:is_malicious::BOOLEAN) AS malicious,
               COUNT_IF($1:is_malicious IS NULL) AS null_label
        FROM {flows}
        """)

    # If `label` reproduces `is_malicious` exactly it is a leak, not a feature.
    show(
        'network_flows: label vs is_malicious', f"""
        SELECT $1:label::STRING AS label,
               $1:is_malicious::BOOLEAN AS is_malicious,
               COUNT(*) AS n
        FROM {flows} GROUP BY 1, 2 ORDER BY 3 DESC
        """)

    show(
        'network_flows: detailed_label', f"""
        SELECT $1:detailed_label::STRING AS detailed_label, COUNT(*) AS n
        FROM {flows} GROUP BY 1 ORDER BY 2 DESC
        """)

    # One IoT-23 scenario is one capture of one malware sample, so a scenario
    # whose malicious rate is 0 or 1 hands the answer to the model.
    show(
        'network_flows: malicious rate per scenario', f"""
        SELECT $1:scenario_id::STRING AS scenario_id,
               COUNT(*) AS n,
               AVG(IFF($1:is_malicious::BOOLEAN, 1.0, 0.0)) AS malicious_rate
        FROM {flows} GROUP BY 1 ORDER BY 2 DESC
        """)

    if args.full:
        # Ethereum's parts were each a full-history ~10% sample despite their
        # "daily" naming. Whether these are time-contiguous decides whether one
        # part is a usable standalone slice.
        show(
            'network_flows: per-part shape', f"""
            SELECT SPLIT_PART(METADATA$FILENAME, '/', -1) AS part,
                   COUNT(*) AS n_rows,
                   TO_TIMESTAMP_NTZ(MIN($1:event_time::NUMBER(38,0)), 6)
                       AS min_time,
                   TO_TIMESTAMP_NTZ(MAX($1:event_time::NUMBER(38,0)), 6)
                       AS max_time,
                   AVG(IFF($1:is_malicious::BOOLEAN, 1.0, 0.0)) AS mal_rate,
                   APPROX_COUNT_DISTINCT($1:scenario_id::STRING) AS scenarios
            FROM '@{STAGE}/network_flows/' {FF}
            GROUP BY 1 ORDER BY 1
            """)

        # The graph joins both IPs into `hosts`. If the flows reference IPs that
        # `hosts` does not contain, the edge dangles; if `hosts` contains far
        # more than the flows reference, most of those 258M nodes are dead
        # weight that still has to be ingested.
        show(
            'endpoint IPs actually referenced by flows', f"""
            SELECT APPROX_COUNT_DISTINCT(ip) AS approx_distinct_endpoint_ips
            FROM (
                SELECT $1:source_ip::STRING AS ip
                FROM '@{STAGE}/network_flows/' {FF}
                UNION ALL
                SELECT $1:destination_ip::STRING
                FROM '@{STAGE}/network_flows/' {FF}
            )
            """)


if __name__ == '__main__':
    main()
