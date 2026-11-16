# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Fixture-shape probe for BCCC-CSE-CIC-IDS2018: every matrix claim, checked.

Rows 1 and 2 both deviated from the matrix -- ethereum's declared primary key
had 304 duplicates, and iot23's row count turned out to be the sum over both
tables -- so nothing here is taken on trust either.

What this fixture makes new, and why each question is asked:

* **It is multiclass.** So the first questions are how many classes there are,
  how skewed they are, and whether the tail is long enough that a uniform hash
  sample can lose a class outright. A class present in train and absent from the
  holdout, or the reverse, is not a metric problem -- it changes what the task
  *is* between tiers.
* **It is 325 columns wide**, against iot23's 21. 48M rows x 325 columns is
  106 GB uncompressed against 20 GB of parquet, so the binding cost here may be
  the *width* rather than the row count. Which columns are constant, all-null,
  or perfectly correlated with a neighbour decides how much of that width is
  real.
* **22 columns infer as VARIANT**, meaning ``INFER_SCHEMA`` saw more than one
  physical type across the 34 part files. Almost always that is a float column
  carrying ``Infinity``/``NaN`` as a *string* in some files. ``TYPEOF`` says
  which, and it has to be settled before the cast list is written: casting a
  string ``'Infinity'`` with ``::FLOAT`` succeeds and yields infinity, which
  then poisons every normalisation statistic downstream.
* **Leakage.** ``label`` is the target; ``is_attack`` is visibly its binary
  collapse; ``capture_id`` is the iot23 ``scenario_id`` problem by another name.
  Each is cross-tabbed against the target rather than assumed.
* **Per-part shape.** iot23's parts were one capture each and ethereum's were
  full-history samples. Which of the two this is decides whether a prefix slice
  is a usable fixture or one attack day dressed up as one.

``--full`` runs the whole-fixture scans (all 34 parts, ~20 GB read through the
external stage); without it the scans are restricted to one part so the script
stays a few minutes. ``--raw`` runs the scans that need the materialized
``NETWORK_FLOWS_RAW`` table instead of the stage, which is far cheaper per scan.
"""
import argparse
import importlib.util
import time
from pathlib import Path

from snowflake.snowpark import Session

#: The build script owns the column lists, so they are read from it rather than
#: kept in a second copy that can drift out of step with the cast list.
_spec = importlib.util.spec_from_file_location(
    'build_bccc_tables',
    Path(__file__).with_name('build_bccc_tables.py'))
assert _spec is not None and _spec.loader is not None
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

STAGE = 'HYU.KAGGLE_FIXTURES.BCCC_IDS_20GB'
FF = '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.PARQUET_FF)'
RAW = 'HYU.BCCC.NETWORK_FLOWS_RAW'
#: 275 MB and 34 parts in, so neither the biggest part nor one of the tiny ones.
ONE_PART = 'part-00003.parquet'

#: The columns `INFER_SCHEMA` could not settle on a single physical type for.
#: See the module docstring: each is a float statistic that some part files
#: carry as a string.
VARIANT_COLUMNS = [
    'payload_bytes_skewness',
    'payload_bytes_cov',
    'fwd_payload_bytes_skewness',
    'fwd_payload_bytes_cov',
    'bwd_payload_bytes_skewness',
    'bwd_payload_bytes_cov',
    'fwd_skewness_header_bytes',
    'bwd_skewness_header_bytes',
    'packets_IAT_skewness',
    'fwd_packets_IAT_skewness',
    'bwd_packets_IAT_skewness',
    'delta_start',
    'handshake_duration',
    'skewness_packets_delta_time',
    'skewness_packets_delta_len',
    'skewness_header_bytes_delta_len',
    'skewness_payload_bytes_delta_len',
    'cov_payload_bytes_delta_len',
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--raw', action='store_true',
                        help='run the scans that read NETWORK_FLOWS_RAW')
    parser.add_argument('--warehouse', default='KUMO_L')
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
            print(f'  ERROR {type(error).__name__}: {str(error)[:400]}',
                  flush=True)
            return
        for row in rows[:60]:
            print(' ', row, flush=True)
        if len(rows) > 60:
            print(f'  ... {len(rows) - 60} more', flush=True)
        print(f'  ({time.monotonic() - start:.1f}s)', flush=True)

    if args.raw:
        # Stype inference is a function of distinct counts, so these numbers
        # decide the model. `kumo_core.stype_inference`: a string column with
        # more than MAX_CATEGORIES (4000) distinct values is *text*, which pulls
        # in a text encoder; an integer column is categorical only if it is
        # under 4000 distinct *and* under UNIQUE_INT_THRESHOLD (5%) unique.
        # Floats are numerical regardless, so only the integer columns can flip
        # between tiers -- and a flipped stype changes the graph key and makes
        # two tiers incomparable.
        #
        # Inference reads a 10,000-row sample, so a column near the 4,000
        # boundary can land on either side depending on the tier. With 56
        # integer columns that is a near-certainty rather than a risk, which is
        # why every one of them is measured here against the whole fact table
        # and then pinned in `run_bccc_local.py`.
        #
        # `APPROX_COUNT_DISTINCT` is deliberately not used: its error is ~1%,
        # and 1% of 4,000 is exactly the resolution this question needs.
        ints = ['src_port', 'dst_port'] + list(build.INT_FEATURES)
        counts = ',\n                   '.join(f'COUNT(DISTINCT {c}) AS {c}'
                                               for c in ints)
        print('\n--- RAW: distinct counts per integer column ---', flush=True)
        start = time.monotonic()
        row = session.sql(f"""
            SELECT COUNT(*) AS n_rows, {counts},
                   COUNT(DISTINCT protocol) AS protocol,
                   COUNT(DISTINCT src_ip) AS src_ip,
                   COUNT(DISTINCT dst_ip) AS dst_ip,
                   COUNT(DISTINCT label) AS label
            FROM {RAW}
            """).collect()[0].as_dict()
        n_rows = row.pop('N_ROWS')
        print(f'  n_rows = {n_rows:,}', flush=True)
        print(
            f'  {"column":<40} {"distinct":>12} {"unique%":>8} '
            f'{"stype":>12}', flush=True)
        pins = {}
        for name, distinct in row.items():
            unique = distinct / n_rows
            if name in ('SRC_IP', 'DST_IP'):
                stype = 'ID (foreign key)'
            elif name in ('PROTOCOL', 'LABEL'):
                stype = 'categorical' if distinct <= 4000 else 'text'
            else:
                stype = ('categorical' if distinct <= 4000 and unique < 0.05
                         else 'numerical')
                pins[name] = stype
            print(
                f'  {name:<40} {distinct:>12,} {100 * unique:>7.3f}% '
                f'{stype:>12}', flush=True)
        print(f'  ({time.monotonic() - start:.1f}s)', flush=True)
        # Emitted ready to paste into `run_bccc_local.INT_STYPES`, so the pin
        # map is a transcription of a measurement rather than a judgement call.
        print('\nINT_STYPES = {', flush=True)
        for name, stype in sorted(pins.items()):
            print(f"    '{name}': Stype.{stype},", flush=True)
        print('}', flush=True)
        # A uniform hash sample keeps a class only in proportion to its size, so
        # the rarest class decides the smallest usable tier. This is the number
        # that says whether `smoke` is a multiclass problem at all.
        show(
            'RAW: class balance, rarest first', f"""
            SELECT label, COUNT(*) AS n,
                   100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS pct
            FROM {RAW} GROUP BY 1 ORDER BY 2 ASC
            """)
        # `is_attack` and `capture_id` are dropped from RAW as leaks, so the
        # claim that they *are* leaks has to be checked against the stage. One
        # aggregate over 20 GB, so it is worth stating precisely: this counts
        # the rows where `is_attack` disagrees with `label <> 'Benign'`. Zero
        # means the column is the target's binary collapse and nothing else.
        show(
            'STAGE: rows where is_attack != (label <> Benign)', f"""
            SELECT COUNT(*) AS n_rows,
                   COUNT_IF($1:is_attack::BOOLEAN
                            <> ($1:label::STRING <> 'Benign')) AS disagreements
            FROM '@{STAGE}/network_flows/' {FF}
            """)

        # iot23's `scenario_id` was a perfect predictor because one capture is
        # one malware sample. `capture_id` here is one capture session of the
        # CSE-CIC-IDS2018 testbed, and each session ran one scripted attack, so
        # the same failure mode is likely. Purity, not correlation, is the test.
        #
        # It is recovered from `flow_id` -- which is literally
        # `'<capture_id>:<5-tuple>_<timestamp>'` -- rather than read back from
        # the stage. That makes the check free: it runs over a native Snowflake
        # table instead of costing another 20 GB pass. It also demonstrates the
        # point of S3 in FINDINGS, which is that dropping the column does not
        # remove the information from the fixture.
        show(
            'RAW: label purity within capture_id, recovered from flow_id', f"""
            SELECT capture_id, SUM(n) AS rows_,
                   COUNT(*) AS n_labels,
                   MAX(n) / SUM(n) AS majority_share,
                   MAX_BY(label, n) AS majority_label
            FROM (
                SELECT SPLIT_PART(flow_id, ':', 1) AS capture_id,
                       label, COUNT(*) AS n
                FROM {RAW} WHERE label IS NOT NULL GROUP BY 1, 2
            ) GROUP BY 1 ORDER BY 2 DESC
            """)
        # A constant column costs a full column of index for zero information;
        # at 325 columns it is worth knowing how many there are before ingest.
        show(
            'RAW: how many endpoint IPs the flows actually reference', f"""
            SELECT COUNT(DISTINCT ip) AS distinct_endpoint_ips FROM (
                SELECT src_ip AS ip FROM {RAW}
                UNION ALL SELECT dst_ip FROM {RAW})
            """)
        return

    show('network_flows: one raw row', f'SELECT $1 FROM {flows} LIMIT 1')
    show('hosts: three raw rows',
         f"SELECT $1 FROM '@{STAGE}/hosts/' {FF} LIMIT 3")

    # `hosts` is 407,244 rows per the manifest -- three orders of magnitude
    # smaller than iot23's 258M -- so the exact distinct count is cheap.
    show(
        'hosts: key uniqueness', f"""
        SELECT COUNT(*) AS n_rows,
               COUNT(DISTINCT $1:host_id::STRING) AS distinct_host_ids,
               COUNT_IF($1:host_id IS NULL) AS null_host_id
        FROM '@{STAGE}/hosts/' {FF}
        """)

    show(
        'network_flows: key uniqueness + null keys', f"""
        SELECT COUNT(*) AS n_rows,
               COUNT(DISTINCT $1:flow_id::STRING) AS flow_ids,
               COUNT_IF($1:flow_id IS NULL) AS null_flow_id
        FROM {flows}
        """)

    # `timestamp` is an integer of unknown unit and there is no float-seconds
    # sibling to settle it against, as `ts` did for iot23. But `flow_id` ends in
    # the same instant *rendered as a string*, so printing the two side by side
    # settles the unit exactly rather than by plausibility.
    #
    # Every candidate unit is rendered `TO_VARCHAR` rather than as a timestamp.
    # Reading microseconds as seconds gives the year 48,129,749, and a bare
    # `TO_TIMESTAMP_NTZ(..., 0)` builds that value happily and then fails in the
    # *client's* arrow decoder -- "year 48129749 is out of range" -- which kills
    # the probe rather than answering it.
    show(
        'network_flows: timestamp unit', f"""
        SELECT MIN($1:timestamp::NUMBER(38,0)) AS min_raw,
               MAX($1:timestamp::NUMBER(38,0)) AS max_raw,
               COUNT_IF($1:timestamp IS NULL) AS null_timestamp,
               TO_VARCHAR(TRY_TO_TIMESTAMP_NTZ(
                   MIN($1:timestamp::NUMBER(38,0)), 0)) AS as_sec,
               TO_VARCHAR(TRY_TO_TIMESTAMP_NTZ(
                   MIN($1:timestamp::NUMBER(38,0)), 3)) AS as_milli,
               TO_VARCHAR(TRY_TO_TIMESTAMP_NTZ(
                   MIN($1:timestamp::NUMBER(38,0)), 6)) AS as_micro,
               TO_VARCHAR(TRY_TO_TIMESTAMP_NTZ(
                   MIN($1:timestamp::NUMBER(38,0)), 9)) AS as_nano
        FROM {flows}
        """)

    # The key ends in the flow's start time as text, so this is the unit
    # question answered by the fixture itself.
    show(
        'network_flows: the instant, as key text and as an integer', f"""
        SELECT $1:flow_id::STRING AS flow_id,
               $1:timestamp::NUMBER(38,0) AS raw,
               TO_VARCHAR(TRY_TO_TIMESTAMP_NTZ(
                   $1:timestamp::NUMBER(38,0), 6)) AS as_micro
        FROM {flows} LIMIT 3
        """)

    show(
        'network_flows: class balance of label', f"""
        SELECT $1:label::STRING AS label, COUNT(*) AS n
        FROM {flows} GROUP BY 1 ORDER BY 2 DESC
        """)

    show(
        'network_flows: label vs is_attack', f"""
        SELECT $1:label::STRING AS label,
               $1:is_attack::BOOLEAN AS is_attack, COUNT(*) AS n
        FROM {flows} GROUP BY 1, 2 ORDER BY 3 DESC
        """)

    show(
        'network_flows: capture_id vs label', f"""
        SELECT $1:capture_id::STRING AS capture_id,
               $1:label::STRING AS label, COUNT(*) AS n
        FROM {flows} GROUP BY 1, 2 ORDER BY 3 DESC
        """)

    # Which physical types actually appear in the 18 unsettled columns. This
    # distinguishes "all doubles, some files just wrote them as ints" from "some
    # rows hold the string 'Infinity'", and only the second needs handling in
    # the cast.
    #
    # Written as one `ARRAY_AGG(DISTINCT ...)` per column rather than a
    # `UNION ALL` of eighteen `GROUP BY`s: the union form re-scans the part
    # eighteen times, at ninety seconds each, and was abandoned after half an
    # hour. Eighteen aggregates over one scan cost one scan.
    aggs = ',\n               '.join(
        f'ARRAY_AGG(DISTINCT TYPEOF($1:{c})) AS {c}' for c in VARIANT_COLUMNS)
    show('network_flows: TYPEOF of the VARIANT columns',
         f'SELECT {aggs} FROM {flows}')

    if args.full:
        # Ethereum's parts were full-history ~10% samples; iot23's were one
        # capture each. Which of the two this is decides whether a prefix is a
        # coherent slice, and whether a single part can be entirely one class.
        show(
            'network_flows: per-part shape', f"""
            SELECT SPLIT_PART(METADATA$FILENAME, '/', -1) AS part,
                   COUNT(*) AS n_rows,
                   TO_TIMESTAMP_NTZ(MIN($1:timestamp::NUMBER(38,0)), 6)
                       AS min_time,
                   TO_TIMESTAMP_NTZ(MAX($1:timestamp::NUMBER(38,0)), 6)
                       AS max_time,
                   COUNT(DISTINCT $1:capture_id::STRING) AS captures,
                   COUNT(DISTINCT $1:label::STRING) AS labels
            FROM '@{STAGE}/network_flows/' {FF}
            GROUP BY 1 ORDER BY 1
            """)


if __name__ == '__main__':
    main()
