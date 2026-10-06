# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Materializes the ethereum-fraud fixture as Snowflake tables.

The local backend's source area reads Snowflake *tables*, not stages, so the
staged parquet has to land in a schema before the SDK can see it. This script
builds one scale at a time into ``HYU.ETH_FRAUD``.

Scales:

``smoke``
    ``LIMIT``-ed transactions plus exactly the addresses those transactions
    touch, so the slice is a coherent graph with no dangling foreign keys.
``part``
    One whole part file (40M transactions) against every address. The parts are
    each a full-history ~10% sample rather than a day range, so this is a
    representative slice of the fixture rather than a prefix of it.
``full``
    Every transaction (380M) against every address.

Two columns are dropped everywhere. ``description`` is free text that names the
scam type, so it is both a text-encoder dependency and a label leak.
``tx_hash`` is a per-row unique string with no predictive content.

``labeled_address_ids`` is not built: transactions carry raw address strings, so
it contributes no edge, and its ``node_id`` holds unsigned 64-bit hashes that
overflow ``int64``.

``--holdout-pct`` is what makes the fixture scoreable. For a *static* node
target, Kumo compiles the prediction table to ``WHERE target IS NULL`` -- train
on the labeled rows, score the unlabeled ones -- and the published fixture
labels every address, so batch prediction over it selects nothing. Blanking a
deterministic slice of ``is_scam`` restores the intended shape. The blanked
rows keep their true label in a side table so scores can still be checked.
"""
import argparse
import time

from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.ETHEREUM_FRAUD_30GB'
FF = '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.PARQUET_FF)'
SCHEMA = 'HYU.ETH_FRAUD'

#: Every transactions part, or one representative part, or a bounded prefix.
SCALES = {
    'smoke':
    dict(source='transactions_daily_part-00009.parquet', limit=5_000_000,
         all_addresses=False),
    'part':
    dict(source='transactions_daily_part-00000.parquet', limit=None,
         all_addresses=True),
    'full':
    dict(source='', limit=None, all_addresses=True),
}

#: Read out of the staged parquet's variant column, cast to a real type.
TRANSACTION_COLUMNS = """
    $1:from_address::STRING       AS from_address,
    $1:to_address::STRING         AS to_address,
    $1:block_number::NUMBER(38,0) AS block_number,
    $1:value_wei::FLOAT           AS value_wei,
    $1:tx_fee_wei::FLOAT          AS tx_fee_wei,
    $1:nonce::NUMBER(38,0)        AS nonce,
    $1:input_data_size::NUMBER(38,0) AS input_data_size,
    $1:contract_creation::BOOLEAN AS contract_creation,
    $1:day::DATE                  AS day
"""

ADDRESS_COLUMNS = """
    $1:address::STRING              AS address,
    $1:is_scam::BOOLEAN             AS is_scam,
    $1:is_contract::BOOLEAN         AS is_contract,
    $1:activity_start_ts::NUMBER(38,0) AS activity_start_ts,
    $1:activity_end_ts::NUMBER(38,0)   AS activity_end_ts
"""

#: Deterministic in the address itself, so the same rows are held out on every
#: rebuild and at every scale.
HELD_OUT = "MOD(ABS(HASH(address)), 100) < {pct}"


def run(session: Session, label: str, sql: str) -> None:
    start = time.monotonic()
    print(f'[{label}] running...', flush=True)
    session.sql(sql).collect()
    print(f'[{label}] done in {time.monotonic() - start:.1f}s', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', choices=sorted(SCALES), default='smoke')
    parser.add_argument('--warehouse', default=None)
    parser.add_argument(
        '--holdout-pct', type=int, default=20,
        help='percent of addresses whose is_scam is blanked '
        'so batch prediction has rows to score')
    args = parser.parse_args()

    scale = SCALES[args.scale]
    suffix = args.scale.upper()
    tx_table = f'{SCHEMA}.TRANSACTIONS_{suffix}'
    addr_table = f'{SCHEMA}.ADDRESSES_{suffix}'

    session = Session.builder.config('connection_name', 'tlb').create()
    if args.warehouse:
        session.sql(f'USE WAREHOUSE {args.warehouse}').collect()
    session.sql(f'CREATE SCHEMA IF NOT EXISTS {SCHEMA}').collect()
    print('warehouse:',
          session.sql('SELECT CURRENT_WAREHOUSE()').collect()[0][0],
          flush=True)

    location = f"@{STAGE}/transactions/{scale['source']}"
    limit = f"LIMIT {scale['limit']}" if scale['limit'] else ''
    run(
        session, f'transactions -> {tx_table}', f"""
        CREATE OR REPLACE TABLE {tx_table} AS
        SELECT {TRANSACTION_COLUMNS}
        FROM '{location}' {FF}
        {limit}
    """)

    # `address` is not unique in the source -- 304 of 3,074,926 rows repeat --
    # and a primary key has to be. The most recently active row wins.
    dedup = f"""
        SELECT {ADDRESS_COLUMNS}
        FROM '@{STAGE}/addresses/' {FF}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY $1:address::STRING
            ORDER BY $1:activity_end_ts::NUMBER(38,0) DESC NULLS LAST) = 1
    """
    if scale['all_addresses']:
        addresses_sql = dedup
    else:
        # Keep only the addresses this transaction slice actually touches, so
        # the smoke graph has no foreign key pointing at a missing row.
        addresses_sql = f"""
            WITH touched AS (
                SELECT from_address AS address FROM {tx_table}
                UNION
                SELECT to_address AS address FROM {tx_table}
            ), deduped AS ({dedup})
            SELECT deduped.* FROM deduped
            JOIN touched ON touched.address = deduped.address
        """
    truth_table = f'{addr_table}_TRUTH'
    run(session, f'addresses (labels intact) -> {truth_table}',
        f'CREATE OR REPLACE TABLE {truth_table} AS {addresses_sql}')

    held_out = HELD_OUT.format(pct=args.holdout_pct)
    run(
        session, f'addresses ({args.holdout_pct}% held out) -> {addr_table}',
        f"""
        CREATE OR REPLACE TABLE {addr_table} AS
        SELECT address,
               IFF({held_out}, NULL, is_scam) AS is_scam,
               is_contract, activity_start_ts, activity_end_ts
        FROM {truth_table}
        """)

    for name in (tx_table, addr_table):
        count = session.sql(f'SELECT COUNT(*) FROM {name}').collect()[0][0]
        print(f'{name}: {count:,} rows', flush=True)
    balance = session.sql(f"""SELECT SUM(IFF(is_scam, 1, 0)) AS labeled_scam,
                   COUNT_IF(is_scam IS NOT NULL) AS labeled,
                   COUNT_IF(is_scam IS NULL) AS held_out
            FROM {addr_table}""").collect()[0]
    print(
        f'{addr_table}: {balance[0]:,} scam of {balance[1]:,} labeled '
        f'({100 * balance[0] / balance[1]:.2f}%), '
        f'{balance[2]:,} held out for scoring', flush=True)


if __name__ == '__main__':
    main()
