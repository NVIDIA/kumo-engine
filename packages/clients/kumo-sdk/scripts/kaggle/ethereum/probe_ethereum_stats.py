# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transaction-side probe for the ethereum-fraud fixture.

The parts are named ``transactions_daily_part-*``, so the question that decides
how to slice the fixture is whether a part is a contiguous day range. A day
range keeps each address's neighbourhood intact, which a random row sample
would not.
"""
from snowflake.snowpark import Session

STAGE = 'HYU.KAGGLE_FIXTURES.ETHEREUM_FRAUD_30GB'
FF = '(FILE_FORMAT => HYU.KAGGLE_FIXTURES.PARQUET_FF)'

session = Session.builder.config('connection_name', 'tlb').create()


def show(title: str, sql: str) -> None:
    print(f'\n--- {title} ---', flush=True)
    for row in session.sql(sql).collect():
        print(' ', row, flush=True)


show(
    'transactions: sample row', f"SELECT $1 FROM '@{STAGE}/transactions/"
    f"transactions_daily_part-00000.parquet' {FF} LIMIT 2")

show(
    'transactions: day range per part', f"""
    SELECT SPLIT_PART(METADATA$FILENAME, '/', -1) AS part,
           MIN($1:day::DATE) AS min_day,
           MAX($1:day::DATE) AS max_day,
           COUNT(*) AS n_rows
    FROM '@{STAGE}/transactions/' {FF}
    GROUP BY 1 ORDER BY 1
    """)
