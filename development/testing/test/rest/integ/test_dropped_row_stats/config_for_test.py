# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test configs for dropped row stats tests.
The source table has 1000 row copied from the original articles table. The
table has article_id1 and time_col1 with no NAs. The table has article_id2 and
time_col2 with 100 NAs each. The 100 NAs are in non-overlapping rows meaning
there are 200 rows with NAs.
"""

dropped_rows_no_na_args = {
    "source_name": "articles",
    "table_name": "dropped_rows_no_na",
    "pkey": "article_id1",
    "time_cols": "time_col1",
    "expected_dropped_rows": 0,
}

dropped_rows_200_na_args = {
    "source_name": "articles",
    "table_name": "dropped_rows_200_na",
    "pkey": "article_id2",
    "time_cols": "time_col2",
    "expected_dropped_rows": 200,
}
