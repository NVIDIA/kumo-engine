#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script is used to convert a json fixture into a REST fixture that can be
# loaded by fixture_utils. This lets you trigger the fixture through the REST
# API. The resulting code should be copied into rest/routers/fixture_XXX.py
# See rest/routers/fixture_worldcup.py for an example.
#
# Example Usage:
#
#   ./rest/scripts/fixture2python.py s3://example-bucket/ux_e2e_stability_tests/fixtures/test/MANY_TABLES_SNOWFLAKE.json # noqa
#

import argparse

from io_utils import json
from rest.config.api_command import FixtureDefinition
from rest.config.table_v2 import (
    BigQuerySourceTable,
    DatabricksSourceTable,
    GCSSourceTable,
    S3SourceTable,
    SnowflakeSourceTable,
    TableDefinition,
)


def run(path: str) -> None:
    fixture = json.load_json(path, FixtureDefinition)

    if not fixture.graphs:
        raise ValueError(f"Fixture at {path} has no graphs")
    graph0 = fixture.graphs[0]

    def _v2_source_identity_str(table_def: TableDefinition) -> str:
        st = table_def.source_table
        if isinstance(st, (S3SourceTable, GCSSourceTable)):
            return str(st.table_name)
        if isinstance(st, SnowflakeSourceTable):
            return str(st.table)
        if isinstance(st, DatabricksSourceTable):
            return str(st.table)
        if isinstance(st, BigQuerySourceTable):
            return str(st.table_name)
        raise TypeError(f"Unsupported source_table type: {type(st)}")

    def _v2_logical_table_name(table_def: TableDefinition) -> str:
        for key, td in graph0.tables.items():
            if td is table_def or td.fpid == table_def.fpid:
                return key
        return _v2_source_identity_str(table_def)

    print("_TABLE_SCHEMAS = [")
    for table_def in fixture.tables:
        name = _v2_logical_table_name(table_def)
        source_identity = _v2_source_identity_str(table_def)
        print("    fixture_utils.Table(")
        print(f"        name='{name}',")
        if name != source_identity:
            print(f"        source_table_name='{source_identity}',")
        if table_def.pkey is not None:
            print(f"        pkey='{table_def.pkey}',")
        if table_def.time_col is not None:
            print(f"        time_col='{table_def.time_col}',")
        print("    ),")
    print("]")
    print("")

    print("_COL_GROUPS: List[List[fixture_utils.TableDotCol]] = [")
    for grp in graph0.col_groups:
        print("    [")
        for col in grp.columns:
            table_dot_col = f"{col.table_name}.{col.col_name}"
            print(f"        '{table_dot_col}',")
        print("    ],")
    print("]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('PATH')
    args = parser.parse_args()

    run(args.PATH)
