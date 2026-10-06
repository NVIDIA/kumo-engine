# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from databricks.sql.client import Cursor


def column_types(
    cursor: Cursor,
    *,
    catalog: str,
    schema: Optional[str],
    target_schema: str,
) -> dict[tuple[str, str], str]:
    frame = cursor.columns(catalog_name=catalog,
                           schema_name=schema).fetchall_arrow().to_pandas()
    columns = {}
    for _, row in frame.iterrows():
        # The API may return catalog-wide rows despite a schema filter.
        row_schema = str(row['TABLE_SCHEM'])
        if (not row_schema or row_schema.casefold() == 'information_schema'
                or row_schema.casefold() != target_schema.casefold()):
            continue
        columns[str(row['TABLE_NAME']), row['COLUMN_NAME']] = row['TYPE_NAME']
    return columns


def table_names(cursor: Cursor, *, catalog: str, schema: Optional[str],
                target_schema: str) -> list[str]:
    columns = column_types(cursor, catalog=catalog, schema=schema,
                           target_schema=target_schema)
    return sorted({table for table, _ in columns})
