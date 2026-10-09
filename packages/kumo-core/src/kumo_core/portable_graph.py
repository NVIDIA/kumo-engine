# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project a resolved API graph into the portable core payload."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from kumoapi.graph import Edge, GraphDefinition
from kumoapi.table import Column


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _project_portable_column(column: Column) -> dict[str, str]:
    result = {
        "name": str(column.name),
        "dtype": _enum_value(column.dtype),
        "stype": _enum_value(column.stype),
    }
    timestamp_format = getattr(column, "timestamp_format", None)
    if timestamp_format is not None:
        result["timestamp_format"] = _enum_value(timestamp_format)
    multicategorical_delimiter = getattr(column, "multicategorical_delimiter",
                                         None)
    if multicategorical_delimiter is not None:
        result["multicategorical_delimiter"] = _enum_value(
            multicategorical_delimiter)
    return result


def _project_portable_graph(
    definition: GraphDefinition,
    edges: Sequence[Edge],
    source_table_names: Mapping[str, str],
    connector_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return the portable payload for a graph with resolved source names.

    Source coordinates belong to the caller because they depend on bound
    connectors. This function owns the representation shared after those
    coordinates have been resolved.
    """
    tables: list[dict[str, Any]] = []

    for table_name, table in definition.tables.items():
        columns = {
            column.name: _project_portable_column(column)
            for column in table.cols
        }
        special = {table.pkey, table.time_col, table.end_time_col} - {None}
        for name in special:
            if name not in columns:
                raise ValueError(
                    f"Table {table_name!r} names {name!r} as a key or time "
                    f"column, but it is not among its columns.")

        foreign_keys: list[dict[str, str]] = []
        for edge in edges:
            if edge.src_table != table_name or edge.src_table == edge.dst_table:
                continue
            destination = definition.tables[edge.dst_table]
            if destination.pkey is None:
                raise ValueError(
                    f"Edge {edge.src_table!r}.{edge.fkey!r} -> "
                    f"{edge.dst_table!r} is invalid: destination table "
                    f"{edge.dst_table!r} requires a primary key.")
            # Only retained relations assign an identifier role. These are
            # fresh dictionaries, so the source table stays unchanged.
            columns[edge.fkey]["stype"] = "ID"
            columns[edge.fkey].pop("multicategorical_delimiter", None)
            foreign_keys.append({
                **columns[edge.fkey],
                "dst_table": edge.dst_table,
                "dst_key": destination.pkey,
            })

        entry: dict[str, Any] = {
            "table_name":
            table_name,
            "source_table_name":
            source_table_names[table_name],
            "cols":
            [value for name, value in columns.items() if name not in special],
            "fkeys":
            foreign_keys,
        }
        for key, name in (("pkey", table.pkey), ("time_col", table.time_col),
                          ("end_time_col", table.end_time_col)):
            if name is not None:
                entry[key] = columns[name]
        if connector_ids and table_name in connector_ids:
            entry["connector_id"] = connector_ids[table_name]
        tables.append(entry)

    return {"tables": tables}
