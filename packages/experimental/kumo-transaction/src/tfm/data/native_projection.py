# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated fixed-FK history projections, independent of prediction anchors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.graph import STATIC_TIME
from tfm.contracts.relational import RelationalForeignKey
from tfm.contracts.table_source import TableSource
from tfm.data.native_source import EVENT_US, ROW_ID, NativeSourceFiles


def history_projection(
    normalized: NativeSourceFiles,
    *,
    root: str,
    paths: Mapping[str, tuple[str, ...]],
    directory: Path,
    exclusive_terminal_paths: bool = False,
) -> tuple[TableSource, dict[str, tuple[str, ...]]]:
    """Carry row/link observability through fixed ownership, without window copies.

    Each projected terminal row keeps its original event time and can be used
    at any prediction cutoff. Its entire path must have been observable by that
    event time. The pinned native API cannot apply two independent history
    clocks; late-known paths fail explicitly before native fitting. Timed
    intermediate paths with fan-in need a different native traversal contract.
    """
    source = normalized.source
    tables = {table.name: table for table in source.tables}
    roles = {role.name: role for role in source.relationships}
    output_tables, output_roles, files = (
        list(source.tables),
        list(source.relationships),
        dict(source.files),
    )
    output_paths, labels, terminals = {}, dict(source.event_labels), set()
    directory.mkdir(parents=True, exist_ok=True)
    for index, (name, path) in enumerate(paths.items()):
        current, destinations, forward = root, [], True
        for role_name in path:
            role = roles.get(role_name)
            if (
                role is None
                or current not in {role.parent, role.child}
                or role.parent == role.child
            ):
                raise SchemaError(
                    f"history_paths.{name}.{role_name}: disconnected or ambiguous relationship"
                )
            if current != role.parent:
                forward = False
            current = role.child if current == role.parent else role.parent
            if tables[current].role != "input":
                raise SchemaError(f"history_paths.{name}: label tables cannot supply inputs")
            destinations.append(current)
        if not path or tables[current].event_time is None:
            raise SchemaError(f"history_paths.{name}: terminal event clock is required")
        if current in terminals and not exclusive_terminal_paths:
            raise UnsupportedCombinationError(
                f"history_paths.{name}: native history cannot deduplicate multiple paths to {current}"
            )
        terminals.add(current)
        if not normalized.same_clocks[current]:
            raise UnsupportedCombinationError(
                f"history_paths.{name}: native histories cannot preserve independent event/availability clocks on {current}; select data.loader: python"
            )
        terminal = tables[current]
        if not any(not tables[table].immutable for table in destinations[:-1]):
            # Native traversal is already valid, and can follow either FK
            # direction through immutable intermediates. Equal clock values may
            # have different physical column names; ingestion proved equality.
            output_paths[name] = path
            continue
        if not forward:
            raise UnsupportedCombinationError(
                f"history_paths.{name}: timed intermediate projection requires a fixed forward FK ownership path"
            )
        aliases = [f"l{i}" for i in range(len(path))]
        joins = " ".join(
            f"JOIN links {aliases[i]} ON {aliases[i]}.parent={aliases[i - 1]}.child"
            for i in range(1, len(path))
        )
        availability = ",".join(
            f"COALESCE({alias}.available_us,{STATIC_TIME})" for alias in ["r", *aliases]
        )
        conditions = " AND ".join(f"{alias}.role=?" for alias in aliases)
        query = f"SELECT {aliases[-1]}.child,l0.parent,MAX({availability}) FROM links l0 {joins} JOIN identities r ON r.table_name=? AND r.code=l0.parent WHERE {conditions} ORDER BY {aliases[-1]}.child"
        cursor = normalized.connection.execute(query, (root, *path))
        found = next(cursor, None)
        table_name = f"__tfm_history_{index}"
        role_name = f"__tfm_history_role_{index}"
        output = directory / f"history-{index}.parquet"
        writer, count = None, 0
        try:
            for block in pq.ParquetFile(source.files[current]).iter_batches(batch_size=4096):
                owners, selected = [], []
                for position, (code, event) in enumerate(
                    zip(
                        block.column(ROW_ID).to_pylist(),
                        block.column(EVENT_US).to_pylist(),
                        strict=True,
                    )
                ):
                    if found is None or found[0] != code:
                        continue
                    if found[2] > event:
                        raise UnsupportedCombinationError(
                            f"history_paths.{name}: row/link ownership was learned after the terminal event; native history cannot preserve both clocks"
                        )
                    selected.append(position)
                    owners.append(found[1])
                    found = next(cursor, None)
                projected = (
                    pa.Table.from_batches([block])
                    .take(pa.array(selected, type=pa.int64()))
                    .append_column("__tfm_history_root", pa.array(owners, type=pa.int64()))
                )
                if writer is None:
                    writer = pq.ParquetWriter(output, projected.schema)
                writer.write_table(projected)
                count += projected.num_rows
        finally:
            if writer is not None:
                writer.close()
        if not count:
            raise UnsupportedCombinationError(
                f"history_paths.{name}: native projection contains no reachable source rows"
            )
        if found is not None:
            raise SchemaError(
                f"history_paths.{name}: ownership projection is not one row per terminal"
            )
        output_tables.append(replace(terminal, name=table_name, available_at=terminal.event_time))
        output_roles.append(
            RelationalForeignKey(role_name, root, table_name, ("__tfm_history_root",))
        )
        files[table_name] = output
        output_paths[name] = (role_name,)
        if current in labels:
            labels[table_name] = labels[current]
    return replace(
        source,
        tables=tuple(output_tables),
        relationships=tuple(output_roles),
        files=files,
        event_labels=labels,
    ), output_paths
