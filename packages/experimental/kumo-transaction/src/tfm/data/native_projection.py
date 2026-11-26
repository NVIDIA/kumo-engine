# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated fixed-FK history projections, independent of prediction anchors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.graph import STATIC_TIME
from tfm.contracts.relational import RelationalForeignKey
from tfm.contracts.table_source import TableSource
from tfm.data.native_source import EVENT_US, KEY_ORDER, ROW_ID, NativeSourceFiles
from tfm.data.native_views import SourceRows, select_source_rows
from tfm.data.relational import key_token


def _write_associations(
    block: pa.RecordBatch,
    selected: list[int],
    owners: list[int],
    identities: list[str],
    output: Path,
    writer: pq.ParquetWriter | None,
) -> tuple[pq.ParquetWriter | None, int]:
    if not selected:
        return writer, 0
    projected = pa.Table.from_batches([block]).take(pa.array(selected, type=pa.int64()))
    projected = projected.set_column(
        projected.schema.get_field_index(ROW_ID),
        ROW_ID,
        pa.array(identities, type=pa.string()),
    )
    projected = projected.append_column("__tfm_history_root", pa.array(owners, type=pa.int64()))
    if writer is None:
        writer = pq.ParquetWriter(output, projected.schema)
    writer.write_table(projected)
    count = projected.num_rows
    selected.clear()
    owners.clear()
    identities.clear()

    return writer, count


def history_projection(
    normalized: NativeSourceFiles,
    *,
    root: str,
    paths: Mapping[str, tuple[str, ...]],
    directory: Path,
    exclusive_terminal_paths: bool = False,
    rules: Mapping[str, SourceRows] | None = None,
) -> tuple[TableSource, dict[str, tuple[str, ...]]]:
    """Union fixed paths by root/physical event before native history truncation.

    One projected source row represents one distinct root/event association,
    regardless of the number of paths or prediction anchors. A self-link is
    counted once; an event associated with two roots remains visible to both.
    Row/link knowledge must already hold at event time because the pinned
    history backend cannot filter independent availability and event clocks.
    """
    source = normalized.source
    rules = rules or {}
    tables = {table.name: table for table in source.tables}
    roles = {role.name: role for role in source.relationships}
    output_tables, output_roles, files = (
        list(source.tables),
        list(source.relationships),
        dict(source.files),
    )
    output_paths, labels = {}, dict(source.event_labels)
    groups: dict[tuple[str, str], list[tuple[str, tuple[str, ...], tuple[str, ...]]]] = defaultdict(
        list
    )
    directory.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        current, destinations = root, []
        for hop, role_name in enumerate(path):
            role = roles.get(role_name)
            if (
                role is None
                or current not in {role.parent, role.child}
                or role.parent == role.child
            ):
                raise SchemaError(
                    f"history_paths.{name}.{role_name}: disconnected or ambiguous relationship"
                )
            current = role.child if current == role.parent else role.parent
            if tables[current].role != "input":
                raise SchemaError(f"history_paths.{name}: label tables cannot supply inputs")
            destinations.append(current)
            rule = rules.get(current)
            if (
                rule is not None
                and (rule.before_us is not None or rule.partition_column is not None)
                and hop < len(path) - 1
            ):
                raise UnsupportedCombinationError(
                    f"history_paths.{name}.{role_name}: source view filter on {current} "
                    "needs an explicit projected prefix for intermediate/profile rows"
                )
        terminal_rule = rules.get(current)
        if (
            name.startswith("__profile_")
            and any(not tables[t].immutable for t in destinations[:-1])
            and terminal_rule is not None
            and (terminal_rule.before_us is not None or terminal_rule.partition_column is not None)
        ):
            raise UnsupportedCombinationError(
                f"history_paths.{name}: source view filter on {current} "
                "needs an explicit projected prefix for profile rows"
            )
        if not path or tables[current].event_time is None:
            raise SchemaError(f"history_paths.{name}: terminal event clock is required")
        if not normalized.same_clocks[current]:
            raise UnsupportedCombinationError(
                f"history_paths.{name}: native histories cannot preserve independent event/availability clocks on {current}; select data.loader: python"
            )
        # Dated profile facts have their own declared meanings. Ordinary
        # history aliases reaching the same physical table share one union.
        purpose = name if name.startswith("__profile_") or exclusive_terminal_paths else "history"
        groups[purpose, current].append((name, path, tuple(destinations)))
    normalized.connection.create_function(
        "tfm_history_identity",
        2,
        lambda table, token: key_token((table, token)),
        deterministic=True,
    )
    normalized.connection.execute(
        "CREATE TEMP TABLE IF NOT EXISTS history_ranks(table_name TEXT,code INTEGER,rank INTEGER,PRIMARY KEY(table_name,code))"
    )
    ranked = {
        row[0]
        for row in normalized.connection.execute("SELECT DISTINCT table_name FROM history_ranks")
    }
    for index, ((purpose, current), entries) in enumerate(groups.items()):
        if normalized.counts[current] == 0:
            # Retain declared optional empty streams; there are no identities
            # to reorder or associations to deduplicate.
            for name, path, _destinations in entries:
                output_paths[name] = path
            continue
        if (
            purpose != "history"
            and len(entries) == 1
            and not any(not tables[t].immutable for t in entries[0][2][:-1])
        ):
            output_paths[entries[0][0]] = entries[0][1]
            continue
        if current not in ranked:
            # Reference IDs nest the typed primary-key token inside another
            # typed identity. Escaping changes lexical order (e.g. a/a-1).
            normalized.connection.execute(
                "INSERT INTO history_ranks SELECT table_name,code,ROW_NUMBER() OVER (ORDER BY tfm_history_identity(table_name,canonical_key))-1 FROM identities WHERE table_name=?",
                (current,),
            )
            ranked.add(current)
        queries, parameters = [], []
        for _name, path, destinations in entries:
            joins, available = [], [f"COALESCE(r.available_us,{STATIC_TIME})"]
            previous, table = "r.code", root
            for hop, (role_name, destination) in enumerate(zip(path, destinations, strict=True)):
                role = roles[role_name]
                start, end = ("parent", "child") if table == role.parent else ("child", "parent")
                link, node = f"l{hop}", f"n{hop}"
                joins.extend(
                    (
                        f"JOIN links {link} ON {link}.{start}={previous} AND {link}.role=?",
                        f"JOIN identities {node} ON {node}.table_name=? AND {node}.code={link}.{end}",
                    )
                )
                parameters.extend((role_name, destination))
                available.extend(
                    (
                        f"COALESCE({link}.available_us,{STATIC_TIME})",
                        f"COALESCE({node}.available_us,{STATIC_TIME})",
                    )
                )
                previous, table = f"{node}.code", destination
            queries.append(
                f"SELECT {previous} AS child,r.code AS owner,MAX({','.join(available)}) AS known FROM identities r {' '.join(joins)} WHERE r.table_name=?"
            )
            parameters.append(root)
        query = f"SELECT p.child,p.owner,MIN(p.known),k.rank FROM ({' UNION ALL '.join(queries)}) p JOIN identities n ON n.table_name=? AND n.code=p.child JOIN history_ranks k ON k.table_name=n.table_name AND k.code=n.code GROUP BY p.child,p.owner ORDER BY p.child,p.owner"
        cursor = normalized.connection.execute(query, (*parameters, current))
        found = next(cursor, None)
        table_name, role_name = f"__tfm_history_{index}", f"__tfm_history_role_{index}"
        output = directory / f"history-{index}.parquet"
        writer, count = None, 0
        try:
            blocks = (
                pq.ParquetFile(source.files[current]).iter_batches(batch_size=4096)
                if found is not None
                else ()
            )
            for block in blocks:
                eligible = (
                    set(select_source_rows(block, rules[current]).column(ROW_ID).to_pylist())
                    if current in rules
                    else None
                )
                selected, owners, identities = [], [], []

                for position, (code, event) in enumerate(
                    zip(
                        block.column(ROW_ID).to_pylist(),
                        block.column(EVENT_US).to_pylist(),
                        strict=True,
                    )
                ):
                    while found is not None and found[0] == code:
                        if eligible is not None and code not in eligible:
                            found = next(cursor, None)
                            continue
                        if found[2] > event:
                            names = ",".join(item[0] for item in entries)
                            raise UnsupportedCombinationError(
                                f"history_paths.{names}: row/link ownership was learned after the terminal event; native history cannot preserve both clocks"
                            )
                        selected.append(position)
                        owners.append(found[1])
                        # Within one owner's stream this prefix is constant;
                        # the suffix orders the complete reference event identity.
                        identities.append(f"{found[1]}:{found[3]:010d}")
                        found = next(cursor, None)
                        if len(selected) == 4096:
                            writer, added = _write_associations(
                                block, selected, owners, identities, output, writer
                            )
                            count += added
                writer, added = _write_associations(
                    block, selected, owners, identities, output, writer
                )
                count += added
        finally:
            if writer is not None:
                writer.close()
        if not count:
            # DiskGraph needs a physical row to retain an optional stream's
            # namespace. NULL ownership makes this one-hop sentinel unreachable
            # from every real root, including when the terminal table is nonempty.
            first = next(pq.ParquetFile(source.files[current]).iter_batches(batch_size=1))
            terminal = tables[current]
            clocks = {terminal.event_time, terminal.available_at}
            sentinel = pa.Table.from_arrays(
                [
                    pa.array(["__tfm_unreachable"], type=pa.string())
                    if field.name == ROW_ID
                    else pa.array([0], type=field.type)
                    if field.name == KEY_ORDER
                    else first.column(field.name)
                    if field.name in clocks
                    else pa.nulls(1, type=field.type)
                    for field in first.schema
                ],
                names=first.schema.names,
            ).append_column("__tfm_history_root", pa.nulls(1, type=pa.int64()))
            pq.write_table(sentinel, output)
        if found is not None:
            raise SchemaError(
                f"history_paths.{entries[0][0]}: native projection lost a source identity"
            )
        terminal = tables[current]
        output_tables.append(replace(terminal, name=table_name, available_at=terminal.event_time))
        output_roles.append(
            RelationalForeignKey(role_name, root, table_name, ("__tfm_history_root",))
        )
        files[table_name] = output
        for name, _path, _destinations in entries:
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
