# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-level train eligibility checks that do not fit a sequence tokenizer."""

from __future__ import annotations

from collections.abc import Mapping

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.graph import STATIC_TIME
from tfm.data.native_source import NativeSourceFiles


def require_training_amount(
    source: NativeSourceFiles,
    *,
    root: str,
    paths: Mapping[str, tuple[str, ...]],
    amount_key: str,
    observation_seconds: int | None = None,
    split: str = "train",
    exclusive_cutoff: bool = False,
    event_partition: str | None = None,
) -> None:
    """Require a finite configured amount on an observable training-history path.

    SQL joins preserve row/link observation times independently of event times.
    No sampled graph, encoded history, model batch or tokenizer is constructed.
    Individual absent amounts remain missing; this validates source coverage.
    """
    tables = {table.name: table for table in source.source.tables}
    roles = {role.name: role for role in source.source.relationships}
    declared = False
    for name, path in paths.items():
        current = root
        joins, parameters = [], []
        previous = "a.root"
        for index, role_name in enumerate(path):
            role = roles.get(role_name)
            if (
                role is None
                or current not in {role.parent, role.child}
                or role.parent == role.child
            ):
                raise SchemaError(
                    f"history_paths.{name}.{role_name}: disconnected or ambiguous path"
                )
            forward = current == role.parent
            current = role.child if forward else role.parent
            start, end = ("parent", "child") if forward else ("child", "parent")
            link, node = f"l{index}", f"n{index}"
            joins.extend(
                [
                    f"JOIN links {link} ON {link}.{start}={previous} AND {link}.role=? AND COALESCE({link}.available_us,{STATIC_TIME})<=a.cutoff*1000000",
                    f"JOIN identities {node} ON {node}.table_name=? AND {node}.code={link}.{end} AND COALESCE({node}.available_us,{STATIC_TIME})<=a.cutoff*1000000",
                ]
            )
            parameters.extend((role_name, current))
            previous = f"{node}.code"
        if not path or tables[current].role != "input":
            raise SchemaError(f"history_paths.{name}: expected an observed event path")
        if amount_key not in tables[current].numeric:
            continue
        declared = True
        lower = (
            ""
            if observation_seconds is None
            else f"AND n{len(path) - 1}.event_us>=(a.cutoff-{int(observation_seconds)})*1000000"
        )
        upper = f"AND n{len(path) - 1}.event_us < a.cutoff*1000000" if exclusive_cutoff else ""
        partition_join, partition_where, partition_values = "", "", ()
        if event_partition is not None:
            if current != "events":
                raise SchemaError("Book partition validation requires the events history table")
            partition_join = f"JOIN event_partitions p ON p.code={previous}"
            partition_where, partition_values = "AND p.partition=?", (event_partition,)
        query = f"SELECT 1 FROM anchors a {' '.join(joins)} JOIN numbers v ON v.table_name=? AND v.code={previous} AND v.field=? {partition_join} WHERE a.split=? AND v.value IS NOT NULL {lower} {upper} {partition_where} LIMIT 1"
        if (
            source.connection.execute(
                query, (*parameters, current, amount_key, split, *partition_values)
            ).fetchone()
            is not None
        ):
            return
    reason = (
        "No eligible training-history event has a finite"
        if declared
        else "Training history has no declared"
    )
    raise SchemaError(f"{reason} amount_field {amount_key!r}")


def validate_graph_policy(
    source: NativeSourceFiles,
    *,
    root: str,
    paths: Mapping[str, tuple[str, ...]],
    fanout: int,
    observation_seconds: int | None,
    canonical_ties: bool = True,
) -> None:
    """Reject graph rules the pinned native recent sampler cannot express.

    A whole-source bound can prove a lookback immaterial. Otherwise native
    graph lookbacks need eligibility filtering during every hop, not trimming
    a completed graph. Canonical ties are safe when retrieval is uncapped or
    every candidate clock is unique within its physical FK neighborhood.
    """
    tables = {
        table.name: table
        for table in source.source.tables
        if table.role == "input"
        and (source.source.graph_numeric is None or table.name in source.source.graph_numeric)
    }
    intermediate, terminals = set(), set()
    for path in paths.values():
        current = root
        reached = []
        for role_name in path:
            role = next(role for role in source.source.relationships if role.name == role_name)
            current = role.child if current == role.parent else role.parent
            reached.append(current)
        if reached:
            intermediate.update(reached[:-1])
            terminals.add(reached[-1])
    structural = {root} | (intermediate - terminals)
    if observation_seconds is not None:
        cutoff = source.connection.execute("SELECT MAX(cutoff) FROM anchors").fetchone()[0]
        lower = (cutoff - observation_seconds) * 1_000_000
        for name, table in tables.items():
            if table.immutable or name in structural:
                continue
            first = source.connection.execute(
                "SELECT MIN(event_us) FROM identities WHERE table_name=?", (name,)
            ).fetchone()[0]
            if first is not None and first < lower:
                raise UnsupportedCombinationError(
                    f"graph.path.{name}: native sampling cannot apply the configured observation lower bound before every hop; select data.loader: python"
                )
    if canonical_ties:
        for role in source.source.relationships:
            if (
                role.parent not in tables
                or role.child not in tables
                or source.forward_degrees.get(role.name, 0) <= fanout
            ):
                continue
            if tables[role.child].immutable:
                raise UnsupportedCombinationError(
                    f"graph.path.{role.name}: native uniform selection cannot preserve canonical-key fanout on an immutable destination"
                )
            tied = source.connection.execute(
                "SELECT 1 FROM links l JOIN fanouts f ON f.role=l.role AND f.parent=l.parent JOIN identities n ON n.table_name=? AND n.code=l.child WHERE l.role=? AND f.degree>? GROUP BY l.parent,n.event_us HAVING COUNT(*)>1 LIMIT 1",
                (role.child, role.name, fanout),
            ).fetchone()
            if tied is not None:
                raise UnsupportedCombinationError(
                    f"graph.path.{role.name}: native index-local recent ties cannot guarantee canonical_key_asc at the fanout boundary"
                )
