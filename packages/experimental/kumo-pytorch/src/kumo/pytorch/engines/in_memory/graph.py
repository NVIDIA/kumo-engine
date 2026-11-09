# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph sampling over tables held as Python mappings.

Simple indices and deterministic fanout, for the same reason the rest of this
package is simple: it is the behaviour an optimized sampler is checked
against.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from kumo.pytorch.contract.sample import GraphSample, NeighborhoodSpec
from kumo.pytorch.contract.schema import DataSchema, table_clock
from kumo.pytorch.data.records import timestamp_to_epoch
from kumo.pytorch.engine import GraphRoot


class InMemoryGraphSampler:
    """Sample disjoint neighborhoods with simple indices and deterministic fanout."""

    def __init__(
        self,
        schema: DataSchema,
        tables: Mapping[str, Sequence[Mapping[str, Any]]],
    ):
        self._table_specs = {table.name: table for table in schema.tables}
        self._relations = {relation.name: relation for relation in schema.relations}
        self._tables = {
            name: tuple(
                {
                    column: dtype.normalize(row[column], path=f"{name}.{column}")
                    for column, dtype in table.columns.items()
                }
                for row in tables[name]
            )
            for name, table in self._table_specs.items()
        }
        self._indices: dict[tuple[str, str], dict[Any, tuple[dict[str, Any], ...]]] = {}

    def fitting_field_values(
        self,
        root_table: str,
        neighborhood: NeighborhoodSpec,
        spec: Any,
        roots: Sequence[GraphRoot],
    ) -> dict[tuple[str, str], tuple[Any | None, ...]]:
        """Collect node-field values from deterministic preparation neighborhoods."""

        node_rows, _owners, _edges = self._collect(
            root_table,
            neighborhood,
            roots,
            seed=0,
            epoch=0,
        )
        values = {}
        for table, fields in spec.node_fields.items():
            if table not in self._tables:
                raise ValueError(f"graph node fields refer to unknown table {table!r}")
            rows = node_rows.get(table, [])
            for use in fields.fields:
                field = use.field
                if any(field not in row for row in rows):
                    raise ValueError(f"{table!r} node is missing graph field {field!r}")
                values[(table, field)] = tuple(row.get(field) for row in rows)
        return values

    def batch(
        self,
        root_table: str,
        neighborhood: NeighborhoodSpec,
        spec: Any,
        encoders: Mapping[str, Mapping[str, Any]],
        roots: Sequence[GraphRoot],
        *,
        seed: int,
        epoch: int,
    ) -> dict[str, Any]:
        node_rows, node_owners, edge_pairs = self._collect(
            root_table,
            neighborhood,
            roots,
            seed,
            epoch,
        )
        return {
            "nodes": self._node_output(
                root_table, neighborhood, spec, encoders, node_rows, node_owners
            ),
            "edges": self._edge_output(neighborhood, edge_pairs),
        }

    def _collect(
        self,
        root_table: str,
        spec: NeighborhoodSpec,
        roots: Sequence[GraphRoot],
        seed: int,
        epoch: int,
    ) -> tuple[
        dict[str, list[dict[str, Any]]],
        dict[str, list[int]],
        dict[str, list[tuple[int, int]]],
    ]:
        node_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        node_ids: dict[tuple[int, str, Any], int] = {}
        node_owners: dict[str, list[int]] = defaultdict(list)
        edge_pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)
        # Node indices are per sample, so a repeated pair is a repeat within one sample.
        seen_edges: set[tuple[str, int, int]] = set()

        def add_node(owner: int, table: str, row: dict[str, Any]) -> int:
            primary_key = self._table_specs[table].primary_key
            identity = (owner, table, row[primary_key])
            if identity not in node_ids:
                node_ids[identity] = len(node_rows[table])
                node_rows[table].append(row)
                node_owners[table].append(owner)
            return node_ids[identity]

        for owner, root_spec in enumerate(roots):
            root = self._row(root_table, root_spec.entity_id)
            add_node(owner, root_table, root)
            frontier: list[tuple[str, dict[str, Any]]] = [(root_table, root)]
            for layer in spec.layers:
                next_frontier: list[tuple[str, dict[str, Any]]] = []
                for traversal in layer.samples:
                    if traversal.relation not in self._relations:
                        raise ValueError(f"unknown graph relation {traversal.relation!r}")
                    for current_table, current_row in frontier:
                        target_table, neighbors = self._neighbors(
                            current_row,
                            current_table,
                            traversal,
                            root_spec,
                            seed,
                            epoch,
                        )
                        if not neighbors:
                            continue
                        current_index = add_node(owner, current_table, current_row)
                        for neighbor in neighbors:
                            neighbor_index = add_node(owner, target_table, neighbor)
                            pair = (
                                (current_index, neighbor_index)
                                if traversal.direction == "forward"
                                else (neighbor_index, current_index)
                            )
                            if (traversal.relation, *pair) not in seen_edges:
                                seen_edges.add((traversal.relation, *pair))
                                edge_pairs[traversal.relation].append(pair)
                            next_frontier.append((target_table, neighbor))
                frontier = self._deduplicate(next_frontier)

        return node_rows, node_owners, edge_pairs

    def _traversal_clock(self, traversal: GraphSample, target_table: str) -> str | None:
        """Resolve which column is this traversal's clock.

        An explicit ``GraphSample.time_column`` wins; otherwise the destination
        table's declared ``time_column`` is inherited. This answers *which*
        column, not *whether* it filters — ``temporal`` decides that, so a
        deliberately non-temporal traversal can still order by recency.
        """

        if traversal.time_column is not None:
            return traversal.time_column
        return table_clock(self._table_specs[target_table])

    def _index(self, table: str, column: str) -> dict[Any, tuple[dict[str, Any], ...]]:
        key = (table, column)
        if key not in self._indices:
            grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for row in self._tables[table]:
                if column not in row:
                    raise ValueError(f"{table!r} row is missing relation column {column!r}")
                if row[column] is not None:
                    grouped[row[column]].append(row)
            primary_key = self._table_specs[table].primary_key
            self._indices[key] = {
                value: tuple(sorted(rows, key=lambda row: repr(row[primary_key])))
                for value, rows in grouped.items()
            }
        return self._indices[key]

    def _row(self, table: str, value: Any) -> dict[str, Any]:
        key = self._table_specs[table].primary_key
        rows = self._index(table, key).get(value, ())
        if len(rows) != 1:
            raise ValueError(f"expected one {table!r} node for {key}={value!r}")
        return rows[0]

    def _neighbors(
        self,
        row: Mapping[str, Any],
        current_table: str,
        traversal: GraphSample,
        root: GraphRoot,
        seed: int,
        epoch: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        relation = self._relations[traversal.relation]
        if traversal.direction == "forward":
            expected, source_key = relation.left_table, relation.left_key
            target_table, target_key = relation.right_table, relation.right_key
        else:
            expected, source_key = relation.right_table, relation.right_key
            target_table, target_key = relation.left_table, relation.left_key
        if current_table != expected:
            return target_table, []
        if source_key not in row:
            raise ValueError(f"{current_table!r} row is missing relation column {source_key!r}")
        candidates = list(self._index(target_table, target_key).get(row[source_key], ()))
        clock = self._traversal_clock(traversal, target_table)
        time_column = clock if traversal.temporal == "inherit" else None
        if time_column is not None:
            if any(candidate.get(time_column) is None for candidate in candidates):
                raise ValueError(
                    f"{target_table!r} graph nodes are missing time column {time_column!r}"
                )
            candidates = [
                candidate
                for candidate in candidates
                if timestamp_to_epoch(candidate[time_column]) <= root.cutoff
            ]
        primary_key = self._table_specs[target_table].primary_key
        if traversal.strategy == "recent":
            if clock is None:
                raise ValueError(
                    f"recent sampling of {target_table!r} needs a time column: set "
                    f"GraphSample.time_column, or declare TableSpec.time_column"
                )
            candidates.sort(
                key=lambda candidate: (
                    timestamp_to_epoch(candidate[clock]),
                    repr(candidate[primary_key]),
                ),
                reverse=True,
            )
        elif len(candidates) > traversal.fanout:
            salt = f"{seed}:{epoch}:{root.sample_id}:{traversal.relation}:{row[source_key]}"
            candidates.sort(
                key=lambda candidate: hashlib.sha256(
                    f"{salt}:{candidate[primary_key]!r}".encode()
                ).digest()
            )
        return target_table, candidates[: traversal.fanout]

    def _deduplicate(
        self,
        values: Sequence[tuple[str, dict[str, Any]]],
    ) -> list[tuple[str, dict[str, Any]]]:
        unique: dict[tuple[str, Any], tuple[str, dict[str, Any]]] = {}
        for table, row in values:
            key = self._table_specs[table].primary_key
            unique[(table, row[key])] = (table, row)
        return list(unique.values())

    def _node_output(
        self,
        root_table: str,
        neighborhood: NeighborhoodSpec,
        spec: Any,
        encoders: Mapping[str, Mapping[str, Any]],
        node_rows: Mapping[str, list[dict[str, Any]]],
        node_owners: Mapping[str, list[int]],
    ) -> dict[str, Any]:
        relation_names = {
            traversal.relation for layer in neighborhood.layers for traversal in layer.samples
        }
        tables = {root_table, *spec.node_fields}
        for name in relation_names:
            tables.add(self._relations[name].left_table)
            tables.add(self._relations[name].right_table)

        nodes = {}
        for table in sorted(tables):
            rows = node_rows.get(table, [])
            primary_key = self._table_specs[table].primary_key
            raw_ids = [row[primary_key] for row in rows]
            ids: Any = (
                torch.tensor(raw_ids, dtype=torch.long)
                if all(isinstance(value, int) for value in raw_ids)
                else tuple(raw_ids)
            )
            nodes[table] = {
                "ids": ids,
                "batch": torch.tensor(node_owners.get(table, []), dtype=torch.long),
                "features": {
                    field: encoder.encode_values(self._checked_values(table, field, rows))
                    for field, encoder in encoders.get(table, {}).items()
                },
            }
        return nodes

    @staticmethod
    def _checked_values(
        table: str,
        field: str,
        rows: Sequence[Mapping[str, Any]],
    ) -> list[Any]:
        if any(field not in row for row in rows):
            raise ValueError(f"{table!r} node is missing graph field {field!r}")
        return [row[field] for row in rows]

    def _edge_output(
        self,
        spec: NeighborhoodSpec,
        edge_pairs: Mapping[str, list[tuple[int, int]]],
    ) -> dict[str, Any]:
        relation_names = {
            traversal.relation for layer in spec.layers for traversal in layer.samples
        }
        return {
            name: {
                "index": (
                    torch.tensor(edge_pairs.get(name, []), dtype=torch.long).t().contiguous()
                    if edge_pairs.get(name)
                    else torch.empty((2, 0), dtype=torch.long)
                ),
                "source_type": self._relations[name].left_table,
                "destination_type": self._relations[name].right_table,
            }
            for name in sorted(relation_names)
        }
