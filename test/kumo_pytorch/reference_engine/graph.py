# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recent-neighbor graphs with unique edges and layerwise frontier expansion.

Table selection runs natively. Python assembles selected topology and delegates
feature encoding to the fitted Kumo encoders. Baseline defects are not emulated.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import Rows, SampleRequest
from kumo.pytorch.contract.schema import table_clock

from .availability import check_root_availability
from .typed import typed_array


class GraphExecutor:
    """Python assembles selected topology; all table selection stays in the backend."""

    def __init__(self, store, schema, queries):
        self.store, self.queries = store, queries
        self.tables = {table.name: table for table in schema.tables}
        self.relations = {relation.name: relation for relation in schema.relations}

    def _validate(self, sample, spec):
        name = getattr(spec, "source", None)
        neighborhood = sample.neighborhoods.get(name)
        if neighborhood is None:
            raise ValueError(f"unknown graph neighborhood {name!r}")
        for table, fields in spec.node_fields.items():
            if table not in self.tables:
                raise ValueError(f"graph node fields refer to unknown table {table!r}")
            missing = {use.field for use in fields.fields} - set(
                self.tables[table].columns
            )
            if missing:
                raise ValueError(
                    f"graph node fields are missing from {table!r}: {sorted(missing)}"
                )
        for layer in neighborhood.layers:
            seen = set()
            for traversal in layer.samples:
                relation = self.relations.get(traversal.relation)
                if relation is None:
                    raise ValueError(f"unknown graph relation {traversal.relation!r}")
                if traversal.strategy != "recent":
                    raise ValueError(
                        "arena graph execution supports recent sampling only"
                    )
                key = (traversal.relation, traversal.direction)
                if key in seen:
                    raise ValueError(
                        "a graph relation direction cannot be sampled twice in one layer"
                    )
                seen.add(key)
                target = (
                    relation.right_table
                    if traversal.direction == "forward"
                    else relation.left_table
                )
                clock = table_clock(self.tables[target])
                if traversal.time_column is not None and traversal.time_column != clock:
                    raise ValueError(
                        "graph traversal cannot override the destination table clock"
                    )
                if traversal.temporal == "static" and clock is not None:
                    raise ValueError(
                        "graph traversal cannot disable filtering of a timed destination"
                    )
                if clock is None:
                    raise ValueError(
                        "recent graph sampling requires a destination table clock"
                    )
        return neighborhood

    def _roots(self, sample, roots):
        table = sample.entity.table
        primary = self.tables[table].primary_key
        available = sample.entity.available_from
        columns = tuple(dict.fromkeys((primary,) + ((available,) if available else ())))
        values = self.store.fetch(table, [root.entity_id for root in roots], columns)
        check_root_availability(sample, values, tuple(root.cutoff for root in roots))
        return values[primary].to_pylist()

    def _collect(self, sample, spec, roots):
        neighborhood = self._validate(sample, spec)
        roots = tuple(roots)
        self.queries.reset_cache()
        try:
            root_ids = self._roots(sample, roots)
            ids, owners, edges = defaultdict(list), defaultdict(list), defaultdict(list)
            positions, edge_seen = {}, defaultdict(set)

            def add(owner, table, identity):
                key = (owner, table, identity)
                if key in positions:
                    return positions[key], False
                index = len(ids[table])
                positions[key] = index
                ids[table].append(identity)
                owners[table].append(owner)
                return index, True

            frontier = []
            for owner, identity in enumerate(root_ids):
                add(owner, sample.entity.table, identity)
                frontier.append((owner, sample.entity.table, identity))
            for layer_index, layer in enumerate(neighborhood.layers):
                next_frontier = {}
                for traversal_index, traversal in enumerate(layer.samples):
                    relation = self.relations[traversal.relation]
                    forward = traversal.direction == "forward"
                    source_table = (
                        relation.left_table if forward else relation.right_table
                    )
                    target_table = (
                        relation.right_table if forward else relation.left_table
                    )
                    selected = [node for node in frontier if node[1] == source_table]
                    if not selected:
                        continue
                    # Neighborhood selection depends on identity and cutoff, not
                    # owner. Query repeated frontiers once, then expand them back
                    # into the original per-anchor order and ownership below.
                    unique, request_positions = {}, []
                    for owner, _table, identity in selected:
                        key = (identity, roots[owner].cutoff)
                        if key not in unique:
                            unique[key] = len(unique)
                        request_positions.append(unique[key])
                    requests = tuple(
                        SampleRequest(
                            sample_id=f"graph:{layer_index}:{traversal_index}:{index}",
                            entity_id=identity,
                            as_of_time=cutoff,
                        )
                        for index, (identity, cutoff) in enumerate(unique)
                    )
                    result = self.queries.select(
                        source_table,
                        requests,
                        Rows(path=(traversal.relation,)),
                        order="descending",
                        limit=traversal.fanout,
                        directions=(traversal.direction,),
                        columns=(self.tables[target_table].primary_key,),
                    )
                    # Only selected identities/owners cross into Python; no table rows are scanned here.
                    result_positions = result[self.queries.anchor_column].to_numpy()
                    target_ids = result[
                        self.tables[target_table].primary_key
                    ].to_pylist()
                    if (
                        len(result_positions) != len(target_ids)
                        or result_positions.dtype.kind not in "iu"
                        or np.any(result_positions < 0)
                        or np.any(result_positions >= len(requests))
                        or np.any(result_positions[1:] < result_positions[:-1])
                    ):
                        raise RuntimeError(
                            "graph query returned invalid or unordered request positions"
                        )
                    offsets = np.concatenate(
                        (
                            [0],
                            np.cumsum(
                                np.bincount(
                                    result_positions.astype(np.int64, copy=False),
                                    minlength=len(requests),
                                )
                            ),
                        )
                    )
                    for selected_position, request_position in enumerate(
                        request_positions
                    ):
                        owner, _table, source_id = selected[selected_position]
                        source_index = positions[(owner, source_table, source_id)]
                        for identity in target_ids[
                            offsets[request_position]: offsets[request_position + 1]
                        ]:
                            target_index, _fresh = add(owner, target_table, identity)
                            pair = (
                                (source_index, target_index)
                                if forward
                                else (target_index, source_index)
                            )
                            if pair not in edge_seen[traversal.relation]:
                                edges[traversal.relation].append(pair)
                                edge_seen[traversal.relation].add(pair)
                            key = (owner, target_table, identity)
                            next_frontier[key] = key
                frontier = list(next_frontier)
            relation_names = {
                item.relation for layer in neighborhood.layers for item in layer.samples
            }
            table_names = {sample.entity.table, *spec.node_fields}
            for name in relation_names:
                relation = self.relations[name]
                table_names.update((relation.left_table, relation.right_table))
            return ids, owners, edges, sorted(table_names), sorted(relation_names)
        finally:
            self.queries.reset_cache()

    def fit_graph_fields(
        self, sample, spec, roots
    ) -> Mapping[tuple[str, str], tuple[Any, ...]]:
        # Root validation is required even when no encoders need fitting: typed graph-only
        # materialization can reach this method without ordinary sample construction.
        if not any(group.fields for group in spec.node_fields.values()):
            self._validate(sample, spec)
            self.queries.reset_cache()
            try:
                self._roots(sample, tuple(roots))
                return {}
            finally:
                self.queries.reset_cache()
        ids, _owners, _edges, _tables, _relations = self._collect(sample, spec, roots)
        values = {}
        for table, fields in spec.node_fields.items():
            names = tuple(use.field for use in fields.fields)
            fetched = self.store.fetch(table, ids.get(table, ()), names)
            for field in names:
                values[(table, field)] = tuple(
                    typed_array(
                        fetched[field],
                        self.tables[table].columns[field]
                        or DType.from_arrow(fetched[field].type),
                    ).to_pylist()
                )
        return values

    def sample_graph(self, sample, spec, state, roots, seed, epoch):
        del (
            seed,
            epoch,
        )  # Recent sampling is deterministic and does not consume random draws.
        ids, owners, pairs, tables, relations = self._collect(sample, spec, roots)
        nodes = {}
        for table in tables:
            raw_ids = ids.get(table, [])
            encoders = state.get(table, {})
            if set(encoders) != (
                {use.field for use in spec.node_fields[table].fields}
                if table in spec.node_fields
                else set()
            ):
                raise ValueError(
                    f"fitted graph encoders differ from requested fields for {table!r}"
                )
            unique_ids = tuple(dict.fromkeys(raw_ids)) if encoders else ()
            fetched = (
                self.store.fetch(table, unique_ids, tuple(encoders))
                if encoders
                else None
            )
            take = None
            if encoders and len(unique_ids) != len(raw_ids):
                positions = {
                    identity: index for index, identity in enumerate(unique_ids)
                }
                take = torch.tensor(
                    [positions[identity] for identity in raw_ids], dtype=torch.int64
                )
            features = {}
            for field, encoder in encoders.items():
                encoded = encoder.encode_arrow(
                    typed_array(
                        fetched[field],
                        self.tables[table].columns[field]
                        or DType.from_arrow(fetched[field].type),
                    )
                )
                features[field] = (
                    encoded
                    if take is None
                    else {
                        name: values.index_select(0, take)
                        for name, values in encoded.items()
                    }
                )
            node_ids = (
                torch.tensor(raw_ids, dtype=torch.int64)
                if all(
                    isinstance(value, int) and -(2**63) <= value < 2**63
                    for value in raw_ids
                )
                else tuple(raw_ids)
            )
            nodes[table] = {
                "ids": node_ids,
                "batch": torch.tensor(owners.get(table, []), dtype=torch.int64),
                "features": features,
            }
        edges = {}
        for name in relations:
            relation = self.relations[name]
            edges[name] = {
                "index": (
                    torch.tensor(pairs[name], dtype=torch.int64).t().contiguous()
                    if pairs.get(name)
                    else torch.empty((2, 0), dtype=torch.int64)
                ),
                "source_type": relation.left_table,
                "destination_type": relation.right_table,
            }
        return {"nodes": nodes, "edges": edges}
