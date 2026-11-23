# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent keyed-graph answers, including allowed fanout-boundary ties.

Tensor positions are translated to source identities. A graph must be exactly
explainable by a complete layer-by-layer selection; repeated traversals cannot
justify extra edges merely because their endpoints are eligible somewhere.
"""

import json
from collections import Counter
from itertools import combinations

import pyarrow as pa
import pyarrow.compute as pc
import torch

from kumo.pytorch.data.records import cutoff_to_epoch, timestamp_to_epoch


class SourceRows:
    def __init__(self, schema, tables):
        self.schema = {table.name: table for table in schema.tables}
        self.tables = tables
        self.indices = {}
        self.joins = {}

    def row(self, table, key):
        primary = self.schema[table].primary_key
        assert primary is not None, "Keyed graph oracle requires declared source identities"
        source = self.tables[table]
        if table not in self.indices:
            keys = (
                source[primary].to_pylist()
                if isinstance(source, pa.Table)
                else [r[primary] for r in source]
            )
            assert len(set(keys)) == len(keys), f"{table}: duplicate source identities"
            self.indices[table] = dict(zip(keys, range(len(keys)), strict=True))
        position = self.indices[table][key]
        return (
            source.slice(position, 1).to_pylist()[0]
            if isinstance(source, pa.Table)
            else source[position]
        )

    def matching(self, table, column, value):
        if value is None:
            return []
        cache = (table, column, value)
        if cache not in self.joins:
            source = self.tables[table]
            self.joins[cache] = (
                source.filter(pc.equal(source[column], value)).to_pylist()
                if isinstance(source, pa.Table)
                else [row for row in source if row[column] == value]
            )
        return self.joins[cache]


def _list(value):
    return value.tolist() if isinstance(value, torch.Tensor) else list(value)


def _feature_values(node, table, source, state, input_name, declared):
    assert set(node["features"]) == {field.field for field in declared}, f"{table}: feature fields"
    for field in declared:
        name = field.field
        encoded = node["features"][name]
        identity = json.dumps(["inputs", input_name, "graph", table, name], separators=(",", ":"))
        encoder = state["encoders"][identity]
        raw = [source.row(table, key)[name] for key in _list(node["ids"])]
        valid = [value is not None for value in raw]
        assert encoded["valid"].dtype == torch.bool
        assert encoded["valid"].shape == encoded["values"].shape == (len(raw),)
        assert encoded["valid"].tolist() == valid, f"{table}.{name}: validity"
        dtype = (
            torch.int64
            if encoder["kind"] == "vocabulary"
            else getattr(torch, encoder.get("output_dtype", encoder["dtype"]["name"]))
        )
        assert encoded["values"].dtype == dtype, f"{table}.{name}: dtype"
        for i, value in enumerate(raw):
            if value is None:
                continue
            if encoder["kind"] == "vocabulary":
                expected = dict(encoder["entries"]).get(value, encoder["roles"]["unknown"])
            elif encoder["kind"] == "zscore":
                expected = (float(value) - encoder["mean"]) / encoder["scale"]
            else:
                assert encoder["kind"] == "raw", (
                    f"Oracle needs an independent rule for {encoder['kind']}"
                )
                expected = value
            expected = torch.tensor(expected, dtype=dtype).item()
            assert encoded["values"][i].item() == expected, (
                f"{table}.{name}/{_list(node['ids'])[i]}: feature differs from source"
            )


def assert_graph(graph, contract, input_name, requests, source, state):
    """Assert identities, features, oriented edges and full sampling semantics."""
    assert set(graph) == {"nodes", "edges"}
    representation = contract.batch.inputs[input_name].representation
    neighborhood = contract.sample.neighborhoods[contract.batch.inputs[input_name].source.name]
    relations = {relation.name: relation for relation in contract.schema.relations}
    used_relations = {sample.relation for layer in neighborhood.layers for sample in layer.samples}
    assert not set(graph["edges"]) - used_relations, "Undeclared graph relation"
    nodes, edges, ids, owners = [set() for _ in requests], [set() for _ in requests], {}, {}
    for table, node in graph["nodes"].items():
        assert set(node) == {"ids", "batch", "features"}
        key = source.schema[table].primary_key
        assert key is not None, "Graph source identities must be declared"
        if source.schema[table].columns[key].name == "string":
            assert isinstance(node["ids"], (tuple, list)) and all(
                isinstance(value, str) for value in node["ids"]
            )
        else:
            assert isinstance(node["ids"], torch.Tensor), "Numeric graph identities require tensors"
            assert node["ids"].ndim == 1 and node["ids"].dtype == torch.int64, (
                "Graph identity dtype/layout"
            )
        ids[table], owners[table] = _list(node["ids"]), _list(node["batch"])
        assert node["batch"].dtype == torch.int64 and node["batch"].ndim == 1
        assert len(ids[table]) == len(owners[table])
        for key, owner in zip(ids[table], owners[table], strict=True):
            assert 0 <= owner < len(requests), "Invalid graph sample owner"
            assert (table, key) not in nodes[owner], "Duplicate graph node identity"
            source.row(table, key)
            nodes[owner].add((table, key))
        fields = representation.node_fields.get(table)
        _feature_values(
            node, table, source, state, input_name, () if fields is None else fields.fields
        )
    for name, edge in graph["edges"].items():
        assert set(edge) == {"source_type", "destination_type", "index"}
        relation = relations[name]
        left, right = relation.left_table, relation.right_table
        assert (edge["source_type"], edge["destination_type"]) == (left, right), (
            "Graph edge orientation"
        )
        index = edge["index"]
        assert index.dtype == torch.int64 and index.ndim == 2 and index.shape[0] == 2
        for a, b in index.T.tolist():
            assert 0 <= a < len(ids.get(left, ())) and 0 <= b < len(ids.get(right, ())), (
                "Graph edge index outside node table"
            )
            owner = owners[left][a]
            assert owner == owners[right][b], "Graph edge crosses samples"
            identity = (name, ids[left][a], ids[right][b])
            assert identity not in edges[owner], "Duplicate graph edge"
            edges[owner].add(identity)
    for owner, request in enumerate(requests):
        root = (contract.sample.entity.table, request.entity_id)
        assert root in nodes[owner], "Graph root is missing"
        assert _explains(
            neighborhood.layers, relations, source, request, root, nodes[owner], edges[owner]
        ), (
            f"sample {request.sample_id}: graph violates source membership, cutoff, "
            "recency, fanout or layer connectivity"
        )


def _explains(layers, relations, source, request, root, actual_nodes, actual_edges):
    cutoff = cutoff_to_epoch(request.as_of_time)
    incidence = Counter()
    endpoints = {}
    for edge in actual_edges:
        name, left, right = edge
        relation = relations[name]
        ends = ((relation.left_table, left), (relation.right_table, right))
        endpoints[edge] = ends
        incidence.update(set(ends))
    if set(incidence) | {root} != actual_nodes:
        return False
    # A non-root leaf can only be discovered through its sole edge, from the
    # other end. Reverse revisits cannot create extra leaves. This necessary
    # bound rejects excess tied leaves before enumerating selection witnesses.
    required = Counter()
    capacity = Counter()
    for layer in layers:
        for sample in layer.samples:
            capacity[sample.relation, sample.direction] += sample.fanout
    for edge, ends in endpoints.items():
        for index, end in enumerate(ends):
            if end != root and incidence[end] == 1 and end != ends[1 - index]:
                direction = "forward" if index == 1 else "reverse"
                required[edge[0], direction, ends[1 - index]] += 1
    if any(count > capacity[name, direction] for (name, direction, _), count in required.items()):
        return False

    def choices(sample, origin, required):
        relation = relations[sample.relation]
        forward = sample.direction == "forward"
        left, right = (
            (relation.left_table, relation.right_table)
            if forward
            else (relation.right_table, relation.left_table)
        )
        left_key, right_key = (
            (relation.left_key, relation.right_key)
            if forward
            else (relation.right_key, relation.left_key)
        )
        clock = sample.time_column or source.schema[right].time_column
        key = source.schema[right].primary_key
        strategy = sample.strategy or ("recent" if isinstance(clock, str) else "uniform")
        assert key is not None and strategy in {"recent", "uniform"}, "Unsupported graph oracle"
        candidates = {}
        for row in source.matching(right, right_key, source.row(left, origin)[left_key]):
            time = (
                timestamp_to_epoch(row[clock])
                if isinstance(clock, str) and row[clock] is not None
                else None
            )
            if strategy == "recent" and (time is None or time > cutoff):
                continue
            # Uniform sampling admits any subset, including events after the cutoff.
            candidates[row[key]] = time if strategy == "recent" else 0
        count = min(sample.fanout, len(candidates))
        boundary = sorted(candidates.values(), reverse=True)[count - 1] if count else None
        mandatory = {
            k for k, time in candidates.items() if boundary is not None and time > boundary
        }

        def edge(k):
            return (sample.relation, origin, k) if forward else (sample.relation, k, origin)

        forced = {item[2] if forward else item[1] for item in required}
        if forced and boundary is None:
            return
        if any(k not in candidates or candidates[k] < boundary for k in forced):
            return
        mandatory |= forced
        if any(edge(k) not in actual_edges for k in mandatory):
            return
        if len(mandatory) > count:
            return
        tied = [
            k
            for k, time in candidates.items()
            if time == boundary and k not in mandatory and edge(k) in actual_edges
        ]
        for selected in combinations(tied, count - len(mandatory)):
            kept = mandatory | set(selected)
            yield {(right, k) for k in kept}, {edge(k) for k in kept}

    seen = set()

    def expand(depth, frontier, visited, emitted):
        key = (depth, frozenset(frontier), frozenset(visited), frozenset(emitted))
        if key in seen:
            return False
        seen.add(key)
        if depth == len(layers):
            return visited == actual_nodes and emitted == actual_edges
        jobs = []
        for sample in layers[depth].samples:
            relation = relations[sample.relation]
            table = relation.left_table if sample.direction == "forward" else relation.right_table
            for name, identity in sorted(frontier, key=repr):
                if name == table:
                    jobs.append((sample, identity))
        later_relations = {s.relation for layer in layers[depth + 1 :] for s in layer.samples}

        def can_emit(job, edge):
            sample, origin = job
            return (
                sample.relation == edge[0]
                and origin == edge[1 if sample.direction == "forward" else 2]
            )

        def select(position, reached, added):
            if position == len(jobs):
                return expand(depth + 1, reached, visited | reached, emitted | added)
            job = jobs[position]
            # The last traversal capable of emitting an edge must include it.
            # This also avoids enumerating huge tied pools on revisited relations.
            required = {
                edge
                for edge in actual_edges - emitted - added
                if can_emit(job, edge)
                and edge[0] not in later_relations
                and not any(can_emit(other, edge) for other in jobs[position + 1 :])
            }
            for nodes, edges in choices(*job, required):
                if select(position + 1, reached | nodes, added | edges):
                    return True
            return False

        return select(0, set(), set())

    return expand(0, {root}, {root}, set())
