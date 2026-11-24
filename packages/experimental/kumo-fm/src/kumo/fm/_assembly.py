# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build occurrence-aligned FM tables directly from public columnar graphs."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ._types import TYPES


def _parent_positions(child, parent, child_count):
    """Collapse repeated selected edges while enforcing FM's one-parent rule."""
    positions = np.full(child_count, -1, dtype=np.int64)
    positions[child] = parent
    if not np.array_equal(positions[child], parent):
        raise ValueError("Selected child has conflicting parents")
    return pa.array(positions, mask=positions < 0)


def assemble_graph(payload, seeds, graph):
    """Consume ColumnarGraph's declared topology and aligned, owned Arrow columns."""
    from kumo.pytorch.api import ColumnarGraph

    if not isinstance(graph, ColumnarGraph) or graph.root_type != payload.entity:
        raise ValueError("Expected a columnar graph rooted at the FM entity")
    count = sum(len(node.ids) for node in graph.nodes.values()) + sum(len(edge.index) for edge in graph.edges.values())
    if count > payload.budgets.max_candidate_rows:
        raise ValueError("FM sampled node-plus-edge budget exceeded")
    for name in payload.table_names:
        if len(graph.nodes[name].ids) > payload.budgets.max_rows_per_table:
            raise ValueError("Expanded related table exceeds row budget")
    schema_view = payload.schema_view
    root_spec = schema_view.tables[payload.entity]
    if clock := root_spec.get("time_column"):
        roots = graph.nodes[payload.entity].features[clock].slice(0, len(seeds))
        anchors = pa.array([seed["anchor_time"] for seed in seeds], TYPES["timestamp[us]"])
        if not pc.all(pc.fill_null(pc.less_equal(pc.cast(roots, anchors.type), anchors), False)).as_py():
            raise ValueError("Task entity is not eligible at its anchor")

    edge_columns = {}
    for name in payload.relation_names:
        relation = schema_view.relations[name]
        edge = graph.edges[name].index
        # Public edges retain the declared PK endpoint even for self-relations.
        child_column, parent_column = (("destination", "source") if relation["parent_is_source"]
                                       else ("source", "destination"))
        child = edge[child_column].to_numpy(zero_copy_only=False)
        parent = edge[parent_column].to_numpy(zero_copy_only=False)
        parents = graph.nodes[relation["parent"]].features[schema_view.tables[relation["parent"]]["primary_key"]]
        positions = _parent_positions(child, parent, len(graph.nodes[relation["child"]].ids))
        edge_columns["__fm_edge_" + name] = pc.take(parents, positions)

    instance_ids = pa.array([seed["instance_id"] for seed in seeds], pa.int64())
    ready = {}
    for name, spec in payload._metadata["schema"]["related_tables"].items():
        node = graph.nodes[name]
        columns = {}
        for column in spec["columns"]:
            if column == "instance_id":
                columns[column] = pc.take(instance_ids, node.owners)
            elif column in edge_columns:
                columns[column] = edge_columns[column]
            else:
                columns[column] = node.features[column]
        ready[name] = pa.table(columns)
    return ready
