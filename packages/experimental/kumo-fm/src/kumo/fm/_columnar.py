# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt public PyTorch columnar graph batches into retained FM tables."""
import numpy as np
import pyarrow as pa


def adapt_columnar_batch(adapter, workload, seeds, batch):
    """Consume public Arrow columns; only reorder when anchor order requires it."""
    from kumo.pytorch.api import ColumnarGraph
    graph = batch.inputs["graph"]
    if not isinstance(graph, ColumnarGraph) or graph.root_type != workload.entity:
        raise ValueError("Expected a columnar graph rooted at the workload entity")
    permutation = np.asarray(batch.metadata["sample_ids"], dtype=np.int64)
    if not np.array_equal(np.sort(permutation), np.arange(len(seeds))):
        raise ValueError("kumo-pytorch anchor IDs are not a permutation of the request")
    same_order = np.array_equal(permutation, np.arange(len(seeds)))
    nodes, features, edges, directions, inverse = {}, {}, {}, {}, {}
    for name, node in graph.nodes.items():
        type_id = adapter.node_types.index(name)
        positions = node.owners.to_numpy(zero_copy_only=False)
        if (node.owners.null_count or np.any(positions >= len(seeds)) or
                np.any(positions < 0)):
            raise ValueError("kumo-pytorch node owner is out of bounds")
        table = node.features
        if len(table) != len(node.ids) or len(table) != len(positions):
            raise ValueError("kumo-pytorch columnar node/feature alignment mismatch")
        owners = node.owners.cast(pa.uint64()) if same_order else pa.array(permutation[positions], pa.uint64())
        order = None
        if not same_order and name == workload.entity:
            order = np.arange(len(positions))
            order[:len(seeds)] = np.argsort(permutation[positions[:len(seeds)]])
            undo = np.empty(len(order), dtype=np.int64)
            undo[order] = np.arange(len(order))
            inverse[name] = undo
            owners, table = owners.take(pa.array(order)), table.take(pa.array(order))
        nodes[type_id] = [pa.record_batch(
            [owners, pa.array(np.arange(len(positions)), pa.uint64())], names=["id_a", "id_b"])]
        features[type_id] = table.to_batches() or [pa.RecordBatch.from_arrays(
            [table[c].combine_chunks() for c in table.column_names], schema=table.schema)]
    for rel_id, (name, edge) in enumerate(graph.edges.items()):
        rel = adapter.bundle.relations[name]
        orientation = (edge.source_type, edge.destination_type)
        if orientation not in ((rel["child"], rel["parent"]),
                               (rel["parent"], rel["child"])):
            raise ValueError("kumo-pytorch relation orientation differs from the schema")
        native_types = orientation
        columns = []
        for column, table_name in zip(("source", "destination"), native_types, strict=True):
            values = edge.index[column]
            if table_name in inverse:
                indices = values.to_numpy(zero_copy_only=False)
                if values.null_count or np.any(indices < 0) or np.any(indices >= len(inverse[table_name])):
                    raise ValueError("kumo-pytorch edge endpoint is out of bounds")
                values = pa.chunked_array([pa.array(inverse[table_name][indices], pa.uint64())])
            columns.append(values.cast(pa.uint64()))
        if orientation != (rel["child"], rel["parent"]):
            columns.reverse()
        index = pa.Table.from_arrays(columns, names=["id_a", "id_b"])
        directions[rel_id] = name, 'forward', rel["child"], rel["parent"]
        edges[rel_id] = index.to_batches() or [pa.RecordBatch.from_arrays(
            [index[c].combine_chunks() for c in index.column_names], schema=index.schema)]
    count = sum(len(b) for group in (*nodes.values(), *edges.values()) for b in group)
    if count > workload.budgets["max_candidate_rows"]:
        raise ValueError("Native-multihop node-plus-edge budget exceeded")
    adapter.metrics["native_candidate_rows"] += count
    return adapter.assembler._adapt(workload, seeds, np.arange(len(seeds), dtype=np.uint64),
                                 nodes, features, edges, directions)

