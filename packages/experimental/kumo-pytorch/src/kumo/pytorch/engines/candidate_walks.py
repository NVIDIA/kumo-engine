# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch compatible candidate walks while preserving each graph's occurrence order."""

from types import SimpleNamespace

import numpy as np
import torch

from kumo.pytorch.contract.batch import Graph, NeighborhoodSource, Raw
from kumo.pytorch.contract.sample import WalkSample
from kumo.pytorch.encoding import _torch_type
from kumo.pytorch.output import kernel_metadata, offsets


def compatible_walk_groups(assembler, compact):
    """Group only single-stage walks whose projected values have one tensor row per node."""
    groups = []
    for name, prepared in assembler._prepared_inputs.items():
        rep, binding = prepared.spec.representation, prepared.spec.source
        if not (
            isinstance(rep, Graph)
            and rep.topology == "relations"
            and isinstance(binding, NeighborhoodSource)
            and (binding.candidates is None or binding.candidates in compact)
        ):
            continue
        spec = (
            assembler.sample
            if binding.candidates is None
            else assembler.sample.candidate_sets[binding.candidates].sample_spec()
        )
        hood = spec.neighborhoods[binding.name]
        if not (
            len(hood.layers) == 1
            and len(hood.layers[0].samples) == 1
            and isinstance(hood.layers[0].samples[0], WalkSample)
            and all(
                field.destination is None
                and isinstance(field.encoder.config, Raw)
                and _torch_type(field.encoder.dtype) is not None
                for fields in prepared.state.values()
                for field in fields.values()
            )
        ):
            continue
        key = (spec.entity, hood, rep)
        for prev, candidates, anchors in groups:
            if key == prev:
                break
        else:
            candidates, anchors = [], []
            groups.append((key, candidates, anchors))
        if binding.candidates is None:
            anchors.append((name, spec, prepared))
        else:
            candidates.append((name, spec, prepared))
    return [(candidates, anchors) for _, candidates, anchors in groups if len(candidates) > 1]


def _take(value, indices):
    if isinstance(value, torch.Tensor):
        return torch.from_numpy(value.numpy()[indices])
    return tuple(value[i] for i in indices)  # non-numeric public node IDs


def _expand(shared, inverse, root_table, edge_types, deferred_rows):
    """A one-layer walk emits distinct destinations per root, in native score/rank order."""
    nodes, edges = {}, {}
    owners = np.arange(len(inverse), dtype=np.int64)
    for table, node in shared["nodes"].items():
        root = table == root_table
        indices = inverse if root else np.empty(0, dtype=np.int64)
        batch = owners if root else indices
        if table in edge_types:
            name, source_offsets = edge_types[table]
            edge = shared["edges"][name]
            lengths = source_offsets[inverse + 1] - source_offsets[inverse]
            starts = offsets(lengths)
            owner = np.repeat(owners, lengths)
            ordinal = np.arange(len(owner), dtype=np.int64)
            edge_rows = (source_offsets[inverse] - starts[:-1])[owner] + ordinal
            dest = edge["index"][1].numpy()[edge_rows]
            edges[name] = {
                **edge,
                "index": torch.from_numpy(np.stack((owner, len(indices) + ordinal))),
                "score": _take(edge["score"], edge_rows),
                "rank": _take(edge["rank"], edge_rows),
            }
            indices = np.concatenate((indices, dest)) if root else dest
            batch = np.concatenate((batch, owner)) if root else owner
        if table in deferred_rows:
            deferred_rows[table].append(indices)
            ids = None
        else:
            ids = _take(node["ids"], indices) if len(indices) else torch.empty(0, dtype=torch.long)
        nodes[table] = {
            "ids": ids,
            "batch": torch.from_numpy(batch),
            "features": {
                field: None
                if encoded is None
                else {key: _take(value, indices) for key, value in encoded.items()}
                for field, encoded in node["features"].items()
            },
        }
    return {"nodes": nodes, "edges": {name: edges[name] for name in shared["edges"]}}


def _fill_projections(pending):
    # Each table is read once for all matching graphs. The slices in model outputs
    # are disjoint, including when different anchors or candidates select the same row.
    for project, pieces in pending.values():
        ids, features = project(np.concatenate([rows for _, rows in pieces]))
        start = 0
        for node, rows in pieces:
            stop = start + len(rows)
            node["ids"] = ids[start:stop] if len(rows) else torch.empty(0, dtype=torch.long)
            for field, encoded in features.items():
                node["features"][field] = {key: value[start:stop] for key, value in encoded.items()}
            start = stop


def _graph_request(prepared, deferred):
    return SimpleNamespace(
        source=prepared.spec.source.name,
        topology=prepared.spec.representation.topology,
        node_fields=prepared.spec.representation.node_fields,
        deferred_projections=deferred,
    )


def sample_candidate_walks(source, groups, candidate_roots, seed, epoch):
    output = {}
    for candidates, anchors in groups:
        _, spec, prepared = candidates[0]
        inputs = {name: field.spec.source.candidates for name, _, field in candidates}
        unique, rows = {}, []
        occurrences = {}
        for candidate in dict.fromkeys(inputs.values()):
            owners = candidate_roots[candidate]
            inverse = []
            for owner, (cutoff, ids) in enumerate(
                zip(owners.cutoffs, owners.entity_ids, strict=True)
            ):
                for position, entity_id in enumerate(ids):
                    index = unique.setdefault((entity_id, cutoff), len(rows))
                    if index == len(rows):
                        rows.append(owners.root(owner, position, entity_id))
                    inverse.append(index)
            occurrences[candidate] = np.asarray(inverse, dtype=np.int64), owners.bounds

        deferred = {}
        # Native sampling and point-in-time projections both run once for the shared roots.
        shared = source.sample_graph(
            spec,
            _graph_request(prepared, deferred),
            prepared.state,
            rows,
            seed,
            epoch,
        )
        edge_types = {}
        for name, edge in shared["edges"].items():
            starts = offsets(np.bincount(edge["index"][0].numpy(), minlength=len(rows)))
            edge_types[edge["destination_type"]] = name, starts
        deferred_rows = {table: [] for table in deferred}
        for name, candidate in inputs.items():
            inverse, boundaries = occurrences[candidate]
            output[name] = _expand(shared, inverse, spec.entity.table, edge_types, deferred_rows)
            output[name].update(
                kernel_metadata({"candidate_offsets": torch.from_numpy(boundaries.copy())})
            )
        pending = {
            table: (
                project,
                [
                    (output[name]["nodes"][table], dense[indices])
                    for name, indices in zip(inputs, deferred_rows[table], strict=True)
                ],
            )
            for table, (dense, project) in deferred.items()
        }
        for name, anchor_spec, anchor in anchors:
            anchor_fields = {}
            graph = source.sample_graph_batch(
                anchor_spec,
                _graph_request(anchor, anchor_fields),
                anchor.state,
                source._runtime_batch,
            )
            output[name] = graph
            for table, (dense, project) in anchor_fields.items():
                pending.setdefault(table, (project, []))[1].append((graph["nodes"][table], dense))
        _fill_projections(pending)
    return output
