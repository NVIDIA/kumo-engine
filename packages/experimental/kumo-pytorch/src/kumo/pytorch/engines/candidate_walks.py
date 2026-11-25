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
from kumo.pytorch.output import offsets


def compatible_walk_groups(assembler, compact):
    """Group only single-stage walks whose projected values have one tensor row per node."""
    groups = []
    for name, prepared in assembler._prepared_inputs.items():
        rep, binding = prepared.spec.representation, prepared.spec.source
        if not (
            isinstance(rep, Graph)
            and rep.topology == "relations"
            and isinstance(binding, NeighborhoodSource)
            and binding.candidates in compact
        ):
            continue
        spec = assembler.sample.candidate_sets[binding.candidates].sample_spec()
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
        for prev, _, _, inputs in groups:
            if key == prev:
                break
        else:
            inputs = {}
            groups.append((key, spec, prepared, inputs))
        inputs[name] = binding.candidates
    return [(spec, prepared, inputs) for _, spec, prepared, inputs in groups if len(inputs) > 1]


def _take(value, indices):
    if isinstance(value, torch.Tensor):
        return torch.from_numpy(value.numpy()[indices])
    return tuple(value[i] for i in indices)  # non-numeric public node IDs


def _expand(shared, inverse, root_table, edge_types):
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
        nodes[table] = {
            "ids": _take(node["ids"], indices)
            if len(indices)
            else torch.empty(0, dtype=torch.long),
            "batch": torch.from_numpy(batch),
            "features": {
                field: {key: _take(value, indices) for key, value in encoded.items()}
                for field, encoded in node["features"].items()
            },
        }
    return {"nodes": nodes, "edges": {name: edges[name] for name in shared["edges"]}}


def sample_candidate_walks(source, groups, candidate_roots, seed, epoch):
    output = {}
    for spec, prepared, inputs in groups:
        unique, rows = {}, []
        occurrences = {}
        for name, candidate in inputs.items():
            owners = candidate_roots[candidate]
            inverse = []
            for group in owners:
                for root in group:
                    key = (root.entity_id, root.cutoff)
                    index = unique.setdefault(key, len(rows))
                    if index == len(rows):
                        rows.append(root)
                    inverse.append(index)
            occurrences[name] = (
                np.asarray(inverse, dtype=np.int64),
                offsets([len(g) for g in owners]),
            )

        # Native sampling and point-in-time projections both run once for the shared roots.
        shared = source.sample_graph(
            spec,
            SimpleNamespace(
                source=prepared.spec.source.name,
                topology=prepared.spec.representation.topology,
                node_fields=prepared.spec.representation.node_fields,
            ),
            prepared.state,
            rows,
            seed,
            epoch,
        )
        edge_types = {}
        for name, edge in shared["edges"].items():
            starts = offsets(np.bincount(edge["index"][0].numpy(), minlength=len(rows)))
            edge_types[edge["destination_type"]] = name, starts
        for name, (inverse, boundaries) in occurrences.items():
            output[name] = _expand(shared, inverse, spec.entity.table, edge_types)
            output[name]["candidate_offsets"] = torch.from_numpy(boundaries)
    return output
