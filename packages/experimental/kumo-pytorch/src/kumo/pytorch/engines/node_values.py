# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate graph node features on the native IDs and cutoffs of a sampled graph."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import replace
from threading import Lock

import numpy as np
import pyarrow as pa

from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    Aggregate,
    Compare,
    Constant,
    Derived,
    EntitySpec,
    EventSpec,
    Rows,
    SampleSpec,
    graph_node_dtype,
)


def _distinct_pairs(ids, ticks):
    if ticks.min() == ticks.max():
        unique, inverse = np.unique(ids, return_inverse=True)
        return unique, np.full(len(unique), ticks[0], dtype=np.int64), inverse
    order = np.lexsort((ticks, ids))
    sorted_ids, sorted_ticks = ids[order], ticks[order]
    first = np.r_[
        True, (sorted_ids[1:] != sorted_ids[:-1]) | (sorted_ticks[1:] != sorted_ticks[:-1])
    ]
    inverse = np.empty(len(ids), dtype=np.intp)
    inverse[order] = np.cumsum(first) - 1
    return sorted_ids[first], sorted_ticks[first], inverse


class DegreeCache:
    """Exact temporal counts from an immutable task index, at most four 2 MiB programs."""

    def __init__(self):
        self.entries = OrderedDict()
        self.lock = Lock()

    def read(self, key, names, ids, ticks, fetch):
        with self.lock:
            capacity = min(65536, (2 << 20) // (17 + 8 * len(names)))
            if not capacity:
                unique, cutoffs, inverse = _distinct_pairs(ids, ticks)
                return {
                    name: values.take(inverse) for name, values in fetch(unique, cutoffs).items()
                }
            if key not in self.entries:
                if len(self.entries) == 4:
                    self.entries.popitem(last=False)
                self.entries[key] = (
                    np.empty(capacity, dtype=np.uint64),
                    np.empty(capacity, dtype=np.int64),
                    np.zeros(capacity, dtype=bool),
                    {name: np.empty(capacity, dtype=np.uint64) for name in names},
                )
            self.entries.move_to_end(key)
            stored_ids, stored_ticks, occupied, counts = self.entries[key]
            # The hash chooses a slot only; compare the complete ID and signed clock.
            hashed = ids ^ (ticks.view(np.uint64) * np.uint64(11400714819323198485))
            slots = (hashed ^ (hashed >> 32)) % capacity
            hits = occupied[slots] & (stored_ids[slots] == ids) & (stored_ticks[slots] == ticks)
            result = {name: column[slots] for name, column in counts.items()}
            if not hits.all():
                missing = ~hits
                unique, cutoffs, inverse = _distinct_pairs(ids[missing], ticks[missing])
                fetched = fetch(unique, cutoffs)
                # Pick one occurrence per slot, so colliding keys, ticks and values agree.
                targets, keep = np.unique(slots[missing], return_index=True)
                occupied[targets] = False
                for name, values in fetched.items():
                    native = values.to_numpy()
                    result[name][missing] = native[inverse]
                    counts[name][targets] = native[inverse[keep]]
                stored_ids[targets] = ids[missing][keep]
                stored_ticks[targets] = ticks[missing][keep]
                occupied[targets] = True
            return {name: pa.array(values) for name, values in result.items()}


def _query_lengths(source, table, nodes, cutoffs, plan):
    result = source._sampler.query_temporal_lengths({table: nodes}, {table: cutoffs}, plan)
    return {
        f"entity:{seq['name'].removeprefix('event:')}": seq["lengths"]
        for seq in result["sequences"]
    }


def _node_program(source, table, declarations, requested):
    from kumo.pytorch.engines.diskgraph import _TemporalPlanCompiler

    # Only evaluate selected values and the operands they depend on.
    needed = set(requested)
    pending = list(requested)
    while pending:
        value = declarations[pending.pop()]
        if isinstance(value, Derived):
            unseen = set(value.inputs) - needed
            pending.extend(unseen)
            needed.update(unseen)
    sample = SampleSpec(
        entity=EntitySpec(table),
        entity_values={name: val for name, val in declarations.items() if name in needed},
    )
    counts, reductions = {}, {}
    for name, value in sample.entity_values.items():
        aggregate = value.left if isinstance(value, Compare) else value
        if not isinstance(aggregate, Aggregate):
            continue
        rows = aggregate.source
        if (
            aggregate.operation == "count"
            and aggregate.column is None
            and isinstance(rows, Rows)
            and len(rows.path) == 1
        ):
            # A single FK joins each child row to one parent. Neither traversal
            # direction can repeat a destination for the same starting row.
            # The native length kernel can count an indexed window by its bounds.
            if source._ingested_foreign_keys:
                rows = replace(rows, distinct=False)
            counts[name] = EventSpec(rows, {"__length__": Constant(0, DType("int64"))})
        else:
            reductions[name] = value
    lengths = None
    if counts:
        program = _TemporalPlanCompiler(source).compile(SampleSpec(sample.entity, events=counts))
        lengths = json.dumps(program.plan)
    aggregates = None
    if reductions:
        program = _TemporalPlanCompiler(source).compile(replace(sample, entity_values=reductions))
        aggregates = json.dumps(program.plan)
    return sample, lengths, aggregates


def computed_node_arrays(source, neighborhood, table, requested, ids, owners, root_ticks):
    """Read each distinct (node ID, cutoff) once and restore node occurrence order."""
    if not len(ids):
        return {
            name: pa.array(
                [], type=graph_node_dtype(source.schema, neighborhood, table, name).to_arrow()
            )
            for name in requested
        }
    # Anchor materializations apply only to their original roots. Their fields
    # may even use the same names as the values declared for this node table.
    runtime = getattr(source, "_runtime_batch", None)
    cache = None if runtime is None else runtime.get("_degree_cache")
    if getattr(source, "_materialized_values", None) is not None:
        source = source._materialized_reference.source
    declarations = neighborhood.node_values[table]
    key = (id(declarations), table, requested)
    if key not in source._node_value_programs:
        source._node_value_programs[key] = (
            declarations,
            _node_program(source, table, declarations, requested),
        )
    sample, lengths, aggregates = source._node_value_programs[key][1]

    # An AnchorTable without an as_of_time column uses epoch zero.
    ticks = np.asarray([0] if root_ticks is None else root_ticks, dtype=np.int64)
    node_ids = np.asarray(ids, dtype=np.uint64)
    occurrence_ticks = (
        np.full(len(ids), ticks[0], dtype=np.int64)
        if len(ticks) == 1
        else ticks[np.asarray(owners, dtype=np.intp)]
    )
    if (
        cache is not None
        and lengths
        and not aggregates
        and all(isinstance(declarations[name], Aggregate) for name in requested)
    ):
        names = tuple(f"entity:{name}" for name in requested)

        def fetch(nodes, times):
            return _query_lengths(source, table, nodes.tolist(), times.tolist(), lengths)

        counts = cache.read((table, lengths), names, node_ids, occurrence_ticks, fetch)
        # These are independent count aggregates. Respect their declared dtypes
        # without rebuilding the evaluation order for derived expressions.
        return {
            name: source._typed_array(counts[native], declarations[name].dtype)
            for name, native in zip(requested, names, strict=True)
        }
    unique_ids, cutoffs, inverse = _distinct_pairs(node_ids, occurrence_ticks)
    cutoffs = cutoffs.tolist()
    native_ids = unique_ids.tolist()
    scalar_values = {}
    if lengths:
        scalar_values.update(_query_lengths(source, table, native_ids, cutoffs, lengths))
    if aggregates:
        result = source._sampler.query_temporal(
            {table: native_ids}, {table: cutoffs}, aggregates, output_format="model"
        )
        scalar_values.update((a["name"], a["values"]) for a in result["aggregates"])
    direct = source._direct_entity_arrays(sample, native_ids)
    values = source._entity_arrays(sample, direct, scalar_values)
    return {name: values[name].take(inverse) for name in requested}
