# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate graph node features on the native IDs and cutoffs of a sampled graph."""

from __future__ import annotations

import json
from dataclasses import replace

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
    if ticks.min() == ticks.max():
        unique_ids, inverse = np.unique(node_ids, return_inverse=True)
        cutoffs = [int(ticks[0])] * len(unique_ids)
    else:
        # Sort both coordinates at their native widths, including unsigned IDs
        # and negative or nanosecond-precision timestamps.
        cutoffs = ticks[np.asarray(owners, dtype=np.intp)]
        order = np.lexsort((cutoffs, node_ids))
        ordered_ids, ordered_ticks = node_ids[order], cutoffs[order]
        first = np.r_[
            True, (ordered_ids[1:] != ordered_ids[:-1]) | (ordered_ticks[1:] != ordered_ticks[:-1])
        ]
        unique_ids = ordered_ids[first]
        cutoffs = ordered_ticks[first].tolist()
        inverse = np.empty(len(ids), dtype=np.intp)
        inverse[order] = np.cumsum(first) - 1
    native_ids = unique_ids.tolist()
    scalar_values = {}
    if lengths:
        result = source._sampler.query_temporal_lengths(
            {table: native_ids}, {table: cutoffs}, lengths
        )
        scalar_values.update(
            (f"entity:{sequence['name'].removeprefix('event:')}", sequence["lengths"])
            for sequence in result["sequences"]
        )
    if aggregates:
        result = source._sampler.query_temporal(
            {table: native_ids}, {table: cutoffs}, aggregates, output_format="model"
        )
        scalar_values.update((a["name"], a["values"]) for a in result["aggregates"])
    direct = source._direct_entity_arrays(sample, native_ids)
    values = source._entity_arrays(sample, direct, scalar_values)
    return {name: values[name].take(inverse) for name in requested}
