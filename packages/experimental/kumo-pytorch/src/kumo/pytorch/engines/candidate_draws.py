# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched candidate selection from DiskGraph's temporal and sampled graph results."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from kumo.pytorch.contract.dtypes import _arrow_values, _strip_utc_timezone, arrow_float64
from kumo.pytorch.contract.sample import (
    DrawNeighbors,
    DrawRows,
    EventSpec,
    Field,
    SampleSpec,
    destination_table,
)
from kumo.pytorch.contract.schema import table_clock


def draw_candidate_ids(source, sample, roots):
    """Return one group of public candidate keys per root and declared draw."""
    started = time.perf_counter_ns()
    draws = {name: spec.draw for name, spec in sample.candidate_sets.items() if spec.draw}
    row_draws = {name: draw for name, draw in draws.items() if isinstance(draw, DrawRows)}
    results = {}
    if row_draws and roots:
        for name, pool in _row_pools(source, sample, roots, row_draws):
            entity = sample.candidate_sets[name].entity
            pool = _eligible_entities(source, pool, entity, roots, native_ids=False)
            results[name] = _select(pool, roots, row_draws[name], name, source._random_seed)

    graphs = {}
    sets_by_table = defaultdict(list)
    for name, draw in draws.items():
        if not roots:
            results[name] = []
        elif isinstance(draw, DrawNeighbors):
            if draw.neighborhood not in graphs:
                _, events, physical = source._sample_graph_events(sample, draw.neighborhood, roots)
                graphs[draw.neighborhood] = _graph_columns(source, events, physical)
            entity = sample.candidate_sets[name].entity
            table = entity.table
            pool = _neighbor_pool(*graphs[draw.neighborhood], table, draw)
            pool = _eligible_entities(source, pool, entity, roots, native_ids=True)
            results[name] = _select(pool, roots, draw, name, source._random_seed)
            sets_by_table[table].append(name)

    # Native graph endpoints are node IDs. Fetch only the selected primary keys,
    # once per table even when several heads draw from the same neighborhood.
    for table, names in sets_by_table.items():
        ids = list(dict.fromkeys(i for name in names for group in results[name] for i in group))
        keys = source._feature_values(table, ids, source._tables[table].primary_key)
        by_id = dict(zip(ids, keys, strict=True))
        for name in names:
            results[name] = [tuple(by_id[i] for i in group) for group in results[name]]
    source._last_candidate_draw_profile = {
        "samples": len(roots),
        "total_ns": time.perf_counter_ns() - started,
    }
    return results


def _row_pools(source, sample, roots, draws):
    from kumo.pytorch.engines.diskgraph import _TemporalPlanCompiler

    # One native sequence per window, including all requested keys and weights.
    # The temporal compiler pushes the exact bounds into the native traversal.
    by_rows = defaultdict(dict)
    for name, draw in draws.items():
        by_rows[draw.rows][name] = draw
    events = {}
    for rows, sharing in by_rows.items():
        fields = {
            col: Field(col)
            for draw in sharing.values()
            for col in (draw.column, draw.weight)
            if col
        }
        events[next(iter(sharing))] = EventSpec(rows, fields)
    program = _TemporalPlanCompiler(source).compile(SampleSpec(entity=sample.entity, events=events))
    result = source._query_values(
        sample,
        source._root_ids(sample.entity.table, roots),
        [root.cutoff for root in roots],
        program.plan,
        output_format="model",
    )
    for sequence in result["sequences"]:
        event = program.sequences[sequence["name"]].spec
        table = destination_table(source.schema, sample.entity.table, event.rows.path)
        offsets = sequence["offsets"].to_numpy()
        owners = np.repeat(np.arange(len(roots)), np.diff(offsets).astype(np.intp))
        batch = sequence["batch"]
        fields = {
            col: source._restore_feature_array(table, col, batch.column(col))
            for col in event.fields
        }
        for name, draw in by_rows[event.rows].items():
            columns = {"owner": owners, "id": fields[draw.column]}
            if draw.weight:
                columns["weight"] = fields[draw.weight]
            yield name, pa.table(columns)


def _graph_columns(source, events, physical):
    nodes = defaultdict(list)
    edges = defaultdict(list)
    for event in events:
        if event["type"] == "node":
            nodes[source._node_types[event["type_id"]]].append(event["batch"])
        elif event["type"] == "edge":
            relation = source._relations_by_id[event["rel_id"]]
            if relation["name"] in physical:
                edges[physical[relation["name"]]].append((event["batch"], "id_b"))
        elif event["type"] == "walk_edges":
            name = f"{event['walk']}.{event['destination_type']}"
            edges[name].append((event["batch"], "destination"))
    return {name: pa.Table.from_batches(batches) for name, batches in nodes.items()}, edges


def _neighbor_pool(nodes, edges, table, draw):
    parts = []
    for name in draw.edges:
        for batch, destination in edges.get(name, ()):
            if draw.ranks:
                lo, hi = draw.ranks
                ranks = batch.column("rank")
                batch = batch.filter(pc.and_(pc.greater_equal(ranks, lo), pc.less(ranks, hi)))
            if not len(batch):
                continue
            # Both forward and reverse native hops point from the traversal
            # source to its sampled destination. Node positions carry ownership.
            endpoints = nodes[table].take(batch.column(destination))
            columns = {"owner": endpoints["id_a"], "id": endpoints["id_b"]}
            if draw.weight:
                columns["weight"] = batch.column(draw.weight)
            parts.append(pa.table(columns))
    return pa.concat_tables(parts) if parts else None


def _eligible_entities(source, pool, entity, roots, *, native_ids):
    """Check the candidate row itself at the anchor before sampling without replacement."""
    table = source._tables[entity.table]
    clock = table_clock(table)
    if not (clock or entity.available_from) or pool is None or not len(pool):
        return pool
    # Occurrences across several owners can reference one stored entity. Read
    # each clock once per distinct key, then compare at each owner's own time.
    keys = pc.dictionary_encode(pool["id"].combine_chunks())
    if not len(keys.dictionary):
        return pool.slice(0, 0)
    if native_ids:
        dense = keys.dictionary.to_pylist()
    else:
        public = keys.dictionary.cast(table.columns[table.primary_key].to_arrow())
        dense = source._sampler.lookup_node_ids(table.name, public)
        if any(node is None for node in dense):
            raise ValueError(f"candidate draw references an unknown {table.name!r} primary key")
    columns = tuple(dict.fromkeys(col for col in (clock, entity.available_from) if col))
    stored = source._feature_arrays(table.name, dense, columns)
    cutoffs = source._clock_array(pa.array([r.cutoff for r in roots], type=pa.int64()))
    cutoffs = _strip_utc_timezone(cutoffs).take(pool["owner"])
    eligible = None
    for col in columns:
        times = source._typed_array(stored[col], table.columns[col])
        if pa.types.is_date32(times.type):
            times = times.cast(pa.timestamp("s"))
        times = _strip_utc_timezone(times).take(keys.indices)
        compare = pc.less if col == entity.available_from else pc.less_equal
        visible = compare(times, cutoffs)
        eligible = visible if eligible is None else pc.and_(eligible, visible)
    # Arrow drops null mask entries. An unknown timestamp cannot prove eligibility.
    return pool.filter(eligible)


def _select(pool, roots, draw, name, seed):
    """Choose distinct IDs in per-anchor pools without rescanning other anchors."""
    groups = [()] * len(roots)
    if pool is None or not len(pool):
        return groups
    if pool["id"].null_count:
        pool = pool.filter(pc.is_valid(pool["id"]))
    if draw.weight:
        weight = arrow_float64(pool["weight"])
        invalid = pc.or_(pc.less(weight, 0), pc.invert(pc.is_finite(weight)))
        if pc.any(invalid).as_py():
            raise ValueError(
                f"candidate set {name!r} requires finite nonnegative weights in {draw.weight!r}"
            )
        pool = pool.set_column(pool.schema.get_field_index("weight"), "weight", weight)
        pool = pool.filter(pc.greater(weight, 0))
    if not len(pool):
        return groups

    # Fix the pool order independently of native event chunks and batch size.
    pool = pool.sort_by([("owner", "ascending"), ("id", "ascending")])
    owners = pool["owner"].to_numpy()
    ids = pool["id"]
    different_ids = pc.not_equal(ids.slice(1), ids.slice(0, len(ids) - 1)).to_numpy()
    first = np.flatnonzero(np.r_[True, (owners[1:] != owners[:-1]) | different_ids])
    ids = ids.take(first)
    owners = owners[first]
    starts = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])
    sizes = np.diff(starts, append=len(owners))
    if np.all(sizes == 1):
        selected = ids
    else:
        priorities = _priorities(roots, owners[starts], sizes, name, seed)
        if draw.weight:
            # Sum in log space, including repeated weights whose total exceeds
            # float64. Gumbel top-k draws in proportion to remaining mass.
            log_mass = np.logaddexp.reduceat(np.log(pool["weight"].to_numpy()), first)
            priorities = log_mass - np.log(-np.log(priorities))
        ordered = np.lexsort((-priorities, owners))
        ranks = np.arange(len(ordered)) - np.repeat(starts, sizes)
        selected = ids.take(ordered[ranks < draw.count])
    counts = np.zeros(len(roots), dtype=np.intp)
    counts[owners[starts]] = np.minimum(sizes, draw.count)
    offsets = np.r_[0, np.cumsum(counts)]
    values = _arrow_values(selected)
    return [tuple(values[a:b]) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


def _priorities(roots, owners, sizes, name, seed):
    """Generate a uniform (0, 1) priority per distinct key in sorted pool order."""
    if seed is None:
        words = np.random.default_rng().bit_generator.random_raw(sizes.sum())
    else:
        # SHAKE provides a deterministic byte stream of any length per anchor,
        # without constructing a NumPy generator for each one-candidate draw.
        streams = []
        for owner, size in zip(owners, sizes, strict=True):
            root = roots[owner]
            identity = (seed, name, root.sample_id, root.entity_id, root.cutoff)
            streams.append(hashlib.shake_128(repr(identity).encode()).digest(8 * int(size)))
        words = np.frombuffer(b"".join(streams), dtype="<u8")
    return ((words >> 12).astype(np.float64) + 0.5) * 2**-52
