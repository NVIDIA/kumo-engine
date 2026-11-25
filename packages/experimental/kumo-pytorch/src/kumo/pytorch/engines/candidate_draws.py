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


def draw_candidate_ids(source, sample, roots, *, runtime=None):
    """Return one group of public candidate keys per root and declared draw."""
    started = time.perf_counter_ns()
    draws = {name: spec.draw for name, spec in sample.candidate_sets.items() if spec.draw}
    row_draws = {name: draw for name, draw in draws.items() if isinstance(draw, DrawRows)}
    results = {}
    if row_draws and roots:
        history = runtime.get("_candidate_row_history") if runtime is not None else None
        for name, pool in _row_pools(source, sample, roots, row_draws, history):
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
                name_in_runtime = runtime is not None and (
                    draw.neighborhood in runtime["graphs"]
                    or runtime.get("_anchor_neighborhood") == draw.neighborhood
                )
                if name_in_runtime:
                    graph = runtime["graphs"].get(draw.neighborhood, runtime["graphs"]["main"])
                    physical = source._sampling_plan(sample.neighborhoods[draw.neighborhood])[1]
                    graphs[draw.neighborhood] = _native_graph_columns(graph, physical)
                else:
                    _, events, physical = source._sample_graph_events(
                        sample, draw.neighborhood, roots
                    )
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


def candidate_row_windows(draws):
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
        events[f"draw:{next(iter(sharing))}"] = (EventSpec(rows, fields), sharing)
    return events


def _row_pools(source, sample, roots, draws, row_history):
    if row_history is None:
        from kumo.pytorch.engines.diskgraph import _TemporalPlanCompiler

        cached = getattr(source, "_candidate_row_program", None)
        if cached is None or cached[0] is not sample:
            windows = candidate_row_windows(draws)
            program = _TemporalPlanCompiler(source).compile(
                SampleSpec(entity=sample.entity), candidate_windows=windows
            )
            cached = source._candidate_row_program = (sample, windows, program)
        _, windows, program = cached
        result = source._query_values(
            sample,
            source._root_ids(sample.entity.table, roots),
            [root.cutoff for root in roots],
            program.plan,
            output_format="model",
        )
    else:
        windows, result = row_history
    for sequence in result["sequences"]:
        if sequence["name"] not in windows:
            continue
        event, sharing = windows[sequence["name"]]
        table = destination_table(source.schema, sample.entity.table, event.rows.path)
        offsets = sequence["offsets"].to_numpy()
        owners = np.repeat(np.arange(len(roots)), np.diff(offsets).astype(np.intp))
        batch = sequence["batch"]
        fields = {
            col: source._restore_feature_array(table, col, batch.column(col))
            for col in event.fields
        }
        for name, draw in sharing.items():
            columns = {"owner": owners, "id": fields[draw.column]}
            if draw.weight:
                columns["weight"] = fields[draw.weight]
            yield name, pa.table(columns)


def _native_graph_columns(graph, physical):
    # The materialized iterator already exposes node IDs, owners and traversal
    # endpoints as separate Arrow arrays. Skip the intermediate event batches.
    nodes = {name: (graph["batch_dict"][name], ids) for name, ids in graph["node_ids"].items()}
    edges = defaultdict(list)
    for edge in graph["edge_index_dict"]:
        relation = edge["edge_type"]["rel"]
        if relation in physical:
            edges[physical[relation]].append((edge["dst"], None))
    for event in graph.get("walk_edges", ()):
        batch = event["batch"]
        edges[f"{event['walk']}.{event['destination_type']}"].append(
            (batch.column("destination"), batch)
        )
    return nodes, edges


def _graph_columns(source, events, physical):
    nodes = defaultdict(list)
    edges = defaultdict(list)
    for event in events:
        if event["type"] == "node":
            nodes[source._node_types[event["type_id"]]].append(event["batch"])
        elif event["type"] == "edge":
            relation = source._relations_by_id[event["rel_id"]]
            if relation["name"] in physical:
                edges[physical[relation["name"]]].append((event["batch"].column("id_b"), None))
        elif event["type"] == "walk_edges":
            name = f"{event['walk']}.{event['destination_type']}"
            edges[name].append((event["batch"].column("destination"), event["batch"]))
    nodes = {name: pa.Table.from_batches(batches) for name, batches in nodes.items()}
    return {name: (data["id_a"], data["id_b"]) for name, data in nodes.items()}, edges


def _neighbor_pool(nodes, edges, table, draw):
    parts = []
    for name in draw.edges:
        for destination, batch in edges.get(name, ()):
            weights = batch.column(draw.weight) if draw.weight else None
            if draw.ranks:
                lo, hi = draw.ranks
                ranks = batch.column("rank")
                keep = pc.and_(pc.greater_equal(ranks, lo), pc.less(ranks, hi))
                destination = destination.filter(keep)
                if weights is not None:
                    weights = weights.filter(keep)
            if not len(destination):
                continue
            # Forward and reverse hops both point at the sampled node position.
            owners, ids = nodes[table]
            columns = {"owner": owners.take(destination), "id": ids.take(destination)}
            if weights is not None:
                columns["weight"] = weights
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
    if not len(pool):
        return groups
    weights = None
    if draw.weight:
        weight = arrow_float64(pool["weight"])
        weights = (weight.fill_null(0) if weight.null_count else weight).to_numpy()
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(
                f"candidate set {name!r} requires finite nonnegative weights in {draw.weight!r}"
            )
    # Fix the pool order independently of native event chunks and batch size.
    owners, ids = pool["owner"].to_numpy(), pool["id"]
    if pa.types.is_integer(ids.type):
        ids = ids.to_numpy()
        order = np.lexsort((ids, owners))
    else:
        order = pc.sort_indices(
            pool, sort_keys=[("owner", "ascending"), ("id", "ascending")]
        ).to_numpy()
    if weights is not None:
        if (weights == 0).any():
            order = order[weights[order] > 0]
            if not len(order):
                return groups
        weights = weights[order]
    owners, ids = owners[order], ids.take(order)
    different_ids = (
        ids[1:] != ids[:-1]
        if isinstance(ids, np.ndarray)
        else pc.not_equal(ids.slice(1), ids.slice(0, len(ids) - 1)).to_numpy()
    )
    distinct = (owners[1:] != owners[:-1]) | different_ids
    first = None
    if not distinct.all():
        first = np.concatenate(([0], np.flatnonzero(distinct) + 1))
        ids = ids.take(first)
        owners = owners[first]
    # Keys already occur in contiguous owner blocks. Sorting within these
    # blocks preserves their boundaries, so slice the first k in each block.
    sizes = np.bincount(owners.astype(np.intp, copy=False), minlength=len(roots))
    if sizes.max() > 1:
        present = np.flatnonzero(sizes)
        priorities = _priorities(roots, present, sizes[present], name, seed)
        if draw.weight:
            # Repeated weights can exceed float64 in linear space.
            log_mass = np.log(weights)
            if first is not None:
                log_mass = np.logaddexp.reduceat(log_mass, first)
            priorities = log_mass - np.log(-np.log(priorities))
        ordered = np.lexsort((-priorities, owners))
        if sizes.max() > draw.count:
            counts = np.minimum(sizes, draw.count)
            skipped = sizes - counts
            # For each retained key, skip only the excess keys in earlier blocks.
            take = np.arange(counts.sum()) + np.repeat(skipped.cumsum() - skipped, counts)
            ordered = ordered[take]
            sizes = counts
        ids = ids.take(ordered)
    values = ids.tolist() if isinstance(ids, np.ndarray) else _arrow_values(ids)
    position = 0
    for owner, size in enumerate(sizes.tolist()):
        groups[owner] = tuple(values[position : position + size])
        position += size
    return groups


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
