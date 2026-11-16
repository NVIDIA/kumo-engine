"""Graph neighborhoods (``NeighborhoodSpec``), sampled on the GPU for many samples at once.

The rules follow Kumo PyTorch's G-GRAPH-001 contract. Each sample's graph grows layer by layer from
its entity row:

- A traversal (``GraphSample``) expands the frontier rows of its start table
  (the relation's left table going forward, its right table in reverse) to the
  rows of the other table whose key matches; a NULL key matches nothing.
- ``temporal="inherit"`` keeps the candidates dated at or before the sample's
  cutoff, on the traversal's clock (its ``time_column``, else the destination
  table's); ``"static"`` keeps them all.
- ``strategy="recent"`` keeps the ``fanout`` most recent candidates, ties to the
  larger primary key by printed value. A
  candidate with no time on the clock is refused.
- ``strategy="uniform"`` keeps every candidate when they fit in ``fanout``, and
  otherwise ``fanout`` of them drawn uniformly at random without replacement,
  listed by printed primary key. The draw is this
  backend's own (G-GRAPH-001 leaves it to each engine): a 64-bit hash
  (SplitMix64) of the batches' ``seed`` and ``epoch`` (0 and 0 when fitting),
  the relation, the sample's entity key and cutoff, and the frontier row's key
  (a text key by its xxHash64), so it follows from the data alone: not from row
  order, files, batches, a split or which other samples are sampled with it. When every candidate fits, no random draw is needed.
- The next frontier is the distinct rows the layer reached, in the order first
  reached; a row reached again expands again.

Nodes are unique per sample (table, primary key). Each edge points from its
relation's left row to its right row, whichever way it was traversed, and
appears once per sample. A batch's graph is upstream's: per table (by name),
``ids`` (primary keys: a tensor for integers and booleans, else a tuple of the
Python values, `TypedIds`, for a table that has nodes in the batch) and ``batch``
(each node's sample); per relation (by
name), ``index`` (2 x edges: positions into its two tables' nodes),
``source_type`` and ``destination_type``. Nodes and edges are ordered by sample, then as reached (layer, traversal, frontier
position, then most recent first or by printed primary key).

A frontier row's candidates are a range of an index of the destination table's
rows sorted once by (key, seconds on the clock, printed primary key), or by
(key, printed primary key) for a traversal that reads no time, found with two
binary searches (a text key as its code among the destination column's texts,
`Sampler.hop_keys`; printed order is `test.kumo_pytorch.reference_engine.gpu.data.printed`'s). The last ``fanout``
rows at or before the cutoff are the most recent; a uniform draw hashes every
candidate only when there are at most twice ``fanout`` of them, and otherwise
draws positions until ``fanout`` differ. So a layer costs per neighbor kept, not
per candidate, however many neighbors a row has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler
    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

__all__ = ["NeighborIndex", "Neighborhoods", "sample_neighborhoods"]

_U64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15  # SplitMix64's increment


class NeighborIndex:
    """A table's rows by (key, seconds on `clock`, printed primary key), sorted once, rows with no time set
    apart (`untimed`: their keys, sorted); with no `clock`, by (key, printed primary key), every row kept.
    NULL and NaN keys are left out; keys are their identities (`test.kumo_pytorch.reference_engine.gpu.data.keys`). Printed order is
    `test.kumo_pytorch.reference_engine.gpu.data.printed.key_columns`'; `uniform` lists each row's candidates in it as it draws them, so no rank
    is kept."""

    def __init__(self, table: GpuTable, key: str, clock: str | None, primary_key: str) -> None:
        import cupy
        import pylibcudf as plc

        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column, _view
        from test.kumo_pytorch.reference_engine.gpu.data.keys import key_space
        from test.kumo_pytorch.reference_engine.gpu.data.printed import key_columns
        from test.kumo_pytorch.reference_engine.gpu.sampling.calculations import _seconds

        space = key_space(table, key)
        rows = cupy.arange(table.num_rows, dtype=cupy.int64)
        matchable = space.matchable_at()
        if matchable is not None:
            rows = rows[matchable]
        timed = table.present(clock) if clock is not None else None
        self.untimed = cupy.zeros(0, dtype=cupy.int64)
        if timed is not None:
            self.untimed = cupy.sort(space.at(rows[~timed[rows]]))
            rows = rows[timed[rows]]
        keys = space.at(rows)
        self._table, self._primary_key = table, primary_key
        columns = [_column(keys), *map(_column, key_columns(table, primary_key, rows))]
        self.seconds = None
        if clock is not None:
            self.seconds = _seconds(table.dtypes[clock], table.values(clock)[rows])
            columns.insert(1, _column(self.seconds))
        order = _view(plc.sorting.stable_sorted_order(
            plc.Table(columns), [plc.types.Order.ASCENDING] * len(columns), [plc.types.NullOrder.AFTER] * len(columns)))
        del columns
        self.keys, self.rows = keys[order], rows[order]
        if self.seconds is not None:
            self.seconds = self.seconds[order]

    def ranges(self, keys: Any, cutoffs: Any) -> tuple[Any, Any]:
        """Each key's candidates as a range of the index: those at or before its cutoff (every one if
        `cutoffs` is None)."""

        import cupy
        import pylibcudf as plc

        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column, _view

        start = cupy.searchsorted(self.keys, keys, side="left")
        if cutoffs is None:
            return start, cupy.searchsorted(self.keys, keys, side="right")
        stop = _view(plc.search.upper_bound(
            plc.Table([_column(self.keys), _column(self.seconds)]),
            plc.Table([_column(cupy.ascontiguousarray(keys)), _column(cupy.ascontiguousarray(cutoffs))]),
            [plc.types.Order.ASCENDING] * 2, [plc.types.NullOrder.AFTER] * 2)).astype(cupy.int64)
        return start, stop

    def recent(self, keys: Any, cutoffs: Any, fanout: int) -> tuple[Any, Any, Any]:
        """Each key's `fanout` most recent rows (at or before its cutoff; every time if `cutoffs` is None),
        most recent first, as (needle, rank, row) triples grouped by needle."""

        import cupy

        start, stop = self.ranges(keys, cutoffs)
        needle, rank = _expand(cupy.minimum(stop - start, fanout))
        return needle, rank, self.rows[stop[needle] - 1 - rank]

    def uniform(self, keys: Any, cutoffs: Any, fanout: int, salts: Any) -> tuple[Any, Any, Any]:
        """Each key's candidates (as `ranges`), all of them if they fit in `fanout`, else `fanout` drawn
        uniformly without replacement, seeded by each key's salt; listed by printed primary key, as (needle,
        rank, row) triples grouped by needle."""

        import cupy
        import pylibcudf as plc

        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column, _view
        from test.kumo_pytorch.reference_engine.gpu.data.printed import key_columns

        start, stop = self.ranges(keys, cutoffs)
        needle, position = _draw(stop - start, fanout, salts)
        index = start[needle] + position
        rows = self.rows[index]
        # by needle, then printed primary key, then row (equal keys: the table's order)
        columns = [_column(cupy.ascontiguousarray(needle)),
                   *map(_column, key_columns(self._table, self._primary_key, rows)), _column(rows)]
        order = _view(plc.sorting.stable_sorted_order(
            plc.Table(columns), [plc.types.Order.ASCENDING] * len(columns), [plc.types.NullOrder.AFTER] * len(columns)))
        needle, rows = needle[order], rows[order]
        rank = cupy.arange(needle.size, dtype=cupy.int64) - cupy.searchsorted(needle, needle, side="left")
        return needle, rank, rows

    def untimed_among(self, keys: Any) -> Any:
        """A GPU flag: does any of `keys` match a row with no time? None when every row has one."""

        import cupy

        if self.untimed.size == 0:
            return None
        return (cupy.searchsorted(self.untimed, keys, side="right")
                > cupy.searchsorted(self.untimed, keys, side="left")).any()


def _expand(counts: Any) -> tuple[Any, Any]:
    """(i, j) for every i and every j < counts[i], grouped by i."""

    import cupy

    ends = cupy.cumsum(counts)
    total = int(ends[-1]) if ends.size else 0
    if total == 0:
        return cupy.zeros(0, dtype=cupy.int64), cupy.zeros(0, dtype=cupy.int64)
    slot = cupy.arange(total, dtype=cupy.int64)
    needle = cupy.searchsorted(ends, slot, side="right")
    return needle, slot - (ends[needle] - counts[needle])


def _mix(values: Any) -> Any:
    """SplitMix64's finalizer, on uint64 values (wrapping)."""

    import cupy

    values = values ^ (values >> cupy.uint64(30))
    values = values * cupy.uint64(0xBF58476D1CE4E5B9)
    values = values ^ (values >> cupy.uint64(27))
    values = values * cupy.uint64(0x94D049BB133111EB)
    return values ^ (values >> cupy.uint64(31))


def _mixed(value: int) -> int:
    """SplitMix64's finalizer on one Python int."""

    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _U64
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _U64
    return value ^ (value >> 31)


def _hash(seed: int, *parts: Any) -> Any:
    """A uint64 hash of a seed and Python ints, then int64 arrays, element by element (the ints mixed on the
    host)."""

    import cupy

    value: Any = seed & _U64
    for part in parts:
        if isinstance(part, int):
            value = _mixed(value ^ (part & _U64))
        else:
            value = _mix(cupy.uint64(value) ^ part.astype(cupy.uint64) if isinstance(value, int)
                         else value ^ part.astype(cupy.uint64))
    return value


def _stream(salts: Any, index: Any) -> Any:
    """The `index`-th value of each salt's SplitMix64 stream."""

    import cupy

    return _mix(salts + (index.astype(cupy.uint64) + cupy.uint64(1)) * cupy.uint64(_GOLDEN))


def _draw(counts: Any, fanout: int, salts: Any) -> tuple[Any, Any]:
    """For each count, every position below it if they fit in `fanout`, else `fanout` distinct positions drawn
    uniformly from its salt's stream, as (needle, position) pairs, grouped by needle (in no particular order).

    Up to twice `fanout` candidates, each position is hashed and the `fanout` smallest are kept; beyond, the
    stream's values give positions (modulo the count) until `fanout` differ: the first `fanout` distinct ones,
    a uniform choice either way."""

    import cupy

    needles, positions = [], []
    needle, position = _expand(cupy.where(counts <= fanout, counts, 0))
    needles.append(needle)
    positions.append(position)
    needle, position = _expand(cupy.where((counts > fanout) & (counts <= 2 * fanout), counts, 0))
    if needle.size:
        order = cupy.lexsort(cupy.stack([_stream(salts[needle], position), needle.astype(cupy.uint64)]))
        needle, position = needle[order], position[order]
        keep = cupy.arange(needle.size, dtype=cupy.int64) - cupy.searchsorted(needle, needle, side="left") < fanout
        needles.append(needle[keep])
        positions.append(position[keep])
    many = cupy.flatnonzero(counts > 2 * fanout)
    drawn_needle = drawn_position = drawn_attempt = cupy.zeros(0, dtype=cupy.int64)
    first = 0
    while many.size:  # each round draws 2 x fanout more for the rows still short of `fanout` distinct positions
        who, attempt = _expand(cupy.full(many.size, 2 * fanout, dtype=cupy.int64))
        who, attempt = many[who], attempt + first
        drawn = (_stream(salts[who], attempt) % counts[who].astype(cupy.uint64)).astype(cupy.int64)
        drawn_needle = cupy.concatenate([drawn_needle, who])
        drawn_position = cupy.concatenate([drawn_position, drawn])
        drawn_attempt = cupy.concatenate([drawn_attempt, attempt])
        first += 2 * fanout
        # each position once, at its first attempt; then each row's positions in attempt order
        order = cupy.lexsort(cupy.stack([drawn_attempt, drawn_position, drawn_needle]))
        drawn_needle, drawn_position, drawn_attempt = drawn_needle[order], drawn_position[order], drawn_attempt[order]
        once = cupy.ones(order.size, dtype=bool)
        once[1:] = (drawn_needle[1:] != drawn_needle[:-1]) | (drawn_position[1:] != drawn_position[:-1])
        drawn_needle, drawn_position, drawn_attempt = drawn_needle[once], drawn_position[once], drawn_attempt[once]
        order = cupy.lexsort(cupy.stack([drawn_attempt, drawn_needle]))
        drawn_needle, drawn_position, drawn_attempt = drawn_needle[order], drawn_position[order], drawn_attempt[order]
        rank = cupy.arange(drawn_needle.size, dtype=cupy.int64) - cupy.searchsorted(drawn_needle, drawn_needle,
                                                                                      side="left")
        enough = cupy.zeros(counts.size, dtype=bool)
        enough[drawn_needle[rank == fanout - 1]] = True
        done = enough[drawn_needle]
        keep = done & (rank < fanout)
        needles.append(drawn_needle[keep])
        positions.append(drawn_position[keep])
        drawn_needle, drawn_position, drawn_attempt = drawn_needle[~done], drawn_position[~done], drawn_attempt[~done]
        many = many[~enough[many]]
    return cupy.concatenate(needles), cupy.concatenate(positions)


def _first(ids: Any) -> Any:
    """Positions of each distinct value's first occurrence in `ids`, in position order."""

    import cupy

    order = cupy.argsort(ids, kind="stable")
    ordered = ids[order]
    first = cupy.ones(ids.size, dtype=bool)
    first[1:] = ordered[1:] != ordered[:-1]
    return cupy.sort(order[first])


@dataclass
class Neighborhoods:
    """Sampled graphs of consecutive samples: per table, nodes (sample, row, primary key) and per relation,
    edges (sample, left node, right node), in traversal order; `bounds` give where each sample's
    nodes and edges start (host arrays). A batch's ``features`` are left empty: they are its input's
    (`test.kumo_pytorch.reference_engine.gpu.batching`)."""

    tables: list[str]
    relations: dict[str, tuple[str, str]]  # relation name -> (left table, right table)
    owners: dict[str, Any]
    ids: dict[str, Any]
    rows: dict[str, Any]  # each node's row of its table
    node_bounds: dict[str, Any]
    edges: dict[str, tuple[Any, Any]]  # relation -> (left positions, right positions) among all samples' nodes
    edge_bounds: dict[str, Any]

    def batch(self, start: int, stop: int) -> dict[str, Any]:
        """Samples `start` to `stop` as one batch's graph, in upstream's form."""

        import cupy
        import torch

        nodes = {}
        for table in self.tables:
            low, high = int(self.node_bounds[table][start]), int(self.node_bounds[table][stop])
            nodes[table] = {"ids": _ids_slice(self.ids[table], low, high),
                            "batch": torch.from_dlpack(self.owners[table][low:high] - start),
                            "features": {}}
        edges = {}
        for name, (left_table, right_table) in self.relations.items():
            low, high = int(self.edge_bounds[name][start]), int(self.edge_bounds[name][stop])
            left, right = self.edges[name]
            index = cupy.stack([left[low:high] - int(self.node_bounds[left_table][start]),
                                right[low:high] - int(self.node_bounds[right_table][start])])
            edges[name] = {"index": torch.from_dlpack(index), "source_type": left_table,
                           "destination_type": right_table}
        return {"nodes": nodes, "edges": edges}


def sample_neighborhoods(sampler: Sampler, spec: Any, node_tables: Any, entity_rows: Any, cutoffs: Any,
                         flags: list[tuple[Any, Any]], *, seed: int = 0, epoch: int = 0) -> Neighborhoods:
    """Each sample's graph for neighborhood `spec`, grown from its entity row (`entity_rows`, with the sample's
    cutoff in seconds); `node_tables` adds tables to the output (the input's ``node_fields``); `seed` and
    `epoch` seed uniform draws. Checks are queued on `flags`."""

    import hashlib

    import cupy
    from kumo.pytorch.contract.schema import table_clock

    schema, tables, specs = sampler.contract.schema, sampler.tables, sampler.specs
    relations = {relation.name: relation for relation in schema.relations}
    root = sampler.entity
    used = {traversal.relation: relations[traversal.relation] for layer in spec.layers for traversal in layer.samples}
    names = sorted({root, *node_tables, *(r.left_table for r in used.values()), *(r.right_table for r in used.values())})
    code = {name: number for number, name in enumerate(names)}
    sizes = {name: tables[name].num_rows if name in tables else 0 for name in names}
    count = int(entity_rows.size)
    samples = cupy.arange(count, dtype=cupy.int64)
    entity_keys = _salt_keys(tables[root], specs[root].primary_key, entity_rows)

    # nodes first reached, per table: (sample, row, layer, order within the layer); `seen` holds their sorted keys
    reached: dict[str, list[tuple[Any, Any, int, Any]]] = {name: [] for name in names}
    seen: dict[str, Any] = {name: cupy.zeros(0, dtype=cupy.int64) for name in names}
    rows = entity_rows.astype(cupy.int64)
    reached[root].append((samples, rows, 0, samples))
    seen[root] = cupy.sort(samples * sizes[root] + rows)
    edge_parts: dict[str, list[tuple[Any, Any, Any, int, Any]]] = {name: [] for name in used}
    frontier = (samples, cupy.full(count, code[root], dtype=cupy.int64), rows)

    for depth, layer in enumerate(spec.layers, start=1):
        parts = []  # per traversal: (traversal, frontier entry, rank, destination code, destination row, source row)
        for number, traversal in enumerate(layer.samples):
            relation = used[traversal.relation]
            forward = traversal.direction == "forward"
            start_table, source_key = (relation.left_table, relation.left_key) if forward else \
                (relation.right_table, relation.right_key)
            end_table, target_key = (relation.right_table, relation.right_key) if forward else \
                (relation.left_table, relation.left_key)
            entry = cupy.flatnonzero(frontier[1] == code[start_table])
            if entry.size == 0:
                continue
            source = tables[start_table]
            source_rows = frontier[2][entry]
            present = source.present(source_key)
            if present is not None:
                keep = present[source_rows]
                entry, source_rows = entry[keep], source_rows[keep]
            found = sampler.hop_keys(start_table, source_key, source_rows, end_table, target_key)
            if found is None:
                continue  # an integer key and a text key: no candidate
            keys, valid = found
            if valid is not None:  # keys with no equal value on the other side (1.5 for an integer key)
                entry, source_rows, keys = entry[valid], source_rows[valid], keys[valid]
            clock = traversal.time_column or table_clock(specs[end_table])
            inherit, recent = traversal.temporal == "inherit", traversal.strategy == "recent"
            # a uniform traversal with no time filter reads no time: every candidate, timed or not
            index = _neighbor_index(sampler, end_table, target_key, clock if inherit or recent else None)
            missing = index.untimed_among(keys)
            if missing is not None:
                flags.append((f"{end_table!r} graph nodes are missing time column {clock!r}", missing))
            owner = frontier[0][entry]
            if recent:
                needle, rank, found = index.recent(keys, cutoffs[owner] if inherit else None, traversal.fanout)
            else:
                relation_code = int.from_bytes(hashlib.blake2b(traversal.relation.encode(), digest_size=8).digest(),
                                               "little")
                salts = _hash(seed, epoch, relation_code, entity_keys[owner], cutoffs[owner],
                              _salt_keys(source, source_key, source_rows))
                needle, rank, found = index.uniform(keys, cutoffs[owner] if inherit else None, traversal.fanout,
                                                    salts)
            parts.append((number, entry[needle], rank, code[end_table], found, source_rows[needle]))
        if not parts or not any(p[1].size for p in parts):
            break  # nothing reached: no later layer reaches anything either
        traversal_of = cupy.concatenate([cupy.full(p[1].size, p[0], dtype=cupy.int64) for p in parts])
        entry = cupy.concatenate([p[1] for p in parts])
        rank = cupy.concatenate([p[2] for p in parts])
        target = cupy.concatenate([cupy.full(p[1].size, p[3], dtype=cupy.int64) for p in parts])
        found = cupy.concatenate([p[4] for p in parts])
        source_rows = cupy.concatenate([p[5] for p in parts])
        owner = frontier[0][entry]
        # order: by sample, then traversal, frontier position and recency
        order = cupy.lexsort(cupy.stack([rank, entry, traversal_of, owner]))
        traversal_of, owner, target, found, source_rows = (traversal_of[order], owner[order], target[order],
                                                           found[order], source_rows[order])
        sequence = cupy.arange(order.size, dtype=cupy.int64)
        for number, traversal in enumerate(layer.samples):
            relation = used[traversal.relation]
            pick = cupy.flatnonzero(traversal_of == number)
            if pick.size == 0:
                continue
            left, right = (source_rows[pick], found[pick]) if traversal.direction == "forward" else \
                (found[pick], source_rows[pick])
            edge_parts[traversal.relation].append((owner[pick], left, right, depth, sequence[pick]))
        # each node once per sample: the layer's distinct rows are the next frontier; those not reached before
        # are new nodes
        keys = owner * cupy.asarray([sizes[name] for name in names], dtype=cupy.int64)[target] + found
        distinct = cupy.zeros(order.size, dtype=bool)
        for name in names:
            pick = cupy.flatnonzero(target == code[name])
            if pick.size == 0:
                continue
            firsts = pick[_first(keys[pick])]
            distinct[firsts] = True
            known = seen[name]
            if known.size:
                place = cupy.minimum(cupy.searchsorted(known, keys[firsts]), known.size - 1)
                firsts = firsts[known[place] != keys[firsts]]
            if firsts.size:
                reached[name].append((owner[firsts], found[firsts], depth, sequence[firsts]))
                seen[name] = cupy.sort(cupy.concatenate([known, keys[firsts]]))
        nexts = cupy.flatnonzero(distinct)
        frontier = (owner[nexts], target[nexts], found[nexts])

    owners, ids, node_rows, node_bounds, positions = {}, {}, {}, {}, {}
    for name in names:
        parts_of = reached[name]
        owner = cupy.concatenate([p[0] for p in parts_of]) if parts_of else cupy.zeros(0, dtype=cupy.int64)
        rows = cupy.concatenate([p[1] for p in parts_of]) if parts_of else cupy.zeros(0, dtype=cupy.int64)
        layer = cupy.concatenate([cupy.full(p[0].size, p[2], dtype=cupy.int64) for p in parts_of]) \
            if parts_of else cupy.zeros(0, dtype=cupy.int64)
        sequence = cupy.concatenate([p[3] for p in parts_of]) if parts_of else cupy.zeros(0, dtype=cupy.int64)
        if owner.size:
            order = cupy.lexsort(cupy.stack([sequence, layer, owner]))
            owner, rows = owner[order], rows[order]
        owners[name], node_rows[name] = owner, rows
        ids[name] = _node_ids(tables[name], specs[name].primary_key, rows) if rows.size else \
            cupy.zeros(0, dtype=cupy.int64)
        node_bounds[name] = cupy.searchsorted(owner, cupy.arange(count + 1, dtype=cupy.int64)).get()
        node_key = owner * sizes[name] + rows
        by_key = cupy.argsort(node_key)
        positions[name] = (node_key[by_key], by_key)

    def position(name: str, owner: Any, rows: Any) -> Any:
        keys, places = positions[name]
        return places[cupy.searchsorted(keys, owner * sizes[name] + rows)]

    edges, edge_bounds = {}, {}
    for name, relation in sorted(used.items()):
        parts_of = edge_parts[name]
        if parts_of:
            owner, left, right = (cupy.concatenate([p[i] for p in parts_of]) for i in range(3))
            layer = cupy.concatenate([cupy.full(p[0].size, p[3], dtype=cupy.int64) for p in parts_of])
            sequence = cupy.concatenate([p[4] for p in parts_of])
            # once per sample: the first time reached
            order = cupy.lexsort(cupy.stack([sequence, layer, right, left, owner]))
            first = cupy.ones(order.size, dtype=bool)
            same = (owner[order][1:] == owner[order][:-1]) & (left[order][1:] == left[order][:-1]) & \
                (right[order][1:] == right[order][:-1])
            first[1:] = ~same
            kept = order[first]
            kept = kept[cupy.lexsort(cupy.stack([sequence[kept], layer[kept], owner[kept]]))]
            owner, left, right = owner[kept], left[kept], right[kept]
        else:
            owner = left = right = cupy.zeros(0, dtype=cupy.int64)
        edges[name] = (position(relation.left_table, owner, left), position(relation.right_table, owner, right))
        edge_bounds[name] = cupy.searchsorted(owner, cupy.arange(count + 1, dtype=cupy.int64)).get()
    return Neighborhoods(names, {name: (r.left_table, r.right_table) for name, r in sorted(used.items())},
                         owners, ids, node_rows, node_bounds, edges, edge_bounds)


def _salt_keys(table: GpuTable, column: str, rows: Any) -> Any:
    """Keys at `rows` as uniform draws hash them, CuPy int64, from the value alone (never from what else a table
    holds: a split's partitions hold less): an integer, boolean, date or timestamp as its stored integer, a float
    by its bits (-0.0 as 0.0), a decimal by its words mixed, a text or bytes by the xxHash64 of its bytes, each
    distinct one hashed once."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column, _view

    dtype = table.dtypes[column]
    if dtype.name.startswith("decimal"):
        from test.kumo_pytorch.reference_engine.gpu.data.decimals import gathered_words

        held = gathered_words(table.columns[column], rows).view(cupy.uint64)
        mixed = held[:, 0].copy()
        for k in range(1, held.shape[1]):
            mixed = _mix(mixed ^ held[:, k])
        return mixed.view(cupy.int64)
    if dtype.name not in ("string", "bytes"):
        from test.kumo_pytorch.reference_engine.gpu.data.keys import key_space

        return key_space(table, column).at(rows)
    values = table.values(column)[rows]
    distinct, inverse = cupy.unique(values, return_inverse=True)
    if distinct.size == 0:
        return cupy.zeros(0, dtype=cupy.int64)
    texts = plc.copying.gather(plc.Table([table.dictionaries[column]]), _column(distinct.astype(cupy.int32)),
                               plc.copying.OutOfBoundsPolicy.DONT_CHECK)
    return _view(plc.hashing.xxhash_64(texts, 0)).view(cupy.int64)[inverse.reshape(values.shape)]


class TypedIds:
    """Nodes' primary keys of a type upstream gives as Python values (floats, dates, timestamps, decimals, bytes):
    the values as held (CuPy, or a libcudf column), turned into Python only for a batch (`python`)."""

    def __init__(self, dtype: Any, values: Any) -> None:
        self.dtype, self.values = dtype, values

    def python(self, low: int, high: int) -> list[Any]:
        import pylibcudf as plc

        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
        from test.kumo_pytorch.reference_engine.gpu.sampling.host_values import _values_python

        if isinstance(self.values, plc.Column):
            picked = plc.copying.slice(self.values, [low, high])[0]
            if self.dtype.name == "bytes":
                from test.kumo_pytorch.reference_engine.gpu.data.text import python_texts

                return python_texts(picked, self.dtype)
            return _values_python(Values(self.dtype, None, column=picked))
        return _values_python(Values(self.dtype, self.values[low:high]))

    @staticmethod
    def join(parts: list[TypedIds]) -> TypedIds:
        import cupy
        import pylibcudf as plc

        if isinstance(parts[0].values, plc.Column):
            return TypedIds(parts[0].dtype, plc.concatenate.concatenate([part.values for part in parts]))
        return TypedIds(parts[0].dtype, cupy.concatenate([part.values for part in parts]))


def _node_ids(table: GpuTable, key: str, rows: Any) -> Any:
    """Nodes' primary keys at `rows` (some), as their own copy (so parts of a split, each with its own values, join
    end to end): CuPy int64 for integers and booleans (upstream's tensor: a bool is an int); a libcudf strings
    column of the texts for text; else `TypedIds`."""

    import cupy
    import pylibcudf as plc

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import _column

    dtype = table.dtypes[key]
    if dtype.integer or dtype.name == "bool":
        ids = table.values(key)[rows]
        if dtype.name == "uint64" and bool((ids.view(cupy.int64) < 0).any()):
            raise ValueError(f"graph nodes of {table.name!r} keyed by uint64 values of 2**63 or more do not fit "
                             f"upstream's int64 `ids` tensor ({table.name}.{key})")
        return ids.astype(cupy.int64)
    if dtype.name.startswith("decimal"):
        picked = plc.copying.gather(plc.Table([table.columns[key]]), _column(cupy.ascontiguousarray(rows.astype(cupy.int32))),
                                    plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
        return TypedIds(dtype, picked)
    if dtype.name not in ("string", "bytes"):
        return TypedIds(dtype, table.values(key)[rows].copy())
    codes = cupy.ascontiguousarray(table.values(key)[rows].astype(cupy.int32))
    texts = plc.copying.gather(plc.Table([table.dictionaries[key]]), _column(codes),
                               plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
    return texts if dtype.name == "string" else TypedIds(dtype, texts)


def _ids_slice(ids: Any, low: int, high: int) -> Any:
    """Nodes `low` to `high` of `ids` (`Neighborhoods.ids`) as upstream gives them: a tensor on the GPU, a tuple of
    strings for text keys; no node at all is an empty int64 tensor whatever the key (upstream's: every one of no
    values is an int)."""

    import cupy
    import pylibcudf as plc
    import torch

    if isinstance(ids, cupy.ndarray):
        return torch.from_dlpack(ids[low:high])
    if high <= low:
        return torch.from_dlpack(cupy.zeros(0, dtype=cupy.int64))
    if isinstance(ids, TypedIds):
        return tuple(ids.python(low, high))
    return tuple(plc.copying.slice(ids, [low, high])[0].to_arrow().to_pylist())


def _neighbor_index(sampler: Sampler, table: str, key: str, clock: str | None) -> NeighborIndex:
    """The plan's index of `table` by (`key`, `clock`), built on first use."""

    cache = sampler.graph_indexes
    if (table, key, clock) not in cache:
        cache[(table, key, clock)] = NeighborIndex(sampler.tables[table], key, clock,
                                                   sampler.specs[table].primary_key)
    return cache[(table, key, clock)]
