# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a dataset takes on the GPU, and whether it is loaded whole or split: `choose_route`.

The data is counted as the GPU holds it, per table and per row, from the Parquet
footers and the contract's sample, reading no rows (`_loads`): each column read
at its width on the GPU, the indexes the sample builds on the table, and the most
more the table needs at once, while it is loaded or while one of its indexes is
built. A dataset that fits beside the model is loaded whole
(`test.kumo_pytorch.reference_engine.gpu.data.tables.load_tables`); one that does not is split by its root entity
(`test.kumo_pytorch.reference_engine.gpu.split`) into partitions sized by the same counts, and streamed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from test.kumo_pytorch.reference_engine.gpu.sampling.paths import sample_routes
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.contract.dtypes import DType
    from kumo.pytorch.contract.sample import SampleSpec
    from kumo.pytorch.contract.schema import DataSchema
    from kumo.pytorch.dataset import Dataset, Parquet

__all__ = ["DIRECT_FRACTION", "RESERVE_FRACTION", "Route", "choose_route"]

# Without the model's peak memory, the data gets a fixed share of the GPU:

#: The share of one GPU's memory the data may take, leaving the rest for the model
#: and the work done on the data: the whole dataset, to be loaded directly; or else
#: the shared tables and two partitions (the one in use and the next, read ahead).
DIRECT_FRACTION = 0.5

# With the model's peak memory, the data gets what the model leaves:

#: The share of one GPU's memory kept free, beside the model's peak, for the
#: sampler's and batch builder's temporary memory. Without the model's peak, the
#: data's own peak (`_loads`) may take all the rest.
RESERVE_FRACTION = 0.1

# What the data takes on the GPU besides its columns (`_loads`), per row of a table, measured on
# pragma-synthetic-medium's tables (up to 612 million rows) and on 20 million rows with 19-digit keys:

#: Bytes a row keeps in a `KeyIndex` (a relation's or an event's) or the entity's `EntityIndex`: its sorted keys
#: and rows, 8 bytes each (measured: exactly 16).
KEY_INDEX_BYTES = 16

#: Bytes a row keeps in a graph hop's `NeighborIndex`: keys and rows, 8 bytes each, and 8 more for its seconds
#: when the hop reads time (measured: exactly 16 and 24).
NEIGHBOR_INDEX_BYTES = 16
NEIGHBOR_INDEX_CLOCK_BYTES = 8

#: Bytes a row of a relation's source table keeps when the relation's keys are text or bytes: the int32 code of
#: each distinct source value among the destination's, at most one per row (`test.kumo_pytorch.reference_engine.gpu.data.keys.translate`; a bound).
KEY_CODES_BYTES = 4
_CODED = ("string", "bytes")

#: Bytes a row needs, beyond those it keeps, while one index on its table is built (measured: 24.2 for a
#: `KeyIndex`; 20 for a `NeighborIndex`, 28 when it reads time, whatever the primary keys' digits).
KEY_INDEX_BUILD_BYTES = 32
NEIGHBOR_INDEX_BUILD_BYTES = 40

#: A float key's index also works out its keys' bits (`test.kumo_pytorch.reference_engine.gpu.data.keys`): what a row needs while it is built, in
#: place of `KEY_INDEX_BUILD_BYTES` (measured on 2 million float64 and float32 keys: 33.2 beyond the 16 kept).
FLOAT_KEY_BUILD_BYTES = 40

#: A decimal key column keeps each row's place among its distinct values (8) and those values (32 bytes each, at
#: most one a row) on its table (`test.kumo_pytorch.reference_engine.gpu.data.keys.KeySpace`), and needs more while they are worked out, in place of
#: `KEY_INDEX_BUILD_BYTES` (measured on 2 million decimal128 and decimal256 keys: 40 kept, 73 more at the peak).
DECIMAL_KEY_BYTES = 40
DECIMAL_KEY_BUILD_BYTES = 80

#: A decimal key that keys of another number type meet (an integer, a float, a decimal of another scale) also
#: keeps its distinct values as a search table (`test.kumo_pytorch.reference_engine.gpu.data.keys.KeySpace.searched`), 32 bytes each (at most one a
#: row); a decimal key that meets another number type keeps, besides its places there (8), whether each of its
#: distinct values has an equal there (1).
SEARCHED_DECIMAL_BYTES = 32

#: A `NeighborIndex` on a table keyed by text also prints and sorts its keys (`test.kumo_pytorch.reference_engine.gpu.data.printed`): bytes a row
#: needs beyond `NEIGHBOR_INDEX_BUILD_BYTES`, `TEXT_KEY_BUILD_BYTES` and `TEXT_KEY_BUILD_FACTOR` per byte a row
#: of the key column takes stored (measured on 2 million keys of 4 to 128 bytes: 25 to 243 beyond the base for
#: ASCII keys, 1.4 to 2.3 a stored byte; 82 to 546 when most are not ASCII, printed by Python and copied in, up
#: to 5.05 a stored byte, the most for the shortest).
TEXT_KEY_BUILD_BYTES = 20
TEXT_KEY_BUILD_FACTOR = 4.5

#: A `NeighborIndex` on a table keyed by dates or timestamps sorts by their calendar parts as printed digits
#: (`test.kumo_pytorch.reference_engine.gpu.data.printed.key_columns`, on the GPU): bytes a row needs beyond those it keeps, in place of
#: `NEIGHBOR_INDEX_BUILD_BYTES` (measured on 2 million keys: 120 for dates, 116 for timestamps).
PRINTED_DATE_BUILD_BYTES = 128

#: The same for a table keyed by decimals, printed by Python on the host, each row's key gathered, then ranked
#: (measured on 2 million keys: 48 for decimal128 and decimal32; 100 for a decimal256 over precision 38, held as
#: 32 bytes a value). Floats and bytes, ranked so too, fit within `NEIGHBOR_INDEX_BUILD_BYTES` (28 measured).
PRINTED_DECIMAL_BUILD_BYTES = 56
PRINTED_WIDE_DECIMAL_BUILD_BYTES = 104

#: A table's peak while it is loaded (`test.kumo_pytorch.reference_engine.gpu.data.tables.load_tables` reads it whole, texts as text, then codes
#: them) over what it holds once loaded (measured: 2.55 to 2.83).
LOAD_PEAK_FACTOR = 3.0


@dataclass(frozen=True)
class Route:
    """How a dataset gets onto the GPU: loaded directly, or split and streamed.

    :param route: ``"direct"`` or ``"partitioned"``.
    :param dataset_bytes: what the data holds on the GPU: the columns read, and the
        indexes the sample builds on them (`_loads`), from the Parquet footers.
    :param gpu_bytes: the GPU's memory.
    :param num_partitions: 1 when direct; otherwise K.
    :param model_bytes: the model's peak memory the route was chosen with, or
        None when it was not given.
    :param partition_bytes: the room each partition may hold when split; None when direct.
    :param peak_bytes: the most the whole dataset takes at once (`_peak`): while a
        table is loaded, or while an index is built once all are.
    :param partition_peak_bytes: when split, the room for two partitions' held bytes
        and one partition's peak beyond them (the shared tables left out); None when direct.
    """

    route: str
    dataset_bytes: int
    gpu_bytes: int
    num_partitions: int
    model_bytes: int | None = None
    partition_bytes: int | None = None
    peak_bytes: int | None = None
    partition_peak_bytes: int | None = None


def choose_route(schema: DataSchema, dataset: Dataset, root: str, *, sample: SampleSpec | None = None,
                 model_bytes: int | None = None) -> Route:
    """Load the dataset directly when it fits on the GPU; otherwise split it by `root`.

    Reads the files' footers only, and asks the GPU its memory size. The data is
    counted as the GPU holds it (`_loads`): each column read, at its width on the
    GPU, and the indexes `sample` builds on each table (none without `sample`);
    and at its peak (`_peak`), while a table is loaded or an index built.

    :param sample: the contract's sample, whose routes and neighborhoods say which
        indexes are built.
    :param model_bytes: the model's peak GPU memory while training (weights,
        gradients, optimizer state and one batch's activations), as measured on
        a real training step. The data, peak included, then gets what the model
        and a reserve of `RESERVE_FRACTION` of the GPU leave: the whole dataset if
        it fits there; otherwise two partitions (the one in use and the next)
        beside the shared tables, which stay on the GPU throughout. A model that
        leaves no room for that is refused. Without it, the data may hold
        `DIRECT_FRACTION` of the GPU, and peak at all but the reserve, in the
        same way: the whole dataset, or the shared tables and two partitions;
        shared tables that leave no room are refused.
    """

    from test.kumo_pytorch.reference_engine.gpu.data.sources import (
        resolve_sources,
    )
    from test.kumo_pytorch.reference_engine.gpu.split.partition_plan import (
        plan_partitions,
    )

    if model_bytes is not None and (type(model_bytes) is not int or model_bytes < 0):
        raise ValueError(f"model_bytes must be a non-negative number of bytes, not {model_bytes!r}")
    plan = plan_partitions(schema, root)
    sources = resolve_sources(schema, dataset)
    gpu = _gpu_memory()
    loads = _loads(schema, sample, sources)
    held, highest = _held(loads, list(sources)), _peak(loads, list(sources))
    reserve = gpu * RESERVE_FRACTION
    room, peak = (gpu * DIRECT_FRACTION, gpu - reserve) if model_bytes is None else (gpu - model_bytes - reserve,) * 2
    if held <= room and highest <= peak:
        return Route("direct", int(held), gpu, 1, model_bytes, peak_bytes=int(highest))
    shared_names = [name for name in sources if name not in plan.owned]
    shared, shared_peak = _held(loads, shared_names), _peak(loads, shared_names)
    per_partition, pair = (room - shared) / 2, peak - shared  # peak >= room, so pair > 0 when per_partition is
    if per_partition <= 0 or shared_peak > peak:
        given = (f"without the model's peak it may hold {DIRECT_FRACTION:.0%} of the GPU "
                 f"({room / 1e9:.1f} GB) and peak at {peak / 1e9:.1f} GB" if model_bytes is None else
                 f"the model's peak ({model_bytes / 1e9:.1f} GB) and the reserve for batch building "
                 f"({reserve / 1e9:.1f} GB) leave it {room / 1e9:.1f} GB")
        raise ValueError(
            f"the data has no room on this {gpu / 1e9:.1f} GB GPU: {given}, and the shared tables "
            f"({shared / 1e9:.1f} GB with their indexes, {shared_peak / 1e9:.1f} GB at their peak, "
            f"loaded whole on every GPU) leave none of it for partitions: relate the biggest of them to {root!r}, "
            "so that they are split" + (", or give the model's peak" if model_bytes is None else "")
        )
    owned = held - shared
    k = max(2, math.ceil(owned / per_partition), math.ceil((2 * owned + _extra(loads, list(plan.owned))) / pair))
    return Route("partitioned", int(held), gpu, k, model_bytes, int(per_partition), int(highest), int(pair))


def _gpu_memory() -> int:
    import cupy

    return int(cupy.cuda.runtime.memGetInfo()[1])


@dataclass(frozen=True)
class _Load:
    """What a table takes on the GPU (`_loads`), per row: its columns once loaded; those and the indexes built on
    it (`held`); and the most more a row needs at once while the table is loaded (`load`) or while one of its
    indexes is built (`build`)."""

    rows: int
    columns: float
    held: float
    load: float
    build: float

    @property
    def extra(self) -> float:
        return max(self.load, self.build)


def _printed_build(dtype: DType | None) -> float:
    """What a row needs, beyond what it keeps, while a `NeighborIndex` orders its table by a primary key of
    `dtype` printed (dates and timestamps on the GPU, decimals on the host); 0 for the rest (text: `_loads`)."""

    name = getattr(dtype, "name", "")
    if name in ("date32", "timestamp"):
        return PRINTED_DATE_BUILD_BYTES
    if name == "decimal256" and (dtype.precision or 0) > 38:
        return PRINTED_WIDE_DECIMAL_BUILD_BYTES
    return PRINTED_DECIMAL_BUILD_BYTES if name.startswith("decimal") else 0.0


def _loads(schema: DataSchema, sample: SampleSpec | None,
           sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]]) -> dict[str, _Load]:
    """Each table's `_Load`, from its files' footers (`_column_rows`) and the indexes `sample` builds on it
    (`_index_rows`; none without `sample`)."""

    types = {name: columns for name, (_, columns) in sources.items()}
    indexes = _index_rows(schema, sample, types)
    specs = {table.name: table for table in schema.tables}
    ranked = {table for table, _, _ in _neighbor_indexes(schema, sample)
              if getattr(types.get(table, {}).get(specs[table].primary_key), "name", None) == "string"}
    stored: dict[tuple[str, str], float] = {}
    loads = {}
    for name, (rows, columns) in _column_rows(list(sources), sources, stored).items():
        kept, build = indexes.get(name, (0.0, 0.0))
        if name in ranked:  # its text keys printed and sorted while its neighbor index is built
            build = max(build, NEIGHBOR_INDEX_BUILD_BYTES + TEXT_KEY_BUILD_BYTES
                        + TEXT_KEY_BUILD_FACTOR * stored.get((name, specs[name].primary_key), 0.0))
        loads[name] = _Load(rows, columns, columns + kept, (LOAD_PEAK_FACTOR - 1) * columns, build)
    return loads


def _held(loads: Mapping[str, _Load], tables: Sequence[str]) -> float:
    """What `tables` hold on the GPU once loaded, indexes included."""

    return sum(loads[name].rows * loads[name].held for name in tables)


def _peak(loads: Mapping[str, _Load], tables: Sequence[str]) -> float:
    """The most `tables` take at once, loaded together: while the last is loaded, beside the others' columns
    (the biggest load counted, whichever comes last); or, once all are loaded, while one index is built beside
    every table's columns and indexes."""

    columns = sum(loads[name].rows * loads[name].columns for name in tables)
    loading = max((loads[name].rows * loads[name].load for name in tables), default=0.0)
    building = max((loads[name].rows * loads[name].build for name in tables), default=0.0)
    return max(columns + loading, _held(loads, tables) + building)


def _extra(loads: Mapping[str, _Load], tables: Sequence[str]) -> float:
    """The most more one of `tables` needs at once beyond what they all hold, indexes included: one table is
    loaded, or one index built, at a time (for partitions, the next one read ahead while one is in use)."""

    return max((loads[name].rows * loads[name].extra for name in tables), default=0.0)


def _partition_bytes(loads: Mapping[str, _Load], rows: Mapping[str, int]) -> dict[str, int]:
    """A partition's ``held_bytes`` and ``extra_bytes`` (as `_held` and `_extra`), from its rows per table."""

    return {"held_bytes": int(sum(count * loads[name].held for name, count in rows.items())),
            "extra_bytes": int(max((count * loads[name].extra for name, count in rows.items()), default=0))}


def _index_rows(schema: DataSchema, sample: SampleSpec | None,
                types: Mapping[str, Mapping[str, DType]] | None = None) -> dict[str, tuple[float, float]]:
    """Per table, the bytes a row keeps in the indexes `sample` builds on it, and the most more a row needs while
    one of them is built: a `KeyIndex` on each column a relation route lands on (`test.kumo_pytorch.reference_engine.gpu.sampling.paths`, neighborhoods
    left out) and the entity's `EntityIndex` on its primary key, with the codes of a text key a route leaves
    from; a `NeighborIndex` on each column a graph hop lands on, by the clock it reads, if any. Column types are
    `types`' (the schema's declared ones without it)."""

    from dataclasses import replace

    if sample is None:
        return {}
    specs = {table.name: table for table in schema.tables}
    relations = {relation.name: relation for relation in schema.relations}
    types = types if types is not None else {table.name: table.columns or {} for table in schema.tables}
    hops = {hop for route in sample_routes(schema, replace(sample, neighborhoods={})) for hop in route}
    keyed = {(sample.entity.table, specs[sample.entity.table].primary_key)} | {(hop.table, hop.column) for hop in hops}
    coded = {(hop.from_table, hop.from_column, hop.table, hop.column) for hop in hops
             if getattr(types.get(hop.from_table, {}).get(hop.from_column), "name", None) in _CODED}
    neighbors = _neighbor_indexes(schema, sample)
    leaving: set[tuple[str, str, str, str]] = set()  # graph hops leaving from a decimal key
    pairs = {(hop.from_table, hop.from_column, hop.table, hop.column) for hop in hops}  # every pair of keys meeting
    for neighborhood in sample.neighborhoods.values():  # a graph hop leaving from a text or bytes key keeps codes
        for layer in neighborhood.layers:
            for traversal in layer.samples:
                relation = relations[traversal.relation]
                ends = [(relation.left_table, relation.left_key), (relation.right_table, relation.right_key)]
                (start, column), (table, key) = ends if traversal.direction == "forward" else ends[::-1]
                pairs.add((start, column, table, key))
                if getattr(types.get(start, {}).get(column), "name", None) in _CODED:
                    coded.add((start, column, table, key))
                elif getattr(types.get(start, {}).get(column), "name", "").startswith("decimal"):
                    leaving.add((start, column, table, key))  # its places on the other side, and its own
    kept: dict[str, float] = defaultdict(float)
    build: dict[str, float] = defaultdict(float)

    def kind(table: str, column: str) -> str:
        return getattr(types.get(table, {}).get(column), "name", "")

    for table, column in keyed:
        kept[table] += KEY_INDEX_BYTES
        build[table] = max(build[table], FLOAT_KEY_BUILD_BYTES if kind(table, column).startswith("float")
                           else KEY_INDEX_BUILD_BYTES)
    for table, *_ in coded:
        kept[table] += KEY_CODES_BYTES
    # decimal keys: each column's places and distinct values, once, and a relation's places on its other side
    decimals = {(table, column) for table, column in keyed if kind(table, column).startswith("decimal")}
    for hop in hops:
        if kind(hop.from_table, hop.from_column).startswith("decimal"):
            decimals.add((hop.from_table, hop.from_column))
            kept[hop.from_table] += 8
    for start, column, *_ in leaving:
        decimals.add((start, column))
        kept[start] += 8

    for table, column, clock in neighbors:
        kept[table] += NEIGHBOR_INDEX_BYTES + (NEIGHBOR_INDEX_CLOCK_BYTES if clock is not None else 0)
        floats = 8 if kind(table, column).startswith("float") else 0  # its keys' bits, while built
        printed = _printed_build(types.get(table, {}).get(specs[table].primary_key))  # text keys: `_loads`
        build[table] = max(build[table], NEIGHBOR_INDEX_BUILD_BYTES + floats, printed)
        if kind(table, column).startswith("decimal"):
            decimals.add((table, column))
    for table, _ in decimals:
        kept[table] += DECIMAL_KEY_BYTES
        build[table] = max(build[table], DECIMAL_KEY_BUILD_BYTES)
    from test.kumo_pytorch.reference_engine.gpu.data.keys import (
        comparable,
        never_equal,
    )

    searched = set()  # keys of different number types meeting (`test.kumo_pytorch.reference_engine.gpu.data.keys.Translation`)
    for start, column, table, key in pairs:
        source, destination = types.get(start, {}).get(column), types.get(table, {}).get(key)
        if source is None or destination is None or never_equal(source, destination) or comparable(source, destination):
            continue
        if source.name.startswith("decimal"):
            kept[start] += 1
        if destination.name.startswith("decimal"):
            searched.add((table, key))
    for table, _ in searched:
        kept[table] += SEARCHED_DECIMAL_BYTES
    return {table: (kept[table], build[table]) for table in kept}


def _neighbor_indexes(schema: DataSchema, sample: SampleSpec | None) -> set[tuple[str, str, str | None]]:
    """The `NeighborIndex` each graph hop of `sample` lands on: (table, key, the clock it reads, if any)."""

    from kumo.pytorch.contract.schema import table_clock

    if sample is None:
        return set()
    specs = {table.name: table for table in schema.tables}
    relations = {relation.name: relation for relation in schema.relations}
    neighbors = set()
    for neighborhood in sample.neighborhoods.values():
        for layer in neighborhood.layers:
            for traversal in layer.samples:
                relation = relations[traversal.relation]
                table, key = ((relation.right_table, relation.right_key) if traversal.direction == "forward" else
                              (relation.left_table, relation.left_key))
                timed = traversal.temporal == "inherit" or traversal.strategy == "recent"
                clock = traversal.time_column or table_clock(specs[table]) if timed else None
                neighbors.add((table, key, clock))
    return neighbors


def _column_rows(tables: list[str], sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]],
                 stored_text: dict[tuple[str, str], float] | None = None) -> dict[str, tuple[int, float]]:
    """Each table's rows, and the bytes a row of the columns read takes on the GPU once loaded, from its files'
    footers: a column's width as the GPU holds it (`_gpu_width`); a text column's int32 codes, and at most its
    stored size for its dictionary of distinct texts; a decimal's float64 numbers too (`GpuTable.numbers`); and,
    where the statistics show NULLs or show nothing, a null mask and the bools `GpuTable.present` keeps. A
    column of any other type counts its stored size. With `stored_text`, each text column's stored bytes a row
    are put there, by (table, column)."""

    import pyarrow.parquet as pq

    from kumo.pytorch.dataset import _local_parquet_files

    found = {}
    for name in tables:
        source, columns = sources[name]
        renamed = {logical: physical for physical, logical in source.columns.items()}
        wanted = {renamed.get(column, column) for column in columns}
        logical = {renamed.get(column, column): column for column in columns}
        rows, total, texts = 0, 0.0, defaultdict(float)
        for path in _local_parquet_files(source.path):
            file = pq.ParquetFile(path)
            metadata, arrow = file.metadata, file.schema_arrow
            rows += metadata.num_rows
            stored: dict[str, int] = defaultdict(int)
            nullable: set[str] = set()
            for group in range(metadata.num_row_groups):
                row_group = metadata.row_group(group)
                for i in range(row_group.num_columns):
                    chunk = row_group.column(i)
                    if chunk.path_in_schema not in wanted:
                        continue
                    stored[chunk.path_in_schema] += chunk.total_uncompressed_size
                    statistics = chunk.statistics
                    if statistics is None or not statistics.has_null_count or statistics.null_count:
                        nullable.add(chunk.path_in_schema)
            for column in wanted:
                arrow_type = arrow.field(column).type
                width = _gpu_width(arrow_type)
                if width is None:
                    total += stored[column]
                    continue
                total += metadata.num_rows * (width + (1 + 1 / 8 if column in nullable else 0))
                if _is_text(arrow_type):
                    total += stored[column]
                    texts[logical[column]] += stored[column]
        found[name] = (rows, total / max(1, rows))
        if stored_text is not None:
            stored_text.update({(name, column): size / max(1, rows) for column, size in texts.items()})
    return found


def _is_text(arrow_type: Any) -> bool:
    """Whether a Parquet column of `arrow_type` is held as text is: strings, and bytes (`test.kumo_pytorch.reference_engine.gpu.data.tables`)."""

    import pyarrow as pa

    if pa.types.is_dictionary(arrow_type):
        arrow_type = arrow_type.value_type
    kind = pa.types
    return (kind.is_string(arrow_type) or kind.is_large_string(arrow_type) or kind.is_binary(arrow_type)
            or kind.is_large_binary(arrow_type))


def _gpu_width(arrow_type: Any) -> int | None:
    """Bytes a value of a Parquet column, read as `arrow_type`, takes on the GPU (`test.kumo_pytorch.reference_engine.gpu.data.tables`): timestamps
    as int64 seconds, dates as int32 days, booleans a byte each, text as int32 codes (`_with_codes`), decimals as
    up to 16 bytes plus their float64 numbers (a decimal256 over precision 38: 32 bytes and an offset of up to 8,
    `test.kumo_pytorch.reference_engine.gpu.data.decimal_storage._wide_decimal`). None for any other type."""

    import pyarrow as pa

    kind = pa.types
    if kind.is_dictionary(arrow_type):
        arrow_type = arrow_type.value_type
    if _is_text(arrow_type):
        return 4
    if kind.is_boolean(arrow_type):
        return 1
    if kind.is_integer(arrow_type) or kind.is_floating(arrow_type):
        return max(arrow_type.bit_width // 8, 4 if kind.is_floating(arrow_type) else 1)
    if kind.is_timestamp(arrow_type) or kind.is_date64(arrow_type) or kind.is_duration(arrow_type) \
            or kind.is_time64(arrow_type):
        return 8
    if kind.is_date32(arrow_type) or kind.is_time32(arrow_type):
        return 4
    if kind.is_decimal(arrow_type):
        from test.kumo_pytorch.reference_engine.gpu.data.decimal_storage import (
            WIDE_DECIMAL_BYTES,
        )

        if arrow_type.precision > 38:
            return WIDE_DECIMAL_BYTES + 8 + 8
        return 16 + 8
    return None


def _decoded_bytes(tables: list[str], sources: Mapping[str, tuple[Parquet, Mapping[str, DType]]]) -> int:
    """What the columns `tables` read take on the GPU once loaded (`_column_rows`), indexes left out."""

    return int(sum(rows * per_row for rows, per_row in _column_rows(tables, sources).values()))
