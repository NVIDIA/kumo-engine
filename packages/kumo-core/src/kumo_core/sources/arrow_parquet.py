# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .cpu import encode_workers
from .errors import SchemaChangedError
from .parallel import WorkCancelled, run_in_parallel
from .transfer import discard_destination

DEFAULT_MAX_ROWS_PER_SHARD = 250_000
DEFAULT_MAX_UNCOMPRESSED_BYTES_PER_SHARD = 64 * 1024 * 1024
#: Dictionaries shrank a 2.36B-row staged table 3.7x at the same CPU; each
#: column chunk's dictionary stops at 1 MiB, then falls back to plain.
WRITER_OPTIONS = {
    "compression": "zstd",
    "compression_level": 1,
    "use_dictionary": True,
}

ArrowBatch = pa.Table | pa.RecordBatch


@dataclass(frozen=True)
class WriteResult:
    rows: int
    uncompressed_bytes: int
    paths: tuple[str, ...]


def _is_string(dtype: pa.DataType) -> bool:
    """Parquet strings come back as ``string`` or ``large_string`` depending
    on the reader's version; the two are the same column to Kumo."""
    return pa.types.is_string(dtype) or pa.types.is_large_string(dtype)


def _is_supported(dtype: pa.DataType) -> bool:
    if (pa.types.is_boolean(dtype) or pa.types.is_integer(dtype)
            or pa.types.is_floating(dtype) or pa.types.is_null(dtype)
            or _is_string(dtype) or pa.types.is_date(dtype)
            or pa.types.is_timestamp(dtype) or pa.types.is_time(dtype)
            or pa.types.is_duration(dtype) or pa.types.is_binary(dtype)
            or pa.types.is_decimal(dtype)):
        return True
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return (pa.types.is_integer(dtype.value_type)
                or pa.types.is_floating(dtype.value_type)
                or _is_string(dtype.value_type)
                or pa.types.is_boolean(dtype.value_type))
    if pa.types.is_dictionary(dtype):
        return pa.types.is_integer(dtype.index_type) and _is_string(
            dtype.value_type)
    return False


def unsupported_columns(schema: pa.Schema) -> list[tuple[str, pa.DataType]]:
    """Columns Kumo cannot ingest, each with the type that makes it so."""
    return [(name, dtype) for name, dtype in zip(schema.names, schema.types)
            if not _is_supported(dtype)]


UNSUPPORTED_ADVICE = (
    "Change the data type to a boolean, integer, float, string, timestamp, "
    "duration, decimal (less than 18 precision long or float), list of float, "
    "int, string or string categorical types. Nested data types such as "
    "structs and maps are not supported.")


def count_rows(paths: Sequence[str]) -> int:
    """Counted from the footers because asking the warehouse again scans a
    view whole, and charges for it."""
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def materialize_arrow_streams(
    streams: Sequence[Iterable[ArrowBatch]],
    destination: str,
    *,
    worker_ceiling: int | None = None,
    max_rows_per_shard: int = DEFAULT_MAX_ROWS_PER_SHARD,
    max_uncompressed_bytes_per_shard: int = (
        DEFAULT_MAX_UNCOMPRESSED_BYTES_PER_SHARD),
    uniform: bool = True,
) -> WriteResult:
    """``destination`` is emptied by the caller that owns it, not here. One
    shared writer serialises every producer behind a single encode and
    measured 2.1x slower; naming parts for the stream index needs no lock.
    """
    _check_bounds(max_rows_per_shard, max_uncompressed_bytes_per_shard)
    directory = Path(destination)
    if not directory.is_dir():
        raise ValueError(f"{destination} must exist before a read writes it")
    completed = False
    stop = threading.Event()
    try:
        drained: list[_Drained] = []
        if streams:
            drained = run_in_parallel(
                [
                    partial(_drain, stream, directory, f"part-{index:05d}",
                            max_rows_per_shard,
                            max_uncompressed_bytes_per_shard, stop)
                    for index, stream in enumerate(streams)
                ],
                workers=min(encode_workers(worker_ceiling), len(streams)),
                stop=stop,
            )
        result = _combine(drained, directory, uniform)
        completed = True
        return result
    finally:
        if not completed:
            discard_destination(destination)


@dataclass(frozen=True)
class _Drained:
    prefix: str
    rows: int
    uncompressed_bytes: int
    paths: tuple[str, ...]
    schema: pa.Schema | None
    empty: pa.Table | None


def _drain(
    batches: Iterable[ArrowBatch],
    directory: Path,
    prefix: str,
    max_rows: int,
    max_uncompressed_bytes: int,
    stop: threading.Event,
) -> _Drained:
    rows = 0
    uncompressed_bytes = 0
    paths: list[str] = []
    empty: pa.Table | None = None
    schema: pa.Schema | None = None
    pending: list[pa.Table] = []
    pending_rows = 0
    pending_bytes = 0
    for batch in batches:
        if stop.is_set():
            # Pulling this stream to its end would keep the warehouse
            # serving a result nobody will read.
            raise WorkCancelled(f"{prefix} abandoned after a stream failed")
        table = _as_table(batch)
        if schema is None:
            schema = table.schema
        elif not table.schema.equals(schema):
            raise SchemaChangedError(
                f"source changed schema mid-read: {schema} became "
                f"{table.schema}")
        rows += table.num_rows
        uncompressed_bytes += table.nbytes
        if table.num_rows == 0:
            if empty is None:
                empty = table
            continue
        pending.append(table)
        pending_rows += table.num_rows
        pending_bytes += table.nbytes
        if pending_rows < max_rows and pending_bytes < max_uncompressed_bytes:
            continue
        _flush(pending, directory, prefix, paths, max_rows,
               max_uncompressed_bytes)
        pending, pending_rows, pending_bytes = [], 0, 0

    if pending:
        _flush(pending, directory, prefix, paths, max_rows,
               max_uncompressed_bytes)
    return _Drained(prefix, rows, uncompressed_bytes, tuple(paths), schema,
                    empty)


def _combine(drained: Sequence[_Drained], directory: Path,
             uniform: bool) -> WriteResult:
    schemas = [d.schema for d in drained if d.schema is not None]
    for schema in schemas[1:] if uniform else []:
        if not schema.equals(schemas[0]):
            raise SchemaChangedError(
                f"sources disagreed on schema: {schemas[0]} beside {schema}")
    paths = [path for d in drained for path in d.paths]
    if not paths:
        empty = next((d.empty for d in drained if d.empty is not None), None)
        if empty is not None:
            name = f"{drained[0].prefix}-00000"
            paths.append(_write_shard(empty, directory, name))
    return WriteResult(
        rows=sum(d.rows for d in drained),
        uncompressed_bytes=sum(d.uncompressed_bytes for d in drained),
        paths=tuple(paths),
    )


def _flush(
    pending: list[pa.Table],
    directory: Path,
    prefix: str,
    paths: list[str],
    max_rows: int,
    max_uncompressed_bytes: int,
) -> None:
    paths.extend(
        write_parquet_shards(_concat(pending), directory, prefix,
                             start=len(paths), max_rows=max_rows,
                             max_uncompressed_bytes=max_uncompressed_bytes))


def write_parquet_shards(
    table: pa.Table,
    directory: Path,
    prefix: str,
    *,
    start: int = 0,
    max_rows: int = DEFAULT_MAX_ROWS_PER_SHARD,
    max_uncompressed_bytes: int = DEFAULT_MAX_UNCOMPRESSED_BYTES_PER_SHARD,
) -> Iterator[str]:
    _check_bounds(max_rows, max_uncompressed_bytes)
    for number, shard in enumerate(
            bounded_slices(table, max_rows, max_uncompressed_bytes), start):
        yield _write_shard(shard, directory, f"{prefix}-{number:05d}")


def _check_bounds(max_rows: int, max_uncompressed_bytes: int) -> None:
    if max_rows <= 0 or max_uncompressed_bytes <= 0:
        raise ValueError("Arrow export shard limits must be positive")


def _as_table(batch: ArrowBatch) -> pa.Table:
    if isinstance(batch, pa.Table):
        return batch
    if isinstance(batch, pa.RecordBatch):
        return pa.Table.from_batches([batch])
    raise TypeError("an Arrow stream yields Table or RecordBatch only")


def _concat(tables: list[pa.Table]) -> pa.Table:
    return tables[0] if len(tables) == 1 else pa.concat_tables(tables)


def bounded_slices(
    table: pa.Table,
    max_rows: int,
    max_uncompressed_bytes: int,
) -> Iterator[pa.Table]:
    """Variable-width columns can be heavily skewed, so an average row width
    cannot enforce the byte threshold; a single row is indivisible."""
    offset = 0
    while offset < table.num_rows:
        upper = min(max_rows, table.num_rows - offset)
        if table.slice(offset, upper).nbytes <= max_uncompressed_bytes:
            shard_rows = upper
        elif table.slice(offset, 1).nbytes > max_uncompressed_bytes:
            shard_rows = 1
        else:
            lower = 1
            while lower < upper:
                candidate = (lower + upper + 1) // 2
                if (table.slice(offset, candidate).nbytes
                        <= max_uncompressed_bytes):
                    lower = candidate
                else:
                    upper = candidate - 1
            shard_rows = lower
        yield table.slice(offset, shard_rows)
        offset += shard_rows


def write_parquet_file(
    batches: Iterable[ArrowBatch],
    output: str | Path | BinaryIO | pa.NativeFile,
    *,
    schema: pa.Schema | None = None,
) -> int:
    writer = (pq.ParquetWriter(output, schema, **WRITER_OPTIONS)
              if schema is not None else None)
    rows = 0
    try:
        for batch in batches:
            table = _as_table(batch)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema,
                                          **WRITER_OPTIONS)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows


def _write_shard(shard: pa.Table, directory: Path, name: str) -> str:
    path = directory / f"{name}.parquet"
    write_parquet_file((shard, ), path)
    return str(path)


_PARQUET_PREVIEW_BUFFER_SIZE = 1 << 20
#: Row groups a preview is spread over, evenly spaced across the fragments.
SAMPLE_ROW_GROUPS = 10


def _sampled_row_groups(fragments: Sequence[Any]) -> list[tuple[Any, int, int]]:
    """Up to ``SAMPLE_ROW_GROUPS`` row groups with rows, each with its row
    count, evenly spaced over the fragments in order and within each chosen
    fragment, so a table sorted or split into parts on a column is sampled
    across that column rather than from its top. An empty part passes its
    slot to the next part with rows, which gives one more row group for
    each slot it takes in; a single row group is read from its head, as
    before."""
    slots: dict[int, int] = {}
    count = min(len(fragments), SAMPLE_ROW_GROUPS)
    each = -(-SAMPLE_ROW_GROUPS // count) if count else 0
    for index in range(count):
        position = (index * len(fragments)) // count
        while (position < len(fragments)
               and not fragments[position].metadata.num_rows):
            position += 1
        if position < len(fragments):
            slots[position] = slots.get(position, 0) + 1
    chosen: list[tuple[Any, int, int]] = []
    for position, received in slots.items():
        fragment = fragments[position]
        metadata = fragment.metadata
        sizes = [
            metadata.row_group(group).num_rows
            for group in range(metadata.num_row_groups)
        ]
        filled = [(group, rows) for group, rows in enumerate(sizes) if rows]
        take = min(len(filled), each * received)
        chosen.extend((fragment, *filled[(slot * len(filled)) // take])
                      for slot in range(take))
    return chosen


def _quotas(sizes: Sequence[int], total: int) -> list[int]:
    """An equal share of ``total`` rows per row group, a group shorter than
    its share passing the rest on, so the sample is as large as the groups
    allow."""
    quotas = [0] * len(sizes)
    left = total
    order = sorted(range(len(sizes)), key=sizes.__getitem__)
    for rank, index in enumerate(order):
        quotas[index] = min(sizes[index], -(-left // (len(sizes) - rank)))
        left -= quotas[index]
    return quotas


def read_bounded_dataset(
    dataset: pq.ParquetDataset | ds.Dataset,
    columns: Sequence[str] | None,
    num_rows: int,
    *,
    spread: bool = False,
) -> pa.Table:
    """Read a bounded preview without prefetching complete column chunks:
    the first rows in fragment order, or with ``spread`` an equal share of
    the first rows of row groups spaced across the dataset
    (:func:`_sampled_row_groups`), which is what type inference samples."""
    fragments = (dataset.fragments if isinstance(dataset, pq.ParquetDataset)
                 else list(dataset.get_fragments()))
    partition_columns = {
        name
        for fragment in fragments
        for name in ds.get_partition_keys(fragment.partition_expression)
    }
    fields = (dataset.schema if columns is None else
              [dataset.schema.field(column) for column in columns])
    schema = pa.schema([
        field.with_type(field.type.value_type) if
        (field.name in partition_columns and pa.types.is_dictionary(field.type)
         and not (pa.types.is_string(field.type.value_type) or
                  pa.types.is_large_string(field.type.value_type))) else field
        for field in fields
    ], metadata=dataset.schema.metadata)
    requested_columns = schema.names
    remaining = max(num_rows, 0)
    if spread:
        plan = _sampled_row_groups(fragments)
        quotas = _quotas([rows for _, _, rows in plan], remaining)
    else:
        plan = [(fragment, None, None) for fragment in fragments]
        quotas = [remaining] * len(plan)
    tables: list[pa.Table] = []
    merged = schema
    for (fragment, group, _), quota in zip(plan, quotas):
        if not remaining:
            break
        if not quota:
            continue
        physical_columns = [
            column for column in requested_columns
            if column in fragment.physical_schema.names
        ]
        partition_keys = ds.get_partition_keys(fragment.partition_expression)
        rows = min(quota, remaining)
        part: list[pa.RecordBatch] = []
        with fragment.open() as source:
            parquet_file = pq.ParquetFile(
                source,
                metadata=fragment.metadata,
                pre_buffer=False,
                buffer_size=_PARQUET_PREVIEW_BUFFER_SIZE,
            )
            for batch in parquet_file.iter_batches(
                    batch_size=rows,
                    row_groups=None if group is None else [group],
                    columns=physical_columns, use_threads=False):
                batch = batch.slice(0, rows)
                arrays = [
                    batch.column(field.name)
                    if field.name in batch.schema.names else pa.repeat(
                        pa.scalar(partition_keys.get(field.name)),
                        batch.num_rows,
                    ).cast(field.type) for field in schema
                ]
                part.append(
                    pa.RecordBatch.from_arrays(arrays, names=schema.names
                                               ) if arrays else batch)
                rows -= batch.num_rows
                remaining -= batch.num_rows
                if not rows:
                    break
        if not part:
            continue
        table = pa.Table.from_batches(part)
        try:
            merged = pa.unify_schemas(
                [merged, table.schema],
                promote_options="permissive") if tables else table.schema
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            break
        tables.append(table)
    if not tables:
        return schema.empty_table()
    return pa.Table.from_batches([
        batch for table in tables for batch in (
            table if table.schema.equals(merged) else table.cast(merged)
        ).to_batches()
    ], merged.with_metadata(schema.metadata))
