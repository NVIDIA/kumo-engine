# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native projected Polars tables for the arena engine's root-row foundation."""

from __future__ import annotations

import copy
import warnings
from collections import OrderedDict
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from kumo.pytorch.contract.dtypes import DType

from ..parquet import local_parquet_files


def _schema(files, table):
    """Validate every shard before a native reader can coerce mismatched types."""
    expected, count = None, 0
    for path in files:
        with pq.ParquetFile(path) as file:
            physical = file.schema_arrow
            missing = set(table.columns) - set(physical.names)
            if missing:
                raise ValueError(
                    f"Parquet table {table.name!r} is missing columns {sorted(missing)}"
                )
            fields = []
            for name in table.columns:
                if physical.names.count(name) != 1:
                    raise ValueError(
                        f"Parquet table {table.name!r} repeats column {name!r}"
                    )
                field = physical.field(name)
                kind = field.type
                declared = table.columns[name]
                if declared is not None and DType.from_arrow(kind) != declared:
                    raise ValueError(
                        f"Declared dtype differs from Parquet type for {table.name}.{name}"
                    )
                if pa.types.is_decimal256(kind):
                    raise TypeError(
                        f"Polars engine does not support Decimal256 column {table.name}.{name}; "
                        "use a backend with exact Decimal256 support"
                    )
                if not (
                    pa.types.is_integer(kind)
                    or kind
                    in (
                        pa.float32(),
                        pa.float64(),
                        pa.bool_(),
                        pa.string(),
                        pa.binary(),
                        pa.date32(),
                    )
                    or pa.types.is_timestamp(kind)
                    or pa.types.is_decimal128(kind)
                ):
                    raise TypeError(
                        f"Unsupported Parquet type for {table.name}.{name}: {kind}"
                    )
                fields.append(field)
            selected = pa.schema(fields)
            if expected is not None and not selected.equals(
                expected, check_metadata=False
            ):
                raise ValueError(
                    f"Parquet schema differs between shards for table {table.name!r}"
                )
            expected = selected
            count += file.metadata.num_rows
    return expected, count


def _keys(values, kind):
    """Preserve numeric identity before and after Arrow's inference and casts."""

    def numeric(value):
        return (
            pa.types.is_integer(value)
            or pa.types.is_floating(value)
            or pa.types.is_decimal(value)
        )

    def require_exact(keys):
        if not numeric(keys.type):
            return
        if pa.types.is_floating(keys.type) and not pc.all(pc.is_finite(keys)).as_py():
            raise ValueError("non-finite requested primary key")
        for original, converted in zip(values, keys.to_pylist(), strict=True):
            # NumPy scalars otherwise promote the comparison and can hide integer rounding.
            if isinstance(original, Integral):
                original = int(original)
            elif isinstance(original, Real):
                original = float(original)
            if original != converted:
                raise ValueError("numeric primary key conversion would lose precision")

    try:
        try:
            keys = pa.array(values)
        except OverflowError:
            # Python uint64 values can exceed inferred int64. Never use this fallback
            # for floating types: constructing them directly can round before safe cast.
            if not pa.types.is_unsigned_integer(kind) or not all(
                isinstance(value, Integral) and not isinstance(value, bool)
                for value in values
            ):
                raise
            keys = pa.array(values, type=kind, safe=True)
        if keys.null_count or not (
            keys.type == kind or numeric(keys.type) and numeric(kind)
        ):
            raise ValueError("null or incompatible key type")
        require_exact(keys)
        result = pc.cast(keys, kind, safe=True)
        require_exact(result)
        return result
    except (pa.ArrowException, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"Requested primary keys must match {kind} without loss"
        ) from error


class PolarsStore:
    """A source owns these tables; returned batches retain their own gathered buffers."""

    def __init__(
        self, dataset, schema, *, num_threads=None, artifact_dir=None, reuse=None
    ):
        from kumo.pytorch.contract.schema import table_clock

        if num_threads is not None and (
            type(num_threads) is not int or num_threads < 1
        ):
            raise ValueError("num_threads must be a positive integer or None")
        actual_threads = pl.thread_pool_size()
        if num_threads is not None and num_threads != actual_threads:
            raise ValueError(
                f"Polars uses a process-wide pool of {actual_threads} threads; "
                f"set POLARS_MAX_THREADS={num_threads} before importing Polars"
            )
        self._closed = False
        self._relation_indexes = OrderedDict()
        self._relation_index_limit = max(8, 2 * len(schema.relations))
        self._mapped_files = []
        self._tables, self._schemas, self._primary_keys, self._positions = (
            {},
            {},
            {},
            {},
        )
        self._description = {
            "backend": "polars",
            "num_threads": actual_threads,
            "tables": {},
        }
        try:
            if reuse is not None:
                from ..artifacts import decode_schemas

                self._schemas = decode_schemas(reuse)
                self._description["tables"] = copy.deepcopy(
                    reuse["description"]["tables"]
                )
                for index, table in enumerate(schema.tables):
                    mapped = pa.memory_map(
                        str(Path(artifact_dir) / f"table-{index}.arrow"), "r"
                    )
                    self._mapped_files.append(mapped)
                    frame = pl.from_arrow(
                        pa.ipc.open_file(mapped).read_all(), rechunk=False
                    )
                    self._tables[table.name] = frame
                    self._primary_keys[table.name] = table.primary_key
                    self._positions[table.name] = {
                        key: i
                        for i, key in enumerate(frame[table.primary_key].to_list())
                    }
                return
            for table_index, table in enumerate(schema.tables):
                source = dataset.tables.get(table.name)
                if getattr(source, "kind", None) != "parquet":
                    raise TypeError(
                        f"Polars table store requires local Parquet for {table.name!r}"
                    )
                if source.columns:
                    raise ValueError(
                        f"Polars table store does not support physical column renames: {table.name!r}"
                    )
                files = local_parquet_files(source.path)
                arrow_schema, source_rows = _schema(files, table)
                frame = (
                    pl.scan_parquet(list(files), hive_partitioning=False)
                    .select(tuple(table.columns))
                    .collect()
                )
                clock = table_clock(table)
                if clock is not None:
                    frame = frame.filter(pl.col(clock).is_not_null())
                keys = frame[table.primary_key]
                if (
                    pa.types.is_floating(arrow_schema.field(table.primary_key).type)
                    and not keys.is_finite().all()
                ):
                    raise ValueError(
                        f"Primary key {table.name}.{table.primary_key} must contain only finite values"
                    )
                if keys.null_count() or keys.n_unique() != len(frame):
                    raise ValueError(
                        f"Primary key {table.name}.{table.primary_key} must be unique and non-null"
                    )
                if artifact_dir is not None:
                    frame.write_ipc(
                        Path(artifact_dir) / f"table-{table_index}.arrow",
                        compression="uncompressed",
                    )
                self._tables[table.name] = frame
                self._schemas[table.name] = arrow_schema
                self._primary_keys[table.name] = table.primary_key
                self._positions[table.name] = {
                    key: index for index, key in enumerate(keys.to_list())
                }
                self._description["tables"][table.name] = {
                    "source_rows": source_rows,
                    "stored_rows": len(frame),
                    "dropped_null_clock_rows": source_rows - len(frame),
                    "primary_key": table.primary_key,
                    "clock": clock,
                    "columns": list(table.columns),
                }
            dropped = dict(self.dropped_rows_counts)
            if any(dropped.values()):
                warnings.warn(
                    f"Polars excluded source rows with NULL table clocks: {dropped}",
                    UserWarning,
                    stacklevel=2,
                )
        except BaseException:
            self.close()
            raise

    @property
    def dropped_rows_counts(self):
        return MappingProxyType(
            {
                name: row["dropped_null_clock_rows"]
                for name, row in self._description["tables"].items()
            }
        )

    def describe(self):
        return copy.deepcopy(self._description)

    def fetch(self, table_name, entity_ids, columns):
        if self._closed:
            raise RuntimeError("Polars table store is closed")
        if table_name not in self._tables:
            raise ValueError(f"Unknown table {table_name!r}")
        if isinstance(columns, (str, bytes)) or isinstance(entity_ids, (str, bytes)):
            raise TypeError("columns and entity_ids must be sequences, not strings")
        columns, entity_ids = tuple(columns), tuple(entity_ids)
        schema = self._schemas[table_name]
        if len(set(columns)) != len(columns) or set(columns) - set(schema.names):
            raise ValueError(
                f"Requested columns must be distinct declared columns of {table_name!r}"
            )
        output_schema = pa.schema([schema.field(name) for name in columns])
        if not entity_ids:
            return pa.Table.from_arrays(
                [pa.array([], type=field.type) for field in output_schema],
                schema=output_schema,
            )
        primary_key = self._primary_keys[table_name]
        keys = _keys(entity_ids, schema.field(primary_key).type).to_pylist()
        lookup = self._positions[table_name]
        missing = [key for key in keys if key not in lookup]
        if missing:
            raise ValueError(
                f"Unknown primary keys in table {table_name!r}: {missing[:8]!r}"
            )
        positions = [lookup[key] for key in keys]
        # Project before gathering; the source columns are never rebuilt as row dictionaries.
        gathered = (
            self._tables[table_name]
            .select(columns or (primary_key,))[positions]
            .to_arrow()
        )
        selected = gathered.select(columns)
        return selected.cast(output_schema, safe=True) if columns else selected

    def relation_index(self, table_name, key, clock):
        """Bounded source-owned sorted row positions for repeated relation lookups.

        Payload columns stay in the original frame. Indexes contain only row
        positions, normalized clocks and one range per distinct relation key.
        They are lazy because a schema may declare unused relation directions.
        """
        if self._closed:
            raise RuntimeError("Polars table store is closed")
        identity = (table_name, key, clock)
        if identity in self._relation_indexes:
            self._relation_indexes.move_to_end(identity)
            return self._relation_indexes[identity]
        from .temporal import _epoch

        frame = self._tables[table_name]
        # Non-finite foreign keys have backend-specific join equivalence rules;
        # retain the relational fallback for those uncommon physical columns.
        if (
            frame.schema[key] in (pl.Float32, pl.Float64)
            and not frame[key].drop_nulls().is_finite().all()
        ):
            return None
        indexed = (
            frame.select(
                pl.col(key).alias("key"),
                pl.col(self._primary_keys[table_name]).alias("pk"),
                (
                    _epoch(clock, frame.schema[clock])
                    if clock
                    else pl.lit(0, dtype=pl.Int64)
                ).alias("time"),
                pl.Series("row", np.arange(len(frame), dtype=np.int64)),
            )
            .filter(pl.col("key").is_not_null() & pl.col("time").is_not_null())
            .sort("key", "time", "pk")
        )
        groups = (
            indexed.with_row_index("offset")
            .group_by("key")
            .agg(pl.col("offset").min().alias("start"), pl.len().alias("length"))
        )
        index = {
            "positions": indexed["row"].to_numpy(),
            "times": indexed["time"].to_numpy(),
            "groups": {
                key: (start, start + length)
                for key, start, length in groups.iter_rows()
            },
        }
        self._relation_indexes[identity] = index
        # Bound retained auxiliary indexes independently of the number of query
        # shapes. Eviction changes preparation work, never supported semantics.
        while len(self._relation_indexes) > self._relation_index_limit:
            self._relation_indexes.popitem(last=False)
        return index

    def close(self):
        if not getattr(self, "_closed", True):
            self._relation_indexes.clear()
            self._tables.clear()
            self._schemas.clear()
            self._primary_keys.clear()
            self._positions.clear()
            for mapped in self._mapped_files:
                mapped.close()
            self._mapped_files.clear()
            self._closed = True
