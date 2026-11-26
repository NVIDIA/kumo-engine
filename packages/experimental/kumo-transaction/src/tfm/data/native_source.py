# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded source-table normalization, without histories or model minibatches.

Native graph tensors expose integer row identities. A disk-backed key registry
binds those identities to source primary/foreign keys once at ingestion. Clocks
remain separate from model features. Files contain one row per source row,
independently of the number of prediction anchors.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as arrow_compute
import pyarrow.parquet as pq

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.schema import ensure_utc
from tfm.contracts.table_source import TableSource
from tfm.data.validation.schema import PROHIBITED_INPUT_KEYS

ROW_ID = "__tfm_row_id"
AVAILABLE_US = "__tfm_available_us"
EVENT_US = "__tfm_event_us"

# Arrow populates compute functions dynamically.
pc: Any = arrow_compute


def _key(values: tuple[Any, ...], path: str) -> str:
    if not values or any(value is None for value in values):
        raise SchemaError(f"{path}: primary/foreign key contains a null component")
    encoded = []
    for value in values:
        if isinstance(value, datetime):
            value = {"__tfm_datetime__": ensure_utc(value).isoformat()}
        elif isinstance(value, date):
            value = {"__tfm_date__": value.isoformat()}
        elif not isinstance(value, (str, bool, int, float)):
            raise UnsupportedCombinationError(f"{path}: unsupported source key scalar")
        encoded.append(value)
    try:
        return json.dumps(encoded, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except ValueError as error:
        raise UnsupportedCombinationError(f"{path}: source keys require finite scalars") from error


def source_key_values(encoded: str) -> tuple[Any, ...]:
    """Decode persisted identity metadata, preserving date and timestamp types."""
    values = []
    for value in json.loads(encoded):
        if isinstance(value, dict):
            if set(value) == {"__tfm_date__"}:
                value = date.fromisoformat(value["__tfm_date__"])
            elif set(value) == {"__tfm_datetime__"}:
                value = ensure_utc(datetime.fromisoformat(value["__tfm_datetime__"]))
            else:
                raise SchemaError("Invalid persisted source identity scalar")
        values.append(value)
    return tuple(values)


def file_sha256(path: Path) -> str:
    """Fingerprint source/anchor files in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def source_entity_id(values: tuple[Any, ...]) -> str:
    """Display raw root keys consistently in native anchor and target metadata."""
    return str(values[0]) if len(values) == 1 else _key(values, "root.key")


def _clocks(batch: pa.RecordBatch, column: str, path: str) -> pa.Array:
    values = batch.column(column)
    if pa.types.is_date(values.type):
        values = pc.cast(values, pa.timestamp("us", tz="UTC"))
    if values.null_count or not pa.types.is_timestamp(values.type):
        raise SchemaError(f"{path}: clocks must be non-null UTC timestamps")
    if values.type.tz not in {"UTC", "+00:00"}:
        raise SchemaError(f"{path}: clock timezone must be UTC")
    micros = pc.cast(pc.cast(values, pa.timestamp("us", tz="UTC")), pa.int64())
    # The pinned native temporal API uses integer-second windows. Reject rather
    # than round away ordering/availability at subsecond boundaries.
    seconds = pc.multiply(pc.floor(pc.divide(micros, 1_000_000)), 1_000_000)
    if not pc.all(pc.equal(micros, seconds)).as_py():
        raise UnsupportedCombinationError(
            f"{path}: native retrieval requires whole-second clocks; select data.loader: python"
        )
    return micros


def validate_feature_bindings(
    source: TableSource, *, extra_forbidden: frozenset[str] = frozenset()
) -> None:
    """Validate original metadata columns before key/FK normalization hides their names."""
    forbidden = PROHIBITED_INPUT_KEYS | {name.strip().casefold() for name in extra_forbidden}
    for table in source.tables:
        if table.role == "labels":
            continue
        metadata = {
            *table.key,
            table.event_time,
            table.available_at,
            *source.event_labels.get(table.name, ()),
        }
        for role in source.relationships:
            if role.child == table.name:
                metadata.update(role.columns)
                metadata.add(role.available_at)
        structural = {name.strip().casefold() for name in metadata if name is not None}
        graph = {} if source.graph_numeric is None else source.graph_numeric.get(table.name, {})
        for mapping in (table.numeric, table.categorical, table.text, graph):
            for alias, column in mapping.items():
                if (
                    alias.strip().casefold() in forbidden
                    or column.strip().casefold() in forbidden | structural
                ):
                    raise SchemaError(
                        f"{table.name}.{column}: keys, clocks and label-derived fields cannot be model inputs"
                    )


class NativeSourceFiles(AbstractContextManager["NativeSourceFiles"]):
    """Normalize source files in bounded Arrow blocks and retain key state on disk.

    The caller owns ``directory`` and may persist it with a run. This object
    closes only its SQLite connection; it never removes caller-owned files.
    """

    def __init__(self, source: TableSource, directory: Path, *, block_size: int = 65_536) -> None:
        validate_feature_bindings(source)
        if block_size < 1:
            raise ValueError("Native source block_size must be positive")
        if source.ownership_versions is not None:
            raise UnsupportedCombinationError(
                "relationships.ownership_versions: native source traversal cannot preserve event-time versioned ownership; select data.loader: python"
            )
        self.original, self.directory, self.block_size = source, directory, block_size
        directory.mkdir(parents=True, exist_ok=True)
        database = directory / "keys.sqlite"
        if database.exists():
            raise ValueError(f"Native source directory already has a key registry: {directory}")
        self.connection = sqlite3.connect(database)
        self.connection.execute(
            "CREATE TABLE identities (table_name TEXT, source_key TEXT, code INTEGER, "
            "available_us INTEGER, event_us INTEGER, PRIMARY KEY(table_name,source_key), UNIQUE(table_name,code))"
        )
        self.connection.execute(
            "CREATE TABLE fanouts (role TEXT, parent INTEGER, degree INTEGER, PRIMARY KEY(role,parent))"
        )
        self.connection.execute(
            "CREATE TABLE links (role TEXT,parent INTEGER,child INTEGER,available_us INTEGER,PRIMARY KEY(role,child))"
        )
        self.connection.execute(
            "CREATE TABLE numbers (table_name TEXT,code INTEGER,field TEXT,value REAL,PRIMARY KEY(table_name,code,field))"
        )
        self.connection.execute(
            "CREATE TABLE event_labels(table_name TEXT,code INTEGER,value REAL,available_us INTEGER,PRIMARY KEY(table_name,code))"
        )
        self.same_clocks: dict[str, bool] = {}
        self.counts: dict[str, int] = {}
        self.source_identity: dict[str, str] = {}
        self._source_stats: dict[str, tuple[int, ...]] = {}
        try:
            self._index_keys()
            self.source = self._write_sources()
            self._verify_snapshot()
            self.connection.execute("CREATE INDEX links_forward ON links(role,parent)")
            self.connection.commit()
            self.forward_degrees = dict(
                self.connection.execute("SELECT role,MAX(degree) FROM fanouts GROUP BY role")
            )
        except BaseException:
            self.close()
            raise

    def _blocks(self, name: str, columns: list[str] | None = None) -> Iterator[pa.RecordBatch]:
        yield from pq.ParquetFile(self.original.files[name]).iter_batches(
            batch_size=self.block_size, columns=columns
        )

    @staticmethod
    def _snapshot_stat(path: Path) -> tuple[int, ...]:
        info = path.stat()
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def _verify_snapshot(self) -> None:
        for name, path in self.original.files.items():
            expected = self._source_stats[name]
            if (
                self._snapshot_stat(path) != expected
                or file_sha256(path) != self.source_identity[name]
                or self._snapshot_stat(path) != expected
            ):
                raise SchemaError(f"{name}: source snapshot changed during normalization")

    def _index_keys(self) -> None:
        for table in self.original.tables:
            path = self.original.files[table.name]
            self._source_stats[table.name] = self._snapshot_stat(path)
            self.source_identity[table.name] = file_sha256(path)
            if self._snapshot_stat(path) != self._source_stats[table.name]:
                raise SchemaError(f"{table.name}: source snapshot changed during normalization")
            if table.role == "labels":
                self.counts[table.name] = pq.ParquetFile(
                    self.original.files[table.name]
                ).metadata.num_rows
                continue
            count = 0
            for block in self._blocks(table.name, list(table.key)):
                values = zip(*(block.column(c).to_pylist() for c in table.key), strict=True)
                rows = (
                    (table.name, _key(tuple(value), f"{table.name}.key"), count + i)
                    for i, value in enumerate(values)
                )
                try:
                    self.connection.executemany(
                        "INSERT INTO identities(table_name,source_key,code) VALUES (?,?,?)", rows
                    )
                except sqlite3.IntegrityError as error:
                    raise SchemaError(f"{table.name}.key: duplicate source primary key") from error
                count += block.num_rows
            self.counts[table.name] = count
        self.connection.commit()

    def code(self, table: str, key: tuple[Any, ...]) -> int:
        """Resolve an anchor/FK through the persisted source identity registry."""
        row = self.connection.execute(
            "SELECT code FROM identities WHERE table_name=? AND source_key=?",
            (table, _key(key, f"{table}.key")),
        ).fetchone()
        if row is None:
            raise SchemaError(f"{table}.key: reference to a missing source row")
        return int(row[0])

    def _write_sources(self) -> TableSource:
        tables, files = [], {}
        relationships = tuple(
            replace(role, columns=(f"__tfm_fk_{i}",), available_at=None)
            for i, role in enumerate(self.original.relationships)
        )
        for table in self.original.tables:
            if table.role == "labels":
                tables.append(table)
                path = self.directory / f"labels-{len(tables)}.parquet"
                shutil.copyfile(self.original.files[table.name], path)
                files[table.name] = path
                continue
            path = self.directory / f"table-{len(tables)}.parquet"
            writer = None
            count = 0
            self.same_clocks[table.name] = True
            try:
                for block in self._blocks(table.name):
                    if any(name.startswith("__tfm_") for name in block.schema.names):
                        raise SchemaError(
                            f"{table.name}: __tfm_ columns are reserved for ingestion"
                        )
                    projected = pa.Table.from_batches([block]).append_column(
                        ROW_ID, pa.array(range(count, count + block.num_rows), type=pa.int64())
                    )
                    # Native numeric encoders distinguish NULL (missing) from
                    # non-finite observations. Normalize only declared numeric
                    # inputs; their presence masks remain explicit at delivery.
                    for column in set(table.numeric.values()):
                        values = projected.column(column)
                        projected = projected.set_column(
                            projected.schema.get_field_index(column),
                            column,
                            pc.if_else(pc.is_finite(values), values, None),
                        )
                    if table.name in self.original.event_labels:
                        value_column, known_column = self.original.event_labels[table.name]
                        for i, (value, known) in enumerate(
                            zip(
                                block.column(value_column).to_pylist(),
                                block.column(known_column).to_pylist(),
                                strict=True,
                            )
                        ):
                            if (value is None) != (known is None) or (
                                value is not None and value not in (0, 1)
                            ):
                                raise SchemaError(
                                    f"{table.name}: event labels require binary values with availability clocks"
                                )
                            if value is not None:
                                self.connection.execute(
                                    "INSERT INTO event_labels VALUES (?,?,?,?)",
                                    (table.name, count + i, value, known),
                                )
                    for alias, column in table.numeric.items():
                        self.connection.executemany(
                            "INSERT INTO numbers VALUES (?,?,?,?)",
                            (
                                (table.name, count + i, alias, value)
                                for i, value in enumerate(projected.column(column).to_pylist())
                            ),
                        )
                    for column, output in (
                        (table.available_at, AVAILABLE_US),
                        (table.event_time, EVENT_US),
                    ):
                        if column is not None:
                            micros = _clocks(block, column, f"{table.name}.{column}")
                            projected = projected.append_column(output, micros)
                            # Date clocks mean UTC midnight in the shared contract.
                            # Native indexes require timestamp storage explicitly.
                            projected = projected.set_column(
                                projected.schema.get_field_index(column),
                                column,
                                pc.cast(micros, pa.timestamp("us", tz="UTC")),
                            )
                    if (
                        not table.immutable
                        and not pc.all(
                            pc.greater_equal(
                                projected.column(AVAILABLE_US), projected.column(EVENT_US)
                            )
                        ).as_py()
                    ):
                        raise SchemaError(f"{table.name}: observation predates event time")
                    if not table.immutable:
                        self.same_clocks[table.name] &= bool(
                            pc.all(
                                pc.equal(projected.column(AVAILABLE_US), projected.column(EVENT_US))
                            ).as_py()
                        )
                    for i, role in enumerate(self.original.relationships):
                        if role.child != table.name:
                            continue
                        if role.available_at is not None:
                            known = _clocks(block, role.available_at, f"{table.name}.{role.name}")
                            if (
                                table.available_at is None
                                or not pc.all(
                                    pc.equal(known, projected.column(AVAILABLE_US))
                                ).as_py()
                            ):
                                raise UnsupportedCombinationError(
                                    f"{table.name}.{role.name}: native fixed-FK availability must equal row availability"
                                )
                        keys = zip(
                            *(block.column(c).to_pylist() for c in role.columns), strict=True
                        )
                        codes = [
                            None
                            if all(v is None for v in key)
                            else self.code(role.parent, tuple(key))
                            for key in keys
                        ]
                        projected = projected.append_column(
                            f"__tfm_fk_{i}", pa.array(codes, type=pa.int64())
                        )
                        observed = (
                            [None] * block.num_rows
                            if table.immutable
                            else projected.column(AVAILABLE_US).to_pylist()
                        )
                        self.connection.executemany(
                            "INSERT INTO links VALUES (?,?,?,?)",
                            (
                                (role.name, parent, count + i, known)
                                for i, (parent, known) in enumerate(
                                    zip(codes, observed, strict=True)
                                )
                                if parent is not None
                            ),
                        )
                        counts = Counter(code for code in codes if code is not None)
                        self.connection.executemany(
                            "INSERT INTO fanouts VALUES (?,?,?) ON CONFLICT(role,parent) DO UPDATE SET degree=degree+excluded.degree",
                            ((role.name, parent, degree) for parent, degree in counts.items()),
                        )
                    if not table.immutable:
                        self.connection.executemany(
                            "UPDATE identities SET available_us=?,event_us=? WHERE table_name=? AND code=?",
                            (
                                (known, event, table.name, count + i)
                                for i, (known, event) in enumerate(
                                    zip(
                                        projected.column(AVAILABLE_US).to_pylist(),
                                        projected.column(EVENT_US).to_pylist(),
                                        strict=True,
                                    )
                                )
                            ),
                        )
                    if writer is None:
                        writer = pq.ParquetWriter(path, projected.schema)
                    writer.write_table(projected)
                    count += block.num_rows
            finally:
                if writer is not None:
                    writer.close()
            if writer is None:
                empty = pa.Table.from_batches(
                    [], schema=pq.read_schema(self.original.files[table.name])
                )
                empty = empty.append_column(ROW_ID, pa.array([], type=pa.int64()))
                if not table.immutable:
                    empty = empty.append_column(
                        AVAILABLE_US, pa.array([], type=pa.int64())
                    ).append_column(EVENT_US, pa.array([], type=pa.int64()))
                for i, role in enumerate(self.original.relationships):
                    if role.child == table.name:
                        empty = empty.append_column(f"__tfm_fk_{i}", pa.array([], type=pa.int64()))
                pq.write_table(empty, path)
            tables.append(replace(table, key=(ROW_ID,)))
            files[table.name] = path
        self.connection.commit()
        return replace(
            self.original, tables=tuple(tables), relationships=relationships, files=files
        )

    def close(self) -> None:
        """Release the disk registry after producers/tasks have been closed."""
        self.connection.close()

    def __exit__(self, *args: Any) -> None:
        self.close()
