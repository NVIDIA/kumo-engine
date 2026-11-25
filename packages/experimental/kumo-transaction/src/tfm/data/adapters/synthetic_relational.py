# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic synthetic tables demonstrating one mapping for all recipe modes."""

from collections.abc import Iterator
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tfm.config import DataConfig
from tfm.contracts.relational import RelationalDataset, RelationalForeignKey, RelationalTable
from tfm.contracts.table_source import TableSource
from tfm.data.adapters.registry import Adapter
from tfm.data.adapters.result import AdapterResult

ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)


def _schema() -> tuple[tuple[RelationalTable, ...], tuple[RelationalForeignKey, ...]]:
    return (
        (
            RelationalTable("entities", ("id",), immutable=True),
            RelationalTable("containers", ("id",), event_time="created", available_at="known"),
            RelationalTable(
                "activity",
                ("id",),
                numeric={"amount": "value"},
                event_time="at",
                available_at="known",
            ),
            RelationalTable("outcomes", ("id",), immutable=True, role="labels"),
        ),
        (
            RelationalForeignKey("owns", "entities", "containers", ("owner",)),
            RelationalForeignKey("activity", "containers", "activity", ("container",)),
        ),
    )


def _records(config: DataConfig) -> Iterator[tuple[str, dict[str, Any]]]:
    if config.n_entities < 8:
        raise ValueError("The relational synthetic example needs at least eight entities")
    for i in range(config.n_entities):
        owner, group = f"entity{i}", f"container{i}"
        split = (
            "train"
            if i < config.n_entities // 2
            else "validation"
            if i < config.n_entities * 3 // 4
            else "test"
        )
        cutoff = ORIGIN + timedelta(days={"train": 10, "validation": 11, "test": 12}[split])
        yield "entities", {"id": owner}
        yield (
            "containers",
            {"id": group, "owner": owner, "created": ORIGIN, "known": ORIGIN + timedelta(days=1)},
        )
        for j, day in enumerate((2, 4, 12, 13)):
            yield (
                "activity",
                {
                    "id": f"activity{i}-{j}",
                    "container": group,
                    "at": ORIGIN + timedelta(days=day),
                    "known": ORIGIN + timedelta(days=day),
                    "value": float((i + 1) * (j + 1)),
                },
            )
        if config.load_labels:
            yield (
                "outcomes",
                {
                    "id": f"label{i}",
                    "owner": owner,
                    "cutoff": cutoff,
                    "outcome": float(i % 2),
                    "available": cutoff + timedelta(days=3),
                    "split": split,
                },
            )


def declaration(config: DataConfig) -> RelationalDataset:
    """The small Python reference fixture; keys remain metadata, never features."""
    tables, roles = _schema()
    rows: dict[str, list[dict[str, Any]]] = {table.name: [] for table in tables}
    for table, row in _records(config):
        rows[table].append(row)
    return RelationalDataset(
        tables,
        roles,
        {table: tuple(values) for table, values in rows.items()},
        observed_through=ORIGIN + timedelta(days=30),
    )


def load(
    config: DataConfig, *, seed: int, flat_fields: bool = False, fixture_dir: Path | None = None
) -> AdapterResult:
    """The Python object fixture, sharing the same source declaration."""
    del seed, flat_fields, fixture_dir
    return AdapterResult(relational=declaration(config))


def source_files(config: DataConfig, *, seed: int, fixture_dir: Path) -> TableSource:
    """Write bounded blocks of source rows; prepare no event objects or histories."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    del seed
    tables, roles = _schema()
    timestamp = pa.timestamp("us", tz="UTC")
    schemas = {
        "entities": pa.schema([("id", pa.string())]),
        "containers": pa.schema(
            [
                ("id", pa.string()),
                ("owner", pa.string()),
                ("created", timestamp),
                ("known", timestamp),
            ]
        ),
        "activity": pa.schema(
            [
                ("id", pa.string()),
                ("container", pa.string()),
                ("at", timestamp),
                ("known", timestamp),
                ("value", pa.float64()),
            ]
        ),
        "outcomes": pa.schema(
            [
                ("id", pa.string()),
                ("owner", pa.string()),
                ("cutoff", timestamp),
                ("outcome", pa.float64()),
                ("available", timestamp),
                ("split", pa.string()),
            ]
        ),
    }
    fixture_dir.mkdir(parents=True, exist_ok=True)
    files = {table.name: fixture_dir / f"{table.name}.parquet" for table in tables}
    with ExitStack() as stack:
        writers = {
            name: stack.enter_context(pq.ParquetWriter(files[name], schema))
            for name, schema in schemas.items()
        }
        buffers: dict[str, list[dict[str, Any]]] = {name: [] for name in schemas}
        for name, row in _records(config):
            buffers[name].append(row)
            if len(buffers[name]) == 4096:
                writers[name].write_table(pa.Table.from_pylist(buffers[name], schema=schemas[name]))
                buffers[name].clear()
        for name, rows in buffers.items():
            if rows:
                writers[name].write_table(pa.Table.from_pylist(rows, schema=schemas[name]))
    return TableSource(tables, roles, files, observed_through=ORIGIN + timedelta(days=30))


adapter = Adapter(load, table_source=source_files)
