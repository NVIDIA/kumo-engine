# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate or replay file-backed native anchors, without constructing histories."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tfm.config import RelationalTaskConfig
from tfm.contracts.errors import SchemaError
from tfm.contracts.graph import utc_us
from tfm.contracts.table_source import TableSource
from tfm.data.native_source import NativeSourceFiles
from tfm.data.relational import timestamp


def _time(value: Any, path: str) -> datetime:
    return timestamp(value, path)


def _identity(key: tuple[Any, ...]) -> str:
    return json.dumps(key, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _source_rows(source: TableSource, table: str) -> Iterator[dict[str, Any]]:
    for batch in pq.ParquetFile(source.files[table]).iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def task_anchors(source: NativeSourceFiles, task: RelationalTaskConfig, output: Path) -> Path:
    """One anchor declaration serves sequence, graph and hybrid tasks.

    Labels are joined only as task metadata. Label horizons and fitting cutoffs
    are checked independently of the feature cutoff. SQLite validates uniqueness
    and performs replay joins with bounded Python memory.
    """
    tables = {table.name: table for table in source.original.tables}
    if task.root not in tables or tables[task.root].role != "input":
        raise SchemaError(f"Root path {task.root}: declare an input table")
    root = tables[task.root]
    db = source.connection
    db.execute(
        "CREATE TABLE task_labels (root INTEGER, cutoff INTEGER, split TEXT, target REAL, available INTEGER, fitting INTEGER, PRIMARY KEY(root,cutoff,split))"
    )
    labels = task.labels
    if labels is not None:
        if labels.table not in tables or tables[labels.table].role != "labels":
            raise SchemaError(f"Label path {labels.table}: declare a labels-role table")
        if len(labels.root_columns) != len(root.key):
            raise SchemaError(f"Label path {labels.table}: root key arity differs")
        for row in _source_rows(source.original, labels.table):
            key = tuple(row[c] for c in labels.root_columns)
            code = source.code(task.root, key)
            cutoff = _time(row[labels.timestamp], f"{labels.table}.{labels.timestamp}")
            available = _time(row[labels.available_at], f"{labels.table}.{labels.available_at}")
            split = str(row[labels.split])
            if split not in task.label_cutoffs:
                raise SchemaError(f"Label path {labels.table}: declare label_cutoffs[{split!r}]")
            fitting = _time(task.label_cutoffs[split], f"label_cutoffs.{split}")
            target = row.get(labels.target)
            if target not in (0, 1) or available > fitting:
                raise SchemaError(
                    f"Label path {labels.table}: binary label unavailable at declared fitting cutoff"
                )
            if available < cutoff + timedelta(days=labels.horizon_days + labels.label_delay_days):
                raise SchemaError(
                    f"Label path {labels.table}: label observed before declared horizon/delay"
                )
            try:
                db.execute(
                    "INSERT INTO task_labels VALUES (?,?,?,?,?,?)",
                    (
                        code,
                        utc_us(cutoff),
                        split,
                        float(target),
                        utc_us(available),
                        utc_us(fitting),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise SchemaError(
                    f"Label path {labels.table}: duplicate root/cutoff/split"
                ) from error
        db.commit()

    def candidates() -> Iterator[dict[str, Any]]:
        if task.anchors is not None:
            for batch in pq.ParquetFile(task.anchors).iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    yield {**row, "split": row.get("split", "predict")}
        elif task.cutoffs:
            for split, raw_cutoff in task.cutoffs.items():
                cutoff = _time(raw_cutoff, f"cutoffs.{split}")
                for encoded, _code, known in db.execute(
                    "SELECT source_key,code,available_us FROM identities WHERE table_name=? ORDER BY code",
                    (task.root,),
                ):
                    if known is not None and known > utc_us(cutoff):
                        continue
                    key = tuple(json.loads(encoded))
                    yield {
                        **dict(zip(root.key, key, strict=True)),
                        "sample_id": _identity((_identity(key), cutoff.isoformat(), split)),
                        "cutoff": cutoff,
                        "split": split,
                    }
        elif labels is not None:
            for encoded, at, split in db.execute(
                "SELECT i.source_key,l.cutoff,l.split FROM task_labels l JOIN identities i ON i.table_name=? AND i.code=l.root ORDER BY l.rowid",
                (task.root,),
            ):
                key = tuple(json.loads(encoded))
                cutoff = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=at)
                yield {
                    **dict(zip(root.key, key, strict=True)),
                    "sample_id": _identity((_identity(key), cutoff.isoformat(), split)),
                    "cutoff": cutoff,
                    "split": split,
                }
        else:
            raise SchemaError("Native task requires label timestamps, cutoffs, or replay anchors")

    # Preserve each raw primary-key dtype. Normalization resolves composite keys
    # through the same registry used for source FKs, without string coercion.
    root_schema = pq.read_schema(source.original.files[task.root])
    schema = pa.schema(
        [
            *((c, root_schema.field(c).type) for c in root.key),
            ("sample_id", pa.string()),
            ("cutoff", pa.timestamp("us", tz="UTC")),
            ("split", pa.string()),
            ("observation_start", pa.timestamp("us", tz="UTC")),
            ("target", pa.float32()),
            ("label_available_us", pa.int64()),
            ("label_cutoff_us", pa.int64()),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(output, schema) as writer:
        buffer = []
        for row in candidates():
            cutoff = _time(row["cutoff"], "anchors.cutoff")
            split = str(row["split"])
            if labels is not None:
                code = source.code(task.root, tuple(row[c] for c in root.key))
                label = db.execute(
                    "SELECT target,available,fitting FROM task_labels WHERE root=? AND cutoff=? AND split=?",
                    (code, utc_us(cutoff), split),
                ).fetchone()
                if label is None and split != "predict":
                    raise SchemaError(
                        f"Anchor {row.get('sample_id')}: missing available entity label"
                    )
                if label is not None:
                    row = {
                        **row,
                        "target": label[0],
                        "label_available_us": label[1],
                        "label_cutoff_us": label[2],
                    }
            buffer.append({**row, "cutoff": cutoff})
            if len(buffer) == 4096:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
    return output
