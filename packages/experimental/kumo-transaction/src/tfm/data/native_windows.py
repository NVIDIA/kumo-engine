# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed, nonoverlapping source groups for flat pretraining windows.

Each original event is stored once and gains one fixed group FK. Prediction
anchors select groups; the index never stores a copy of every requested
history. Whole model batches remain an iterator concern.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from tfm.config import RecipeConfig
from tfm.contracts.errors import SchemaError
from tfm.contracts.graph import utc_us
from tfm.contracts.relational import RelationalForeignKey
from tfm.contracts.table_source import TableSource
from tfm.data.splits.temporal import parse_duration, parse_timestamp

PREFIX = "__tfm_window/"


def flat_window_source(source: TableSource, config: RecipeConfig, directory: Path) -> TableSource:
    """Add fixed ownership for training chunks; retain the original owner FK too."""
    windows = config.data.pretrain_windows
    if windows is None:
        return source
    directory.mkdir(parents=True, exist_ok=True)
    events = pq.ParquetFile(source.files["events"])
    if config.data.split_protocol == "precomputed":
        maximum = None
        for block in events.iter_batches(batch_size=4096, columns=["at", "partition"]):
            times = [
                at
                for at, part in zip(
                    block.column(0).to_pylist(), block.column(1).to_pylist(), strict=True
                )
                if part == "train"
            ]
            if times:
                maximum = max(maximum, max(times)) if maximum else max(times)
        if maximum is None:
            raise SchemaError("Precomputed source has no training events")
        cutoff = maximum
    else:
        cutoff = parse_timestamp(config.data.temporal_split.train_end)
    lower = utc_us(cutoff - parse_duration(config.data.observation_window))
    with (
        TemporaryDirectory(prefix="groups-", dir=directory) as temporary,
        sqlite3.connect(Path(temporary) / "groups.sqlite") as db,
    ):
        db.execute(
            "CREATE TABLE events(id TEXT PRIMARY KEY,owner TEXT,at INTEGER,eligible INTEGER,payload TEXT,group_id TEXT)"
        )
        timestamp_columns = [f.name for f in events.schema_arrow if pa.types.is_timestamp(f.type)]
        for block in events.iter_batches(batch_size=4096):
            records = []
            for row in block.to_pylist():
                if str(row["owner"]).startswith(PREFIX):
                    raise SchemaError(
                        f"Source owner uses reserved pretraining group prefix {PREFIX!r}"
                    )
                stamp = utc_us(row["at"])
                eligible = (
                    row["partition"] == "train"
                    if config.data.split_protocol == "precomputed"
                    else stamp < utc_us(cutoff)
                )
                records.append(
                    (
                        row["id"],
                        row["owner"],
                        stamp,
                        int(eligible and stamp >= lower),
                        json.dumps(row, default=lambda value: value.isoformat()),
                    )
                )
            db.executemany(
                "INSERT INTO events(id,owner,at,eligible,payload) VALUES (?,?,?,?,?)", records
            )
        db.execute(
            "CREATE TABLE assignments AS SELECT id,owner,CAST((ROW_NUMBER() OVER(PARTITION BY owner ORDER BY at,id)-1)/? AS INTEGER) AS ordinal FROM events WHERE eligible=1",
            (windows.events,),
        )
        db.execute("CREATE UNIQUE INDEX assignment_ids ON assignments(id)")
        db.execute(
            "UPDATE events SET group_id=COALESCE((SELECT ? || owner || '/' || ordinal FROM assignments a WHERE a.id=events.id),? || owner || '/unused')",
            (PREFIX, PREFIX),
        )
        db.commit()
        output_schema = events.schema_arrow.append(pa.field("window_owner", pa.string()))
        output = directory / "events.parquet"
        with pq.ParquetWriter(output, output_schema) as writer:
            cursor = db.execute("SELECT payload,group_id FROM events ORDER BY owner,at,id")
            while records := cursor.fetchmany(4096):
                rows = []
                for payload, group in records:
                    row = json.loads(payload)
                    for name in timestamp_columns:
                        if row[name] is not None:
                            row[name] = datetime.fromisoformat(row[name])
                    rows.append({**row, "window_owner": group})
                writer.write_table(pa.Table.from_pylist(rows, schema=output_schema))
        users = pq.ParquetFile(source.files["users"])
        roots = directory / "users.parquet"
        with pq.ParquetWriter(roots, users.schema_arrow) as writer:
            for block in users.iter_batches(batch_size=4096):
                writer.write_table(pa.Table.from_batches([block]))
            cursor = db.execute("SELECT DISTINCT group_id FROM events ORDER BY group_id")
            while groups := cursor.fetchmany(4096):
                writer.write_table(
                    pa.Table.from_pylist(
                        [{"id": group, "bias": 1.0} for (group,) in groups],
                        schema=users.schema_arrow,
                    )
                )
    return replace(
        source,
        files={**source.files, "events": output, "users": roots},
        relationships=(
            *source.relationships,
            RelationalForeignKey("window_events", "users", "events", ("window_owner",)),
        ),
    )
