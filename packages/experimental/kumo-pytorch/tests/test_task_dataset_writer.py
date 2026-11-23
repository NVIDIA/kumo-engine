# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

dg = pytest.importorskip("diskgraph")


def test_rotating_task_parts_preserve_order_seconds_and_binding(tmp_path):
    schema = pa.schema(
        [("time", pa.timestamp("s", tz="UTC")), ("x", pa.int64())], metadata={b"binding": b"task"}
    )
    table = pa.Table.from_arrays(
        [pa.array([-1, 0, 1, 42, 99], type=schema.field(0).type), pa.array([5, 4, 3, 2, 1])],
        schema=schema,
    )
    with dg.TaskParquetWriter(str(tmp_path / "parts"), max_rows_per_part=2) as writer:
        for batch in table.to_batches(max_chunksize=3):
            writer.write_batch(batch)
    parts = sorted((tmp_path / "parts").glob("*.parquet"))
    assert [p.name for p in parts] == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]
    assert [pq.ParquetFile(p).metadata.num_rows for p in parts] == [2, 2, 1]
    restored = pa.concat_tables([pq.read_table(p) for p in parts])
    assert restored.cast(schema).equals(table)
    assert restored.schema.metadata == schema.metadata


def test_empty_task_part_keeps_schema(tmp_path):
    schema = pa.schema([("x", pa.int64())], metadata={b"binding": b"empty"})
    with dg.TaskParquetWriter(str(tmp_path / "empty")) as writer:
        writer.write_batch(
            pa.RecordBatch.from_arrays([pa.array([], type=pa.int64())], schema=schema)
        )
    saved = pq.read_table(tmp_path / "empty")
    assert saved.num_rows == 0
    assert saved.schema == schema
