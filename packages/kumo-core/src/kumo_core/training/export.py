# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exported rows of a generated training table."""

from __future__ import annotations

import json
import os
import posixpath
from typing import Any, Iterator, Mapping, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from kumo_core.train.autoregression import (
    AUTOREGRESSION_COL,
    FUTURE_LABELS_COL,
    YOY_COL,
)
from kumo_core.training import custom_table
from pyarrow.fs import FileSystem

SPLIT_COLUMN = "SPLIT"
STAGE_SPLITS = {"train": 0, "val": 1, "test": 2}
INGEST_SPEC = "tt_ingest_spec.json"
# Bump when the exported columns change.
FORMAT = "tt-export-v1"
_KEPT = (AUTOREGRESSION_COL, FUTURE_LABELS_COL, YOY_COL,
         "__kumo_internal_frequency_column__")


def stage_parts(spec: Mapping[str, Any]) -> dict[int, list[str]]:
    """Seed parts relative to the bundle, keyed by split value."""
    stages = {s["key"]: list(s["parts"]) for s in spec.get("split_stages") or []}
    if not any(stages.values()) or set(stages) - set(STAGE_SPLITS):
        raise ValueError(f"Unexpected training table stages: {sorted(stages)}")
    return {
        split: stages[key]
        for key, split in STAGE_SPLITS.items() if key in stages
    }


def identity(spec: Mapping[str, Any]) -> str:
    return f"{spec['tt_id']}:{FORMAT}"


def _exported(name: str) -> bool:
    weight = custom_table.contract()["weight_output_column"]
    return name in _KEPT or custom_table.column_role(
        name, weight_col=weight) != "other"


def export_schema(seed_schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [f.with_nullable(True) for f in seed_schema if _exported(f.name)] +
        [pa.field(SPLIT_COLUMN, pa.int64(), nullable=False)])


def read_spec(bundle_dir: str,
              filesystem: Optional[FileSystem] = None) -> dict[str, Any]:
    path = posixpath.join(bundle_dir, INGEST_SPEC)
    if filesystem is not None:
        with filesystem.open_input_stream(path) as handle:
            return json.loads(handle.read())
    if not os.path.exists(path):
        raise ValueError(f"{bundle_dir!r} is not a generated training table")
    with open(path) as handle:
        return json.load(handle)


def iter_export_parts(
    bundle_dir: str,
    output_dir: str,
    filesystem: Optional[FileSystem] = None,
) -> Iterator[str]:
    """Write every staged row with its ``SPLIT``, yielding each local part."""
    spec = read_spec(bundle_dir, filesystem)
    parts = [(split, posixpath.join(bundle_dir, part))
             for split, paths in stage_parts(spec).items() for part in paths]
    schema = export_schema(
        pq.read_schema(parts[0][1], filesystem=filesystem))
    columns = schema.names[:-1]
    os.makedirs(output_dir, exist_ok=True)
    written = 0
    for split, part in parts:
        source = pq.ParquetFile(part, filesystem=filesystem)
        try:
            if export_schema(source.schema_arrow) != schema:
                raise ValueError(
                    f"Training table part {part!r} does not match the "
                    f"schema of {parts[0][1]!r}")
            if source.metadata.num_rows == 0:
                continue
            path = os.path.join(output_dir, f"part-{written:05d}.parquet")
            with pq.ParquetWriter(path, schema) as writer:
                for batch in source.iter_batches(columns=columns):
                    values = [batch.column(name) for name in columns]
                    values.append(
                        pa.repeat(pa.scalar(split, pa.int64()),
                                  batch.num_rows))
                    writer.write_table(
                        pa.Table.from_arrays(values, schema=schema))
        finally:
            source.close()
        written += 1
        yield path
    if not written:
        path = os.path.join(output_dir, "part-00000.parquet")
        pq.write_table(schema.empty_table(), path)
        yield path


def export_train_table(bundle_dir: str, output_dir: str) -> list[str]:
    return list(iter_export_parts(bundle_dir, output_dir))
