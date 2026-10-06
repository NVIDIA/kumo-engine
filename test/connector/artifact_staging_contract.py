# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class CancelledByRuntime(BaseException):
    """Test-only cancellation sentinel that ordinary handlers must not mask."""


def local_artifact_dataset(
    tmp_path: Path,
    *,
    shards: int = 1,
    embeddings: bool = False,
    nested: bool = False,
) -> Path:
    """Create the canonical local prediction/embedding Parquet dataset."""
    path = tmp_path / ('processed_embeddings.parquet'
                       if embeddings else 'processed_predictions.parquet')
    path.mkdir()
    values = ([[0.1, 0.2], [0.3, 0.4]] if embeddings else [0.2, 0.8])
    column = 'EMBEDDING' if embeddings else 'SCORE'
    table = pa.table({'ENTITY_ID': [1, 2], column: values})
    for index in range(shards):
        parent = path / f'nested-{index}' if nested else path
        parent.mkdir(exist_ok=True)
        filename = 'part.parquet' if nested else f'part-{index:05d}.parquet'
        pq.write_table(table.slice(index % 2, 1), parent / filename)
    return path


def assert_indexed_parquet_parts(paths: list[str], suffix: str = '.parquet') -> str:
    """Assert unique staging plus deterministic shard names; return prefix."""
    prefixes = {path.rsplit('/', 1)[0] for path in paths}
    assert len(prefixes) == 1
    assert {path.rsplit('/', 1)[-1] for path in paths} == {
        f'part-{index:08d}{suffix}'
        for index in range(len(paths))
    }
    return prefixes.pop()
