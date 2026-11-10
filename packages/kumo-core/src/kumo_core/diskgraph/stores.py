# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DiskGraph runtime sizing and bounded node-column reads."""

import json
import multiprocessing
import os
import sys
from typing import Any, Iterator

import numpy as np
import pyarrow as pa

from kumo_core.diskgraph.metadata import load_diskgraph_schema
from kumo_core.sources.cpu import cpu_budget
from kumo_core.training.config import shm_safe_num_workers


def diskgraph_auto_config(
    cpu_count: int,
    ram_bytes: int,
    index_bytes: int,
    estimated_edges: int,
) -> dict:
    """Auto-configure diskgraph sampling parameters.

    Returns dict with keys:
    - ``mode``: ``"fused"`` or ``"standalone"``
    - ``num_workers``: recommended DataLoader workers
    - ``rayon_threads``: rayon threads per worker

    ``estimated_edges`` is the per-batch I/O volume estimate:
    ``batch_size * per_seed_fanout * 0.25`` (dedup/sparsity factor).

    Heuristic based on empirical sweep data across subgraph sizes,
    worker counts, and rayon thread counts on a production ranking job's
    graph (500GB index, 64 vCPUs, 500GB RAM).
    """
    # On non-Linux platforms (macOS, Windows) multiprocessing uses 'spawn',
    # where re-importing the whole module graph per worker makes worker startup
    # expensive enough that single-process loading is the better default.
    if multiprocessing.get_start_method(allow_none=True) in (None, 'spawn') \
            and sys.platform != 'linux':
        return {"mode": "fused", "num_workers": 0, "rayon_threads": 1}

    if estimated_edges < 5_000:
        return {
            "mode": "fused",
            "num_workers": shm_safe_num_workers(max(2, min(cpu_count // 4,
                                                           16))),
            "rayon_threads": 1,
        }

    usable_memory = int(ram_bytes * 0.7)
    cache_ratio = usable_memory / max(index_bytes, 1)
    effective_cpus = (int(cpu_count *
                          0.85) if cache_ratio > 0.9 else int(cpu_count * 0.5))

    if estimated_edges < 100_000:
        num_workers = min(effective_cpus // 2, 16)
    else:
        num_workers = min(effective_cpus // 4, 8)
    num_workers = max(2, num_workers)

    rayon_threads = max(2, min(effective_cpus // num_workers, 8))
    num_workers = shm_safe_num_workers(num_workers)
    if num_workers > 0:
        rayon_threads = max(2, min(effective_cpus // num_workers, 8))

    return {
        "mode": "standalone",
        "num_workers": num_workers,
        "rayon_threads": rayon_threads,
    }


def diskgraph_index_size_bytes(index_path: str) -> int:
    """Read the finalized DiskGraph snapshot size without walking its tree.

    The snapshot manifest is written atomically only after ingest succeeds. A
    directory walk is unsafe because a pool stage/evict can mutate the live
    directory while a trainer is selecting its runtime configuration.
    """
    manifest_path = os.path.join(index_path, "_MANIFEST.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return int(sum(entry["size_bytes"] for entry in manifest["files"]))


def _decimal_primary_key_arrow_type(
    index_path: str,
    table_name: str,
    primary_key_name: str,
) -> pa.DataType | None:
    """Recover decimal PK metadata lost by the feature-fetch string view."""
    config_path = os.path.join(index_path, 'diskgraph_config.json')
    try:
        with open(config_path) as file:
            config = json.load(file)
    except FileNotFoundError:
        return None

    for table in config.get('tables', []):
        if table.get('name') != table_name:
            continue
        for column in table.get('columns', []):
            if column.get('name') != primary_key_name:
                continue
            dtype = column.get('ingested_dtype')
            if dtype not in {'decimal128', 'decimal256'}:
                return None
            try:
                precision = int(column['precision'])
                scale = int(column['scale'])
                constructor = (pa.decimal128
                               if dtype == 'decimal128' else pa.decimal256)
                return constructor(precision, scale)
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"DiskGraph decimal primary key '{table_name}."
                    f"{primary_key_name}' is missing valid precision/scale"
                ) from error
    return None


_NODE_COLUMN_READ_BATCH = 1 << 20


def _iter_node_column(
    engine: Any,
    table_name: str,
    column: str,
    start: int,
    stop: int,
) -> Iterator[tuple[int, pa.Array]]:
    for chunk_start in range(start, stop, _NODE_COLUMN_READ_BATCH):
        node_ids = list(
            range(chunk_start, min(chunk_start + _NODE_COLUMN_READ_BATCH,
                                   stop)))
        request = [{
            'node_type': table_name,
            'node_ids': node_ids,
            'feature_names': [column],
        }]
        batches = [
            event['batch'] for event in engine.fetch_features(request)
            if event['type'] == 'feature'
        ]
        if not batches:
            raise RuntimeError(
                f"DiskGraph returned no '{table_name}.{column}' values for "
                f'node IDs [{node_ids[0]}, {node_ids[-1]}]')
        values = pa.Table.from_batches(batches)
        if column not in values.column_names:
            raise RuntimeError(f"DiskGraph response for '{table_name}' "
                               f"omitted column '{column}'")
        if values.num_rows != len(node_ids):
            raise RuntimeError(
                f"DiskGraph returned {values.num_rows} '{table_name}.{column}' "
                f'values for {len(node_ids)} requested node IDs')
        yield chunk_start, values.column(column).combine_chunks()


def _node_column_engine(index_path: str) -> Any:
    from diskgraph import SamplerEngine
    return SamplerEngine(index_path, num_threads=cpu_budget())


def read_node_column(
    index_path: str,
    table_name: str,
    column: str,
    num_nodes: int,
    engine: Any | None = None,
) -> pa.ChunkedArray:
    """``column`` for node IDs ``[0, num_nodes)``, scanned on a multi-threaded
    engine."""
    chunks = [
        values for _, values in _iter_node_column(
            engine or _node_column_engine(index_path), table_name, column, 0,
            num_nodes)
    ]
    return pa.chunked_array(chunks, type=None if chunks else pa.null())


def iter_primary_key_batches(
    index_path: str,
    table_name: str,
    *,
    start: int = 0,
    stop: int,
    batch_size: int = 50_000,
    engine: Any | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield canonical node IDs and primary keys without graph sampling."""
    if start < 0 or stop < start:
        raise ValueError('primary-key range must satisfy 0 <= start <= stop')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')

    schema = load_diskgraph_schema(index_path)
    pkey_name = schema.pkey_dict.get(table_name)
    if not pkey_name:
        raise KeyError(
            f"DiskGraph table '{table_name}' has no primary-key column")
    decimal_primary_key_type = _decimal_primary_key_arrow_type(
        index_path,
        table_name,
        pkey_name,
    )

    feature_engine = engine or _node_column_engine(index_path)
    for chunk_start, primary_key_column in _iter_node_column(
            feature_engine, table_name, pkey_name, start, stop):
        if (decimal_primary_key_type is not None
                and primary_key_column.type != decimal_primary_key_type):
            try:
                primary_key_column = primary_key_column.cast(
                    decimal_primary_key_type)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
                raise RuntimeError(
                    f"DiskGraph returned invalid values for decimal primary "
                    f"key '{table_name}.{pkey_name}'") from error
        for offset in range(0, len(primary_key_column), batch_size):
            batch = primary_key_column.slice(offset, batch_size)
            node_ids = list(
                range(chunk_start + offset,
                      chunk_start + offset + len(batch)))
            canonical = np.asarray(
                feature_engine.canonical_node_mask(table_name, node_ids,
                                                   batch),
                dtype=np.bool_,
            )
            if canonical.shape != (len(node_ids), ):
                raise RuntimeError(
                    f"DiskGraph returned an invalid canonical-node mask for "
                    f"'{table_name}': {canonical.shape}")
            yield (np.asarray(node_ids, dtype=np.uint64)[canonical],
                   batch.to_pandas().to_numpy(copy=False)[canonical])
