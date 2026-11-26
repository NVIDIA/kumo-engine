# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build native recent indexes with an exact canonical-key secondary order."""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.graph import utc_us
from tfm.data.native_source import KEY_ORDER
from tfm.extras.kumo_pytorch import native_api


def _clock_anchor(table: dict[str, Any], column: str) -> int:
    """Choose an exact one-second origin; never quantize or wrap timestamp keys."""
    pc: Any = importlib.import_module("pyarrow.compute")
    import pyarrow.parquet as pq

    bounds = []
    for path in table["source"]["paths"]:
        file = pq.ParquetFile(path)
        index = next(
            i for i in range(file.metadata.num_columns) if file.schema.column(i).path == column
        )
        for group in range(file.num_row_groups):
            statistics = file.metadata.row_group(group).column(index).statistics
            if statistics is not None and statistics.has_min_max:
                bounds.extend(
                    (utc_us(statistics.min) // 1_000_000, utc_us(statistics.max) // 1_000_000)
                )
            else:
                values = file.read_row_group(group, columns=[column]).column(column)
                limit = pc.min_max(values).as_py()
                bounds.extend(utc_us(limit[key]) // 1_000_000 for key in ("min", "max"))
    if not bounds:
        return 0
    lower, upper = min(bounds), max(bounds)
    if upper - lower > 2**32 - 1:
        raise UnsupportedCombinationError(
            f"{table['name']}.{column}: exact native canonical sampling needs a clock span below 2**32 seconds"
        )
    return (lower + upper + 1) // 2


class CanonicalDiskGraphEngine:
    """Preserve native clocks and sort equal-time edges by canonical source key.

    The source normalizer supplies a uint32 inverse canonical rank. DiskGraph
    packs it after the timestamp in its native edge sort key, before fanout is
    applied. It is metadata only; no encoder exposes it as a model feature.
    This adapter uses the matching pinned API's index compiler and public
    completed-index binding. It does not replace native sampling or batching.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self._dataset_snapshot: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.engine, name)

    @property
    def name(self) -> str:
        return str(self.engine.name)

    def _check_binding(self, dataset: Any, schema: Any) -> Any:
        if self._dataset_snapshot is not None and dataset is not None:
            materialization = importlib.import_module("kumo.pytorch.materialization")

            if materialization._dataset_payload(dataset) != self._dataset_snapshot:
                raise SchemaError("Canonical native index source binding changed")
            return self.engine._check_binding(None, schema)
        return self.engine._check_binding(dataset, schema)

    def _resolve_index_schema(self, *args: Any, **kwargs: Any) -> Any:
        return self.engine._resolve_index_schema(*args, **kwargs)

    def materialize_anchors(self, *args: Any, **kwargs: Any) -> Any:
        return self.engine.materialize_anchors(*args, **kwargs)

    def open(self, dataset: Any, schema: Any, **kwargs: Any) -> Any:
        """Ingest once, then use the API's verified native index source."""
        if dataset is None:
            return self.engine.open(dataset, schema, **kwargs)
        if self._dataset_snapshot is not None:
            self._check_binding(dataset, schema)
            return self.engine.open(None, schema, **kwargs)
        import diskgraph

        api = native_api()
        materialization = importlib.import_module("kumo.pytorch.materialization")

        index = Path(self.engine.index)
        if index.exists():
            raise ValueError("Canonical native index must use a fresh owned build")
        snapshot = materialization._dataset_payload(dataset)
        config, _relations, _types = materialization._compile_diskgraph_config(
            schema, dataset, num_shards=self.engine.num_shards, dataset_name="default"
        )
        anchors: dict[tuple[str, str], int] = {}
        for table in config["tables"]:
            for role in table["fkeys"]:
                for direction, destination in (
                    ("forward", role["dst_table"]),
                    ("reverse", table["name"]),
                ):
                    ordering = role.get(direction)
                    if ordering is None or "time" not in ordering:
                        continue
                    target = next(item for item in config["tables"] if item["name"] == destination)
                    column = next((c for c in target["columns"] if c["name"] == KEY_ORDER), None)
                    if column is None or column["ingested_dtype"] != "u_int32":
                        raise SchemaError(
                            f"{destination}: native canonical order requires uint32 key metadata"
                        )
                    clock = ordering["time"]["column"]
                    identity = (destination, clock)
                    if identity not in anchors:
                        anchors[identity] = _clock_anchor(target, clock)
                    ordering["time"].update(bitwidth=32, anchor_time=anchors[identity], unit="s")
                    ordering["weight"] = {
                        "table": destination,
                        "expr": KEY_ORDER,
                        "dtype": "uint32",
                    }
        threads = self.engine.num_threads
        thread_options = (
            {}
            if threads is None
            else {
                name: threads
                for name in (
                    "context_threads",
                    "encode_threads",
                    "pk_map_threads",
                    "edges_map_threads",
                    "index_threads",
                )
            }
        )
        started = time.perf_counter()
        diskgraph.ingest(
            json.dumps(config, sort_keys=True),
            str(index),
            col_stats=False,
            cat_count_strategy="none",
            index_build_id=self.engine.index_build_id,
            **thread_options,
        )
        if materialization._dataset_payload(dataset) != snapshot:
            raise SchemaError("Native source files changed during canonical ingestion")
        # Publish the pinned API's typed completion/binding metadata last,
        # after native ingestion and source verification have succeeded.
        manifest = {
            "manifest_version": materialization._MANIFEST_VERSION,
            "artifact_kind": "diskgraph_base_index",
            "schema_fingerprint": materialization._fingerprint(
                materialization._schema_payload(schema)
            ),
            "dataset_fingerprint": materialization._fingerprint(snapshot),
            "config_fingerprint": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
            "relation_mapping": _relations,
            "column_types": {
                table.name: {name: asdict(dtype) for name, dtype in table.columns.items()}
                for table in schema.tables
            },
            "source_types": _types,
            "num_shards": self.engine.num_shards,
            "ingest_seconds": time.perf_counter() - started,
            "graph_order": "canonical_key_asc_uint32_v1",
        }
        staged = index / "index_manifest.json.partial"
        staged.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        staged.replace(index / "index_manifest.json")
        self.engine = api.DiskGraphEngine.from_snapshot(
            SimpleNamespace(index_path=index, id=None), num_threads=threads
        )
        self._dataset_snapshot = snapshot
        return self.engine.open(None, schema, **kwargs)
