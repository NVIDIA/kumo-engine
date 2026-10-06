# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-dataset bundle writer (mock).

Pure-Python implementation of the post-rework bundle layout from
`runtime-dataset-contracts.md` §Contract 1.  The bundle wraps a TT
in-memory or on-disk plus split bytes, and produces one table-ref per
stage under `ingested_stage_tables`.  In the mock each table-ref uses
`storage_kind="mock_parquet"` and points to a parquet file.  The real
implementation is expected to emit equivalent stage table refs with
`storage_kind="diskgraph_table"`, where `table_name` is the DG-ingested
stage table name reopened by the runtime.

Bundle layout:

    <artifact_root>/
      manifest.json
      metadata.json
      split_info.json
      stats.json
      feature_schema.json
      graph_runtime_metadata.json
      ingested_stage_tables/
        <table_name>__train.parquet
        <table_name>__val.parquet
        <table_name>__test.parquet
      pre_ingest/
        train_table.parquet           (optional, for inspection)
        split_assignments.parquet     (optional, for inspection)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ._exceptions import DiskgraphConfigError, DiskgraphDataError, DiskgraphSystemError

TASK_DATASET_SCHEMA_VERSION = "v1"
TASK_DATASET_IMPLEMENTATION_KIND = "python_mock"

# Stage byte vocabulary, matching the real diskgraph TT pipeline.
_STAGE_BYTE_TRAIN = 0
_STAGE_BYTE_VAL = 1
_STAGE_BYTE_TEST = 2
_STAGE_BYTE_UNUSED = 255

_STAGE_BYTE_TO_NAME = {
    _STAGE_BYTE_TRAIN: "train",
    _STAGE_BYTE_VAL: "val",
    _STAGE_BYTE_TEST: "test",
}

_VALID_SPLIT_BYTES = {0, 1, 2, _STAGE_BYTE_UNUSED}

_SCALAR_NODE_TASK_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "regression",
}


@dataclass
class TrainTableColumnMappingInputs:
    """Per-family TT column mapping recorded on `metadata.json`.
    Defaults match the canonical `ENTITY` / `TIMESTAMP` / `TARGET`
    column names; each ingested stage table carries the same names so the
    runtime knows which columns are seed_id / target / time / weight /
    pkey.  In the mock, those stage tables are parquet-backed."""

    seed_id_column: str = "ENTITY"
    target_column: str = "TARGET"
    time_column: Optional[str] = "TIMESTAMP"
    weight_column: Optional[str] = None
    pkey_column: Optional[str] = None


@dataclass
class TaskDatasetMetadataInputs:
    """Caller-supplied metadata.  Schema version, task family, and row
    count are filled in by the writer."""

    task_type: str
    is_temporal: bool
    feature_spec_version: str = "v1"
    seed_node_type: Optional[str] = None
    target_node_type: Optional[str] = None
    train_table_name: Optional[str] = None
    entity_table_name: Optional[str] = None
    has_time: bool = False
    has_weight: bool = False
    train_table_columns: TrainTableColumnMappingInputs = field(
        default_factory=TrainTableColumnMappingInputs
    )
    available_side_artifacts: list[str] = field(default_factory=list)
    runtime_capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": TASK_DATASET_SCHEMA_VERSION,
                "task_family": "node_prediction",
                "task_type": self.task_type,
                "is_temporal": self.is_temporal,
                "row_count": None,  # writer overwrites
                "seed_node_type": self.seed_node_type,
                "target_node_type": self.target_node_type,
                "train_table_name": self.train_table_name,
                "entity_table_name": self.entity_table_name,
                "has_time": self.has_time,
                "has_weight": self.has_weight,
                "train_table_columns": {
                    "seed_id_column": self.train_table_columns.seed_id_column,
                    "target_column": self.train_table_columns.target_column,
                    "time_column": self.train_table_columns.time_column,
                    "weight_column": self.train_table_columns.weight_column,
                    "pkey_column": self.train_table_columns.pkey_column,
                },
                "available_side_artifacts": self.available_side_artifacts,
                "runtime_capabilities": self.runtime_capabilities,
                "feature_spec_version": self.feature_spec_version,
                "metadata": self.metadata,
            }
        )


@dataclass
class SplitInfoInputs:
    split_kind: str
    normalized_split_config: dict[str, Any]
    stage_metadata: dict[str, Any] = field(default_factory=dict)
    adjusted_semantics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        # `stage_counts` is filled in by the writer.
        return json.dumps(
            {
                "split_kind": self.split_kind,
                "normalized_split_config": self.normalized_split_config,
                "stage_counts": {},
                "stage_metadata": self.stage_metadata,
                "adjusted_semantics": self.adjusted_semantics,
            }
        )


@dataclass
class FeatureSchemaInputs:
    node_types: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_feature_format: str = "arrow_columnar_v1"
    provenance_mode: str = "node_id_time_keyed_v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "node_types": self.node_types,
                "raw_feature_format": self.raw_feature_format,
                "provenance_mode": self.provenance_mode,
                "metadata": self.metadata,
            }
        )


@dataclass
class GraphRuntimeMetadataInputs:
    graph_index_id: str
    node_types: list[str] = field(default_factory=list)
    edge_types: list[Mapping[str, str]] = field(default_factory=list)
    node_type_facts: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "graph_index_id": self.graph_index_id,
                "node_types": list(self.node_types),
                "edge_types": [dict(t) for t in self.edge_types],
                "node_type_facts": self.node_type_facts,
                "metadata": self.metadata,
            }
        )


def task_dataset_artifact_root(
    *, graph_index_root: str, task_dataset_name: str
) -> str:
    """Return the intended task-dataset bundle root under
    `graph_index_root`.  Production callers should colocate task-dataset
    bundles under the associated graph/index root, as siblings of the
    DG index files: `<graph_index_root>/<task_dataset_name>/`."""
    if not graph_index_root:
        raise ValueError("graph_index_root is required")
    if not task_dataset_name:
        raise ValueError("task_dataset_name is required")
    if (
        task_dataset_name in {".", ".."}
        or os.sep in task_dataset_name
        or "/" in task_dataset_name
    ):
        raise ValueError(
            "task_dataset_name must be a simple child directory name, not a path"
        )
    return os.path.join(graph_index_root, task_dataset_name)


def build_node_prediction_task_dataset_bundle(
    *,
    artifact_root: str,
    task_type: str,
    seed_node_ids: Any,
    targets: Any,
    split_bytes: Any,
    seed_times: Any = None,
    weights: Any = None,
    pkeys: Any = None,
    metadata: TaskDatasetMetadataInputs,
    split_info: SplitInfoInputs,
    stats: dict[str, Any],
    feature_schema: FeatureSchemaInputs,
    graph_runtime_metadata: GraphRuntimeMetadataInputs,
    producer_summary: Optional[dict[str, Any]] = None,
    downsampling: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Write a node-prediction task-dataset bundle to ``artifact_root``.

    `artifact_root` must not exist.  On success returns
    `{"artifact_root": str, "manifest_path": str}`.

    Per the post-rework contract the runtime substrate is the
    **per-split ingested DG tables**.  The manifest records stage -> table-ref
    objects under `ingested_stage_tables`.  In the mock those table refs use
    `storage_kind="mock_parquet"` and point at plain parquet files, one per
    stage.  In the real implementation they should use
    `storage_kind="diskgraph_table"` and `table_name` should be the actual
    DG-ingested stage table.  `node_id` semantics is **table-local**: the row
    position within the ingested table.

    `UNUSED_STAGE_ID` (255) rows are dropped — they are not part of
    any iterable stage.
    """
    if not _is_supported_scalar_node_task_type(task_type):
        raise DiskgraphConfigError(
            f"task_type {task_type!r} is not supported by the scalar "
            "node-prediction task-dataset bundle"
        )

    if metadata.task_type != task_type:
        raise DiskgraphConfigError(
            "metadata.task_type must match task_type "
            f"(got metadata.task_type={metadata.task_type!r}, task_type={task_type!r})"
        )

    if not graph_runtime_metadata.graph_index_id:
        raise DiskgraphConfigError(
            "graph_runtime_metadata.graph_index_id is required"
        )

    if os.path.exists(artifact_root):
        raise DiskgraphConfigError(
            f"artifact_root {artifact_root!r} already exists; refusing to overwrite"
        )

    cols = metadata.train_table_columns
    if metadata.is_temporal or metadata.has_time:
        if seed_times is None:
            raise DiskgraphConfigError(
                "seed_times is required when metadata.is_temporal or "
                "metadata.has_time is true"
            )
        if not cols.time_column:
            raise DiskgraphConfigError(
                "metadata.train_table_columns.time_column is required when "
                "temporal/time metadata is true"
            )
    if seed_times is not None and not cols.time_column:
        raise DiskgraphConfigError(
            "seed_times was supplied but metadata.train_table_columns.time_column "
            "is empty"
        )
    if metadata.has_weight:
        if weights is None:
            raise DiskgraphConfigError(
                "weights is required when metadata.has_weight is true"
            )
        if not cols.weight_column:
            raise DiskgraphConfigError(
                "metadata.train_table_columns.weight_column is required when "
                "metadata.has_weight is true"
            )
    if weights is not None and not cols.weight_column:
        raise DiskgraphConfigError(
            "weights was supplied but metadata.train_table_columns.weight_column "
            "is empty"
        )

    seed_node_ids_arr = _as_int64_array("seed_node_ids", seed_node_ids)
    targets_arr = _as_float64_array("targets", targets)
    n = len(seed_node_ids_arr)
    if len(targets_arr) != n:
        raise DiskgraphDataError(
            f"targets length {len(targets_arr)} != seed_node_ids length {n}"
        )

    seed_times_arr = (
        _as_int64_array("seed_times", seed_times) if seed_times is not None else None
    )
    if seed_times_arr is not None and len(seed_times_arr) != n:
        raise DiskgraphDataError(
            f"seed_times length {len(seed_times_arr)} != seed_node_ids length {n}"
        )

    weights_arr = _as_float64_array("weights", weights) if weights is not None else None
    if weights_arr is not None and len(weights_arr) != n:
        raise DiskgraphDataError(
            f"weights length {len(weights_arr)} != seed_node_ids length {n}"
        )

    pkeys_arr = _as_int64_array("pkeys", pkeys) if pkeys is not None else None
    if pkeys_arr is not None and len(pkeys_arr) != n:
        raise DiskgraphDataError(
            f"pkeys length {len(pkeys_arr)} != seed_node_ids length {n}"
        )

    split_bytes_b = _as_split_bytes("split_bytes", split_bytes)
    if len(split_bytes_b) != n:
        raise DiskgraphDataError(
            f"split_bytes length {len(split_bytes_b)} != seed_node_ids length {n}"
        )

    staging = artifact_root + ".staging"
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging, exist_ok=False)

    try:
        # Per-stage table refs under ingested_stage_tables/. The mock backs
        # each stage table with parquet; the real implementation should use
        # storage_kind="diskgraph_table" for DG-ingested stage tables.
        ingested_dir = os.path.join(staging, "ingested_stage_tables")
        os.makedirs(ingested_dir, exist_ok=False)
        stage_counts: dict[str, int] = {}
        ingested_stage_tables: dict[str, dict[str, str]] = {}
        valid_count = 0
        for byte, name in _STAGE_BYTE_TO_NAME.items():
            mask = (np.frombuffer(split_bytes_b, dtype=np.uint8) == byte)
            stage_count = int(mask.sum())
            stage_counts[name] = stage_count
            if stage_count == 0:
                continue
            valid_count += stage_count
            table_name = f"{metadata.train_table_name or 'task_dataset'}__{name}"
            stage_path = os.path.join(ingested_dir, f"{table_name}.parquet")
            _write_stage_parquet(
                path=stage_path,
                seed_node_ids=seed_node_ids_arr[mask],
                targets=targets_arr[mask],
                seed_times=seed_times_arr[mask] if seed_times_arr is not None else None,
                weights=weights_arr[mask] if weights_arr is not None else None,
                pkeys=pkeys_arr[mask] if pkeys_arr is not None else None,
                source_row_ids=np.flatnonzero(mask).astype(np.int64),
                cols=cols,
            )
            ingested_stage_tables[name] = {
                "table_name": table_name,
                "storage_kind": "mock_parquet",
                "path": os.path.relpath(stage_path, staging),
            }

        unused_count = int(
            (np.frombuffer(split_bytes_b, dtype=np.uint8) == _STAGE_BYTE_UNUSED).sum()
        )
        if unused_count > 0:
            stage_counts["unused"] = unused_count

        # Pre-ingest substrate (optional, inspection only).  Useful for
        # tooling that wants to look at the unsplit TT or the raw split
        # assignments later.  Cheap: same parquet schema, written once.
        pre_ingest_dir = os.path.join(staging, "pre_ingest")
        os.makedirs(pre_ingest_dir, exist_ok=False)
        _write_stage_parquet(
            path=os.path.join(pre_ingest_dir, "train_table.parquet"),
            seed_node_ids=seed_node_ids_arr,
            targets=targets_arr,
            seed_times=seed_times_arr,
            weights=weights_arr,
            pkeys=pkeys_arr,
            source_row_ids=np.arange(n, dtype=np.int64),
            cols=cols,
        )

        # Metadata.
        metadata_dict = json.loads(metadata.to_json())
        metadata_dict["row_count"] = valid_count
        metadata_dict["has_time"] = seed_times_arr is not None
        metadata_dict["has_weight"] = weights_arr is not None
        metadata_dict.setdefault("metadata", {})["implementation_kind"] = (
            TASK_DATASET_IMPLEMENTATION_KIND
        )
        with open(os.path.join(staging, "metadata.json"), "w") as f:
            json.dump(metadata_dict, f, indent=2)

        split_info_dict = json.loads(split_info.to_json())
        split_info_dict["stage_counts"] = stage_counts
        with open(os.path.join(staging, "split_info.json"), "w") as f:
            json.dump(split_info_dict, f, indent=2)

        with open(os.path.join(staging, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        with open(os.path.join(staging, "feature_schema.json"), "w") as f:
            f.write(feature_schema.to_json())

        with open(os.path.join(staging, "graph_runtime_metadata.json"), "w") as f:
            f.write(graph_runtime_metadata.to_json())

        # Optional durability companions.
        producer_summary_rel = None
        if producer_summary is not None:
            p = os.path.join(staging, "producer_summary.json")
            with open(p, "w") as f:
                json.dump(producer_summary, f, indent=2)
            producer_summary_rel = "producer_summary.json"
        downsampling_rel = None
        if downsampling is not None:
            p = os.path.join(staging, "downsampling.json")
            with open(p, "w") as f:
                json.dump(downsampling, f, indent=2)
            downsampling_rel = "downsampling.json"

        manifest = {
            "schema_version": TASK_DATASET_SCHEMA_VERSION,
            "implementation_kind": TASK_DATASET_IMPLEMENTATION_KIND,
            "task_family": "node_prediction",
            "task_type": task_type,
            "is_temporal": metadata.is_temporal,
            "artifact_root": artifact_root,
            "metadata_path": "metadata.json",
            "split_info_path": "split_info.json",
            "stats_path": "stats.json",
            "feature_schema_path": "feature_schema.json",
            "graph_runtime_metadata_path": "graph_runtime_metadata.json",
            "ingested_stage_tables": ingested_stage_tables,
            "feature_payload_root": None,
            "checksums_path": "checksums.json",
            "producer_summary_path": producer_summary_rel,
            "downsampling_metadata_path": downsampling_rel,
            "side_artifacts": {},
            "runtime_capabilities": [],
        }
        manifest_path = os.path.join(staging, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Checksums (sha256) over the bundle artifacts.  Reopen verifies only
        # small control files by default; parquet/stage-table hashes are
        # informational unless full verification is explicitly requested.
        checksums: dict[str, str] = {}
        for rel in [
            "manifest.json",
            "metadata.json",
            "split_info.json",
            "stats.json",
            "feature_schema.json",
            "graph_runtime_metadata.json",
        ]:
            checksums[rel] = _sha256_of_file(os.path.join(staging, rel))
        for stage_ref in ingested_stage_tables.values():
            rel = stage_ref["path"]
            checksums[rel] = _sha256_of_file(os.path.join(staging, rel))
        with open(os.path.join(staging, "checksums.json"), "w") as f:
            json.dump(checksums, f, indent=2)

        # Atomic promote.
        os.rename(staging, artifact_root)
    except Exception:
        if os.path.exists(staging):
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "artifact_root": artifact_root,
        "manifest_path": os.path.join(artifact_root, "manifest.json"),
    }


def _write_stage_parquet(
    *,
    path: str,
    seed_node_ids: np.ndarray,
    targets: np.ndarray,
    seed_times: Optional[np.ndarray],
    weights: Optional[np.ndarray],
    pkeys: Optional[np.ndarray],
    source_row_ids: np.ndarray,
    cols: TrainTableColumnMappingInputs,
) -> None:
    """Write one stage's rows to a parquet using the canonical column
    names recorded on `metadata.train_table_columns`.  In the mock this is
    the parquet backing for one ingested stage table.  Also stamps a
    `__source_row_id__` column carrying the pre-ingest TT row position
    (= `input_id` in the lineage sense), per-row aligned with the
    stage's slice."""
    fields = [pa.field(cols.seed_id_column, pa.int64(), nullable=False)]
    arrays: list[pa.Array] = [pa.array(seed_node_ids, type=pa.int64())]
    if seed_times is not None and cols.time_column:
        fields.append(pa.field(cols.time_column, pa.int64(), nullable=False))
        arrays.append(pa.array(seed_times, type=pa.int64()))
    fields.append(pa.field(cols.target_column, pa.float64(), nullable=False))
    arrays.append(pa.array(targets, type=pa.float64()))
    if weights is not None and cols.weight_column:
        fields.append(pa.field(cols.weight_column, pa.float64(), nullable=True))
        arrays.append(pa.array(weights, type=pa.float64()))
    if pkeys is not None and cols.pkey_column:
        fields.append(pa.field(cols.pkey_column, pa.int64(), nullable=True))
        arrays.append(pa.array(pkeys, type=pa.int64()))
    fields.append(pa.field("__source_row_id__", pa.int64(), nullable=False))
    arrays.append(pa.array(source_row_ids, type=pa.int64()))

    schema = pa.schema(fields)
    table = pa.Table.from_arrays(arrays, schema=schema)
    pq.write_table(table, path)


def _is_supported_scalar_node_task_type(task_type: str) -> bool:
    return task_type.lower() in _SCALAR_NODE_TASK_TYPES


def _as_int64_array(name: str, a: Any) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise DiskgraphDataError(
            f"{name} must be 1-D, got shape {arr.shape}"
        )
    if arr.dtype.kind not in ("i", "u"):
        raise DiskgraphDataError(
            f"{name} must be an integer array; got dtype {arr.dtype}. "
            "Convert upstream to an integer dtype — float / object / "
            "bool inputs would be silently truncated."
        )
    if arr.dtype.kind == "u" and arr.size and int(arr.max()) > np.iinfo(np.int64).max:
        raise DiskgraphDataError(f"{name} contains values exceeding int64 max")
    return arr.astype(np.int64, copy=False)


def _as_float64_array(name: str, a: Any) -> np.ndarray:
    arr = np.asarray(a).astype(np.float64, copy=False)
    if arr.ndim != 1:
        raise DiskgraphDataError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _as_split_bytes(name: str, a: Any) -> bytes:
    if isinstance(a, (bytes, bytearray)):
        out = bytes(a)
        bad = next((v for v in out if v not in _VALID_SPLIT_BYTES), None)
        if bad is not None:
            raise DiskgraphDataError(
                f"{name} contains invalid stage byte {bad}; expected one of "
                "0, 1, 2, or 255"
            )
        return out
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise DiskgraphDataError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype.kind not in ("i", "u"):
        raise DiskgraphDataError(f"{name} must contain integer stage bytes")
    bad = next((int(v) for v in arr.tolist() if int(v) not in _VALID_SPLIT_BYTES), None)
    if bad is not None:
        raise DiskgraphDataError(
            f"{name} contains invalid stage byte {bad}; expected one of "
            "0, 1, 2, or 255"
        )
    return arr.astype(np.uint8, copy=False).tobytes()


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "TASK_DATASET_SCHEMA_VERSION",
    "TASK_DATASET_IMPLEMENTATION_KIND",
    "TaskDatasetMetadataInputs",
    "TrainTableColumnMappingInputs",
    "SplitInfoInputs",
    "FeatureSchemaInputs",
    "GraphRuntimeMetadataInputs",
    "build_node_prediction_task_dataset_bundle",
    "task_dataset_artifact_root",
]
