# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal, anchor-independent DiskGraph index creation for ``prepare()``."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kumo.pytorch.column_stats import StatRequest, stats_request
from kumo.pytorch.contract._frozen import FrozenMapping
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.schema import DataSchema, RelationSpec, table_clock
from kumo.pytorch.dataset import Dataset, Parquet, _local_parquet_files
from kumo.pytorch.engines.unsupported import DISKGRAPH_UNSUPPORTED

_MANIFEST = "index_manifest.json"
_MANIFEST_VERSION = 3


@dataclass(frozen=True)
class _DiskGraphIndex:
    """Internal handle to a persistent DiskGraph index."""

    path: str
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(Path(self.path).resolve()))
        object.__setattr__(self, "manifest", FrozenMapping(dict(self.manifest)))

    @classmethod
    def open(cls, path: str | Path) -> _DiskGraphIndex:
        root = Path(path).resolve()
        manifest_path = root / _MANIFEST
        if not manifest_path.is_file():
            raise ValueError(
                f"{root} is missing the DiskGraph index manifest {_MANIFEST!r}; "
                "remove the directory to rebuild the index"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("manifest_version") != _MANIFEST_VERSION:
            raise ValueError(
                f"unsupported DiskGraph index manifest version {manifest.get('manifest_version')!r}"
            )
        if not (root / "index_metadata.json").is_file():
            raise ValueError(f"{root} is missing DiskGraph index_metadata.json")
        return cls(str(root), manifest)

    def validate_binding(self, schema: DataSchema, dataset: Dataset) -> None:
        expected_schema = _fingerprint(_schema_payload(schema))
        expected_dataset = _fingerprint(_dataset_payload(dataset))
        if self.manifest.get("schema_fingerprint") != expected_schema:
            raise ValueError("DiskGraph index was materialized from a different schema")
        if self.manifest.get("dataset_fingerprint") != expected_dataset:
            raise ValueError("DiskGraph index was materialized from a different dataset")


def ensure_diskgraph_index(
    schema: DataSchema,
    dataset: Dataset,
    destination: str | Path,
    *,
    num_shards: int = 1,
    num_threads: int | None = None,
    dataset_name: str = "default",
    if_missing: bool = True,
    column_stats: tuple[Sequence[StatRequest], Sequence[int | None]] | None = None,
) -> _DiskGraphIndex:
    """Reuse or build the base index while opening a DiskGraph engine.

    Accepts local Parquet with physical column names matching the logical
    schema and PK/FK relations with exactly one primary-key side.
    A matching index is reused. An incompatible or incomplete index is never
    overwritten. ``if_missing=False`` requires the index to exist already.
    An ingest also computes ``column_stats``: requests at cutoffs in epoch nanoseconds.
    """

    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if num_threads is not None and num_threads <= 0:
        raise ValueError("num_threads must be positive when supplied")
    if not dataset_name:
        raise ValueError("dataset_name must be non-empty")

    output = Path(destination).resolve()
    if output.exists():
        artifact = _DiskGraphIndex.open(output)
        artifact.validate_binding(schema, dataset)
        _warn_dropped_rows(output)
        return artifact
    if not if_missing:
        raise FileNotFoundError(f"DiskGraph index does not exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    diskgraph = _import_diskgraph()
    source_snapshot = _dataset_payload(dataset)
    config, relation_mapping, source_types = _compile_diskgraph_config(
        schema,
        dataset,
        num_shards=num_shards,
        dataset_name=dataset_name,
    )
    config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    stats_options = {}
    if column_stats is not None:
        requests, cutoffs = column_stats
        clock = config.get("timestamp_unit", "seconds")
        ticks = list(
            dict.fromkeys(
                None if cutoff is None else diskgraph.convert_ticks(cutoff, "nanoseconds", clock)
                for cutoff in cutoffs
            )
        )
        request = stats_request(requests, ticks, clock)
        stats_options = {"cutoff_col_stats": json.dumps(request)}
    started = time.perf_counter()
    thread_options = (
        {}
        if num_threads is None
        else {
            "context_threads": num_threads,
            "encode_threads": num_threads,
            "pk_map_threads": num_threads,
            "edges_map_threads": num_threads,
            "index_threads": num_threads,
        }
    )
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        build = Path(temporary) / "index"
        diskgraph.ingest(
            config_json,
            str(build),
            col_stats=False,
            cat_count_strategy="none",
            **stats_options,
            **thread_options,
        )
        elapsed = time.perf_counter() - started
        if _dataset_payload(dataset) != source_snapshot:
            raise ValueError("Parquet files changed while DiskGraph was ingesting them")

        manifest = {
            "manifest_version": _MANIFEST_VERSION,
            "artifact_kind": "diskgraph_base_index",
            "schema_fingerprint": _fingerprint(_schema_payload(schema)),
            "dataset_fingerprint": _fingerprint(source_snapshot),
            "config_fingerprint": hashlib.sha256(config_json.encode()).hexdigest(),
            "relation_mapping": relation_mapping,
            "column_types": {
                table.name: {name: asdict(dtype) for name, dtype in table.columns.items()}
                for table in sorted(schema.tables, key=lambda value: value.name)
            },
            "source_types": source_types,
            "num_shards": num_shards,
            "ingest_seconds": elapsed,
        }
        (build / _MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if output.exists():
            raise FileExistsError(f"DiskGraph index appeared during ingest: {output}")
        build.rename(output)
    _warn_dropped_rows(output)
    return _DiskGraphIndex(str(output), manifest)


def _warn_dropped_rows(root: Path) -> None:
    stats_file = root / "ingest_stats.json"
    if not stats_file.is_file():
        return
    counts = json.loads(stats_file.read_text()).get("stats", {}).get("dropped_rows_counts", {})
    dropped = {table: count for table, count in counts.items() if count}
    if dropped:
        warnings.warn(
            f"DiskGraph index {root} dropped source rows {dropped}; rows with NULL "
            "table clocks are excluded. Counts may include other ingest rejections; "
            "see ingest_stats.json.",
            UserWarning,
            stacklevel=3,
        )


def _import_diskgraph():
    try:
        import diskgraph
    except ImportError as error:  # pragma: no cover - depends on optional installation
        raise ImportError(
            "DiskGraph materialization requires the optional 'diskgraph' Python package"
        ) from error
    return diskgraph


def _compile_diskgraph_config(
    schema: DataSchema,
    dataset: Dataset,
    *,
    num_shards: int,
    dataset_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - installed with DiskGraph
        raise ImportError("DiskGraph materialization requires pyarrow") from error

    tables = {table.name: table for table in schema.tables}
    configs: dict[str, dict[str, Any]] = {}
    source_types: dict[str, Any] = {}
    for table in sorted(schema.tables, key=lambda value: value.name):
        source = dataset.tables.get(table.name)
        if not isinstance(source, Parquet):
            kind = None if source is None else source.kind
            raise TypeError(
                f"DiskGraph materialization requires Parquet for table {table.name!r}, got {kind!r}"
            )
        if source.columns:
            raise DISKGRAPH_UNSUPPORTED["DG-COLUMN-RENAME"].error(
                "DiskGraph materialization requires matching physical column names; "
                f"table {table.name!r} declares {dict(source.columns)!r}"
            )
        if "://" in source.path:
            raise DISKGRAPH_UNSUPPORTED["DG-REMOTE-PARQUET"].error(
                f"local Parquet paths are required; got {source.path!r}"
            )
        paths = _local_parquet_files(source.path)
        source_types[table.name] = {}
        for path in paths:
            try:
                physical = pq.read_schema(path)
            except Exception as error:
                raise ValueError(
                    f"table {table.name!r}: cannot read Parquet schema from {path}: {error}"
                ) from error
            missing = sorted(set(table.columns) - set(physical.names))
            if missing:
                raise ValueError(
                    f"Parquet table {table.name!r} in {path!r} is missing columns {missing}"
                )
            source_types[table.name][path] = {}
            for name, declared in table.columns.items():
                actual = physical.field(name).type
                try:
                    resolved = DType.from_arrow(actual)
                except (ValueError, TypeError) as error:
                    raise DISKGRAPH_UNSUPPORTED["DG-NESTED-COLUMN"].error(
                        f"bind {table.name}.{name} in {path!r}: "
                        f"unsupported physical type {actual}; "
                        f"provide a column matching {declared!r}"
                    ) from error
                if declared.temporal and pa.types.is_integer(actual):
                    raise DISKGRAPH_UNSUPPORTED["DG-INT64-CLOCK"].error(
                        f"bind {table.name}.{name} in {path!r}: declared {declared!r}, "
                        f"physical {actual} resolves to {resolved!r}"
                    )
                if resolved != declared:
                    raise ValueError(
                        f"bind {table.name}.{name} in {path!r}: declared {declared!r}, "
                        f"physical {actual} resolves to {resolved!r}; supply matching typed data"
                    )
                source_types[table.name][path][name] = str(actual)
        configs[table.name] = {
            "name": table.name,
            "columns": [
                {"name": name, **_diskgraph_dtype(pa, dtype.to_arrow())}
                for name, dtype in sorted(table.columns.items())
            ],
            "source": {"type": "parquet", "paths": paths},
            "fkeys": [],
        }
        if table.primary_key is not None:
            configs[table.name]["primary_key"] = table.primary_key
        clock = table_clock(table)
        if clock is not None:
            configs[table.name]["create_time"] = clock

    relation_mapping = {}
    for relation in sorted(schema.relations, key=lambda value: value.name):
        fkey_table, fkey_column, pk_table, logical_forward_is_physical = _foreign_key(
            relation, tables
        )
        pk_column = tables[pk_table].primary_key
        base = f"{fkey_table}_{fkey_column}_to_{pk_table}_{pk_column}"
        forward_time = _time_direction(tables[pk_table])
        reverse_time = _time_direction(tables[fkey_table])
        fkey: dict[str, Any] = {
            "src_column": fkey_column,
            "dst_table": pk_table,
            "bidirectional": True,
        }
        if forward_time is not None:
            fkey["forward"] = forward_time
        if reverse_time is not None:
            fkey["reverse"] = reverse_time
        configs[fkey_table]["fkeys"].append(fkey)

        physical_forward = base if logical_forward_is_physical else f"{base}_reverse"
        physical_reverse = f"{base}_reverse" if logical_forward_is_physical else base
        relation_mapping[relation.name] = {
            "left_table": relation.left_table,
            "right_table": relation.right_table,
            "forward": physical_forward,
            "reverse": physical_reverse,
        }

    return (
        {
            "dataset_name": dataset_name,
            "tables": [configs[name] for name in sorted(configs)],
            "num_shards": num_shards,
            "edge_format": "binary",
        },
        relation_mapping,
        source_types,
    )


def _foreign_key(
    relation: RelationSpec,
    tables: Mapping[str, Any],
) -> tuple[str, str, str, bool]:
    left_is_pk = relation.left_key == tables[relation.left_table].primary_key
    right_is_pk = relation.right_key == tables[relation.right_table].primary_key
    if left_is_pk == right_is_pk:
        raise ValueError(
            f"relation {relation.name!r} must have exactly one primary-key side for "
            "DiskGraph materialization"
        )
    if right_is_pk:
        return relation.left_table, relation.left_key, relation.right_table, True
    return relation.right_table, relation.right_key, relation.left_table, False


def _time_direction(table: Any) -> dict[str, Any] | None:
    clock = table_clock(table)
    if clock is None:
        return None
    return {
        "time": {
            "table": table.name,
            "column": clock,
            "unit": "s",
        }
    }


def _diskgraph_dtype(pa: Any, dtype: Any) -> dict[str, Any]:
    simple = (
        (pa.int8(), "int8"),
        (pa.int16(), "int16"),
        (pa.int32(), "int32"),
        (pa.int64(), "int64"),
        (pa.uint8(), "u_int8"),
        (pa.uint16(), "u_int16"),
        (pa.uint32(), "u_int32"),
        (pa.uint64(), "u_int64"),
        (pa.float32(), "float32"),
        (pa.float64(), "float64"),
        (pa.bool_(), "bool"),
        (pa.string(), "string"),
        (pa.binary(), "bytes"),
        (pa.date32(), "date"),
    )
    for arrow_type, name in simple:
        if dtype == arrow_type:
            return {"ingested_dtype": name}
    if pa.types.is_timestamp(dtype):
        return {"ingested_dtype": "timestamp"}
    if pa.types.is_decimal128(dtype):
        return {
            "ingested_dtype": "decimal128",
            "precision": dtype.precision,
            "scale": dtype.scale,
        }
    if pa.types.is_decimal256(dtype):
        return {
            "ingested_dtype": "decimal256",
            "precision": dtype.precision,
            "scale": dtype.scale,
        }
    raise TypeError(f"unsupported Parquet type for DiskGraph materialization: {dtype}")


def _schema_payload(schema: DataSchema) -> dict[str, Any]:
    return {
        "tables": [
            {
                "name": table.name,
                "primary_key": table.primary_key,
                "columns": {name: asdict(dtype) for name, dtype in sorted(table.columns.items())},
                "time_column": table_clock(table),
            }
            for table in sorted(schema.tables, key=lambda value: value.name)
        ],
        "relations": [
            asdict(relation) for relation in sorted(schema.relations, key=lambda value: value.name)
        ],
    }


def _dataset_payload(dataset: Dataset) -> dict[str, Any]:
    payload = {}
    for name, source in sorted(dataset.tables.items()):
        if not isinstance(source, Parquet):
            raise TypeError("DiskGraph index provenance only supports Parquet datasets")
        payload[name] = {
            **source.to_dict(),
            "files": [
                {
                    "path": path,
                    "size": Path(path).stat().st_size,
                    "mtime_ns": Path(path).stat().st_mtime_ns,
                }
                for path in _local_parquet_files(source.path)
            ],
        }
    return payload


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
