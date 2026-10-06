# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable TT-NG task-dataset specification and manifest materialization."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from kumo_core.encoder_selection import _EncoderStat, select_target_encoder
from kumoapi.encoder import EncoderType
from kumoapi.typing import ColStatType, Stype
from pydantic import TypeAdapter

_NODE_TASK_TYPES = frozenset({
    "binary_classification",
    "multiclass_classification",
    "multilabel_classification",
    "multilabel_ranking",
    "regression",
})
_LINK_TASK_TYPES = frozenset({
    "static_link_prediction",
    "temporal_link_prediction",
    "link_prediction",
})

_CATEGORICAL_TASK_TYPES = frozenset({
    "binary_classification",
    "multiclass_classification",
})
_MULTICATEGORICAL_TASK_TYPES = frozenset({
    "multilabel_classification",
    "multilabel_ranking",
})
_NUMERICAL_TASK_TYPES = frozenset({"regression", "forecasting"})
_TARGET_ENCODER_INPUT_FIELDS = frozenset(
    {"column_name", "dtype", "stype", "encoder", "stats"})

# Link-prediction targets are graph node IDs, not train-table columns encoded by
# Kumo. Keep their task-dataset stats bounded to the scalar fields consumed by
# runtime and postprocessing, independent of the number of distinct RHS nodes.
_LINK_TASK_STATS_FIELDS = (
    "version",
    "kind",
    "num_total_rows",
    "num_train_rows",
    "num_entities",
    "num_targets",
    "num_links",
    "num_target_rows",
    "num_timeframes",
    "num_timeframes_post_filter",
    "avg_entities_per_timeframe",
    "avg_targets_per_timeframe",
    "min_date",
    "max_date",
    "anchor_date",
)


@dataclass(frozen=True)
class DiskgraphTaskManifest:
    """Runtime task-dataset manifest fragments consumed by Kumo training.

    ``manifest_path`` identifies the sidecar index used for trainer metadata;
    ``runtime_manifest_path`` identifies the flat manifest consumed by the
    DiskGraph batch iterator.
    """

    manifest_path: str
    runtime_manifest_path: str
    manifest: Mapping[str, Any]
    metadata: Mapping[str, Any]
    feature_schema: Mapping[str, Any]
    graph_runtime_metadata: Mapping[str, Any]
    stats: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, manifest_path: str) -> "DiskgraphTaskManifest":
        manifest = _load_manifest_json(manifest_path)
        artifact_root = manifest.get("artifact_root") or os.path.dirname(
            os.path.abspath(manifest_path))

        def sidecar(key: str) -> Mapping[str, Any]:
            relative_path = manifest.get(key)
            if not relative_path:
                raise ValueError(f"manifest missing required {key}")
            return _load_manifest_json(
                os.path.join(artifact_root, str(relative_path)))

        def optional_sidecar(key: str) -> Mapping[str, Any]:
            relative_path = manifest.get(key)
            if not relative_path:
                return {}
            return _load_manifest_json(
                os.path.join(artifact_root, str(relative_path)))

        runtime_manifest_path = os.path.join(artifact_root,
                                             "runtime_manifest.json")
        if not os.path.isfile(runtime_manifest_path):
            raise ValueError("bundle missing runtime_manifest.json at "
                             f"{runtime_manifest_path}")

        return cls(
            manifest_path=manifest_path,
            runtime_manifest_path=runtime_manifest_path,
            manifest=manifest,
            metadata=sidecar("metadata_path"),
            feature_schema=sidecar("feature_schema_path"),
            graph_runtime_metadata=sidecar("graph_runtime_metadata_path"),
            stats=optional_sidecar("stats_path"),
        )

    @property
    def train_table_name(self) -> str:
        name = self.metadata.get("train_table_name")
        if not name:
            raise ValueError("metadata.train_table_name is required")
        return str(name)

    @property
    def seed_node_type(self) -> str:
        name = self.metadata.get("seed_node_type")
        if not name:
            raise ValueError("metadata.seed_node_type is required")
        return str(name)

    @property
    def target_node_type(self) -> str:
        """RHS entity node type, or the seed type for node-level tasks."""

        name = self.metadata.get("target_node_type")
        return str(name) if name else self.seed_node_type

    @property
    def target_to_entity_paths(self) -> tuple[tuple[tuple[str, str, str], ...],
                                              ...] | None:
        """The query's target->entity paths, or None if none were recorded.

        Bundles materialized before these were written report ``None`` so
        callers can fall back rather than read an empty list as "this task
        has no such path".
        """

        raw = self.metadata.get("target_to_entity_paths")
        if raw is None:
            return None
        return tuple(
            tuple((str(edge[0]), str(edge[1]), str(edge[2])) for edge in path)
            for path in raw)

    @property
    def task_type(self) -> str | None:
        value = self.metadata.get("task_type")
        return str(value) if value else None

    @property
    def train_table_columns(self) -> Mapping[str, Any]:
        return self.metadata.get("train_table_columns") or {}

    @property
    def stage_counts(self) -> Mapping[str, int]:
        """Return per-stage seed counts used for steps-per-epoch sizing."""

        split_info = self._split_info()
        counts = split_info.get("stage_counts") or {}
        return {str(name): int(value) for name, value in counts.items()}

    @property
    def stage_seed_times(self) -> Mapping[str, tuple[int, int]]:
        """Return temporal stage bounds as canonical integer pairs."""

        raw = self._split_info().get("stage_seed_times") or {}
        result: dict[str, tuple[int, int]] = {}
        for stage, bounds in raw.items():
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                result[str(stage)] = (int(bounds[0]), int(bounds[1]))
        return result

    def _split_info(self) -> Mapping[str, Any]:
        relative_path = self.manifest.get("split_info_path")
        if not relative_path:
            return {}
        artifact_root = self.manifest.get("artifact_root") or os.path.dirname(
            os.path.abspath(self.manifest_path))
        return _load_manifest_json(
            os.path.join(artifact_root, str(relative_path)))


def _load_manifest_json(path: str) -> Mapping[str, Any]:
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


@dataclass(frozen=True)
class TaskDatasetMaterializeSpec:
    tt_manifest_path: str
    job_id: str
    artifact_root: str
    task_family: str
    task_type: str
    is_temporal: bool
    seed_node_type: str
    train_table_name: str
    entity_table_name: str
    feature_schema: dict[str, Any]
    graph_node_types: tuple[str, ...]
    graph_edge_types: tuple[tuple[str, str, str], ...]
    rhs_node_type: str | None = None
    #: The query's target->entity paths, in `PredictiveQueryV2` order. The
    #: graph's edge list cannot stand in for them: for link prediction the
    #: first path runs from the *fact* table to the seed entity, which is
    #: neither the seed nor the RHS node type the manifest already names.
    target_to_entity_paths: tuple[tuple[tuple[str, str, str], ...], ...] = ()
    #: Stages the originating split requires to contain at least one row.
    required_stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task_family not in {
                "node_prediction", "link_prediction", "forecasting"
        }:
            raise ValueError(
                "task_family must be node_prediction, link_prediction, or "
                "forecasting")
        for label, value in (
            ("task_type", self.task_type),
            ("seed_node_type", self.seed_node_type),
            ("train_table_name", self.train_table_name),
            ("entity_table_name", self.entity_table_name),
        ):
            _require_bare_name(value, label)
        if self.task_family == "node_prediction":
            supported_task_types = _NODE_TASK_TYPES
        elif self.task_family == "link_prediction":
            supported_task_types = _LINK_TASK_TYPES
        else:
            supported_task_types = frozenset({"forecasting"})
        if self.task_type not in supported_task_types:
            raise ValueError(f"task_type {self.task_type!r} does not belong to "
                             f"task_family {self.task_family!r}")
        if not isinstance(self.is_temporal, bool):
            raise ValueError("is_temporal must be a bool")
        if self.rhs_node_type is not None:
            _require_bare_name(self.rhs_node_type, "rhs_node_type")
        if (len(set(self.required_stages)) != len(self.required_stages)
                or not set(self.required_stages) <= {"train", "val", "test"}):
            raise ValueError(
                "required_stages must contain unique train, val, or test "
                "stage names")
        for index, path in enumerate(self.target_to_entity_paths):
            location = f"target_to_entity_paths[{index}]"
            if len(path) != 1:
                raise ValueError(f"{location} must contain exactly one edge")
            source, relation, destination = path[0]
            _require_bare_name(source, f"{location}.source")
            _require_bare_name(relation, f"{location}.relation")
            _require_bare_name(destination, f"{location}.destination")
        for index, node_type in enumerate(self.graph_node_types):
            _require_bare_name(node_type, f"graph_node_types[{index}]")
        if len(set(self.graph_node_types)) != len(self.graph_node_types):
            raise ValueError("graph_node_types must be unique")
        if not isinstance(self.feature_schema, Mapping):
            raise ValueError("feature_schema must be a mapping")
        if set(self.feature_schema) != {"node_types", "target_encoder_input"}:
            raise ValueError(
                "feature_schema keys must exactly match node_types and "
                "target_encoder_input")
        target_input = self.feature_schema["target_encoder_input"]
        if target_input is not None and (
                not isinstance(target_input, Mapping)
                or set(target_input) != _TARGET_ENCODER_INPUT_FIELDS
                or not isinstance(target_input["stats"], Mapping)):
            raise ValueError(
                "feature_schema.target_encoder_input must be None or a "
                f"mapping with exactly {sorted(_TARGET_ENCODER_INPUT_FIELDS)!r}"
                " and a stats mapping")
        node_schema = self.feature_schema["node_types"]
        if not isinstance(node_schema, Mapping):
            raise ValueError("feature_schema.node_types must be a mapping")
        expected_feature_types = set(self.graph_node_types)
        if self.task_family == "forecasting":
            expected_feature_types.add(self.train_table_name)
        if set(node_schema) != expected_feature_types:
            raise ValueError(
                "feature_schema keys must exactly match the portable task "
                "node types")
        for node_type, feature_config in node_schema.items():
            if not isinstance(feature_config, Mapping):
                raise ValueError(
                    f"feature_schema[{node_type!r}] must be a mapping")
            expected_feature_fields = {
                "column_specs",
                "feature_columns",
                "mapper_key",
                "time_column",
            }
            if set(feature_config) != expected_feature_fields:
                raise ValueError(
                    f"feature_schema[{node_type!r}] fields must exactly match "
                    f"{sorted(expected_feature_fields)!r}")
            feature_columns = feature_config.get("feature_columns")
            if not isinstance(feature_columns, list) or not all(
                    isinstance(column, str) and column
                    for column in feature_columns):
                raise ValueError(
                    f"feature_schema[{node_type!r}].feature_columns must be "
                    "a list of non-empty strings")
            if feature_config.get("mapper_key") != node_type:
                raise ValueError(
                    f"feature_schema[{node_type!r}].mapper_key must exactly "
                    "match its graph node type")
            time_column = feature_config.get("time_column")
            if time_column is not None and (not isinstance(time_column, str)
                                            or not time_column):
                raise ValueError(
                    f"feature_schema[{node_type!r}].time_column must be None "
                    "or a non-empty string")
            column_specs = feature_config.get("column_specs")
            if not isinstance(column_specs, list):
                raise ValueError(
                    f"feature_schema[{node_type!r}].column_specs must be a list"
                )
            if [value.get("column_name")
                    for value in column_specs] != feature_columns:
                raise ValueError(
                    f"feature_schema[{node_type!r}].column_specs must exactly "
                    "match feature_columns")
        for index, edge_type in enumerate(self.graph_edge_types):
            if len(edge_type) != 3:
                raise ValueError(
                    f"graph_edge_types[{index}] must contain three names")
            for part_index, value in enumerate(edge_type):
                _require_bare_name(value,
                                   f"graph_edge_types[{index}][{part_index}]")
            if edge_type[0] not in self.graph_node_types or edge_type[
                    2] not in self.graph_node_types:
                raise ValueError(
                    f"graph_edge_types[{index}] must reference graph_node_types"
                )


def _require_bare_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if "." in value:
        raise ValueError(
            f"{label} must be an exact unqualified portable name; got {value!r}"
        )
    return value


@dataclass(frozen=True)
class TaskDatasetMaterializeResult:
    manifest_path: str
    runtime_manifest_path: str
    bundle_dir: str
    storage_kind: str = "diskgraph_table"


def _normalized_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _validate_binding(
    ingest_manifest: Mapping[str, Any],
    task_dataset: Mapping[str, Any],
    *,
    graph_index_path: str,
    extension_path: str,
    tt_id: str,
) -> None:
    expected_graph = _normalized_path(graph_index_path)
    expected_extension = _normalized_path(extension_path)

    def require_path(container: Mapping[str, Any], field: str,
                     expected: str) -> None:
        value = container.get(field)
        if not isinstance(value, str) or _normalized_path(value) != expected:
            raise RuntimeError(
                f"TT-NG ingest manifest binding mismatch for {field!r}; "
                f"expected {expected!r}, got {value!r}. Re-ingest this task "
                "dataset for the current graph snapshot.")

    if ingest_manifest.get("tt_id") != tt_id:
        raise RuntimeError(
            "TT-NG ingest manifest binding mismatch for 'tt_id'; "
            f"expected {tt_id!r}, got {ingest_manifest.get('tt_id')!r}")
    require_path(ingest_manifest, "graph_index_path", expected_graph)
    require_path(ingest_manifest, "tt_extension_path", expected_extension)
    require_path(task_dataset, "graph_index_path", expected_graph)
    require_path(task_dataset, "tt_extension_path", expected_extension)
    expected_manifest = _normalized_path(
        os.path.join(extension_path, "diskgraph_manifest.json"))
    manifest_path = ingest_manifest.get("diskgraph_manifest_path")
    if (not isinstance(manifest_path, str)
            or _normalized_path(manifest_path) != expected_manifest):
        raise RuntimeError("TT-NG ingest manifest binding mismatch for "
                           "'diskgraph_manifest_path'")
    stages = task_dataset.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise RuntimeError(
            "TT-NG ingest manifest is missing task_dataset.stages")
    for name, stage in stages.items():
        if not isinstance(stage, dict):
            raise RuntimeError(f"malformed task_dataset stage {name!r}")
        require_path(stage, "index_path", expected_extension)


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"JSON artifact at {path!r} is unreadable") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact at {path!r} must be an object")
    return value


def _parse_stats(path: str) -> dict[str, Any]:
    value = _read_json(path)
    if (value.get("version"), value.get("kind")) != (2, "tt_stats"):
        raise RuntimeError("stats artifact version/kind mismatch")
    for field_name in (
            "num_total_rows",
            "num_train_rows",
            "num_entities",
            "num_targets",
    ):
        if not isinstance(value.get(field_name), int):
            raise RuntimeError(
                f"stats artifact missing integer field: {field_name}")
    num_links = value.get("num_links")
    if num_links is not None and not isinstance(num_links, int):
        raise RuntimeError("stats artifact num_links must be int or null")
    return value


def _task_dataset_stats(task_type: str, stats: Mapping[str,
                                                       Any]) -> dict[str, Any]:
    """Project producer stats onto the task-specific runtime contract."""
    if task_type not in _LINK_TASK_TYPES:
        return dict(stats)
    return {
        field_name: stats[field_name]
        for field_name in _LINK_TASK_STATS_FIELDS if field_name in stats
    }


def _target_stats(path: str) -> dict[str, Any]:
    payload = _read_json(path)
    if (payload.get("version"), payload.get("kind"),
            payload.get("stats_format")) != (
                1,
                "tt_train_table_col_stats",
                "kumo_col_stats_v1",
            ):
        raise RuntimeError("train-table target stats artifact mismatch")
    raw_stats = payload.get("stats")
    if not isinstance(raw_stats, dict):
        raise RuntimeError("train-table target stats artifact has no stats")
    result = {}
    for key, raw_value in raw_stats.items():
        if not isinstance(key, str) or not key.startswith(
            ("TARGET_", "TARGET1_")):
            continue
        parts = key.rsplit("__", 2)
        if len(parts) != 3:
            continue
        stat_name = parts[-2]
        if not isinstance(raw_value, dict) or "value" not in raw_value:
            raise RuntimeError(f"train-table target stat {key!r} has no value")
        result[stat_name] = raw_value["value"]
    return result


def resolve_target_encoder_input(
    spec: TaskDatasetMaterializeSpec,
    ingest: Mapping[str, Any],
    extension: str,
) -> dict[str, Any] | None:
    if spec.task_type in _LINK_TASK_TYPES:
        return None
    descriptor = ingest.get("train_table_col_stats")
    configured_path = (descriptor.get("path") if isinstance(
        descriptor, Mapping) else None)
    stats_path = (configured_path if isinstance(configured_path, str) else
                  os.path.join(extension, "train_table_col_stats.json"))
    if not os.path.isabs(stats_path):
        stats_path = os.path.join(extension, stats_path)
    if not os.path.isfile(stats_path):
        raise RuntimeError(
            "non-link task is missing train_table_col_stats.json required "
            "to materialize the TARGET encoder")
    stats = _target_stats(stats_path)
    configured = spec.feature_schema.get("target_encoder_input")
    if configured is not None:  # hosted-built: add the stats it lacks
        return {
            **configured,
            "stats": {
                **stats,
                **configured["stats"]
            },
        }
    if spec.task_type in _CATEGORICAL_TASK_TYPES:
        dtype, stype = "string", "categorical"
    elif spec.task_type in _MULTICATEGORICAL_TASK_TYPES:
        dtype, stype = "stringlist", "multicategorical"
    elif spec.task_type in _NUMERICAL_TASK_TYPES:
        dtype, stype = "float", "numerical"
    else:
        raise RuntimeError(
            f"no TARGET encoder contract for task type {spec.task_type!r}")
    return {
        "column_name": "TARGET",
        "dtype": dtype,
        "stype": stype,
        "encoder": _default_target_encoder(stype, stats),
        "stats": stats,
    }


def _default_target_encoder(
    stype: str,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    known = {stat_type.value: stat_type for stat_type in ColStatType}
    try:
        encoder = select_target_encoder(
            Stype(stype), {
                known[name]: _EncoderStat(value=value, approx=False)
                for name, value in stats.items() if name in known
            })
    except ValueError as error:
        raise RuntimeError(
            f"TARGET encoder selection failed: {error}") from error
    config = TypeAdapter(EncoderType).dump_python(encoder, mode="json")
    config.pop("_stats", None)
    return config


def with_resolved_target_encoder_input(
    spec: TaskDatasetMaterializeSpec,
    ingest: Mapping[str, Any],
    extension: str,
) -> TaskDatasetMaterializeSpec:
    schema = dict(spec.feature_schema)
    schema["target_encoder_input"] = resolve_target_encoder_input(
        spec, ingest, extension)
    return replace(spec, feature_schema=schema)


def _write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=False)
        handle.write("\n")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_thin_manifest(
    spec: TaskDatasetMaterializeSpec,
    *,
    graph_index_id: str,
    graph_index_path: str,
    stats: dict[str, Any],
    stage_counts: dict[str, int],
    stage_table_refs: dict[str, dict[str, Any]],
    has_time: bool,
    has_weight: bool,
    stage_seed_times: dict[str, tuple[int, int]] | None = None,
) -> str:
    metadata = {
        "schema_version": "v1",
        "task_family": spec.task_family,
        "task_type": spec.task_type,
        "is_temporal": spec.is_temporal,
        "seed_node_type": spec.seed_node_type,
        "target_node_type": spec.rhs_node_type or spec.seed_node_type,
        "train_table_name": spec.train_table_name,
        "entity_table_name": spec.entity_table_name,
        "has_time": has_time,
        "has_weight": has_weight,
        "train_table_columns": {
            "seed_id_column":
            "ENTITY",
            "target_column":
            "TARGET",
            "time_column":
            "TIMESTAMP" if has_time else None,
            "weight_column":
            "WEIGHT" if has_weight else None,
            "foreign_key_columns": (["ENTITY", "TARGET"] if spec.task_family
                                    == "link_prediction" else ["ENTITY"]),
        },
        "available_side_artifacts": [],
        "runtime_capabilities": [],
        "feature_spec_version": "v1",
        "metadata": {},
    }

    # Written only when there are paths to record: the reader distinguishes
    # "this producer recorded none" from "this producer records none at
    # all", and persisting an empty list would silently retire the
    # edge-list fallback for every bundle a producer writes without them.
    if spec.target_to_entity_paths:
        metadata["target_to_entity_paths"] = [[list(edge) for edge in path]
                                              for path in
                                              spec.target_to_entity_paths]
    feature_schema = {
        **spec.feature_schema,
        "raw_feature_format": "arrow_columnar_v1",
        "provenance_mode": "node_id_time_keyed_v1",
    }
    graph_metadata = {
        "graph_index_id":
        graph_index_id,
        "graph_index_path":
        graph_index_path,
        "node_types":
        list(spec.graph_node_types),
        "edge_types": [{
            "src": source,
            "rel": relation,
            "dst": destination
        } for source, relation, destination in spec.graph_edge_types],
    }
    split_info: dict[str, Any] = {
        "split_kind": "precomputed",
        "stage_counts": stage_counts,
    }
    if stage_seed_times is not None:
        split_info["stage_seed_times"] = stage_seed_times
    manifest = {
        "schema_version": "v1",
        "task_family": spec.task_family,
        "task_type": spec.task_type,
        "is_temporal": spec.is_temporal,
        "artifact_root": os.path.abspath(spec.artifact_root),
        "metadata_path": "metadata.json",
        "feature_schema_path": "feature_schema.json",
        "graph_runtime_metadata_path": "graph_runtime_metadata.json",
        "split_info_path": "split_info.json",
        "stats_path": "stats.json",
        "ingested_stage_tables": stage_table_refs,
        "feature_payload_root": None,
        "producer_summary_path": None,
        "downsampling_metadata_path": None,
        "side_artifacts": {},
        "runtime_capabilities": [],
    }

    artifact_root = os.path.abspath(spec.artifact_root)
    os.makedirs(os.path.dirname(artifact_root), exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".manifest_staging_",
                               dir=os.path.dirname(artifact_root))
    try:
        sidecars = {
            "metadata.json": metadata,
            "feature_schema.json": feature_schema,
            "graph_runtime_metadata.json": graph_metadata,
            "split_info.json": split_info,
            "stats.json": stats,
        }
        checksums = {}
        for name, value in sidecars.items():
            path = os.path.join(staging, name)
            _write_json(path, value)
            checksums[f"{name}_sha256"] = _sha256(path)
        _write_json(os.path.join(staging, "checksums.json"), checksums)
        manifest["checksums_path"] = "checksums.json"
        _write_json(os.path.join(staging, "manifest.json"), manifest)
        if os.path.exists(artifact_root):
            shutil.rmtree(artifact_root)
        os.rename(staging, artifact_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return os.path.join(artifact_root, "manifest.json")


def write_flat_manifest(
    spec: TaskDatasetMaterializeSpec,
    *,
    bundle_dir: str,
    graph_index_path: str,
    stage_index_path: str,
    stage_counts: dict[str, int],
    stage_table_refs: dict[str, dict[str, Any]],
    has_time: bool,
    has_weight: bool,
) -> str:
    def stage_value(stage: str, count: int) -> dict[str, Any]:
        reference = stage_table_refs.get(stage, {})
        value = {
            "table_name": reference.get("table_name"),
            "index_path": stage_index_path,
            "row_count": count,
        }
        for optional in ("binary_label_index", "sample_weight_index"):
            if reference.get(optional):
                value[optional] = reference[optional]
        return value

    payload = {
        "schema_version": "kumo.diskgraph_task_dataset_manifest.v1",
        "kind": "diskgraph_task_dataset",
        "task": {
            "task_type": spec.task_type,
            "is_temporal": spec.is_temporal,
        },
        "table": {
            "train_table_name": spec.train_table_name,
            "seed_entity_table": spec.entity_table_name,
            "columns": {
                "primary_entity": "ENTITY1",
                "primary_target": "TARGET1",
                "timestamp": "TIMESTAMP" if has_time else None,
                "weight": "WEIGHT" if has_weight else None,
            },
        },
        "graph": {
            "index_path": graph_index_path,
            "node_types": list(spec.graph_node_types),
            "edge_types": [list(edge) for edge in spec.graph_edge_types],
        },
        "features": {
            **spec.feature_schema,
            "raw_feature_format": "arrow_columnar_v1",
            "provenance_mode": "node_id_time_keyed_v1",
        },
        "stages": {
            stage: stage_value(stage, count)
            for stage, count in stage_counts.items()
        },
    }
    target = os.path.join(os.path.abspath(bundle_dir), "runtime_manifest.json")
    descriptor, temporary = tempfile.mkstemp(
        prefix=".runtime_manifest_",
        suffix=".json.tmp",
        dir=os.path.dirname(target),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
        os.rename(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def materialize_task_dataset(
    *,
    graph_index_id: str,
    graph_index_path: str,
    spec: TaskDatasetMaterializeSpec,
) -> TaskDatasetMaterializeResult:
    """Materialize trainer manifests from the authoritative ingest extension."""

    extension = os.path.dirname(spec.tt_manifest_path)
    ingest_path = os.path.join(extension, "diskgraph_manifest.json")
    stats_path = os.path.join(extension, "stats.json")
    if not os.path.exists(ingest_path):
        raise RuntimeError(f"TT-NG ingest manifest missing at {ingest_path}")
    ingest = _read_json(ingest_path)
    task_dataset = ingest.get("task_dataset")
    if not isinstance(task_dataset, dict):
        raise RuntimeError("TT-NG ingest manifest is missing ``task_dataset``")
    _validate_binding(
        ingest,
        task_dataset,
        graph_index_path=graph_index_path,
        extension_path=extension,
        tt_id=os.path.basename(os.path.normpath(extension)),
    )
    spec = with_resolved_target_encoder_input(spec, ingest, extension)
    if not os.path.exists(stats_path):
        raise RuntimeError(
            f"TT-NG ingest stats sidecar missing at {stats_path}")
    stats = _task_dataset_stats(spec.task_type, _parse_stats(stats_path))

    stages = task_dataset.get("stages")
    if not isinstance(stages, dict) or not stages:
        raise RuntimeError("TT manifest task_dataset.stages empty")
    counts: dict[str, int] = {}
    seed_times: dict[str, tuple[int, int]] = {}
    references: dict[str, dict[str, Any]] = {}
    for stage, value in stages.items():
        if stage == "unused" or not isinstance(value, dict):
            continue
        count = value.get("row_count")
        if isinstance(count, int):
            counts[stage] = count
        seed_time = value.get("seed_time")
        if (isinstance(seed_time, (list, tuple)) and len(seed_time) == 2
                and all(isinstance(item, int) for item in seed_time)):
            seed_times[stage] = (seed_time[0], seed_time[1])
        table_name = value.get("table_name")
        index_path = value.get("index_path")
        if not isinstance(table_name, str) or not table_name:
            raise RuntimeError(
                f"task_dataset stage {stage!r} has no table_name")
        if not isinstance(index_path, str) or not index_path:
            raise RuntimeError(
                f"task_dataset stage {stage!r} has no index_path")
        reference = {
            "table_name": table_name,
            "index_path": index_path,
            "storage_kind": "diskgraph_table",
        }
        for optional in ("binary_label_index", "sample_weight_index"):
            if isinstance(value.get(optional), dict):
                reference[optional] = value[optional]
        references[stage] = reference
    if not counts:
        raise RuntimeError("TT manifest stages have no row counts")
    missing_stages = [
        stage for stage in spec.required_stages if stage not in counts
    ]
    if missing_stages:
        rendered = ", ".join(repr(stage) for stage in missing_stages)
        raise RuntimeError(
            f"TT manifest required stage(s) {rendered} have no row counts")
    empty_stages = [
        stage for stage in spec.required_stages if counts[stage] == 0
    ]
    if empty_stages:
        rendered = ", ".join(repr(stage) for stage in empty_stages)
        raise RuntimeError(
            f"TT manifest required stage(s) {rendered} have no rows")
    columns = task_dataset.get("columns") or {}
    has_time = bool(columns.get("timestamp"))
    has_weight = bool(columns.get("weight"))
    stage_index_path = task_dataset.get("tt_extension_path")
    if not isinstance(stage_index_path, str) or not stage_index_path:
        raise RuntimeError("task_dataset has no tt_extension_path")

    manifest_path = write_thin_manifest(
        spec,
        graph_index_id=graph_index_id,
        graph_index_path=graph_index_path,
        stats=stats,
        stage_counts=counts,
        stage_seed_times=seed_times or None,
        stage_table_refs=references,
        has_time=has_time,
        has_weight=has_weight,
    )
    bundle_dir = os.path.dirname(manifest_path)
    runtime_path = write_flat_manifest(
        spec,
        bundle_dir=bundle_dir,
        graph_index_path=graph_index_path,
        stage_index_path=stage_index_path,
        stage_counts=counts,
        stage_table_refs=references,
        has_time=has_time,
        has_weight=has_weight,
    )
    return TaskDatasetMaterializeResult(
        manifest_path=manifest_path,
        runtime_manifest_path=runtime_path,
        bundle_dir=bundle_dir,
    )
