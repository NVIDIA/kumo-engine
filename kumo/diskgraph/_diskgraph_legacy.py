# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform compatibility adapters for the portable DiskGraph package."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime
from glob import glob
from numbers import Integral
from typing import Any

from kumo_core.encoder_selection import feature_column_names
from kumo_core.types.schema import source_name_path_token
from kumoapi.encoder import EncoderType
from pydantic import TypeAdapter


def canonical_graph_snapshot_id(snapshot_id: str) -> str:
    from statedb.config.common import DataSnapshot
    return str(DataSnapshot.GRAPH.canonicalize(snapshot_id))


def infer_connector_id_from_ingest_dir(
    data_persist_dir: str,
    source_table_name: str,
    table_version: str,
) -> str | None:
    """Infer one connector across legacy and schema-aware ingest paths."""
    source_token = source_name_path_token(source_table_name)
    pattern = os.path.join(
        data_persist_dir,
        "ingest",
        f"fpv3.*.{source_token}*",
        str(table_version),
    )
    connector_ids: set[str] = set()
    source_marker = f".{source_token}"
    for match in glob(pattern):
        if not os.path.isdir(match):
            continue
        base = os.path.basename(os.path.dirname(match))
        if not base.startswith("fpv3."):
            continue
        connector_id, separator, schema_suffix = base[len("fpv3."):].rpartition(
            source_marker)
        if not separator:
            continue
        if schema_suffix and not schema_suffix.startswith(".schema_"):
            continue
        if connector_id:
            connector_ids.add(connector_id)
    return connector_ids.pop() if len(connector_ids) == 1 else None


def resolve_split_config(
        split_config: Any,
        split_family: str | None) -> tuple[dict[str, Any], str]:
    """Resolve historical Kumo split objects and aliases before core."""
    if isinstance(split_config, Mapping):
        config = dict(split_config)
    elif is_dataclass(split_config):
        config = asdict(split_config)
    elif callable(getattr(split_config, "model_dump", None)):
        config = split_config.model_dump()
    elif callable(getattr(split_config, "dict", None)):
        config = split_config.dict()
    else:
        raise RuntimeError("legacy split_config must resolve to a mapping")
    target = config.pop("_target_", None)
    if split_family is None and isinstance(target, str) and target:
        split_family = target.rsplit(".", 1)[-1]
    if not isinstance(split_family, str) or not split_family:
        raise RuntimeError("legacy split_config has no resolvable family")
    aliases = {
        "seed": "rnd_seed",
        "timestamp": "backtesting_timestamp",
        "train_value": "manual_train_values",
        "val_value": "manual_val_values",
        "test_value": "manual_test_values",
    }
    for legacy, canonical in aliases.items():
        if legacy not in config:
            continue
        if canonical in config and config[canonical] != config[legacy]:
            raise RuntimeError(
                f"legacy split fields {legacy} and {canonical} conflict")
        config[canonical] = config.pop(legacy)
    if "col" in config:
        column = config.pop("col")
        resolved_column = column.get("name") if isinstance(column,
                                                           Mapping) else column
        if ("split_column" in config
                and config["split_column"] != resolved_column):
            raise RuntimeError("legacy col and split_column conflict")
        config["split_column"] = resolved_column
    for field_name in ("manual_train_values", "manual_val_values",
                       "manual_test_values"):
        value = config.get(field_name)
        if value is not None and not isinstance(value, (list, tuple)):
            config[field_name] = [str(value)]
        elif value is not None:
            config[field_name] = [str(item) for item in value]
    if config.get("backtesting_timestamp") is not None:
        config["backtesting_timestamp"] = _legacy_epoch_seconds(
            config["backtesting_timestamp"])
    if config.get("ranges") is not None:
        config["ranges"] = [[
            _legacy_epoch_seconds(pair[0]),
            _legacy_epoch_seconds(pair[1]),
        ] for pair in config["ranges"]]
    return config, split_family


def _legacy_epoch_seconds(value: Any) -> int:
    if isinstance(value, bool):
        raise RuntimeError("legacy timestamp value must not be bool")
    if isinstance(value, Integral):
        integer = int(value)
        magnitude = abs(integer)
        if magnitude >= 1_000_000_000_000_000_000:
            return integer // 1_000_000_000
        if magnitude >= 1_000_000_000_000_000:
            return integer // 1_000_000
        if magnitude >= 1_000_000_000_000:
            return integer // 1_000
        return integer
    if isinstance(value, datetime):
        return int(value.timestamp())
    try:
        import pandas as pd
        rendered = str(value).removeprefix("kumo_timestamp_")
        return int(pd.Timestamp(rendered).value // 1_000_000_000)
    except Exception as error:
        raise RuntimeError(
            f"invalid legacy timestamp value: {value!r}") from error


def build_task_dataset_spec(
    *,
    task: Any,
    fitted_encoders: Any,
    tt_manifest_path: str,
    job_id: str,
    artifact_root: str,
    rhs_node_type: str | None = None,
) -> Any:
    """Adapt Kumo task objects into the exact portable task schema."""
    from kumo_core.task_dataset import TaskDatasetMaterializeSpec

    task_type = str(getattr(task.task_type, "value", task.task_type)).lower()
    family = task_family(task.task_type)
    feature_schema = build_feature_schema(task, fitted_encoders)
    foreign_keys = list(getattr(task.train_table, "fkeys", None) or [])
    if not foreign_keys:
        raise RuntimeError(
            "expected train_table.fkeys[0] to point at the entity table")
    destination = foreign_keys[0].dst_table or ""
    entity_table = destination.rsplit(".", 1)[-1]
    return TaskDatasetMaterializeSpec(
        tt_manifest_path=tt_manifest_path,
        job_id=str(job_id),
        artifact_root=artifact_root,
        task_family=family,
        task_type=task_type,
        is_temporal=task_is_temporal(task),
        seed_node_type=entity_table,
        train_table_name=task.train_table.table_name,
        entity_table_name=entity_table,
        rhs_node_type=rhs_node_type,
        feature_schema=feature_schema,
        graph_node_types=tuple(table.table_name for table in task.graph.tables),
        graph_edge_types=tuple(
            (table.table_name, foreign_key.name, foreign_key.dst_table_name)
            for table in task.graph.tables for foreign_key in table.fkeys),
    )


def task_family(task_type: Any) -> str:
    value = str(getattr(task_type, "value", task_type)).lower()
    if value == "forecasting":
        return "forecasting"
    if bool(getattr(task_type, "is_link_pred", False)) or value in {
            "static_link_prediction", "temporal_link_prediction",
            "link_prediction"
    }:
        return "link_prediction"
    if value in {
            "binary_classification",
            "multiclass_classification",
            "multilabel_classification",
            "multilabel_ranking",
            "regression",
    }:
        return "node_prediction"
    raise RuntimeError(f"native task dataset does not support {task_type!r}")


def task_is_temporal(task: Any) -> bool:
    if hasattr(task, "is_temporal"):
        return bool(task.is_temporal)
    value = str(getattr(task.task_type, "value", task.task_type)).lower()
    return value == "temporal_link_prediction"


def build_feature_schema(task: Any, fitted_encoders: Any) -> dict[str, Any]:
    family = task_family(task.task_type)
    excluded_by_table = dict(getattr(task, "exclude_cols", {}) or {})
    excluded_by_table.update(getattr(task, "exclude_cols_dict", {}) or {})
    node_types: dict[str, Any] = {}

    def encoder_input(table: Any, column_name: str,
                      encoder: Any) -> dict[str, Any]:
        target_column = getattr(table, "target_col", None)
        if (target_column is not None and target_column.name == column_name):
            column = target_column
        else:
            column = table.col(column_name)
        if column is None:
            raise RuntimeError(
                f"encoder column {column_name!r} is missing from "
                f"table {table.table_name!r}")
        encoder_config = TypeAdapter(EncoderType).dump_python(
            encoder, mode="json")
        encoder_config.pop("_stats", None)
        stats = {
            str(getattr(stat, "value", stat)): value
            for stat, value in encoder._stats.items()
        }
        return {
            "column_name": column_name,
            "dtype": str(getattr(column.dtype, "value", column.dtype)),
            "stype": str(getattr(column.stype, "value", column.stype)),
            "encoder": encoder_config,
            "stats": stats,
        }

    def table_schema(table: Any) -> dict[str, Any]:
        name = table.table_name
        excluded = set(excluded_by_table.get(name, []))
        excluded.update(excluded_by_table.get(getattr(table, "name", name), []))
        encoded = fitted_encoders.encoders.get(name, {})
        columns = feature_column_names(
            encoded,
            primary_key=getattr(table, "pkey_name", None),
            foreign_keys=getattr(table, "fkey_names", None) or (),
            end_time_column=getattr(table, "end_time_col_name", None),
            excluded=excluded,
        )
        has_time = getattr(table, "has_time_col", lambda: False)()
        return {
            "feature_columns":
            columns,
            "mapper_key":
            name,
            "time_column":
            table.time_col_name if has_time else None,
            "column_specs": [
                encoder_input(table, column, encoded[column])
                for column in columns
            ],
        }

    for table in task.graph.tables:
        node_types[table.table_name] = table_schema(table)
    if family == "forecasting":
        node_types[task.train_table.table_name] = table_schema(task.train_table)

    target = getattr(fitted_encoders, "target", None)
    target_encoder_input = None
    if family != "link_prediction" and target is not None:
        target_encoder_input = encoder_input(
            task.train_table,
            target.col_name,
            target.encoder,
        )

    return {
        "node_types": node_types,
        "target_encoder_input": target_encoder_input,
    }
