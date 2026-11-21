# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the schema of a staged custom training or prediction table."""

from pathlib import Path

import pyarrow.parquet as pq


def _validate_source(path, query, graph, *, weight_col, split_col,
                     prediction) -> None:
    """Check source schema; DiskGraph validates rows during ingestion."""
    from kumo_core.pquery import query_is_temporal
    from kumo_core.training import custom_table
    from kumo_core.types.enums import TaskType
    from kumoapi.common import KumoValidationError, ValidationResponse
    from kumoapi.typing import Dtype

    path = Path(path)
    parts = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not parts:
        raise RuntimeError("Custom-table staging produced no parquet files.")
    with pq.ParquetFile(parts[0]) as parquet_file:
        schema = custom_table.stored_schema(
            parquet_file.schema_arrow,
            parquet_file.metadata.metadata,
        )
    task_type = TaskType(query.task_type)
    entity = graph.table(query.entity_table)
    anchored = (query_is_temporal(query, graph) if prediction else
                not custom_table.anchor_is_sentinel(
                    task_type,
                    temporal=query.query_type == "temporal",
                    entity_has_time_column=entity.time_column is not None,
                    list_distinct_target_is_array=(
                        query.list_distinct_target_is_array),
                ))
    rules = custom_table.SuppliedTableRules(
        task_type=task_type,
        weight_col=weight_col,
        split_col=split_col,
        anchored=anchored,
    )
    required = custom_table.contract_required_columns(rules)
    if prediction:
        required = [name for name in required
                    if name != custom_table.target_column()]
    errors = custom_table.missing_column_errors(
        schema.names,
        required=required,
        weight_col=weight_col,
        split_col=split_col,
    )
    errors += custom_table.column_type_errors(schema, rules)
    errors += custom_table.weight_arity_errors(schema, rules)
    primary_key = entity.primary_key
    entity_dtype = (None if not primary_key else
                    Dtype(entity.column(primary_key).dtype))
    errors += custom_table.entity_key_errors(
        schema,
        entity_pk_dtype=entity_dtype,
    )
    if prediction:
        targets = [
            name for name in schema.names
            if custom_table.column_role(name) == "target"
        ]
        if targets:
            errors.append(
                KumoValidationError(
                    title="Target column in prediction table",
                    message=("A custom prediction table selects rows to "
                             "score and must not contain target columns; "
                             f"remove {targets!r}."),
                ))
    if not errors:
        for part in parts[1:]:
            with pq.ParquetFile(part) as part_file:
                part_schema = custom_table.stored_schema(
                    part_file.schema_arrow, part_file.metadata.metadata)
                if not part_schema.equals(schema):
                    raise ValueError(
                        f"Custom-table parquet parts must have matching "
                        f"schemas; {part} differs from {parts[0]}.")
    if errors:
        raise ValueError(ValidationResponse(errors=errors).error_message())


def validate_training_source(path, query, graph, spec) -> None:
    """Validate labeled rows with the same contract as hosted ingestion."""
    split_col = (spec.split_col or "").strip() or None
    _validate_source(
        path,
        query,
        graph,
        weight_col=spec.weight_col,
        split_col=split_col,
        prediction=False,
    )


def validate_prediction_source(path, query, graph) -> None:
    """Validate caller-selected entities and anchors without a target."""
    _validate_source(
        path,
        query,
        graph,
        weight_col=None,
        split_col=None,
        prediction=True,
    )
