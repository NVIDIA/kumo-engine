# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage a directly referenced connector table as local parquet parts."""

import errno
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from kumo.engine.connector.source_table import SourceTable


def stage_source_table(source_table: SourceTable,
                       destination: Path) -> Tuple[Path, Optional[str]]:
    """Read ``source_table`` through its connector and retain its parquet
    parts; also returns the connector's version of the rows read."""
    if not isinstance(source_table, SourceTable):
        raise TypeError("source_table must be a kumo.engine SourceTable")
    destination.mkdir(parents=True, exist_ok=False)
    prepared = source_table.connector.source_backend.prepare_ingest(
        source_table.connector,
        source_table.table_name,
        str(destination / "source"),
        sql=source_table.sql,
    )
    if prepared.credentials is not None:
        raise NotImplementedError(
            "Custom tables require a source backend that stages local "
            "parquet without scoped ingest credentials.")
    if prepared.source.get("type") != "parquet":
        raise NotImplementedError(
            "Custom tables require a source backend that stages local "
            f"parquet; got {prepared.source.get('type')!r}.")
    paths = [Path(path) for path in prepared.source.get("paths", ())]
    if not paths:
        path = prepared.source.get("path")
        if path:
            paths = [Path(path)]
    if not paths or any(not path.exists() for path in paths):
        raise RuntimeError("Custom-table staging produced no parquet files.")
    # DiskGraph accepts one file or a flat directory of parquet parts. Flatten
    # connector directories without re-encoding or combining their row groups.
    output = destination / "parts"
    output.mkdir()
    parts = [
        Path(fragment.path) for path in paths
        for fragment in ds.dataset(str(path), format="parquet").get_fragments()
    ]
    if not parts:
        raise RuntimeError("Custom-table staging produced no parquet files.")
    for index, part in enumerate(parts):
        target = output / f"part-{index:05d}.parquet"
        try:
            # Relative links keep the retained source and parts directory
            # together if the staging tree moves.
            target.symlink_to(os.path.relpath(part.resolve(), output.resolve()))
        except OSError as error:
            if error.errno not in (
                    errno.EPERM, errno.EACCES, errno.EOPNOTSUPP, errno.ENOSYS):
                raise
            shutil.copyfile(part, target)
    return output, prepared.version


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
