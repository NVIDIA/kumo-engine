# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fit preparation shared by local trainers and artifact producers."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from kumoapi.pquery import QueryType
from pyarrow import parquet

from kumo_core.encoder_selection import no_encoder_inference_warnings
from kumo_core.model_plan import inferred_or_missing_type_to_none
from kumo_core.pquery import (
    build_generated_training_task_dataset,
    query_is_temporal,
    target_to_entity_paths,
    to_plan_graph,
)
from kumo_core.training import workflow
from kumo_core.training.custom_table import contract
from kumo_core.training.export import read_spec, stage_parts
from kumo_core.training.table import GeneratedTrainTableResult
from kumo_core.training.workflow import (
    MaterializedTrainingTask,
    TrainingWorkflowRequest,
)
from kumo_core.types import TaskType


def prepare_training_task(
    request: TrainingWorkflowRequest,
    generated: GeneratedTrainTableResult,
    required_stages: tuple[str, ...],
    validate_features: Callable[[dict[str, Any]], None] | None = None,
) -> MaterializedTrainingTask:
    """Validate, ingest and materialize a fit where its graph and staged rows live.

    Args:
        request: Selected graph, query, model plan and fit output locations.
        generated: Existing staged training rows; generation is not repeated.
        required_stages: Splits required by the generated table.
        validate_features: Optional training-node resource check before ingestion.
    Returns:
        Ingested extension and materialized task/model/runtime metadata.
    Raises:
        ValueError: If the model plan or staged rows are invalid.
        OSError: If staged rows or generated artifacts cannot be read or written.
        RuntimeError: If native ingestion or materialization fails.
    """
    parts = stage_parts(read_spec(generated.staging_dir))
    seed_part = next(part for paths in parts.values() for part in paths)
    seed_schema = parquet.read_schema(Path(generated.staging_dir) / seed_part)
    entity_tables = [request.query.entity_table]
    if (
        request.query.rhs_node_type is not None
        and request.query.rhs_node_type not in entity_tables
    ):
        entity_tables.append(request.query.rhs_node_type)
    workflow.resolve_model_plan(
        request.model_plan,
        request.search,
        training_table_generation_plan=request.training_table_generation_plan,
        task_type=TaskType(request.query.task_type),
        query_type=QueryType(request.query.query_type),
        has_train_table_time_col=query_is_temporal(request.query, request.graph),
        entity_table_names=entity_tables,
        target_to_entity_paths=target_to_entity_paths(request.query, request.graph),
        has_train_table_weight_col=contract()["weight_output_column"]
        in seed_schema.names,
        graph=to_plan_graph(request.graph),
    )
    tt_id = f"tt-{request.run_id}"
    options = request.training_table_generation_plan
    lags = inferred_or_missing_type_to_none(options.lag_timesteps)
    year_over_year = inferred_or_missing_type_to_none(options.year_over_year)
    with no_encoder_inference_warnings(
        request.model_plan.column_processing.encoder_overrides or {}
    ):
        task_dataset = build_generated_training_task_dataset(
            query=request.query,
            graph=request.graph,
            graph_index_path=request.prepared_graph.index_path,
            run_id=request.run_id,
            artifact_dir=request.artifact_dir,
            tt_id=tt_id,
            required_stages=required_stages,
            lag_timesteps=None if lags is None else int(lags),
            year_over_year=bool(year_over_year),
        )
    if validate_features is not None:
        validate_features(task_dataset.feature_schema)
    Path(request.artifact_dir).mkdir(parents=True, exist_ok=True)
    materialized = workflow.materialize_generated_training_task(
        request, task_dataset, generated, tt_id=tt_id
    )
    if (
        materialized.task.split.stage_counts.get("test") == 0
        and not materialized.task.task_type.is_link_pred
    ):
        raise ValueError(
            "The test split is empty: shorten the query window or add data."
        )
    return materialized
