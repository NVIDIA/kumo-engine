# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared batch-prediction option decisions."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Sequence, Tuple

from kumoapi.jobs import MetadataField

from .types import TaskType

_THRESHOLDED_CLASSIFICATION_TASK_TYPES = {
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
}


class PredictionTimeOptions(NamedTuple):
    ignore_time_col: bool
    pred_time_col_name: str | None
    include_job_timestamp: bool


def resolve_prediction_time_options(
    task_type: TaskType | str | None,
    extra_fields: Mapping[str, Sequence[str]] | None,
) -> PredictionTimeOptions:
    """Keep forecast horizons separate from optional anchor metadata."""
    requested = {
        field
        for fields in (extra_fields or {}).values()
        for field in fields
    }
    is_forecasting = task_type == TaskType.FORECASTING
    has_anchor = MetadataField.ANCHOR_TIMESTAMP.value in requested
    return PredictionTimeOptions(
        ignore_time_col=not is_forecasting and not has_anchor,
        pred_time_col_name=(MetadataField.ANCHOR_TIMESTAMP.value
                            if has_anchor and not is_forecasting else None),
        include_job_timestamp=MetadataField.JOB_TIMESTAMP.value in requested,
    )


def resolve_classification_threshold(
    task_type: TaskType,
    threshold: float | None,
    *,
    binary_required: bool = False,
) -> float | None:
    """Return the effective classification threshold for a batch request."""

    if task_type not in _THRESHOLDED_CLASSIFICATION_TASK_TYPES:
        return None
    if threshold is None:
        if binary_required and task_type is TaskType.BINARY_CLASSIFICATION:
            raise ValueError(
                "binary_classification_threshold must be specified for "
                "binary classification tasks.")
        return 0.5
    if not 0 <= threshold <= 1:
        raise ValueError(
            "binary_classification_threshold must be between 0 and 1, "
            f"but got '{threshold}'")
    return threshold


def prediction_timestamps(prediction_time: Any) -> tuple[int, ...]:
    """Interpret the public naive timestamp as UTC, matching hosted admission."""
    from datetime import datetime, timezone

    if prediction_time is None:
        moment = datetime.now(timezone.utc)
    else:
        if prediction_time.tzinfo is not None:
            raise ValueError('prediction_time must not include a timezone')
        moment = prediction_time.replace(tzinfo=timezone.utc)
    return (int(moment.timestamp() * 1_000_000_000), )


def apply_binary_threshold(predictions: Any, threshold: float | None) -> None:
    """Preserve positive probabilities and relabel the selected binary output."""
    if threshold is None:
        return
    if predictions is None:
        raise ValueError(
            'binary_classification_threshold requires a predictions artifact')
    if 'True_PROB' not in predictions or 'TARGET_PRED' not in predictions:
        raise ValueError(
            'binary_classification_threshold is only valid for a binary '
            'classification prediction output')
    predictions['TARGET_PRED'] = predictions['True_PROB'].astype('Float64').gt(
        float(threshold))


def prediction_query_contract(
        query: str, graph_config: Mapping[str, Any]) -> Tuple[Any, ...]:
    r"""Return the parsed fields that determine what the model scores.

    A prediction table may select a different population in its ``WHERE``
    clause. Its target, task, ``FOR EACH`` entity, forecast length, and graph
    paths connecting the target to that entity must still match the model.
    """
    from kumo_core.pquery import (
        normalize_graph,
        parse_and_validate_query,
        target_to_entity_paths,
    )

    graph = normalize_graph(graph_config)
    parsed = parse_and_validate_query(query, graph)
    paths = tuple(
        tuple(tuple(edge) for edge in path)
        for path in target_to_entity_paths(parsed, graph))
    return (
        str(parsed.query.target_ast),
        parsed.task_type,
        parsed.entity_table,
        parsed.entity_column,
        parsed.rhs_node_type,
        parsed.forecast_length,
        paths,
    )


def validate_prediction_workers(num_workers: int) -> None:
    """Validate local scorer count before table materialization or spawning."""
    if isinstance(num_workers,
                  bool) or not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError('num_workers must be an integer of at least 1')


def validate_num_classes_to_return(num_classes_to_return: int | None) -> None:
    """Validate the public output count before local prediction work starts."""
    if num_classes_to_return is not None and (
            isinstance(num_classes_to_return, bool)
            or not isinstance(num_classes_to_return, int)
            or num_classes_to_return < 1):
        raise ValueError('num_classes_to_return must be a positive integer')
