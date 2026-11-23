# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persist independent outcomes around an optional prediction explanation replay."""

import logging
from typing import Mapping

from kumo_core.embedding_models import EmbeddingModelStore
from kumo_core.encoders.config import EncodersConfig
from kumo_core.prediction_explanation_result import (
    PredictionExplanationOptions,
    PredictionExplanationResult,
    PredictionExplanationSupport,
    ProducerOutcome,
)
from kumo_core.prediction_explanations import (
    explain_prediction,
    prediction_explanation_support,
    prediction_model_support,
)
from kumo_core.task_dataset import TaskDatasetMaterializeResult
from kumo_core.types.enums import TaskType

from kumo.engine.explain._prediction_manifest import (
    PredictionExplanationManifest,
    write_record,
)
from kumo.engine.explain.results import PredictionExplanations

logger = logging.getLogger(__name__)


def _failed(support: PredictionExplanationSupport,
            error: str) -> PredictionExplanationResult:
    return PredictionExplanationResult(
        ProducerOutcome('failed', error) if support.cohorts else
        ProducerOutcome('unsupported'),
        ProducerOutcome('failed', error) if support.samples else
        ProducerOutcome('unsupported'))


def explain_run(
    *,
    run_root: str,
    run_id: str,
    task_type: TaskType,
    materialized: TaskDatasetMaterializeResult,
    model_dir: str,
    explanations: bool | PredictionExplanationOptions = False,
    top_k: int | None = None,
    binary_threshold: float | None = None,
    task_options: Mapping[str, int | None] | None = None,
    embedding_models: EmbeddingModelStore | None = None,
    encoders: EncodersConfig | None = None,
    accelerator: str = 'auto',
) -> PredictionExplanations:
    """Record actual prediction identity and preserve completed replay outcomes.

    The caller finishes prediction export/tracking and releases scoring resources
    first. ``run_id`` is that prediction's recorded ID, never a directory name.
    A status-write error cannot turn a completed producer into a failed one.
    """
    enabled = bool(explanations)
    support = prediction_explanation_support(task_type)
    record = PredictionExplanationManifest(
        schema_version=1, kind='prediction_explanations',
        run_id=run_id, enabled=enabled,
        state='running' if enabled else 'complete',
        result=(_failed(support,
                        'Explanation replay did not complete') if enabled else None))
    if not enabled:
        try:
            write_record(run_root, record)
        except Exception:
            logger.exception('Could not retain disabled explanation status for '
                             'prediction run %s.', run_id)
        return PredictionExplanations.from_run(run_root, run_id=run_id)

    try:
        support = prediction_model_support(task_type, model_dir)
    except Exception as error:
        outcome = _failed(support, f'{type(error).__name__}: {error}')
    else:
        record = record.model_copy(update={
            'result': _failed(support, 'Explanation replay did not complete')})
        try:
            write_record(run_root, record)
        except Exception as error:
            logger.exception('Prediction outputs remain available; could not '
                             'start explanation status for run %s.', run_id)
            return PredictionExplanations.from_run(
                run_root, enabled=True, run_id=run_id,
                result=_failed(support, f'{type(error).__name__}: {error}'))
        try:
            outcome = explain_prediction(
                materialized=materialized, model_dir=model_dir, run_id=run_id,
                output_dir=f'{run_root}/xai', top_k=top_k,
                binary_threshold=binary_threshold, task_options=task_options,
                embedding_models=embedding_models, encoders=encoders,
                accelerator=accelerator,
                options=(explanations if isinstance(
                    explanations, PredictionExplanationOptions) else None))
        except Exception as error:
            outcome = _failed(support, f'{type(error).__name__}: {error}')

    try:
        write_record(run_root, record.model_copy(update={
            'state': 'complete', 'result': outcome}))
    except Exception:
        logger.exception('Prediction outputs remain available; could not retain '
                         'explanation status for run %s.', run_id)
    if outcome.failure:
        logger.warning('Prediction outputs remain available; explanations failed '
                       'for run %s: %s', run_id, outcome.failure)
    return PredictionExplanations.from_run(
        run_root, enabled=True, result=outcome, run_id=run_id)
