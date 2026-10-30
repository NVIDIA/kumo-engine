# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A `Trainer.fit` as a tracked run's `RunSpec`."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.summaries import Scalar, Summary

from kumo.engine._tracking import RunSpec, _flatten, logged_names, model_label


@dataclass(frozen=True)
class FitContext:
    r"""What a fit trains, known once its model plan is resolved."""

    query: str
    model_plan: Mapping[str, Any]
    inputs: Mapping[str, Mapping[str, Optional[str]]] = field(
        default_factory=dict)
    started: Optional[datetime] = None
    snapshot_id: Optional[str] = None
    task_type: Optional[str] = None
    model_name: Optional[str] = None


#: A model-plan option the plan leaves to its defaults.
_UNSET = '???'


def fit_spec(context: FitContext, *,
             tags: Optional[Mapping[str, Any]] = None) -> RunSpec:
    started = context.started or datetime.now(timezone.utc)
    model = model_label(context.model_name, context.query, context.task_type)
    return RunSpec(
        kind='fit',
        name=f'fit-{started:%Y%m%d-%H%M%S}-{model}',
        model_label=model,
        started=started,
        params={key: value for key, value in _flatten(context.model_plan)
                if value is not None and value != _UNSET},
        parents={'snapshot_id': context.snapshot_id,
                 'task_type': context.task_type,
                 'tune_metric': (context.model_plan.get('training_job')
                                 or {}).get('tune_metric')},
        tags=dict(tags or {}),
        inputs=dict(context.inputs),
        description=f'`{context.query}`',
    )


def fitted_spec(spec: RunSpec, result: Any,
                summaries: Sequence[Summary] = ()) -> RunSpec:
    r"""``spec`` once the fit returned ``result``, a `TrainingResult`:
    its test metrics, since the run's train and validation series show the
    selected trial's epochs, the version it registered, and
    ``summaries``."""
    artifacts = dict(spec.artifacts)
    provenance = Path(result.bundle_dir) / PROVENANCE_FILE
    if provenance.is_file():
        artifacts[PROVENANCE_FILE] = provenance.read_text(encoding='utf-8')
    return replace(
        spec, metrics=holdout_metrics(result),
        parents={**spec.parents, 'bundle_id': result.id,
                 **_registered(result.registered_model)},
        artifacts=artifacts, summaries=summaries)


def holdout_metrics(result: Any) -> Dict[str, float]:
    r"""The trainer's test metrics of ``result``, a `TrainingResult`, as
    `logged_names` names them."""
    return logged_names({
        name: value for name, value in result.metrics.items()
        if not name.startswith(('train_', 'val_'))
    }, refit_full=_fit_mode(result) == 'refit_full')


def model_metrics(result: Any) -> Dict[str, float]:
    r"""The model's own quality, as a logged model and every registry
    record it: `holdout_metrics` and the ``role='model'`` summaries."""
    return {name.replace('@', '_at_'): value for name, value in {
        **holdout_metrics(result),
        **{summary.name: summary.value
           for summary in getattr(result, '_summaries', ())
           if isinstance(summary, Scalar) and summary.role == 'model'}
    }.items()}


def _fit_mode(result: Any) -> Optional[str]:
    return getattr(getattr(result, 'manifest', None), 'fit_mode', None)


def _registered(model: Any) -> Mapping[str, Any]:
    if model is None:
        return {}
    return {'model_name': model.name, 'model_version': model.version}
