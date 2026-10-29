# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A `Trainer.fit` as a tracked run's `RunSpec`."""

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.summaries import Summary

from kumo.engine._tracking import RunSpec, _flatten
from kumo.engine._version import __version__


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


def fit_spec(context: FitContext, *,
             tags: Optional[Mapping[str, Any]] = None) -> RunSpec:
    started = context.started or datetime.now(timezone.utc)
    return RunSpec(
        kind='fit',
        name='-'.join(filter(None, ('fit', context.model_name,
                                    started.strftime('%Y%m%d-%H%M')))),
        params=dict(_flatten(context.model_plan)),
        parents={'snapshot_id': context.snapshot_id,
                 'task_type': context.task_type,
                 'tune_metric': (context.model_plan.get('training_job')
                                 or {}).get('tune_metric'),
                 'engine_version': __version__},
        tags=dict(tags or {}),
        inputs=dict(context.inputs),
        artifacts={
            'query.pql': context.query,
            'model_plan.json': json.dumps(dict(context.model_plan), indent=2,
                                          default=str),
        },
    )


def fitted_spec(spec: RunSpec, result: Any, seconds: Optional[float] = None,
                summaries: Sequence[Summary] = ()) -> RunSpec:
    r"""``spec`` once the fit returned ``result``, a `TrainingResult`,
    after ``seconds``: its test metrics, since the run's train and
    validation series show the selected trial's epochs, the version it
    registered, and ``summaries``."""
    artifacts = dict(spec.artifacts)
    provenance = Path(result.bundle_dir) / PROVENANCE_FILE
    if provenance.is_file():
        artifacts[PROVENANCE_FILE] = provenance.read_text(encoding='utf-8')
    tested = {name: value for name, value in result.metrics.items()
              if not name.startswith(('train_', 'val_'))}
    return replace(
        spec, metrics={
            **tested, **({} if seconds is None else {
                'seconds': seconds
            })
        }, parents={**spec.parents, 'bundle_id': result.id,
                    **_registered(result.registered_model,
                                  result.registered_alias)},
        artifacts=artifacts, summaries=summaries)


def _registered(model: Any, alias: Optional[str]) -> Mapping[str, Any]:
    if model is None:
        return {}
    return {'model_name': model.name, 'model_version': model.version,
            'model_alias': alias}
