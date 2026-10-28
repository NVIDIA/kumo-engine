# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A `Trainer.fit` as a tracked run's `RunSpec`."""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from kumo_core.artifacts import PROVENANCE_FILE

from kumo.engine._tracking import RunSpec, _flatten


@dataclass(frozen=True)
class FitContext:
    r"""What a fit trains, known once its model plan is resolved."""

    run_id: str
    query: str
    model_plan: Mapping[str, Any]
    inputs: Mapping[str, Mapping[str, Optional[str]]] = field(
        default_factory=dict)


def fit_spec(context: FitContext, *,
             tags: Optional[Mapping[str, Any]] = None) -> RunSpec:
    return RunSpec(
        kind='fit',
        run_id=context.run_id,
        params=dict(_flatten(context.model_plan)),
        tags=dict(tags or {}),
        inputs=dict(context.inputs),
        artifacts={
            'query.pql': context.query,
            'model_plan.json': json.dumps(dict(context.model_plan), indent=2,
                                          default=str),
        },
    )


def fitted_spec(spec: RunSpec, result: Any,
                seconds: Optional[float] = None) -> RunSpec:
    r"""``spec`` once the fit returned ``result``, a `TrainingResult`,
    after ``seconds``."""
    artifacts = dict(spec.artifacts)
    provenance = Path(result.bundle_dir) / PROVENANCE_FILE
    if provenance.is_file():
        artifacts[PROVENANCE_FILE] = provenance.read_text(encoding='utf-8')
    return replace(
        spec, metrics={
            **result.metrics, **({} if seconds is None else {
                'seconds': seconds
            })
        }, parents={'bundle_id': result.id}, artifacts=artifacts)
