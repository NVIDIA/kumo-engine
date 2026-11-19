# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The batch prediction widget: the training widget's cohort and subgraph
cards over one batch's retained PRED-stage explanations."""

from dataclasses import asdict
from typing import Any, Optional

from pydantic import TypeAdapter

from kumo.engine._explainability import (
    ExplainabilityView,
    _columns,
    _subgraphs,
)
from kumo.engine._explainability._run import _TASKS, Run
from kumo.engine._manifest import PredictionManifest
from kumo.engine.explain._prediction_manifest import read_record
from kumo.engine.explain.results import PredictionExplanations

_REASONS: dict[Optional[str], str] = dict(
    disabled='Generation was disabled for this batch.',
    unsupported='Not supported for this prediction model.',
    failed='Generation did not complete.')

# Batch cards in order: card id, producer, its label, the card maker and the
# notice when available artifacts yield no card.
_CARDS = (('groups', 'cohorts', 'Cohorts', _columns.groups,
           'no readable column groups were recorded.'),
          ('subgraphs', 'samples', 'Subgraphs', _subgraphs.card,
           'no readable saved examples are available.'))


def _task(source: PredictionExplanations, kind: str) -> Optional[str]:
    """The task type in one producer's first readable artifact."""
    try:
        if kind == 'cohorts':
            return source.column_analysis().task_type.value
        for reference in source.samples():
            store = reference.available and source.sample(
                reference.artifact_id).subgraph_store
            if store:
                return store.task_type.value
    except (OSError, ValueError):
        pass
    return None


def build(explanations: PredictionExplanations,
          prediction: Optional[PredictionManifest],
          run_id: Optional[str]) -> dict[str, Any]:
    """Every number and text the batch widget draws."""
    try:
        # Live metadata may hold nulls in typed fields; only a valid record
        # of this batch is used.
        prediction = TypeAdapter(PredictionManifest).validate_python(
            asdict(prediction)) if prediction else None
    except (TypeError, ValueError):
        prediction = None
    if prediction and run_id not in (None, prediction.id):
        prediction = None
    expected = (run_id, None, None) if prediction is None else (
        prediction.id, prediction.parents.model.id,
        prediction.parents.snapshot.id)
    notices, root = [], explanations.run_root
    try:
        record = read_record(root) if root else None
    except (OSError, ValueError):
        record = None
    if record and any(
            known not in (None, value)
            for known, value in zip(expected, (record.run_id, record.model_id,
                                               record.snapshot_id))):
        notices.append(
            'Retained explanations do not match this prediction batch.')
        record = None
    source = (PredictionExplanations.load(root)
              if record and root else PredictionExplanations.unavailable())
    availability = source.availability
    # A producer status attached to the result outranks retained files.
    for kind, state in explanations.availability.items():
        if state.reason in _REASONS:
            availability[kind] = state
    task = next(
        filter(None, (_task(source, kind) for kind in ('cohorts', 'samples')
                      if availability[kind].available)), None)
    provenance = record.result.provenance if record and record.result else None
    run = Run(task or '', _TASKS.get(task, 'prediction'), None, {}, {}, {},
              None, False, False, None, source, batch=True,
              cohort_rows=provenance and provenance.processed_cohort_rows)
    cards = []
    for card_id, kind, label, make, empty in _CARDS:
        state = availability[kind]
        card = make(run) if state.available else None
        if card is not None:
            cards.append(dict(id=card_id, **card))
        else:
            reason = (empty if state.available else _REASONS.get(
                state.reason, 'Retained artifacts are unavailable.'))
            notices.append(f'{label}: {reason}')
    header: dict[str, Any] = dict(query='Batch prediction explanations')
    prediction = prediction or (record.prediction if record else None)
    identity = [('Batch', expected[0])]
    if prediction is not None:
        table, model = prediction.prediction_table, prediction.parents.model
        rows = table.scored_rows if table.scored_rows is not None else table.rows
        header['summary'] = (f'{rows:,} prediction rows' if rows is not None
                             else 'Prediction row count was not recorded.')
        training = model.parents.training_table if model.parents else None
        task = task or (training.task_type if training else None)
        identity = [
            ('Batch', prediction.id), ('Model ID', model.id),
            ('Snapshot ID', prediction.parents.snapshot.id),
            ('Prediction type',
             task.replace('_', ' ').capitalize() if task else 'Not recorded'),
            ('Fit mode', model.fit_mode),
            ('Batch started', prediction.timings
             and prediction.timings.started_at)
        ]
    header['identity'] = [[k, str(v)] for k, v in identity if v]
    return _subgraphs._json_value(
        dict(
            kind='prediction', model_type=run.model_type, entities=run.entities,
            header=header, tabs=[
                dict(
                    id='explanation', label='Explanation', cards=[
                        dict(id='status', title='Explanation availability',
                             messages=notices)
                    ] * bool(notices) + cards)
            ]))


class PredictionExplainabilityView(ExplainabilityView):
    """Read-only display of one batch's retained PRED-stage explanations."""
    def __init__(self, explanations: PredictionExplanations,
                 prediction: Optional[PredictionManifest] = None,
                 run_id: Optional[str] = None) -> None:
        self._source = (explanations, prediction, run_id)
        self._widget = None

    def _build(self) -> dict[str, Any]:
        return build(*self._source)

    def __repr__(self) -> str:
        return f'<PredictionExplainabilityView {self._source[0].run_root}>'
