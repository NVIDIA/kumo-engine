# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One training run's inputs, read once and shared by every card.

The published bundle is read through its ``TrainingResult``; the retained run
directories, which may be remote, through ``kumo_core.io.filesystem``. A
missing or unreadable optional input reads as ``None`` so the card that needs
it is omitted instead of failing the widget.
"""

import io
import json
import math
import posixpath
import re
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd
from kumo_core.io import filesystem

from kumo.engine import _text
from kumo.engine._explainability._format import numbers, percent

if TYPE_CHECKING:
    from kumoapi.explain.column_analysis import ColumnAnalysisOutput
    from kumoapi.explain.gradient import GraphGradientScore

    from kumo.engine.explain import TrainingExplanations
    from kumo.engine.trainer.trainer import TrainingResult

CLASSIFICATION = 'classification'
REGRESSION = 'regression'
FORECASTING = 'forecasting'
# Link prediction and multilabel ranking share one model type: both rank the
# top K items per entity and report the same @K metric families.
LINK_PREDICTION = 'link_prediction'

CLASS_LABELED = ('multiclass_classification', 'multilabel_classification')

# Kumo task types the widget supports, by the model type it renders.
_TASKS = dict(binary_classification=CLASSIFICATION,
              multiclass_classification=CLASSIFICATION,
              multilabel_classification=CLASSIFICATION, regression=REGRESSION,
              forecasting=FORECASTING, multilabel_ranking=LINK_PREDICTION,
              link_prediction=LINK_PREDICTION,
              temporal_link_prediction=LINK_PREDICTION,
              static_link_prediction=LINK_PREDICTION)

_HOLDOUT_COLUMNS = dict(
    binary_classification=['ENTITY', 'TARGET', 'True_PROB', 'False_PROB'],
    multiclass_classification=['ENTITY', 'TARGET'],
    multilabel_classification=['ENTITY', 'TARGET'],
    regression=['ENTITY', 'TARGET', 'TARGET_PRED'], forecasting=[
        'ENTITY', 'TIMESTAMP', 'ANCHOR_TIMESTAMP', 'TARGET', 'TARGET_PRED'
    ])

_TARGET = re.compile(
    r'^PREDICT\s+(.+?)\s+(?:RANK\s+TOP\b|FORECAST\b|FOR\s+EACH\b)',
    re.IGNORECASE)


class UnsupportedModelError(ValueError):
    """The run's task type has no explainability widget."""


def model_type(task_type: Optional[str]) -> str:
    if task_type not in _TASKS:
        raise UnsupportedModelError(
            f'No explainability view for task type {task_type!r}; supported: '
            'binary, multiclass and multilabel classification, regression, '
            'forecasting, multilabel ranking and link prediction.')
    return _TASKS[task_type]


def read_json(path: Any) -> Any:
    if path is None:
        return None
    try:
        with filesystem.open_file(str(path), 'rt', encoding='utf-8') as src:
            return json.load(src)
    except (OSError, ValueError):
        return None


def read_parquet(path: Any, columns: list[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    try:
        with filesystem.open_file(str(path), 'rb') as src:
            return pd.read_parquet(io.BytesIO(src.read()), columns=columns)
    except (OSError, ValueError, KeyError):
        return None


def mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


@dataclass
class Run:
    task_type: str
    model_type: str
    query: Optional[str]
    plan: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, float]
    holdout: Optional[pd.DataFrame]
    holdout_truncated: bool
    full_refit: bool
    history: Optional[dict[str, Any]]
    explanations: 'TrainingExplanations'
    table: Optional[str] = None
    snapshot: Optional[str] = None
    input_manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def entities(self) -> str:
        return _text.entity(self.query) or 'rows'

    @property
    def training_job(self) -> dict[str, Any]:
        return mapping(self.plan.get('training_job'))

    @property
    def metric_names(self) -> list[str]:
        metrics = self.training_job.get('metrics') or []
        return [m for m in metrics if isinstance(m, str)]

    @property
    def tune(self) -> Optional[str]:
        tune = self.training_job.get('tune_metric')
        return tune if isinstance(tune, str) else None

    @property
    def refit(self) -> bool:
        trial = self.explanations.trial_root
        return trial is not None and trial.endswith('_refit_trainval')

    def target(self, short: bool = False) -> Optional[str]:
        """The query's target expression, for axis titles.

        ``short`` drops the time window, so ``SUM(transactions.price, 0, 30,
        days)`` reads ``SUM(transactions.price)``.
        """
        match = _TARGET.match(self.query or '')
        if not match:
            return None
        expr = match.group(1)
        if short:
            expr = re.sub(r'\(([^,()]*),[^()]*\)', r'(\1)', expr, count=1)
        return expr

    @cached_property
    def splits(self) -> dict[str, dict[str, Any]]:
        """``split_stats.json`` of the test training table, by split."""
        stats = mapping(
            read_json(None if self.table is
                      None else posixpath.join(self.table, 'split_stats.json')))
        return {
            name: mapping(stats.get(f'{name}_table_stats'))
            for name in ('train', 'val', 'test')
        }

    @cached_property
    def gradient(self) -> Optional['GraphGradientScore']:
        try:
            return self.explanations.gradient_scores()
        except (OSError, ValueError):
            return None

    @cached_property
    def cohorts(self) -> Optional['ColumnAnalysisOutput']:
        try:
            return self.explanations.column_analysis()
        except (OSError, ValueError):
            return None

    @cached_property
    def catalog(self) -> Optional[tuple[int, float, str]]:
        """Catalog size, mean test list length and catalog table."""
        if self.table is None or self.snapshot is None:
            return None
        manifest = read_json(
            posixpath.join(self.table, 'diskgraph_manifest.json'))
        rhs = mapping(
            mapping(manifest).get('task_dataset')).get('rhs_node_type')
        stats = read_json(posixpath.join(self.snapshot, 'ingest_stats.json'))
        size = mapping(mapping(
            mapping(stats).get('stats')).get('node_counts')).get(str(rhs))
        seeds = posixpath.join(self.table, 'test_seeds')
        try:
            frames = [
                read_parquet(posixpath.join(seeds, posixpath.basename(name)),
                             ['TARGET1'])
                for name in sorted(filesystem.listdir(seeds))
                if name.endswith('.parquet')
            ]
        except OSError:
            return None
        if (not isinstance(size, int) or size <= 0 or not frames
                or any(frame is None for frame in frames)):
            return None
        lengths = pd.concat(frames)['TARGET1'].map(len).to_numpy(dtype=float)
        return (size, float(lengths.mean()), str(rhs)) if len(lengths) else None

    def covered(self, rows: Optional[int]) -> Optional[str]:
        """A line when only the first ``rows`` test rows were used."""
        total = integer(self.splits['test'].get('num_train_rows'))
        if rows is None or total is None or rows >= total:
            return None
        return f'based on the first {rows:,} of {total:,} test rows'

    @property
    def _batch(self) -> Optional[int]:
        """The evaluation batch size, as the trainer resolves it."""
        return next(
            filter(None,
                   (integer(mapping(self.config.get(key)).get('batch_size'))
                    for key in ('eval_loader_kwargs', 'loader_kwargs'))), None)

    def metrics_coverage(self) -> Optional[str]:
        """Kumo stops test evaluation after ``max_test_steps`` batches."""
        steps = integer(
            mapping(self.plan.get('optimization')).get('max_test_steps'))
        batch = self._batch
        return self.covered(steps *
                            batch if batch and steps and steps > 0 else None)

    def explanation_coverage(self) -> Optional[str]:
        """The explanation pass replays a fixed number of batches at a
        quarter of the evaluation batch size."""
        from kumo_core.training.runtime import LocalExplainRuntime

        batch = self._batch
        return self.covered(None if batch is
                            None else int(LocalExplainRuntime.max_batches) *
                            max(batch // 4, 1))

    def holdout_coverage(self) -> Optional[str]:
        rows = None if self.holdout is None else len(self.holdout)
        if self.holdout is not None and 'ANCHOR_TIMESTAMP' in self.holdout:
            # A forecasting holdout has one row per step; its test rows are
            # its distinct entity and anchor pairs.
            rows = len(self.holdout[['ENTITY',
                                     'ANCHOR_TIMESTAMP']].drop_duplicates())
        line = self.covered(rows)
        if not self.holdout_truncated:
            return line
        return (line + '; retained holdout is truncated' if line else
                'Based on a truncated retained holdout; full coverage is '
                'unavailable.')


def load(result: 'TrainingResult') -> Run:
    from kumo.engine.trainer._spec import read_training_spec
    from kumo.engine.trainer.trainer import _listed_holdout

    paths = result.artifact_paths
    task = mapping(read_json(paths.get('prediction_schema'))).get('task_type')
    kind = model_type(task)
    try:
        query = ' '.join(
            read_training_spec(result.bundle_dir).predictive_query.split())
    except (OSError, ValueError, KeyError, AttributeError):
        query = None
    config = mapping(read_json(paths.get('trainer_config')))
    try:
        history = result.metric_history
    except (OSError, ValueError):
        history = None
    trial = result.explanations.trial_root
    if history is None and trial is not None:
        # The retained run's search progress, beside the selected trial.
        history = read_json(
            posixpath.join(posixpath.dirname(trial),
                           'auto_train_progress.json'))
    manifest = result.manifest
    fit_mode = None if manifest is None else manifest.fit_mode
    full_refit = (fit_mode == 'refit_full' if fit_mode else mapping(
        config.get('refit_cfg')).get('refitted_full') is True)
    columns = _HOLDOUT_COLUMNS.get(task)
    holdout, truncated = None, result.holdout_truncated is True
    if columns is not None:
        holdout = read_parquet(result.holdout_path, columns)
        if holdout is None:
            path, listed = _listed_holdout(result.bundle_dir)
            holdout = read_parquet(path, columns)
            truncated = truncated or listed is True
    root = result.explanations.run_root
    runtime = mapping(
        read_json(None if root is None else posixpath.
                  join(root, 'task_dataset', 'runtime_manifest.json')))
    directories = []
    for index in (mapping(mapping(runtime.get('stages')).get('test')),
                  mapping(runtime.get('graph'))):
        path = index.get('index_path')
        try:
            present = isinstance(path, str) and filesystem.exists(path)
        except OSError:
            present = False
        directories.append(path if present else None)
    return Run(
        task, kind, query, mapping(read_json(paths.get('resolved_model_plan'))),
        config, {
            key: float(value)
            for key, value in result.metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value)
        }, None if holdout is None or holdout.empty else holdout, truncated,
        full_refit, history, result.explanations, *directories,
        input_manifest=mapping(read_json(paths.get('input_manifest'))))


def equal_count_groups(values: np.ndarray, entities: np.ndarray,
                       groups: int = 10) -> np.ndarray:
    """Zero-based group of each row, ascending by value then entity.

    Row ``i`` in sorted order goes to group ``floor(groups * i / n)``, so
    group sizes differ by at most one and the last group holds the highest.
    """
    n = len(values)
    order = np.lexsort((entities, values))
    out = np.empty(n, dtype=int)
    out[order] = np.arange(n) * groups // max(n, 1)
    return out


def tenths(entities: np.ndarray, first: np.ndarray, second: np.ndarray,
           word: str, reference_label: str, *,
           percentage: bool = False) -> dict[str, Any]:
    """Mean of both series in each tenth of ``first``, and the widest gap.

    ``word`` names the second series in the gap callout.
    """
    group = equal_count_groups(first, entities)
    occupied = np.unique(group)
    means = [(float(first[group == k].mean()), float(second[group == k].mean()))
             for k in occupied]
    reference = float(second.mean())
    values = [v for pair in means for v in pair] + [reference]
    texts = [percent(v) for v in values] if percentage else numbers(values)
    worst = int(np.argmax([abs(a - b) for a, b in means]))
    rows = [
        dict(label=str(k + 1), values=list(pair), texts=texts[2 * i:2 * i + 2])
        for i, (k, pair) in enumerate(zip(occupied, means))
    ]
    return dict(
        id='tenths', tenths=rows, reference=reference, highlight=worst,
        reference_label=reference_label, percentage=percentage,
        reference_callout=f'{reference_label}: {texts[-1]}',
        callout=(f'Group {occupied[worst] + 1} of 10: predicted '
                 f'{texts[2 * worst]} · {word} {texts[2 * worst + 1]}'))
