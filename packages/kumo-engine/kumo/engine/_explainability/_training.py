# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Training tab: epoch history and the training table's splits.

The history comes from ``metric-history.json`` in the bundle or the retained
run's ``auto_train_progress.json``. Split facts come from the training
table's ``split_stats.json`` when the run is retained, and otherwise from the
bundle's input manifest.
"""

import math
import re
from typing import Any, Optional

import pandas as pd

from kumo.engine import _data_summaries, _text
from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._run import Run, integer, mapping

Series = dict[str, list[tuple[int, Optional[float]]]]

# The selected trial's directory, with the refit suffix a TEST refit adds.
_SELECTED = re.compile(r'version_(\d+)(?:_refit_trainval)?$')
_SPLITS = (('Training', 'train'), ('Validation', 'val'), ('Holdout', 'test'))
# Static tasks carry a placeholder anchor at the epoch.
_NO_TIME = '1970-01-01'


def _finite(value: Any) -> Optional[float]:
    ok = (isinstance(value, (int, float)) and not isinstance(value, bool)
          and math.isfinite(value))
    return value if ok else None


def _series(trial: Any) -> Series:
    """One trial's scalars with the same tags and epoch labels as the UI."""
    series: Series = {}
    for epoch, record in mapping(mapping(trial).get('metrics')).items():
        if not re.fullmatch(r'[0-9]+', epoch):
            continue
        for field, prefix in (('train_metrics', 'train_'),
                              ('validation_metrics', 'val_')):
            for tag, value in mapping(mapping(record).get(field)).items():
                if tag.startswith(prefix) and '_per_label_' not in tag:
                    series.setdefault(tag, []).append(
                        (int(epoch), _finite(value)))
    for tag, points in series.items():
        points.sort(key=lambda point: point[0])
        # The TensorBoard hook adjusts each series independently.
        if points[0][0] == 0:
            series[tag] = [(epoch + 1, value) for epoch, value in points]
    return series


def _selected(run: Run, trials: dict[str, Series]) -> Optional[str]:
    """The key of the experiment the published model came from.

    The retained run's ``best_trainer.json`` names it as ``version_<key>``.
    Without a retained run, it is the one experiment that recorded the
    bundle's validation score for the tuning metric.
    """
    if len(trials) < 2:
        return None
    match = _SELECTED.search(run.explanations.trial_root or '')
    if match and match[1] in trials:
        return match[1]
    tag = f'val_{run.tune}'
    score = run.metrics.get(tag)
    if run.tune is None or score is None:
        return None
    found = [
        key for key, series in trials.items()
        if any(value is not None and math.isclose(value, score, rel_tol=1e-9)
               for _, value in series.get(tag, []))
    ]
    return found[0] if len(found) == 1 else None


def history_card(run: Run) -> Optional[dict[str, Any]]:
    """Training and validation scalars of every search experiment."""
    trials = {
        key: series
        for key, trial in mapping(mapping(run.history).get(
            'trial_progress')).items() if (series := _series(trial))
    }
    if not trials:
        return None
    best = _selected(run, trials)
    experiments = [
        dict(label=f'Experiment {int(key) + 1 if key.isdigit() else i + 1}',
             best=key == best, series=series)
        for i, (key, series) in enumerate(trials.items())
    ]
    tags = sorted(tag for series in trials.values() for tag in series)
    metrics = list(dict.fromkeys(tag.split('_', 1)[1] for tag in tags))
    if run.tune in metrics:
        metrics.remove(run.tune)
        metrics.insert(0, run.tune)
    return dict(title='Learning Curves', metrics=metrics,
                experiments=experiments)


def _day(value: Any) -> Optional[str]:
    return value[:10] if isinstance(value, str) and value else None


def _seed_day(value: Any) -> Optional[str]:
    """A day from a nanosecond seed time of the bundle's input manifest."""
    number = integer(value)
    try:
        return None if number is None else str(pd.Timestamp(number)).split()[0]
    except (OverflowError, ValueError):
        return None


def _split(run: Run, name: str) -> Optional[dict[str, Any]]:
    stats = run.splits[name]
    plan = mapping(run.input_manifest.get('split'))
    rows = integer(stats.get('num_train_rows'))
    if rows is None:
        rows = integer(mapping(plan.get('stage_counts')).get(name))
    if rows is None:
        return None
    first, last = _day(stats.get('min_date')), _day(stats.get('max_date'))
    seeds = plan.get('stage_seed_times')
    seeds = mapping(seeds).get(name)
    if not (first and last) and isinstance(seeds, list) and len(seeds) == 2:
        first, last = _seed_day(seeds[0]), _seed_day(seeds[1])
    target = mapping(stats.get('target_distribution'))
    counts = mapping(target.get('count'))
    entities = integer(
        _data_summaries._value({
            'num_entities': integer(stats.get('num_entities')),
            'num_train_rows': rows
        }, target, 'entities', {}))
    positive = None
    if counts and set(counts) <= {'true', 'false'} and all(
            integer(value) is not None for value in counts.values()):
        positive = _data_summaries._value(stats, target, 'positive rate', {})
    return dict(
        rows=rows, entities=entities, positive=positive,
        mean=_finite(target.get('target_mean')),
        time=None if not (first and last) else
        first if first == last else f'{first} to {last}')


def _count(values: list[int]) -> list[str]:
    return [f'{value:,}' for value in values]


def _share(values: list[float]) -> list[str]:
    return [fmt.percent(value) for value in values]


def table_card(run: Run) -> Optional[dict[str, Any]]:
    """Rows per split, with entities, target balance and time range."""
    splits = [(label, facts) for label, name in _SPLITS
              if (facts := _split(run, name)) is not None]
    if not splits:
        return None
    if all(facts['time'] in (None, _NO_TIME) for _, facts in splits):
        for _, facts in splits:
            facts['time'] = None
    entity = _text.entity(run.query) or 'entities'
    columns = (('rows', 'Rows',
                _count), ('entities', f'Distinct {entity} (approx.)',
                          _count), ('positive', 'Positive rate', _share),
               ('mean', 'Target mean', fmt.numbers), ('time', 'Anchor times',
                                                      list))
    texts = {}
    for key, label, text in columns:
        present = [f[key] for _, f in splits if f[key] is not None]
        if present:
            shown = iter(text(present))
            texts[label] = [
                None if f[key] is None else next(shown) for _, f in splits
            ]
    return dict(
        title='Training Table', columns=list(texts), rows=[
            dict(name=name,
                 texts=[column[i] or '' for column in texts.values()])
            for i, (name, _) in enumerate(splits)
        ])
