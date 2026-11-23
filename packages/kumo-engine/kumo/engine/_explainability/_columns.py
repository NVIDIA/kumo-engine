# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Column importance and Column groups, from Kumo's own explanation pass.

Both read the evaluated fit's retained explanations: gradient sensitivity per
input column and per-column value groups.
"""

from typing import Any, Optional

import numpy as np
from kumoapi.explain.column_analysis import ColumnCohorts

from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._run import CLASSIFICATION, REGRESSION, Run

TOP = 10
DOMINANT_GROUP = 0.9
GAP_MIN_SHARE = 0.05
COUNT_COLUMN = 'COUNT(*)'


def _scores(run: Run) -> Optional[list[tuple[str, str, float]]]:
    if run.gradient is None:
        return None
    out = sorted(((table, column, score)
                  for table, values in run.gradient.tables.items()
                  for column, score in values.columns.items()), key=lambda r:
                 (-r[2], r[0], r[1]))
    return out if sum(r[2] for r in out) > 0 else None


def cohorts(run: Run) -> Optional[list[ColumnCohorts]]:
    analysis = run.cohorts
    if (run.model_type not in (CLASSIFICATION, REGRESSION) or analysis is None
            or analysis.task_type.value != run.task_type):
        return None
    if run.batch:
        return analysis.cohorts or None
    entries = [c for c in analysis.cohorts if c.targets is not None]
    return entries if any(e.prediction_score > 0 for e in entries) else None


def _key(entry: ColumnCohorts) -> str:
    return f'{entry.table_name}|{entry.hop}|{entry.col_name}'


def _dominant(entry: ColumnCohorts) -> Optional[str]:
    """The value group holding 90% or more of the column's population."""
    pops = np.asarray(entry.populations, dtype=float)
    j = int(np.argmax(pops))
    share = pops[j] / pops.sum()
    return (f'{fmt.percent(share)} in {entry.cohorts[j]}'
            if share >= DOMINANT_GROUP else None)


def _ranking(view_id: str, label: str, items: list[tuple[str, str, float,
                                                         Optional[str]]],
             dominant: dict[str, Optional[str]]) -> dict[str, Any]:
    """Bars as shares of the summed score, top ten then the remainder."""
    total = sum(score for _, _, score, _ in items)
    bars: list[dict[str, Any]] = [
        dict(table=table, label=f'{table}.{column}', share=score / total,
             text=fmt.percent(score / total), group=group,
             callout=dominant.get(group or ''))
        for table, column, score, group in items
    ]
    rest = sum(bar['share'] for bar in bars[TOP:])
    by_table: dict[str, float] = {}
    for bar in bars:
        by_table[bar['table']] = by_table.get(bar['table'], 0) + bar['share']
    tables = sorted(by_table.items(), key=lambda kv: (-kv[1], kv[0]))
    return dict(
        id=view_id, label=label, bars=bars, top=min(TOP, len(bars)),
        remainder=dict(label=f'{len(bars) - TOP} other columns', share=rest,
                       text=fmt.percent(rest)) if len(bars) > TOP else None,
        top_callout=f"{fmt.percent(bars[0]['share'])} of total",
        tables=[dict(table=t, text=f'{t} {fmt.percent(v)}')
                for t, v in tables], legend=view_id == 'sensitivity')


def _link(entries: list[ColumnCohorts], table: str,
          column: str) -> Optional[str]:
    """The cohort key of the column's nearest hop."""
    matches = [
        e for e in entries if e.table_name == table and e.col_name == column
    ]
    return _key(min(matches, key=lambda e: e.hop)) if matches else None


def importance(run: Run) -> Optional[dict[str, Any]]:
    scores = _scores(run)
    if scores is None:
        return None
    entries = cohorts(run) or []
    linked = [(table, column, score, _link(entries, table, column))
              for table, column, score in scores]
    views = [_ranking('sensitivity', 'Sensitivity', linked, {})]
    if entries:
        ranked = sorted(
            entries, key=lambda e:
            (-e.prediction_score, f'{e.table_name}.{e.col_name}'))
        views.append(
            _ranking('variation', 'Variation across groups',
                     [(e.table_name, e.col_name, e.prediction_score, _key(e))
                      for e in ranked],
                     {_key(e): _dominant(e)
                      for e in entries}))
    return dict(title='Column importance', views=views,
                coverage=run.explanation_coverage())


def group_view(run: Run, entry: ColumnCohorts, total_score: float,
               class_name: Optional[str] = None) -> dict[str, Any]:
    pops = np.asarray(entry.populations, dtype=float)
    # A batch shows predictions only.
    predictions, targets = entry.predictions, None if run.batch else entry.targets
    if isinstance(predictions, dict):
        predictions = predictions[class_name]
        targets = targets and targets[class_name]
    series = {
        name: np.asarray(values, dtype=float)
        for name, values in (('actual', targets), ('predicted', predictions))
        if values is not None
    }
    averages = {
        name: float((pops * values).sum() / pops.sum())
        for name, values in series.items()
    }
    eligible = np.where(pops >= GAP_MIN_SHARE)[0]
    gap = None
    if len(eligible) and 'actual' in series:
        pred, actual = series['predicted'], series['actual']
        j = int(eligible[np.argmax(np.abs(pred - actual)[eligible])])
        gap = dict(
            index=j, text=f'predicted {fmt.number(pred[j])} · actual '
            f'{fmt.number(actual[j])}')
    stype = entry.stype or ('count' if entry.col_name == COUNT_COLUMN else '')
    unit = ('of sampled rows' if run.batch else f'of {run.entities}'
            if entry.hop == 0 or entry.col_name == COUNT_COLUMN else
            f'of sampled {entry.table_name} rows')
    texts = fmt.numbers([v for values in series.values()
                         for v in values] + list(averages.values()))
    n = len(pops)
    if run.batch:
        axis = ('average predicted value' if run.model_type == REGRESSION else
                f'average predicted probability of {class_name or True}')
    elif class_name is not None:
        axis = f'share {class_name}'
    elif run.model_type == CLASSIFICATION:
        axis = 'share True'
    else:
        axis = 'average ' + (run.target() or 'value')
    groups = []
    for i in range(n):
        group = dict(label=str(entry.cohorts[i]),
                     share_text=f'{fmt.percent(pops[i])} {unit}')
        for k, (name, values) in enumerate(series.items()):
            group.update({
                name: float(values[i]),
                name + '_text': texts[k * n + i]
            })
        groups.append(group)
    share = entry.prediction_score / total_score if total_score else 0
    return dict(
        header=(f'{entry.table_name}.{entry.col_name} · hop {entry.hop} · '
                f'{stype} · variation {fmt.percent(share)}'), groups=groups,
        axis=axis, **{
            'average_' + name: v
            for name, v in averages.items()
        }, average_callout=' · '.join(f'average {name} {fmt.number(v)}'
                                      for name, v in averages.items()),
        gap_callout=gap)


def groups(run: Run) -> Optional[dict[str, Any]]:
    entries = cohorts(run)
    if entries is None:
        return None
    total = sum(e.prediction_score for e in entries)
    views = {
        _key(entry):
        dict(by_class=[
            dict(label=label, **group_view(run, entry, total, label))
            for label in entry.predictions
        ]) if isinstance(entry.predictions, dict) else group_view(
            run, entry, total)
        for entry in entries
    }
    scores = _scores(run)
    default = next(
        filter(None, (_link(entries, table, column)
                      for table, column, _ in scores or [])), None)
    return dict(
        title='Cohorts' if run.batch else 'Column groups',
        selectable=not scores, columns=views, default=default
        or _key(max(entries, key=lambda e: e.prediction_score)),
        coverage=run.explanation_coverage())
