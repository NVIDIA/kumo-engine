# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Column importance and Column groups, from Kumo's own explanation pass.

Both read the evaluated fit's explanation files: gradient sensitivity per
input column (``explain_test/scores.csv``) and per-column value groups
(``column_analysis/test/column_analysis.json``).
"""

import posixpath
from dataclasses import asdict
from typing import Any, Optional

import numpy as np

from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._header import target
from kumo.engine._explainability._run import (
    CLASSIFICATION,
    REGRESSION,
    Run,
)
from kumo.engine.explain._artifacts import read_cohorts, read_gradient

TOP = 10
MAX_GROUPS = 10
DOMINANT_GROUP = 0.9
GAP_MIN_SHARE = 0.05
COUNT_COLUMN = 'COUNT(*)'


def _trial_file(run: Run, *parts: str) -> Optional[str]:
    trial = run.files.trial
    return None if trial is None else posixpath.join(trial, *parts)


def _scores(run: Run) -> Optional[list[tuple[str, str, float]]]:
    path = _trial_file(run, 'explain_test', 'scores.csv')
    if path is None:
        return None
    try:
        scores = read_gradient(path)
    except (OSError, ValueError):
        return None
    out = [(table, column, score) for table, values in scores.tables.items()
           for column, score in values.columns.items()]
    if any(score < 0 for _, _, score in out):
        return None
    out.sort(key=lambda r: (-r[2], r[0], r[1]))
    return out if sum(r[2] for r in out) > 0 else None


def cohorts(run: Run) -> Optional[list[dict[str, Any]]]:
    if run.model_type not in (CLASSIFICATION, REGRESSION):
        return None
    path = _trial_file(run, 'column_analysis', 'test', 'column_analysis.json')
    if path is None:
        return None
    try:
        artifact = read_cohorts(path)
    except (OSError, ValueError):
        return None
    if artifact.task_type.value != run.task_type:
        return None
    entries = [asdict(c) for c in artifact.cohorts if c.targets is not None]
    return entries if any(e['prediction_score'] > 0 for e in entries) else None


def _key(entry: dict[str, Any]) -> str:
    return f"{entry['table_name']}|{entry['hop']}|{entry['col_name']}"


def _label(table: str, column: str) -> str:
    return f'{table}.{column}'


def _dominant(entry: dict[str, Any]) -> Optional[str]:
    """I3: the value group holding 90% or more of the column's population."""
    pops = np.asarray(entry['populations'], dtype=float)
    total = pops.sum()
    if total <= 0:
        return None
    j = int(np.argmax(pops))
    share = pops[j] / total
    if share < DOMINANT_GROUP:
        return None
    return f"{fmt.percent(share)} in {entry['cohorts'][j]}"


def _ranking(
        items: list[tuple[str, str, float, Optional[str]]],
        dominant: Optional[dict[str, Optional[str]]] = None) -> dict[str, Any]:
    """Bars as shares of the summed score, top ten then the remainder."""
    total = sum(score for _, _, score, _ in items)
    bars = []
    for table, column, score, group in items:
        bar = {
            'table': table,
            'column': column,
            'label': _label(table, column),
            'share': score / total,
            'text': fmt.percent(score / total),
            'group': group,
        }
        if dominant is not None and group is not None:
            bar['callout'] = dominant.get(group)
        bars.append(bar)
    rest = bars[TOP:]
    remainder = None
    if rest:
        share = sum(bar['share'] for bar in rest)
        remainder = {
            'label': f'{len(rest)} other columns',
            'share': share,
            'text': fmt.percent(share),
        }
    by_table: dict[str, float] = {}
    for bar in bars:
        by_table[bar['table']] = by_table.get(bar['table'], 0) + bar['share']
    tables = sorted(by_table.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        'bars':
        bars,
        'top':
        min(TOP, len(bars)),
        'remainder':
        remainder,
        'top_callout':
        f"{fmt.percent(bars[0]['share'])} of total",
        'tables': [{
            'table': t,
            'share': s,
            'text': f'{t} {fmt.percent(s)}'
        } for t, s in tables],
    }


def _first_entry(entries: list[dict[str, Any]], table: str,
                 column: str) -> Optional[dict[str, Any]]:
    matches = [
        e for e in entries
        if e['table_name'] == table and e['col_name'] == column
    ]
    return min(matches, key=lambda e: e['hop']) if matches else None


def importance(run: Run) -> Optional[dict[str, Any]]:
    scores = _scores(run)
    if scores is None:
        return None
    entries = cohorts(run)
    views = []
    linked = []
    for table, column, score in scores:
        entry = None if entries is None else _first_entry(
            entries, table, column)
        linked.append(
            (table, column, score, None if entry is None else _key(entry)))
    sensitivity = _ranking(linked)
    sensitivity['legend'] = True
    views.append({'id': 'sensitivity', 'label': 'Sensitivity', **sensitivity})
    if entries is not None:
        ranked = sorted(
            entries, key=lambda e: (-float(e['prediction_score']),
                                    _label(e['table_name'], e['col_name'])))
        dominant = {_key(e): _dominant(e) for e in entries}
        variation = _ranking([(e['table_name'], e['col_name'],
                               float(e['prediction_score']), _key(e))
                              for e in ranked], dominant)
        variation['legend'] = False
        views.append({
            'id': 'variation',
            'label': 'Variation across groups',
            **variation
        })
    return {
        'title': 'Column importance',
        'views': views,
        'coverage': run.coverage.line(run.coverage.explanation),
    }


def _share_unit(run: Run, entry: dict[str, Any]) -> str:
    if entry['hop'] == 0 or entry['col_name'] == COUNT_COLUMN:
        return f'of {run.entities}'
    return f"of sampled {entry['table_name']} rows"


def group_view(run: Run, entry: dict[str, Any], total_score: float,
               class_name: Optional[str] = None) -> dict[str, Any]:
    names = [str(c) for c in entry['cohorts']][:MAX_GROUPS]
    pops = np.asarray(entry['populations'], dtype=float)[:MAX_GROUPS]
    predictions, targets = entry['predictions'], entry['targets']
    if class_name is not None:
        predictions, targets = predictions[class_name], targets[class_name]
    pred = np.asarray(predictions, dtype=float)[:MAX_GROUPS]
    actual = np.asarray(targets, dtype=float)[:MAX_GROUPS]
    weight = pops.sum()
    avg_actual = float((pops * actual).sum() / weight) if weight else None
    avg_pred = float((pops * pred).sum() / weight) if weight else None
    eligible = np.where(pops >= GAP_MIN_SHARE)[0]
    gap = None
    if len(eligible):
        j = int(eligible[np.argmax(np.abs(pred - actual)[eligible])])
        gap = {
            'index':
            j,
            'text':
            f'predicted {fmt.number(pred[j])} · actual '
            f'{fmt.number(actual[j])}'
        }
    stype = entry.get('stype') or ('count'
                                   if entry['col_name'] == COUNT_COLUMN else '')
    share = float(entry['prediction_score']) / total_score
    header = (f"{_label(entry['table_name'], entry['col_name'])} · hop "
              f"{entry['hop']} · {stype} · variation "
              f"{fmt.percent(share)}")
    unit = _share_unit(run, entry)
    values = fmt.numbers(
        list(pred) + list(actual) +
        [v for v in (avg_actual, avg_pred) if v is not None])
    n = len(names)
    return {
        'key':
        _key(entry),
        'header':
        header,
        'groups': [{
            'label': names[i],
            'share': float(pops[i]),
            'share_text': f'{fmt.percent(pops[i])} {unit}',
            'predicted': float(pred[i]),
            'predicted_text': values[i],
            'actual': float(actual[i]),
            'actual_text': values[n + i],
        } for i in range(n)],
        'axis':
        _axis(run) if class_name is None else f'share {class_name}',
        'average_actual':
        avg_actual,
        'average_predicted':
        avg_pred,
        'average_callout':
        None if avg_actual is None or avg_pred is None else
        (f'average actual {fmt.number(avg_actual)} · '
         f'average predicted {fmt.number(avg_pred)}'),
        'gap_callout':
        gap,
    }


def _axis(run: Run) -> str:
    """The group chart's value axis: what both bar series average."""
    if run.model_type == CLASSIFICATION:
        return 'share True'
    return 'average ' + (target(run) or 'value')


def groups(run: Run) -> Optional[dict[str, Any]]:
    entries = cohorts(run)
    if entries is None:
        return None
    total = sum(float(e['prediction_score']) for e in entries)
    views = {}
    for entry in entries:
        if isinstance(entry['predictions'], dict):
            view = {
                'by_class': [{
                    'label': label,
                    **group_view(run, entry, total, label)
                } for label in entry['predictions']]
            }
        else:
            view = group_view(run, entry, total)
        views[_key(entry)] = view
    default = None
    scores = _scores(run)
    for table, column, _ in scores or []:
        entry = _first_entry(entries, table, column)
        if entry is not None:
            default = _key(entry)
            break
    if default is None:
        default = _key(max(entries, key=lambda e: float(e['prediction_score'])))
    return {
        'title': 'Column groups',
        'selectable': not scores,
        'default': default,
        'columns': views,
        'coverage': run.coverage.line(run.coverage.explanation),
    }
