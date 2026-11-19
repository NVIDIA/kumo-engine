# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The widget header: the predictive query and one line about the holdout."""

import posixpath
import re
from typing import Any, Optional

from kumo.engine._explainability import _files
from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._forecasting import period
from kumo.engine._explainability._metrics import catalog, fits, positive_share
from kumo.engine._explainability._run import (
    CLASSIFICATION,
    FORECASTING,
    LINK_PREDICTION,
    REGRESSION,
    Run,
)


def query(run: Run) -> Optional[str]:
    spec = _files.read_json(
        posixpath.join(run.files.bundle, 'training-spec.json'))
    value = spec.get('predictive_query') if isinstance(spec, dict) else None
    return ' '.join(value.split()) if isinstance(value, str) else None


_TARGET = re.compile(
    r'^PREDICT\s+(.+?)\s+(?:RANK\s+TOP\b|FORECAST\b|FOR\s+EACH\b)',
    re.IGNORECASE)


def target(run: Run, short: bool = False) -> Optional[str]:
    """The query's target expression, for axis titles.

    ``short`` drops the time window, so ``SUM(transactions.price, 0, 30,
    days)`` reads ``SUM(transactions.price)``.
    """
    text = query(run)
    match = _TARGET.match(text) if text else None
    if not match:
        return None
    expr = match.group(1)
    if short:
        expr = re.sub(r'\(([^,()]*),[^()]*\)', r'(\1)', expr, count=1)
    return expr


def _anchor(run: Run) -> Optional[str]:
    stats = _files.split_stats(run.files) or {}
    split = stats.get('test_table_stats')
    if isinstance(split, dict) and split.get('anchor_date'):
        return f'anchor {fmt.date(split["anchor_date"])}'
    return None


def _holdout(run: Run) -> list[str]:
    frame = run.holdout
    if frame is None or not len(frame):
        return []
    if run.model_type == FORECASTING:
        entities = frame['ENTITY'].nunique()
        steps = frame['TIMESTAMP'].nunique()
        word = period(frame['TIMESTAMP'])
        unit = word if steps == 1 else f'{word}s'
        return [
            f'{fmt.count(entities)} {run.entities} × {steps} {unit} = '
            f'{fmt.count(len(frame))} rows'
        ]
    return [f'{fmt.count(len(frame))} {run.entities}']


def _detail(run: Run) -> list[str]:
    if run.model_type == CLASSIFICATION:
        share = positive_share(run)
        return [] if share is None else [f'{fmt.percent(share)} True']
    if run.model_type == REGRESSION and run.holdout is not None and len(
            run.holdout):
        zero = float((run.holdout['TARGET'] == 0).mean())
        return [f'{fmt.percent(zero)} with actual 0'] if zero >= 0.1 else []
    if run.model_type == LINK_PREDICTION:
        cat = catalog(run)
        if cat is None:
            return []
        return [
            f'{fmt.number(cat.mean_targets)} items each',
            f'catalog {fmt.count(cat.size)} {cat.table}'
        ]
    return []


def _link_entities(run: Run) -> list[str]:
    stats = _files.split_stats(run.files) or {}
    split = stats.get('test_table_stats')
    rows = split.get('num_train_rows') if isinstance(split, dict) else None
    return [f'{fmt.count(rows)} {run.entities}'] if isinstance(rows,
                                                               int) else []


def header(run: Run) -> dict[str, Any]:
    """The query and a holdout line; either may be missing."""
    parts = (_link_entities(run)
             if run.model_type == LINK_PREDICTION else _holdout(run))
    anchor = _anchor(run)
    if parts and anchor:
        parts.append(anchor)
    parts += _detail(run)
    line = ('Holdout: ' + ' · '.join(parts)) if parts else None
    if fits(run).full_refit:
        note = 'Diagnostics and explanations describe the model before full refit.'
        line = f'{line} · {note}' if line else note
    return {'query': query(run), 'holdout': line}
