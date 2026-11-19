# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forecasting card: Forecast against actual, from the published model's
holdout predictions, one row per entity and forecast step."""

from typing import Any, Optional

import numpy as np
import pandas as pd

from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._holdout import equal_count_groups
from kumo.engine._explainability._run import Run

# Forecast step lengths in days and the word for one step.
_PERIODS = ((1, 'day'), (7, 'week'), (28, 'month'), (29, 'month'),
            (30, 'month'), (31, 'month'), (365, 'year'), (366, 'year'))


def period(steps: pd.Series) -> str:
    """The calendar word for the spacing of the forecast steps."""
    stamps = pd.to_datetime(steps.drop_duplicates()).sort_values()
    days = stamps.diff().dropna().dt.days
    if days.empty:
        return 'step'
    typical = int(days.median())
    for length, word in _PERIODS:
        if typical == length:
            return word
    return 'step'


def _weeks(frame: pd.DataFrame, word: str) -> dict[str, Any]:
    per = frame.groupby('TIMESTAMP', sort=True).agg(mae=('abs_error', 'mean'),
                                                    zero=('abs_actual', 'mean'),
                                                    bias=('error', 'mean'))
    texts = fmt.numbers(list(per['mae']) + list(per['zero']))
    n = len(per)
    return {
        'id':
        'horizon',
        'label':
        f'By horizon {word}',
        'x':
        f'forecast {word}',
        'y':
        'mean absolute error',
        'series': ['model', 'predict zero'],
        'steps': [{
            'label': fmt.date(str(stamp)),
            'model': float(row.mae),
            'model_text': texts[i],
            'zero': float(row.zero),
            'zero_text': texts[n + i],
            'bias_callout': f'bias {fmt.signed(float(row.bias))}',
        } for i, (stamp, row) in enumerate(per.iterrows())],
    }


def _totals(frame: pd.DataFrame, word: str, entities: str) -> dict[str, Any]:
    per = frame.groupby('TIMESTAMP',
                        sort=True).agg(actual=('TARGET', 'sum'),
                                       forecast=('TARGET_PRED', 'sum'))
    actual, forecast = float(per['actual'].sum()), float(per['forecast'].sum())
    callout = None
    if actual > 0:
        callout = (f'forecast total {fmt.percent(forecast / actual)} of '
                   f'actual over {len(per)} {word}s')
    values = list(per['actual']) + list(per['forecast'])
    target = frame['TARGET'].to_numpy()
    if np.allclose(target, np.round(target), rtol=0, atol=1e-3):
        # A count target totals to whole units; the forecast is shown on the
        # same footing.
        totals = [fmt.count(round(v)) for v in values]
    else:
        totals = fmt.numbers(values)
    n = len(per)
    return {
        'id':
        'totals',
        'label':
        f'{word.capitalize()}ly totals'
        if word in ('week', 'month', 'year') else 'Totals by step',
        'x':
        f'forecast {word}',
        'y':
        f'total over all {entities}',
        'series': ['actual', 'forecast'],
        'steps': [{
            'label': fmt.date(str(stamp)),
            'actual': float(row.actual),
            'actual_text': totals[i],
            'forecast': float(row.forecast),
            'forecast_text': totals[n + i],
        } for i, (stamp, row) in enumerate(per.iterrows())],
        'total_callout':
        callout,
    }


def _tenths(run: Run, frame: pd.DataFrame) -> Optional[dict[str, Any]]:
    per = frame.groupby('ENTITY',
                        sort=True).agg(actual=('TARGET', 'sum'),
                                       forecast=('TARGET_PRED', 'sum'))
    total = float(per['actual'].sum())
    if total <= 0:
        return None
    group = equal_count_groups(per['forecast'].to_numpy(dtype=float),
                               per.index.to_numpy())
    rows = []
    for k in np.unique(group):
        mask = group == k
        share = float(per['actual'].to_numpy()[mask].sum()) / total
        rows.append({
            'label': str(k + 1),
            'entities': int(mask.sum()),
            'share': share,
            'share_text': fmt.percent(share),
        })
    top = rows[-1]
    return {
        'id':
        'tenths',
        'label':
        'By forecast tenth',
        'x':
        'forecast tenth, lowest to highest',
        'y':
        'share of actual units',
        'tenths':
        rows,
        'reference':
        0.1,
        'reference_callout':
        'equal share 10%',
        'top_callout':
        f'{top["share_text"]} of actual units · '
        f'{fmt.count(top["entities"])} {run.entities}',
    }


def forecast_card(run: Run) -> Optional[dict[str, Any]]:
    holdout = run.holdout
    if holdout is None or len(holdout) == 0:
        return None
    frame = holdout.copy()
    frame['TARGET'] = frame['TARGET'].astype(float)
    frame['TARGET_PRED'] = frame['TARGET_PRED'].astype(float)
    frame = frame[np.isfinite(frame['TARGET'])
                  & np.isfinite(frame['TARGET_PRED'])]
    if frame.empty:
        return None
    frame['error'] = frame['TARGET_PRED'] - frame['TARGET']
    frame['abs_error'] = frame['error'].abs()
    frame['abs_actual'] = frame['TARGET'].abs()
    word = period(frame['TIMESTAMP'])
    views = [_weeks(frame, word), _totals(frame, word, run.entities)]
    tenths = _tenths(run, frame)
    if tenths is not None:
        views.append(tenths)
    return {
        'title': 'Forecast against actual',
        'views': views,
        'coverage': run.coverage.holdout_line(),
    }
