# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forecasting card: Forecast against actual, from the published model's
holdout predictions, one row per entity and forecast step."""

from typing import Any, Optional, cast

import numpy as np
import pandas as pd

from kumo.engine._explainability import _format as fmt
from kumo.engine._explainability._run import (
    FORECASTING,
    Run,
    equal_count_groups,
)

# Forecast step lengths in days and the word for one step.
_PERIODS = {
    1: 'day',
    7: 'week',
    **dict.fromkeys(range(28, 32), 'month'), 365: 'year',
    366: 'year'
}


def period(steps: pd.Series) -> str:
    """The calendar word for the spacing of the forecast steps."""
    days = pd.to_datetime(
        steps.drop_duplicates()).sort_values().diff().dropna().dt.days
    return 'step' if days.empty else _PERIODS.get(int(days.median()), 'step')


def _steps(per: pd.DataFrame, texts: list[str]) -> list[dict[str, Any]]:
    """One step per row of the two columns of ``per``, labelled by date."""
    n = len(per)
    return [
        dict(label=str(stamp)[:10], values=[float(a), float(b)],
             texts=[texts[i], texts[n + i]])
        for i, (stamp, (a, b)) in enumerate(per.iterrows())
    ]


def forecast_card(run: Run) -> Optional[dict[str, Any]]:
    if run.model_type != FORECASTING or run.holdout is None:
        return None
    frame = run.holdout.astype({'TARGET': float, 'TARGET_PRED': float})
    frame = cast(
        pd.DataFrame, frame[np.isfinite(frame['TARGET'])
                            & np.isfinite(frame['TARGET_PRED'])])
    if frame.empty:
        return None
    error = frame['TARGET_PRED'] - frame['TARGET']
    word = period(cast(pd.Series, frame['TIMESTAMP']))
    by_step = frame.assign(model=error.abs(), zero=frame['TARGET'].abs(),
                           bias=error).groupby('TIMESTAMP', sort=True)
    horizon = cast(pd.DataFrame, by_step[['model', 'zero', 'bias']].mean())
    steps = _steps(cast(pd.DataFrame, horizon[['model', 'zero']]),
                   fmt.numbers([*horizon['model'], *horizon['zero']]))
    for step, bias in zip(steps, horizon['bias']):
        step['sub'] = f'bias {fmt.signed(float(bias))}'
    totals = cast(pd.DataFrame, by_step[['TARGET', 'TARGET_PRED']].sum())
    values = [*totals['TARGET'], *totals['TARGET_PRED']]
    target = frame['TARGET'].to_numpy()
    # A count target totals to whole units; the forecast is shown on the
    # same footing.
    texts = ([f'{int(round(v)):,}' for v in values] if np.allclose(
        target, np.round(target), rtol=0, atol=1e-3) else fmt.numbers(values))
    actual, forecast = totals['TARGET'].sum(), totals['TARGET_PRED'].sum()
    views = [
        dict(id='horizon', label=f'By horizon {word}', x=f'forecast {word}',
             y='mean absolute error', series=['model',
                                              'predict zero'], steps=steps),
        dict(
            id='totals', x=f'forecast {word}',
            label=(f'{word.capitalize()}ly totals'
                   if word in ('week', 'month', 'year') else 'Totals by step'),
            y=f'total over all {run.entities}', series=['actual', 'forecast'],
            steps=_steps(totals, texts),
            total_callout=(f'forecast total {fmt.percent(forecast / actual)} '
                           f'of actual over {len(totals)} {word}s')
            if actual > 0 else None),
    ]
    per = cast(
        pd.DataFrame,
        frame.groupby('ENTITY', sort=True)[['TARGET', 'TARGET_PRED']].sum())
    total = float(cast(pd.Series, per['TARGET']).sum())
    if total > 0:
        group = equal_count_groups(per['TARGET_PRED'].to_numpy(dtype=float),
                                   per.index.to_numpy())
        rows = []
        for k in np.unique(group):
            share = float(per['TARGET'].to_numpy()[group == k].sum()) / total
            rows.append(
                dict(label=str(k + 1), values=[share],
                     texts=[fmt.percent(share)]))
        views.append(
            dict(
                id='tenths', label='By total forecast',
                x='Entities by total forecast (low to high)',
                y='Share of actual total (%)', tenths=rows, reference=0.1,
                reference_callout='equal share 10%', highlight=len(rows) - 1,
                callout=(f'{rows[-1]["texts"][0]} of actual units · '
                         f'{int((group == group.max()).sum()):,} '
                         f'{run.entities}')))
    return dict(title='Forecast against actual', views=views,
                coverage=run.holdout_coverage())
