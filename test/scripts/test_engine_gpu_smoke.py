# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small guards for malformed GPU smoke outputs; no training or CUDA required."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.agent_local
RUNNER = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/wheel_gpu_smoke.py')


def runner():
    assert RUNNER.is_file(), 'The standalone GPU smoke runner is missing'
    spec = importlib.util.spec_from_file_location('gpu_smoke', RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'foreign', 'nan'])
def test_scalar_outputs_reject_incomplete_or_invalid_entities(change):
    check = runner().prediction_values
    frame = pd.DataFrame({
        'ENTITY': [20, 10],
        'TARGET_PRED': [2., 1.],
        'ANCHOR_TIMESTAMP': ['2025-01-01'] * 2
    })
    assert check('regression', frame, [10, 20],
                 anchor='2025-01-01').tolist() == [[1.], [2.]]
    if change == 'missing':
        frame = frame.iloc[:1]
    elif change == 'duplicate':
        frame.loc[0, 'ENTITY'] = 10
    elif change == 'foreign':
        frame.loc[0, 'ENTITY'] = 30
    else:
        frame.loc[0, 'TARGET_PRED'] = np.nan
    with pytest.raises(ValueError):
        check('regression', frame, [10, 20], anchor='2025-01-01')


def test_class_outputs_require_every_coordinate_and_correct_chosen_class():
    check = runner().prediction_values
    frame = pd.DataFrame({
        'ENTITY': [10, 10, 20, 20],
        'CLASS': ['a', 'b', 'a', 'b'],
        'SCORE': [.8, .2, .3, .7],
        'PREDICTED': [True, False, False, True],
        'ANCHOR_TIMESTAMP': ['2025-01-01'] * 4
    })
    args = dict(classes=['a', 'b'], anchor='2025-01-01')
    assert check('multiclass', frame, [10, 20], **args).tolist() == [[.8, .2],
                                                                     [.3, .7]]
    for broken in (frame.iloc[:3], frame.assign(CLASS=['a'] * 4),
                   frame.assign(PREDICTED=[False, True, False, True]),
                   frame.assign(SCORE=[.8, .9, .3, .7])):
        with pytest.raises(ValueError):
            check('multiclass', broken, [10, 20], **args)


def test_forecasts_require_all_twelve_horizons_at_the_original_anchor():
    check = runner().prediction_values
    times = pd.date_range('2025-01-01', periods=12, freq='7D')
    frame = pd.DataFrame({
        'ENTITY': [10] * 12,
        'TIMESTAMP': times,
        'TARGET_PRED': range(12),
        'ANCHOR_TIMESTAMP': ['2025-01-01'] * 12
    })
    assert check('forecasting', frame, [10],
                 anchor='2025-01-01').shape == (1, 12)
    for broken in (frame.iloc[:11], frame.assign(TIMESTAMP=[times[0]] * 12),
                   frame.assign(ANCHOR_TIMESTAMP='2025-01-02')):
        with pytest.raises(ValueError):
            check('forecasting', broken, [10], anchor='2025-01-01')


def test_links_require_the_complete_catalog_without_repeated_pairs():
    check = runner().prediction_values
    frame = pd.DataFrame({
        'ENTITY': [10, 10],
        'CLASS': [1, 2],
        'SCORE': [-2., 3.]
    })
    assert check('static_link', frame, [10],
                 classes=[1, 2]).tolist() == [[-2., 3.]]
    for broken in (frame.iloc[:1], pd.concat([frame, frame.iloc[:1]]),
                   frame.assign(CLASS=[1, 3])):
        with pytest.raises(ValueError):
            check('static_link', broken, [10], classes=[1, 2])


def test_reload_comparison_measures_drift_and_rejects_unqualified_passes():
    compare = runner().compare_predictions
    result = compare([[0., 1.]], [[0., 1.000001]], atol=0., rtol=0.,
                     qualify=True)
    assert result['max_abs_difference'] == pytest.approx(1e-6)
    assert result['within_tolerance'] is False
    with pytest.raises(ValueError):
        compare([[0., 1.]], [[0., 1.000001]], atol=0., rtol=0.)
    assert compare([[0., 1.]], [[0., 1.000001]], atol=2e-6,
                   rtol=0.)['within_tolerance']
    for before, after in (([], []), ([[1.]], [[1., 2.]]), ([[1.]], [[np.inf]])):
        with pytest.raises(ValueError):
            compare(before, after, atol=1., rtol=1., qualify=True)


@pytest.mark.parametrize('task', ['binary', 'multiclass', 'multilabel'])
def test_reload_rejects_changed_labels_even_with_close_scores(task):
    smoke = runner()
    if task == 'binary':
        before = pd.DataFrame({
            'ENTITY': [10],
            'False_PROB': [.5000001],
            'True_PROB': [.4999999],
            'TARGET_PRED': [False]
        })
        after = pd.DataFrame({
            'ENTITY': [10],
            'False_PROB': [.4999999],
            'True_PROB': [.5000001],
            'TARGET_PRED': [True]
        })
        classes = []
    else:
        scores = [.5, .5] if task == 'multiclass' else [.4999999, .5000001]
        before = pd.DataFrame({
            'ENTITY': [10, 10],
            'CLASS': ['a', 'b'],
            'SCORE':
            scores,
            'PREDICTED':
            [True, False] if task == 'multiclass' else [False, True]
        })
        after = before.assign(
            PREDICTED=[False, True] if task == 'multiclass' else [True, True],
            SCORE=scores if task == 'multiclass' else [.5000001, .5000001])
        classes = ['a', 'b']
    before['ANCHOR_TIMESTAMP'] = after['ANCHOR_TIMESTAMP'] = '2025-01-01'
    values = [
        smoke.prediction_values(task, frame, [10], classes=classes,
                                anchor='2025-01-01')
        for frame in (before, after)
    ]
    decisions = [
        smoke.prediction_decisions(task, frame, [10], classes=classes)
        for frame in (before, after)
    ]
    options = dict(atol=1e-5, rtol=0., before_decisions=decisions[0],
                   after_decisions=decisions[1])
    assert smoke.compare_predictions(*values, qualify=True,
                                     **options)['decisions_equal'] is False
    with pytest.raises(ValueError, match='decisions'):
        smoke.compare_predictions(*values, **options)


@pytest.mark.parametrize('task',
                         ['multilabel_ranking', 'static_link', 'temporal_link'])
def test_reload_rejects_changed_ranks_even_with_identical_aligned_scores(task):
    smoke = runner()
    before = pd.DataFrame({
        'ENTITY': [20, 20, 10, 10],
        'CLASS': [1, 2, 1, 2],
        'SCORE': [.6, .4, .7, .3],
        'ANCHOR_TIMESTAMP': ['2025-01-01'] * 4
    })
    after = before.iloc[[1, 0, 2, 3]]
    options = dict(classes=[1, 2], anchor='2025-01-01')
    values = [
        smoke.prediction_values(task, frame, [10, 20], **options)
        for frame in (before, after)
    ]
    decisions = [
        smoke.prediction_decisions(task, frame, [10, 20], classes=[1, 2])
        for frame in (before, after)
    ]
    assert decisions[0].tolist() == [[0, 1], [0, 1]]
    np.testing.assert_array_equal(values[0], values[1])
    with pytest.raises(ValueError, match='decisions'):
        smoke.compare_predictions(*values, atol=0., rtol=0.,
                                  before_decisions=decisions[0],
                                  after_decisions=decisions[1])
