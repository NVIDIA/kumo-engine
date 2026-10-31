# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small guards for malformed GPU smoke outputs; no training or CUDA required."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

RUNNER = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/test/wheel_gpu_smoke.py')


def runner():
    assert RUNNER.is_file(), 'The standalone GPU smoke runner is missing'
    spec = importlib.util.spec_from_file_location('gpu_smoke', RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(('task', 'targets'), [
    ('multiclass',
     ['breakfast', 'dinner', 'lunch', 'late', 'early', 'afternoon']),
    ('multilabel', ['breakfast|dinner', '', 'lunch|late', 'early|afternoon']),
    ('multilabel_ranking',
     ['breakfast|dinner', '', 'lunch|late', 'early|afternoon']),
    ('multilabel', [['breakfast', 'dinner'], [], ['lunch', 'late'],
                    ['early', 'afternoon']]),
    ('multilabel_ranking', [['breakfast', 'dinner'], [], ['lunch', 'late'],
                            ['early', 'afternoon']]),
])
def test_target_classes_decode_native_tokens_instead_of_string_characters(
        task, targets):
    assert runner().target_classes(task, targets) == [
        'afternoon', 'breakfast', 'dinner', 'early', 'late', 'lunch'
    ]


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


def test_reload_tolerance_accepts_boundary_and_rejects_larger_or_invalid_values(
):
    compare = runner().compare_predictions
    assert compare([[0.]], [[1e-5]])['within_tolerance']
    for before, after in (
        ([[0.]], [[np.nextafter(1e-5, np.inf)]]),
        ([[1.]], [[1.00002]]),
        ([], []),
        ([[1.]], [[1., 2.]]),
        ([[np.nan]], [[1.]]),
        ([[1.]], [[np.inf]]),
    ):
        with pytest.raises(ValueError):
            compare(before, after)


def test_training_proof_rejects_zero_work_and_incomplete_fits():
    check = runner().verify_training_work
    valid = dict(fits=1, completed_fits=1, training_batches=2,
                 optimizer_steps=4)
    check(valid, 3)
    for field in valid:
        with pytest.raises(ValueError,
                           match='optimizer updates or CUDA batches'):
            check(dict(valid, **{field: 0}), 3)
    for steps in (None, 0, 5):
        with pytest.raises(ValueError):
            check(valid, steps)


def test_custom_reload_tolerance_stays_enforced_and_rejects_invalid_bounds():
    compare = runner().compare_predictions
    assert compare([[0.]], [[.004]], atol=.005)['within_tolerance']
    with pytest.raises(ValueError, match='exceeds'):
        compare([[0.]], [[.006]], atol=.005)
    with pytest.raises(ValueError, match='decisions'):
        compare([[0.]], [[.004]], atol=.005, before_decisions=[True],
                after_decisions=[False])
    for bound in (-1., np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match='finite and nonnegative'):
            compare([[0.]], [[0.]], atol=bound)


def test_rtx_workflow_accepts_observed_repeat_difference_and_rejects_drift():
    root = RUNNER.parents[3]
    workflow = yaml.safe_load(
        (root / '.github/workflows/kumo-engine-pr.yml').read_text())
    action = yaml.safe_load(
        (root / '.github/actions/engine-gpu-smoke/action.yml').read_text())
    step = next(step for step in workflow['jobs']['gpu']['steps']
                if step.get('uses') == './.github/actions/engine-gpu-smoke')
    bound = float(step['with'].get('reload-atol',
                                   action['inputs']['reload-atol']['default']))
    compare = runner().compare_predictions
    assert compare([[0.]], [[3.975629806518555e-5]],
                   atol=bound)['within_tolerance']
    with pytest.raises(ValueError, match='exceeds'):
        compare([[0.]], [[1.0001e-4]], atol=bound)


@pytest.mark.parametrize('task', [
    'binary', 'multiclass', 'multilabel', 'multilabel_ranking', 'regression',
    'forecasting', 'static_link', 'temporal_link'
])
def test_every_task_rejects_corrupted_reload_values_with_unchanged_decisions(
        task):
    width = 12 if task == 'forecasting' else 2
    before = np.ones((1, width))
    after = before.copy()
    after[0, 0] += .01
    decisions = None if task in ('regression', 'forecasting') else [[0, 1]]
    with pytest.raises(ValueError, match='exceeds'):
        runner().compare_predictions(before, after, before_decisions=decisions,
                                     after_decisions=decisions)


@pytest.mark.parametrize('task', [
    'binary', 'multiclass', 'multilabel', 'multilabel_ranking', 'static_link',
    'temporal_link'
])
def test_changed_labels_or_ranks_fail_even_when_scores_are_within_tolerance(
        task):
    smoke = runner()
    if task == 'binary':
        before = pd.DataFrame(
            dict(ENTITY=[10], False_PROB=[.5000001], True_PROB=[.4999999],
                 TARGET_PRED=[False]))
        after = before.assign(False_PROB=.4999999, True_PROB=.5000001,
                              TARGET_PRED=True)
        classes = []
    else:
        before = pd.DataFrame(
            dict(ENTITY=[10, 10], CLASS=['a', 'b'], SCORE=[.5, .5],
                 PREDICTED=[True, False], RANK=[0, 1]))
        if task == 'multilabel':
            before['SCORE'] = [.5000001, .4999999]
        after = before.assign(PREDICTED=[False, True], RANK=[1, 0])
        if task == 'multilabel':
            after['SCORE'] = [.4999999, .5000001]
        classes = ['a', 'b']
    before['ANCHOR_TIMESTAMP'] = after['ANCHOR_TIMESTAMP'] = '2025-01-01'
    options = dict(classes=classes, anchor='2025-01-01')
    values = [
        smoke.prediction_values(task, frame, [10], **options)
        for frame in (before, after)
    ]
    decisions = [
        smoke.prediction_decisions(task, frame, [10], classes=classes)
        for frame in (before, after)
    ]
    with pytest.raises(ValueError, match='decisions'):
        smoke.compare_predictions(*values, before_decisions=decisions[0],
                                  after_decisions=decisions[1])
