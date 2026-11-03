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


def test_forecasts_accept_the_graphs_exported_time_column():
    check = runner().prediction_values
    frame = pd.DataFrame({
        'ENTITY': [10] * 12,
        'signup_time': pd.date_range('2025-01-01', periods=12, freq='7D'),
        'TARGET_PRED': range(12),
        'ANCHOR_TIMESTAMP': ['2025-01-01'] * 12
    })
    assert check('forecasting', frame, [10], anchor='2025-01-01').tolist() == [
        list(range(12))
    ]
    for broken in (frame.drop(columns='signup_time'),
                   frame.assign(other_time=frame.signup_time)):
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
    bound = 1e-4
    assert compare([[0.]], [[bound]], task='regression',
                   atol=bound)['within_tolerance']
    for before, after in (
        ([[0.]], [[np.nextafter(bound, np.inf)]]),
        ([[1.]], [[1.001]]),
        ([], []),
        ([[1.]], [[1., 2.]]),
        ([[np.nan]], [[1.]]),
        ([[1.]], [[np.inf]]),
    ):
        with pytest.raises(ValueError):
            compare(before, after, task='regression', atol=bound)


def test_measurement_retains_failed_bounds_but_rejects_invalid_coordinates():
    compare = runner().compare_predictions
    result = compare([[1.]], [[1.01]], task='regression', atol=.0001,
                     measure_only=True)
    assert result['status'] == 'FAIL'
    assert result['max_abs_difference'] > .009
    with pytest.raises(ValueError):
        compare([[1.]], [[np.nan]], task='regression', atol=.0001,
                measure_only=True)
    with pytest.raises(ValueError):
        compare([[.7, .3]], [[.69, .31]], task='binary', atol=.0001,
                before_decisions=np.array([True]),
                after_decisions=np.array([False]), measure_only=True)


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
    assert compare([[0.]], [[.004]], task='regression',
                   atol=.005)['within_tolerance']
    with pytest.raises(ValueError, match='exceeds'):
        compare([[0.]], [[.006]], task='regression', atol=.005)
    for bound in (-1., np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match='finite and nonnegative'):
            compare([[0.]], [[0.]], task='regression', atol=bound)


def test_rtx_callers_supply_a_tolerance_enforced_at_its_boundary():
    root = RUNNER.parents[3]
    workflow = yaml.safe_load(
        (root / '.github/workflows/kumo-engine-pr.yml').read_text())
    action = yaml.safe_load(
        (root / '.github/actions/engine-gpu-smoke/action.yml').read_text())
    step = next(step for step in workflow['jobs']['gpu']['steps']
                if step.get('uses') == './.github/actions/engine-gpu-smoke')
    assert action['inputs']['reload-atol']['required']
    bound = float(step['with']['reload-atol'])
    compare = runner().compare_predictions
    assert compare([[0.]], [[bound]], task='regression',
                   atol=bound)['within_tolerance']
    with pytest.raises(ValueError, match='exceeds'):
        compare([[0.]], [[np.nextafter(bound, np.inf)]],
                task='regression', atol=bound)


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
        runner().compare_predictions(before, after, task=task, atol=1e-4,
                                     before_decisions=decisions,
                                     after_decisions=decisions)


@pytest.mark.parametrize('task', [
    'binary', 'multiclass', 'multilabel', 'multilabel_ranking', 'static_link',
    'temporal_link'
])
def test_near_tie_label_or_rank_changes_pass_with_bounded_scores(
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
    result = smoke.compare_predictions(*values, task=task, atol=1e-4,
                                       before_decisions=decisions[0],
                                       after_decisions=decisions[1])
    assert result['within_tolerance'] and not result['decisions_equal']


@pytest.mark.parametrize('task,before,after,left,right', [
    ('binary', [[.8, .2]], [[.8, .2]], [False], [True]),
    ('multilabel', [[.8, .2]], [[.8, .2]], [[True, False]], [[False, True]]),
    ('multiclass', [[.8, .2]], [[.8, .2]], [[True, False]], [[False, True]]),
    ('static_link', [[3., 1.]], [[3., 1.]], [[0, 1]], [[1, 0]]),
    ('temporal_link', [[.5, .5]], [[.5, .5]], [[0, 1]], [[0, 0]]),
    ('multilabel_ranking', [[.5, .5]], [[.5, .5]], [[0, 1]], [[0, np.nan]]),
])
def test_decisions_cannot_change_arbitrarily_even_with_identical_scores(
        task, before, after, left, right):
    with pytest.raises(ValueError, match='decisions'):
        runner().compare_predictions(before, after, task=task, atol=1e-4,
                                     before_decisions=left,
                                     after_decisions=right)


@pytest.mark.parametrize('gap,accepted', [
    (1e-4, True), (np.nextafter(1e-4, np.inf), False),
])
def test_rank_swaps_use_the_tolerance_boundary(gap, accepted):
    compare = runner().compare_predictions
    args = dict(task='static_link', atol=1e-4,
                before_decisions=[[0, 1]], after_decisions=[[1, 0]])
    if accepted:
        assert compare([[gap, 0.]], [[0., gap]], **args)['within_tolerance']
    else:
        with pytest.raises(ValueError):
            compare([[gap, 0.]], [[0., gap]], **args)


def test_rank_changes_cannot_hide_a_wide_reversal_inside_adjacent_ties():
    with pytest.raises(ValueError, match='decisions'):
        runner().compare_predictions(
            [[3e-4, 2e-4, 1e-4, 0.]], [[3e-4, 2e-4, 1e-4, 0.]],
            task='static_link', atol=2e-4,
            before_decisions=[[0, 1, 2, 3]], after_decisions=[[2, 3, 0, 1]])


def test_tolerance_is_required_by_the_runner_cli_and_pure_comparison(tmp_path):
    import subprocess
    import sys

    with pytest.raises(TypeError):
        runner().compare_predictions([[0.]], [[0.]], task='regression')
    result = subprocess.run([
        sys.executable, str(RUNNER), '--data-dir', '.', '--graph-metadata', '.',
        '--output-dir', str(tmp_path), '--expected-version', 'test',
    ], capture_output=True, text=True)
    assert result.returncode == 2 and '--reload-atol' in result.stderr
