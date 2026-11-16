# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from kumo_core.training.validation import (
    MAX_NUM_CLASSES,
    validate_anchor_time,
    validate_target_classes,
    validate_train_offsets,
)


@pytest.mark.parametrize('unit', ['days', 'months'])
@pytest.mark.parametrize('start,end', [(10, 60), (60, 60)])
def test_training_offsets_reject_reversed_and_equal_bounds(unit, start, end):
    error = validate_train_offsets(pd.DateOffset(**{unit: start}),
                                   pd.DateOffset(**{unit: end}))
    assert error.title == 'Train start/end offsets error'
    assert f'train_start_offset={start} {unit}' in error.message


@pytest.mark.parametrize('start,end', [(60, 10), (None, 10), (60, None)])
def test_training_offsets_allow_valid_and_unspecified_bounds(start, end):
    assert validate_train_offsets(
        None if start is None else pd.DateOffset(days=start),
        None if end is None else pd.DateOffset(days=end)) is None


@pytest.mark.parametrize('anchor', ['2000-01-01', '2024-01-01', '2030-01-01'])
def test_anchor_time_only_rejects_dates_before_source_data(anchor):
    error = validate_anchor_time(pd.Timestamp(anchor),
                                 pd.Timestamp('2024-01-01'), 'orders')
    if anchor.startswith('2000'):
        assert error.title == 'Invalid Prediction Anchor Time'
        assert 'orders (2024-01-01' in error.message
    else:
        assert error is None


def test_anchor_time_without_source_bound_is_allowed():
    assert validate_anchor_time(pd.Timestamp('2000-01-01'), None,
                                'orders') is None


@pytest.mark.parametrize(
    'classes,title',
    [([], 'No positive samples'), ([False], 'Too few classes to predict'),
     ([True], 'Too few classes to predict'),
     (['only-label'], 'Too few classes to predict')],
)
def test_degenerate_classification_targets_are_rejected(classes, title):
    errors = validate_target_classes(1, classes, classify=True)
    assert len(errors) == 1
    assert errors[0].title == title


def test_two_classes_and_class_limit_boundary_are_allowed():
    assert validate_target_classes(2, [False, True], classify=True) == []
    assert validate_target_classes(MAX_NUM_CLASSES, [], classify=True) == []


@pytest.mark.parametrize('classify', [False, True])
def test_class_limit_preserves_classify_hint(classify):
    errors = validate_target_classes(MAX_NUM_CLASSES + 1, [], classify=classify)
    assert errors[0].title == 'Too many classes to predict'
    assert ('RANK instead of CLASSIFY' in errors[0].message) == classify
