# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from cases import common


def scramble(frame, task):
    function = getattr(common, 'scramble_labels', None)
    assert function is not None, 'The label-scrambling control is missing'
    actual = function(frame, task)
    reordered = function(
        frame.sample(frac=1, random_state=5).reset_index(drop=True), task)
    keys = ['ENTITY1', 'SPLIT', 'TIMESTAMP']
    pd.testing.assert_frame_equal(
        actual.set_index(keys).sort_index(),
        reordered.set_index(keys).sort_index())
    return actual


@pytest.mark.parametrize('task', ['regression', 'forecasting', 'temporal_link'])
def test_scramble_keeps_answer_tuples_in_their_stage_and_anchor(task):
    frame = pd.DataFrame([
        (entity, anchor, split, 100 * group + entity + 1)
        for group, (split,
                    anchor) in enumerate(((split, anchor)
                                          for split in ['train', 'val', 'test']
                                          for anchor in [10, 20]), 1)
        for entity in range(3)
    ], columns=['ENTITY1', 'TIMESTAMP', 'SPLIT', 'TARGET1'])
    answers = ['TARGET1']
    if task == 'temporal_link':
        frame['__kumo_internal_frequency_column__'] = [
            [value + 10, value + 20] for value in frame['TARGET1']
        ]
        frame['TARGET1'] = [[value, value + 1000] for value in frame['TARGET1']]
        answers.append('__kumo_internal_frequency_column__')
    if task == 'forecasting':
        frame['future_labels'] = [[[value + 1], [value + 2]]
                                  for value in frame['TARGET1']]
        answers.append('future_labels')
    frame['autoregression_labels'] = [[value] for value in frame['ENTITY1']]
    frame['year_over_year_labels'] = [[value] for value in frame['TIMESTAMP']]
    original = frame.copy(deep=True)

    actual = scramble(frame, task)

    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(actual, scramble(frame, task))
    pd.testing.assert_frame_equal(actual.drop(columns=answers),
                                  original.drop(columns=answers))
    permutations = set()
    for positions in original.groupby(['SPLIT', 'TIMESTAMP']).indices.values():
        before, after = original.iloc[positions][answers], actual.iloc[
            positions][answers]
        assert sorted(map(repr,
                          after.itertuples(index=False, name=None))) == sorted(
                              map(repr,
                                  before.itertuples(index=False, name=None)))
        assert not after.equals(before)
        labels = after['TARGET1'].str[0] if task == 'temporal_link' else after[
            'TARGET1']
        permutations.add(tuple(labels % 100))
    assert len(permutations) > 1


def test_static_link_scramble_moves_complete_matching_entity_trajectories():
    profiles = [
        (['train', 'val', 'test'], [(1, 2, 3), (2, 3, 1), (3, 1, 2)]),
        (['train', 'val'], [(4, 5), (5, 6), (6, 4)]),
        (['train', 'test'], [(7, 8), (8, 9), (9, 7)]),
    ]
    frame = pd.DataFrame(
        [(group * 3 + entity, stage, [label], 0)
         for group, (stages, trajectories) in enumerate(profiles)
         for entity, trajectory in enumerate(trajectories)
         for stage, label in zip(stages, trajectory)],
        columns=['ENTITY1', 'SPLIT', 'TARGET1', 'TIMESTAMP'])
    original = frame.copy(deep=True)

    actual = scramble(frame, 'static_link')

    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(actual, scramble(frame, 'static_link'))
    pd.testing.assert_frame_equal(actual.drop(columns='TARGET1'),
                                  original.drop(columns='TARGET1'))
    for entity, rows in actual.groupby('ENTITY1'):
        labels = tuple(value[0] for value in rows['TARGET1'])
        assert labels in profiles[entity // 3][1]
        assert len(set(labels)) == len(labels)
    assert not actual['TARGET1'].equals(original['TARGET1'])


@pytest.mark.parametrize('weights', [None, [[1], [2], [3]]])
def test_scramble_rejects_constant_labels_even_if_weights_change(weights):
    frame = pd.DataFrame({
        'ENTITY1': [0, 1, 2],
        'SPLIT': ['train'] * 3,
        'TIMESTAMP': [0] * 3,
        'TARGET1': [[7]] * 3
    })
    if weights is not None:
        frame['__kumo_internal_frequency_column__'] = weights
    with pytest.raises(ValueError):
        scramble(frame, 'temporal_link')
