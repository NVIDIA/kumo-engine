# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd
import pytest
from kumo_core.sources.preview import sample_dataframe
from kumo_core.stype_inference import column_statistics


@pytest.mark.parametrize('dtype', ['Int64', 'UInt64'])
def test_nullable_integers_keep_exact_values_and_other_schema_types(dtype):
    values = [2**53, 2**53 + 1, None]
    frame = pd.DataFrame({
        'id': pd.Series(values, dtype=dtype),
        'time': pd.date_range('2024-01-01', periods=3),
        'label': ['first', 'second', None],
    })
    frame.index.name = 'row'

    restored = sample_dataframe(frame.to_json(orient='table'))

    assert restored is not None
    assert restored['id'].dropna().tolist() == values[:2]
    pd.testing.assert_series_equal(restored['time'], frame['time'])
    assert restored.index.name == 'row'
    assert column_statistics(restored)['id'] == (2, 2 / 3)


def test_nullable_integer_primary_key_is_not_rounded():
    frame = pd.DataFrame({
        'id': pd.Series([2**53, 2**53 + 1], dtype='Int64'),
        'value': [10, 20],
    }).set_index('id')

    restored = sample_dataframe(frame.to_json(orient='table'))

    assert restored is not None
    assert restored.index.tolist() == [2**53, 2**53 + 1]
    assert restored['value'].tolist() == [10, 20]


def test_scalar_statistics_skip_native_lists_but_keep_null_denominator():
    frame = pd.DataFrame({
        'labels': [['a', 'b'], ['a'], None],
        'value': [1, 2, None],
        'empty': [None, None, None],
    })

    assert column_statistics(frame) == {'value': (2, 2 / 3)}


def test_unrelated_unhashable_values_are_not_suppressed():
    with pytest.raises(TypeError):
        column_statistics(pd.DataFrame({'bad': [{'a': 1}]}))


def test_malformed_integer_schema_is_still_rejected():
    payload = json.loads(
        pd.DataFrame({
            'value': pd.Series([1, None], dtype='Int64')
        }).to_json(orient='table'))
    payload['data'][0]['value'] = 'not-an-integer'

    with pytest.raises(ValueError):
        sample_dataframe(json.dumps(payload))


@pytest.mark.parametrize('sample', [None, ''])
def test_absent_sample_is_not_an_empty_dataframe(sample):
    assert sample_dataframe(sample) is None
