# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumo_core.types import ColStatType
from kumo_core.types.schema import BaseColumnConfig
from kumoapi.typing import Dtype, Stype

from kumo.stats.stats import Stats
from kumo.validation import ValidationResponse
from rest.backends.data_backend.validation import _validate_feature_dimensions


class _FeatureDimensionTable:
    def __init__(self, cols):
        self.cols = cols
        self._cols_by_name = {col.name: col for col in cols}

    def col_to_base_config(self, name):
        return self._cols_by_name[name]


def _sequence_table():
    cols = [
        BaseColumnConfig('id', dtype=Dtype.int, stype=Stype.ID),
        BaseColumnConfig('seed_embedding', dtype=Dtype.floatlist,
                         stype=Stype.sequence),
    ]
    return _FeatureDimensionTable(cols), cols


def test_validate_feature_dimensions_accepts_approx_sequence_length_stats():
    table, cols = _sequence_table()
    test_version = 'approx-sequence-lengths'
    col_stats = {
        col: {
            ColStatType.NUM_UNIQUE:
            Stats(value=100, approx=False, data_version=test_version),
            ColStatType.COUNT:
            Stats(value=100, approx=False, data_version=test_version),
        }
        for col in cols
    }
    col_stats[cols[1]][ColStatType.SEQUENCE_MIN_LENGTH] = Stats(
        value=3, approx=True, data_version=test_version)
    col_stats[cols[1]][ColStatType.SEQUENCE_MAX_LENGTH] = Stats(
        value=3, approx=True, data_version=test_version)

    response = ValidationResponse()
    _validate_feature_dimensions(table, col_stats, response, 'test_table',
                                 max_dimensions=4097)

    assert all(error.title != 'Missing embedding statistics'
               for error in response.errors)
    assert all(error.title != 'Inconsistent embedding dimensions'
               for error in response.errors)


def test_validate_feature_dimensions_reports_missing_sequence_length_stats():
    table, cols = _sequence_table()
    test_version = 'missing-sequence-lengths'
    col_stats = {
        cols[0]: {
            ColStatType.NUM_UNIQUE:
            Stats(value=100, approx=False, data_version=test_version),
            ColStatType.COUNT:
            Stats(value=100, approx=False, data_version=test_version),
        }
    }

    response = ValidationResponse()
    _validate_feature_dimensions(table, col_stats, response, 'test_table',
                                 max_dimensions=4097)

    assert any(error.title == 'Missing embedding statistics'
               for error in response.errors)
