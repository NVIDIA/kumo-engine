# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from kumoapi.source_table import SourceColumn, SourceTableDataResponse
from kumoapi.typing import Dtype, Stype

from kumo.engine._infer import _sample_frame, resolve_table_columns


def test_inference_uses_bounded_sample_and_counts_null_rows():
    calls = []
    column = SourceColumn(name='value', dtype=Dtype.int64, stype=None,
                          is_primary=False)
    frame = pd.DataFrame(
        {'value': pd.Series([1, 2, 3, 4] + [None] * 96, dtype='Int64')})

    class Source:
        def get_sample(self, connector, table_name, sample_rows):
            calls.append((table_name, sample_rows))
            return SourceTableDataResponse(
                table_name=table_name, cols=[column],
                sample_rows=frame.to_json(orient='table'))

    columns = resolve_table_columns(SimpleNamespace(source_backend=Source()),
                                    'events', {'value': column},
                                    column_names=None, primary_key=None,
                                    time_column=None, end_time_column=None)

    assert calls == [('events', 10_000)]
    assert columns[0].dtype == Dtype.int64
    assert columns[0].stype == Stype.categorical


def test_nullable_integer_sample_preserves_values_above_float_precision():
    values = [2**53, 2**53 + 1, None]
    sample = pd.DataFrame({
        'value': pd.Series(values, dtype='Int64')
    }).to_json(orient='table')

    restored = _sample_frame(sample)

    assert restored is not None
    assert restored['value'].dropna().tolist() == values[:2]
    assert restored['value'].nunique() == 2


def test_native_list_sample_does_not_require_hashable_values():
    column = SourceColumn(name='labels', dtype=Dtype.stringlist, stype=None,
                          is_primary=False)
    frame = pd.DataFrame({'labels': [['a', 'b'], ['b', 'c'], None]})

    class Source:
        def get_sample(self, connector, table_name, sample_rows):
            return SourceTableDataResponse(
                table_name=table_name, cols=[column],
                sample_rows=frame.to_json(orient='table'))

    columns = resolve_table_columns(SimpleNamespace(source_backend=Source()),
                                    'events', {'labels': column},
                                    column_names=None, primary_key=None,
                                    time_column=None, end_time_column=None)

    assert columns[0].dtype == Dtype.stringlist
    assert columns[0].stype == Stype.multicategorical


def test_empty_sample_reports_the_diskgraph_row_requirement():
    column = SourceColumn(name='value', dtype=Dtype.int64, stype=None,
                          is_primary=False)
    frame = pd.DataFrame({'value': pd.Series([], dtype='Int64')})

    class Source:
        def get_sample(self, connector, table_name, sample_rows):
            return SourceTableDataResponse(
                table_name=table_name, cols=[column],
                sample_rows=frame.to_json(orient='table'))

    from kumo_core.sources.errors import PermanentSourceError

    with pytest.raises(PermanentSourceError, match='at least one row'):
        resolve_table_columns(SimpleNamespace(source_backend=Source()),
                              'events', {'value': column}, column_names=None,
                              primary_key=None, time_column=None,
                              end_time_column=None)


def test_unresolved_column_error_names_its_table():
    column = SourceColumn(name='net_amount_v3', dtype=Dtype.float64,
                          stype=None, is_primary=False)

    class Source:
        def get_sample(self, connector, table_name, sample_rows):
            raise RuntimeError('unreachable')

    with pytest.raises(
            ValueError,
            match=("Column 'net_amount_v3' in table "
                   "'drift_records_v3_whole_seconds' cannot be resolved: "
                   "it has dtype float64")):
        resolve_table_columns(SimpleNamespace(source_backend=Source()),
                              'drift_records_v3_whole_seconds',
                              {'net_amount_v3': column}, column_names=None,
                              primary_key=None, time_column=None,
                              end_time_column=None)


def test_sample_does_not_mask_unrelated_table_schema_errors():
    sample = json.loads(
        pd.DataFrame({'value': [1]}).to_json(orient='table'))
    sample['schema']['fields'][1]['type'] = 'integer'
    sample['data'][0]['value'] = 'not-an-integer'

    with pytest.raises(ValueError):
        _sample_frame(json.dumps(sample))
