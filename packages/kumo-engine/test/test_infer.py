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
    column = SourceColumn(name='measurement', dtype=Dtype.int64, stype=None,
                          is_primary=False)
    frame = pd.DataFrame(
        {'measurement': pd.Series([1, 2, 3, 4] + [None] * 96, dtype='Int64')})

    class Source:
        def get_sample(self, connector, table_name, sample_rows, sql=None):
            calls.append((table_name, sample_rows))
            return SourceTableDataResponse(
                table_name=table_name, cols=[column],
                sample_rows=frame.to_json(orient='table'))

    columns = resolve_table_columns(SimpleNamespace(source_backend=Source()),
                                    'events', {'measurement': column},
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
        def get_sample(self, connector, table_name, sample_rows, sql=None):
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
    column = SourceColumn(name='measurement', dtype=Dtype.int64, stype=None,
                          is_primary=False)
    frame = pd.DataFrame({'measurement': pd.Series([], dtype='Int64')})

    class Source:
        def get_sample(self, connector, table_name, sample_rows, sql=None):
            return SourceTableDataResponse(
                table_name=table_name, cols=[column],
                sample_rows=frame.to_json(orient='table'))

    from kumo_core.sources.errors import PermanentSourceError

    with pytest.raises(PermanentSourceError, match='at least one row'):
        resolve_table_columns(SimpleNamespace(source_backend=Source()),
                              'events', {'measurement': column},
                              column_names=None, primary_key=None,
                              time_column=None, end_time_column=None)


def test_unresolved_column_error_names_its_table():
    column = SourceColumn(name='net_amount_v3', dtype=Dtype.float64, stype=None,
                          is_primary=False)

    class Source:
        def get_sample(self, connector, table_name, sample_rows, sql=None):
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
    sample = json.loads(pd.DataFrame({'value': [1]}).to_json(orient='table'))
    sample['schema']['fields'][1]['type'] = 'integer'
    sample['data'][0]['value'] = 'not-an-integer'

    with pytest.raises(ValueError):
        _sample_frame(json.dumps(sample))


@pytest.mark.parametrize('delimiter', [',', '|', ';', '\t'])
def test_scalar_multicat_inference_persists_detected_delimiter(delimiter):
    from itertools import combinations

    values = [delimiter.join(pair) for pair in combinations('abcde', 2)] * 12
    frame = pd.DataFrame({'tags': values + [None]})
    source_column = SourceColumn(name='tags', dtype=Dtype.string, stype=None,
                                 is_primary=False)
    calls = []

    class Source:
        def get_sample(self, connector, table_name, sample_rows, sql=None):
            calls.append(sample_rows)
            return SourceTableDataResponse(
                table_name=table_name, cols=[source_column],
                sample_rows=frame.to_json(orient='table'))

    columns = resolve_table_columns(SimpleNamespace(source_backend=Source()),
                                    'users', {'tags': source_column},
                                    column_names=None, primary_key=None,
                                    time_column=None, end_time_column=None)
    assert columns[0].stype == Stype.multicategorical
    assert columns[0].multicategorical_delimiter == delimiter
    assert calls == [10_000]


@pytest.mark.parametrize(
    'values',
    [
        ['scalar'] * 1000,
        ['a|b'] + ['scalar'] * 999,
        ['["a", "b"]', '["b", "c"]'] * 50,
        ['2026-01-01', '2026-01-02'] * 50,
        ['a|b', 'b|c'] *
        50,  # Whole-string cardinality does not imply multicat.
    ])
def test_scalar_strings_do_not_become_multicat_from_punctuation(values):
    from kumo_core.stype_inference import (
        column_statistics,
        multicategorical_separator,
        multicategorical_token_counts,
    )

    from kumo.engine._infer import _stype_for_column

    frame = pd.DataFrame({'tags': values})
    separator = multicategorical_separator(frame['tags'])
    count = (None if separator is None else len(
        multicategorical_token_counts(values, separator)))
    stype = _stype_for_column(Dtype.string,
                              column_statistics(frame)['tags'], count, separator
                              is not None)
    assert stype == Stype.categorical


@pytest.mark.parametrize('num_tokens, expected', [
    (500, Stype.multicategorical),
    (501, Stype.categorical),
])
def test_multicat_token_vocabulary_boundary(num_tokens, expected):
    from kumo_core.stype_inference import (
        column_statistics,
        multicategorical_separator,
        multicategorical_token_counts,
    )

    from kumo.engine._infer import _stype_for_column

    values = [
        f'{a}|{b}' for a in range(num_tokens) for b in range(a + 1, a + 3)
        if b < num_tokens
    ]
    frame = pd.DataFrame({'tags': values})
    separator = multicategorical_separator(frame['tags'])
    count = len(multicategorical_token_counts(values, separator))
    assert count == num_tokens
    assert _stype_for_column(Dtype.string,
                             column_statistics(frame)['tags'], count,
                             True) == expected


def test_local_declared_multicat_infers_delimiter_but_preserves_explicit_choice(
        tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from kumo.engine import FileConnector, Table

    pq.write_table(pa.table({
        'id': [1, 2],
        'tags': ['a|b', 'b|c']
    }), tmp_path / 'users.parquet')
    table = Table.from_source_table(
        FileConnector(root_dir=str(tmp_path))['users'], primary_key='id')
    assert table.definition.cols[1].stype == Stype.categorical
    inferred = table.with_column('tags', stype=Stype.multicategorical)
    assert inferred.definition.cols[1].multicategorical_delimiter == '|'
    explicit = table.with_column('tags', stype=Stype.multicategorical,
                                 multicategorical_delimiter=';')
    assert explicit.definition.cols[1].multicategorical_delimiter == ';'
    assert table.definition.cols[1].stype == Stype.categorical
    with pytest.raises(ValueError):
        table.with_column('tags', stype=Stype.multicategorical,
                          multicategorical_delimiter=':')


def test_inferred_multicat_delimiter_reaches_real_snapshot(
        tmp_path, monkeypatch):
    from itertools import combinations

    import pyarrow as pa
    import pyarrow.parquet as pq

    from kumo.engine import FileConnector, Graph, Table

    monkeypatch.setattr('kumo.engine._scratch._root', tmp_path / 'scratch')
    source = tmp_path / 'source'
    source.mkdir()
    values = ['|'.join(pair) for pair in combinations('abcde', 2)] * 12
    pq.write_table(pa.table({
        'id': range(len(values)),
        'tags': values
    }), source / 'users.parquet')
    table = Table.from_source_table(
        FileConnector(root_dir=str(source))['users'], primary_key='id')
    snapshot = Graph([table]).snapshot()
    column = next(col for col in snapshot.graph_config['tables'][0]['cols']
                  if col['name'] == 'tags')
    assert column['stype'] == 'multicategorical'
    assert column['multicategorical_delimiter'] == '|'
    metadata = json.loads(
        (snapshot.index_path / 'diskgraph_config.json').read_text())
    native_column = next(col for col in metadata['tables'][0]['columns']
                         if col['name'] == 'tags')
    assert native_column['multicategorical_delimiter'] == '|'
    assert native_column['semantic_type'] == 'multicategorical'
    text = table.with_column('tags', stype=Stype.text)
    assert text.definition.cols[1].multicategorical_delimiter is None
    assert text.definition.cols[1].stype == Stype.text


@pytest.mark.parametrize('name,dtype,declared,expected', [
    ('user_id', Dtype.int64, None, Stype.ID),
    ('USER_ID', Dtype.string, None, Stype.ID),
    ('user_id', Dtype.float64, None, Stype.numerical),
    ('grid', Dtype.int64, None, Stype.numerical),
    ('age', Dtype.int64, None, Stype.numerical),
    ('user_id', Dtype.int64, Stype.numerical, Stype.numerical),
    ('user_id', Dtype.bool, None, Stype.categorical),
])
def test_identifier_name_inference_preserves_explicit_types(
        name, dtype, declared, expected):
    column = SourceColumn(name=name, dtype=dtype, stype=declared,
                          is_primary=False)
    frame = pd.DataFrame({name: range(100)})
    calls = []

    class Source:
        def get_sample(self, connector, table_name, sample_rows, sql=None):
            calls.append(table_name)
            return SourceTableDataResponse(
                table_name=table_name, cols=[column],
                sample_rows=frame.to_json(orient='table'))

    columns = resolve_table_columns(SimpleNamespace(source_backend=Source()),
                                    'events', {name: column}, column_names=None,
                                    primary_key=None, time_column=None,
                                    end_time_column=None)
    assert columns[0].stype == expected
    assert calls == (['events'] if declared is None
                     and expected == Stype.numerical else [])


@pytest.mark.parametrize('role,expected', [('primary_key', Stype.ID),
                                           ('time_column', Stype.timestamp)])
def test_explicit_column_roles_take_precedence_over_id_names(role, expected):
    column = SourceColumn(name='user_id', dtype=Dtype.int64,
                          stype=Stype.numerical, is_primary=False)
    roles = dict(primary_key=None, time_column=None, end_time_column=None)
    roles[role] = 'user_id'
    columns = resolve_table_columns(SimpleNamespace(), 'events',
                                    {'user_id': column}, column_names=None,
                                    **roles)
    assert columns[0].stype == expected


@pytest.mark.parametrize('name,dtype,expected', [
    ('num_items', Dtype.int64, Stype.numerical),
    ('price', Dtype.float64, Stype.numerical),
    ('score', Dtype.int64, Stype.numerical),
    ('num_users_id', Dtype.int64, Stype.ID),
    ('num_items', Dtype.bool, Stype.categorical),
])
def test_named_types_resolve_without_sampling(name, dtype, expected):
    column = SourceColumn(name=name, dtype=dtype, stype=None, is_primary=False)
    columns = resolve_table_columns(SimpleNamespace(), 'events', {name: column},
                                    column_names=None, primary_key=None,
                                    time_column=None, end_time_column=None)
    assert columns[0].stype == expected
