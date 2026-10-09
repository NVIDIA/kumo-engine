# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pandas as pd
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.types import ColStatType
from kumoapi.encoder import Index, MultiCategorical
from kumoapi.graph import ColumnKey, ColumnKeyGroup, GraphDefinition
from kumoapi.source_table import UnavailableSourceTable
from kumoapi.table import Column as ColumnDefinition
from kumoapi.table import TableDefinition
from kumoapi.typing import Dtype
from kumoapi.typing import Stype
from kumoapi.typing import Stype as KumoStype

from kumo.graph.column import Column
from kumo.graph.table import Table
from kumo.graph.utils import (
    _coerce_category_value,
    _normalize_category_counts_for_dtype,
    _normalize_encoder_category_stats_for_dtype,
    fetch_and_set_encoder_stats,
    make_graph_mock_object,
    map_to_api_graph_definition,
    read_parquet,
)
from kumo.stats.stats import Stats


def get_graph_definition():
    return GraphDefinition(
        tables={
            'USERS':
            TableDefinition(
                cols=[
                    ColumnDefinition(name='USER_ID', stype=Stype.ID,
                                     dtype=Dtype.int),
                    ColumnDefinition(name='AGE', stype=Stype.numerical,
                                     dtype=Dtype.int),
                    ColumnDefinition(name='LOCATION_ID', stype=Stype.ID,
                                     dtype=Dtype.int),
                ],
                source_table=UnavailableSourceTable(table='USERS'),
                pkey='USER_ID',
            ),
            'LOCATIONS':
            TableDefinition(
                cols=[
                    ColumnDefinition(name='LOCATION_ID', stype=Stype.ID,
                                     dtype=Dtype.int),
                    ColumnDefinition(name='NAME', stype=Stype.categorical,
                                     dtype=Dtype.string),
                    ColumnDefinition(name='DESCRIPTION', stype=Stype.text,
                                     dtype=Dtype.string),
                ],
                source_table=UnavailableSourceTable(table='LOCATIONS'),
                pkey='LOCATION_ID',
            ),
            'TRANSACTIONS':
            TableDefinition(
                cols=[
                    ColumnDefinition(name='USER_ID', stype=Stype.ID,
                                     dtype=Dtype.int),
                    ColumnDefinition(name='AMOUNT', stype=Stype.numerical,
                                     dtype=Dtype.float),
                    ColumnDefinition(name='CATEGORY', stype=Stype.categorical,
                                     dtype=Dtype.int),
                    ColumnDefinition(name='IS_ONLINE', stype=Stype.categorical,
                                     dtype=Dtype.bool),
                    ColumnDefinition(name='DATE', stype=Stype.timestamp,
                                     dtype=Dtype.time),
                    ColumnDefinition(name='TYPE', stype=Stype.categorical,
                                     dtype=Dtype.string),
                ],
                source_table=UnavailableSourceTable(table='TRANSACTIONS'),
                time_col='DATE',
            ),
        }, col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey(table_name='LOCATIONS', col_name='LOCATION_ID'),
                ColumnKey(table_name='USERS', col_name='LOCATION_ID'),
            )),
            ColumnKeyGroup(columns=(
                ColumnKey(table_name='TRANSACTIONS', col_name='USER_ID'),
                ColumnKey(table_name='USERS', col_name='USER_ID'),
            )),
        ])


def test_preloaded_int_category_counts_match_target_values():
    encoder = Index(min_occ=1, na_strategy="separate")
    encoder._stats = {
        ColStatType.CATEGORY_COUNTS: [["0", 10], ["23", 5]],
    }

    _normalize_encoder_category_stats_for_dtype(encoder, Dtype.int64)
    assert encoder._stats[ColStatType.CATEGORY_COUNTS] == [[0, 10], [23, 5]]

    mapped = TensorMapper.from_encoder(encoder)(pd.Series(
        [0, 23], dtype="Int64")).view(-1)
    assert mapped.tolist() == [0, 1]


def test_normalize_category_counts_for_string_dtype_preserves_labels():
    category_counts = [["001", 10], ["23", 5]]
    assert (_normalize_category_counts_for_dtype(
        category_counts, Dtype.string) == category_counts)


def test_normalize_category_counts_canonicalizes_tied_counts():
    category_counts = [["west", 249], ["east", 249], ["south", 249],
                       ["north", 249]]
    assert (_normalize_category_counts_for_dtype(
        category_counts, Dtype.string) == [["east", 249], ["north", 249],
                                           ["south", 249], ["west", 249]])


def test_normalize_category_counts_preserves_descending_frequency_priority():
    category_counts = [["b", 2], ["c", 3], ["a", 2]]
    assert (_normalize_category_counts_for_dtype(category_counts,
                                                 Dtype.string) == [["c", 3],
                                                                   ["a", 2],
                                                                   ["b", 2]])


def test_normalize_category_counts_for_float_dtype():
    # JSON round-trip of float categorical labels can produce strings.
    category_counts = [["1.5", 10], ["2.0", 5], [3.14, 1]]
    assert (_normalize_category_counts_for_dtype(category_counts,
                                                 Dtype.float) == [[1.5, 10],
                                                                  [2.0, 5],
                                                                  [3.14, 1]])


def test_normalize_category_counts_for_bool_dtype():
    # Bool labels can be stringified ("true"/"false") or stored as 0/1.
    category_counts = [["true", 10], ["False", 5], [1, 3], [0, 2]]
    assert (_normalize_category_counts_for_dtype(category_counts,
                                                 Dtype.bool) == [[True, 10],
                                                                 [False, 5],
                                                                 [True, 3],
                                                                 [False, 2]])


def test_normalize_category_counts_preserves_tuple_form():
    # Some upstream producers materialize counts as tuples; coercion must
    # keep the tuple shape so downstream code sees identical structure.
    category_counts = [("0", 10), ("23", 5)]
    normalized = _normalize_category_counts_for_dtype(category_counts,
                                                      Dtype.int64)
    assert normalized == [(0, 10), (23, 5)]
    assert all(isinstance(item, tuple) for item in normalized)


def test_normalize_category_counts_for_int_list_dtype():
    # The dtype helper unwraps list element types so list-of-int columns also
    # get their categorical labels coerced back to ints.
    category_counts = [["0", 10], ["7", 5]]
    assert (_normalize_category_counts_for_dtype(category_counts,
                                                 Dtype.intlist) == [[0, 10],
                                                                    [7, 5]])


def test_coerce_category_value_handles_invalid_and_nan():
    # Unparseable strings fall back to the original value.
    assert _coerce_category_value("not_a_number",
                                  Dtype.int64) == ("not_a_number")
    # NaN is preserved across coercion.
    import math
    assert math.isnan(_coerce_category_value(float("nan"), Dtype.float))
    # None is preserved.
    assert _coerce_category_value(None, Dtype.int64) is None


def test_normalize_encoder_category_stats_handles_string_keyed_stats():
    # ``Encoder._stats`` may be populated from a JSON-deserialized cache where
    # ColStatType keys come back as their string ``.value`` rather than enum.
    encoder = Index(min_occ=1, na_strategy="separate")
    encoder._stats = {
        ColStatType.CATEGORY_COUNTS.value: [["0", 10], ["23", 5]],
    }

    _normalize_encoder_category_stats_for_dtype(encoder, Dtype.int64)
    assert (encoder._stats[ColStatType.CATEGORY_COUNTS.value] == [[0, 10],
                                                                  [23, 5]])


def test_normalize_encoder_category_stats_for_multicategorical():
    # ``MultiCategorical`` uses ``MULTI_CATEGORY_COUNTS`` rather than
    # ``CATEGORY_COUNTS``; the normalizer must handle that stat as well.
    encoder = MultiCategorical(min_occ=1)
    encoder._stats = {
        ColStatType.MULTI_CATEGORY_COUNTS: [["1", 10], ["2", 5]],
    }
    _normalize_encoder_category_stats_for_dtype(encoder, Dtype.intlist)
    assert (encoder._stats[ColStatType.MULTI_CATEGORY_COUNTS] == [[1, 10],
                                                                  [2, 5]])


def test_normalize_encoder_category_stats_noop_for_none_encoder():
    # Defensive: a column may have ``encoder is None`` (e.g. Null-encoded
    # columns); the helper must accept and skip it.
    _normalize_encoder_category_stats_for_dtype(None, Dtype.int64)


def _make_mock_table():
    mock_table = MagicMock(spec=Table)
    mock_table.fingerprint.return_value = 'fp'
    return mock_table


def _make_int_target_column():
    return Column(
        name='target',
        stype=KumoStype.categorical,
        dtype=Dtype.int64,
        encoder=Index(min_occ=1, na_strategy="separate"),
    )


def test_fetch_and_set_encoder_stats_coerces_fetched_category_counts():
    # End-to-end through the real materialization helper: simulate the
    # backend cache returning stringified integer categories (as occurs after
    # a JSON cache reload) and confirm the encoder ends up with int-typed
    # labels and that ``TensorMapper.from_encoder`` then maps the real target
    # values correctly.
    col = _make_int_target_column()

    mock_cache = MagicMock()
    mock_cache.get_col_stats.return_value = {
        col.base_config(): {
            ColStatType.CATEGORY_COUNTS:
            Stats(value=[["0", 10], ["23", 5]], approx=False),
        },
    }

    with patch('kumo.graph.utils.get_active_cache', return_value=mock_cache):
        fetch_and_set_encoder_stats([col], _make_mock_table())

    mock_cache.load_col_stats_cache.assert_called_once_with('fp', None)
    mock_cache.get_col_stats.assert_called_once()
    assert (col.encoder._stats[ColStatType.CATEGORY_COUNTS] == [[0, 10],
                                                                [23, 5]])
    mapped = TensorMapper.from_encoder(col.encoder)(pd.Series(
        [0, 23], dtype="Int64")).view(-1)
    assert mapped.tolist() == [0, 1]


def test_fetch_and_set_encoder_stats_coerces_preloaded_cached_stats():
    # The reviewer's concern: when the column encoder already carries
    # pre-loaded stats (e.g. deserialized from the persisted col-stats cache
    # before ``fetch_and_set_encoder_stats`` runs), the integer keys can be
    # stringified by JSON. Verify the preload normalization branch coerces
    # them in place even though no backend fetch is required.
    col = _make_int_target_column()
    col.encoder._stats[ColStatType.CATEGORY_COUNTS] = [["0", 10], ["23", 5]]

    mock_cache = MagicMock()
    # No new stats need to be fetched — preload covers required_stats.
    mock_cache.get_col_stats.return_value = {}

    with patch('kumo.graph.utils.get_active_cache', return_value=mock_cache):
        fetch_and_set_encoder_stats([col], _make_mock_table())

    mock_cache.load_col_stats_cache.assert_not_called()
    mock_cache.get_col_stats.assert_not_called()
    assert (col.encoder._stats[ColStatType.CATEGORY_COUNTS] == [[0, 10],
                                                                [23, 5]])
    mapped = TensorMapper.from_encoder(col.encoder)(pd.Series(
        [0, 23], dtype="Int64")).view(-1)
    assert mapped.tolist() == [0, 1]


def test_fetch_and_set_encoder_stats_preserves_string_labels_through_fetch():
    # String-typed categorical columns must not have their labels coerced
    # to numbers (e.g. preserve leading zeros) — exercises the fetched-stats
    # branch with a string dtype.
    col = Column(
        name='name',
        stype=KumoStype.categorical,
        dtype=Dtype.string,
        encoder=Index(min_occ=1, na_strategy="separate"),
    )

    mock_cache = MagicMock()
    mock_cache.get_col_stats.return_value = {
        col.base_config(): {
            ColStatType.CATEGORY_COUNTS:
            Stats(value=[["001", 10], ["23", 5]], approx=False),
        },
    }

    with patch('kumo.graph.utils.get_active_cache', return_value=mock_cache):
        fetch_and_set_encoder_stats([col], _make_mock_table())

    assert (col.encoder._stats[ColStatType.CATEGORY_COUNTS] == [["001", 10],
                                                                ["23", 5]])


def test_make_graph_mock_object():
    graph_def = get_graph_definition()
    graph = make_graph_mock_object(graph_def)
    assert set(graph.table_names) == {
        'dummy_connector.LOCATIONS', 'dummy_connector.USERS',
        'dummy_connector.TRANSACTIONS'
    }
    assert graph['USERS'].pkey_name == 'USER_ID'
    assert graph['USERS'].col_names == ['AGE']
    assert set(graph['USERS'].fkey_names) == {'LOCATION_ID'}
    assert graph['LOCATIONS'].pkey_name == 'LOCATION_ID'
    assert set(graph['LOCATIONS'].col_names) == {'NAME', 'DESCRIPTION'}
    assert graph['LOCATIONS'].fkey_names == []
    assert graph['TRANSACTIONS'].time_col_name == 'DATE'
    assert set(graph['TRANSACTIONS'].col_names) == {
        'AMOUNT', 'CATEGORY', 'IS_ONLINE', 'TYPE', 'DATE'
    }
    assert graph['TRANSACTIONS'].fkey_names == ['USER_ID']


def test_map_to_api_graph_definition(create_graph):
    graph = create_graph(name='FINANCIAL')
    graph.infer_metadata()
    graph_def = map_to_api_graph_definition(graph)
    # Sort col groups to prevent test flakiness
    graph_def.col_groups.sort(key=lambda x: str(x))
    assert graph_def == GraphDefinition(
        tables={
            'ACCOUNT':
            TableDefinition(
                cols=[
                    ColumnDefinition(name='ACCOUNT_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='DISTRICT_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='FREQUENCY', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None),
                    ColumnDefinition(name='DATE', stype=Stype.timestamp,
                                     dtype=Dtype.date, timestamp_format=None)
                ], source_table=UnavailableSourceTable(table='ACCOUNT'),
                pkey='ACCOUNT_ID', time_col=None, end_time_col=None),
            'CARD':
            TableDefinition(
                cols=[
                    ColumnDefinition(name='CARD_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='DISP_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='TYPE', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None),
                    ColumnDefinition(name='ISSUED', stype=Stype.timestamp,
                                     dtype=Dtype.date, timestamp_format=None)
                ], source_table=UnavailableSourceTable(table='CARD'),
                pkey='CARD_ID', time_col=None, end_time_col=None),
            'CLIENT':
            TableDefinition(
                cols=[
                    ColumnDefinition(name='CLIENT_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='DISTRICT_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='GENDER', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None),
                    ColumnDefinition(name='BIRTH_DATE', stype=Stype.timestamp,
                                     dtype=Dtype.date, timestamp_format=None)
                ], source_table=UnavailableSourceTable(table='CLIENT'),
                pkey='CLIENT_ID', time_col=None, end_time_col=None),
            'DISP':
            TableDefinition(
                cols=[
                    ColumnDefinition(name='DISP_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='CLIENT_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='ACCOUNT_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='TYPE', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None)
                ], source_table=UnavailableSourceTable(table='DISP'),
                pkey='DISP_ID', time_col=None, end_time_col=None),
            'DISTRICT':
            TableDefinition(
                cols=[
                    ColumnDefinition(name='DISTRICT_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='A3', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None),
                    ColumnDefinition(name='A4', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A5', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A6', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A7', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A8', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A9', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A10', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A11', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A12', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A13', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A14', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A15', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A16', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='A2', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None)
                ], source_table=UnavailableSourceTable(table='DISTRICT'),
                pkey='DISTRICT_ID', time_col=None, end_time_col=None),
            'LOAN':
            TableDefinition(
                cols=[
                    ColumnDefinition(name='LOAN_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='ACCOUNT_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='DATE', stype=Stype.timestamp,
                                     dtype=Dtype.date, timestamp_format=None),
                    ColumnDefinition(name='AMOUNT', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='DURATION', stype=Stype.categorical,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='PAYMENTS', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='STATUS', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None)
                ], source_table=UnavailableSourceTable(table='LOAN'),
                pkey='LOAN_ID', time_col=None, end_time_col=None),
            'ORDERS':
            TableDefinition(
                cols=[
                    ColumnDefinition(name='ORDERS_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='ACCOUNT_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='BANK_TO', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None),
                    ColumnDefinition(name='AMOUNT', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='K_SYMBOL', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None),
                    ColumnDefinition(name='ACCOUNT_TO', stype=Stype.numerical,
                                     dtype=Dtype.int64, timestamp_format=None)
                ], source_table=UnavailableSourceTable(table='ORDERS'),
                pkey='ORDERS_ID', time_col=None, end_time_col=None),
            'TRANS':
            TableDefinition(
                cols=[
                    ColumnDefinition(name='TRANS_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='ACCOUNT_ID', stype=Stype.ID,
                                     dtype=Dtype.int, timestamp_format=None),
                    ColumnDefinition(name='DATE', stype=Stype.timestamp,
                                     dtype=Dtype.date, timestamp_format=None),
                    ColumnDefinition(name='TYPE', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None),
                    ColumnDefinition(name='OPERATION', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None),
                    ColumnDefinition(name='AMOUNT', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='BALANCE', stype=Stype.numerical,
                                     dtype=Dtype.float, timestamp_format=None),
                    ColumnDefinition(name='K_SYMBOL', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None),
                    ColumnDefinition(name='BANK', stype=Stype.categorical,
                                     dtype=Dtype.string, timestamp_format=None),
                    ColumnDefinition(name='ACCOUNT', stype=Stype.numerical,
                                     dtype=Dtype.int64, timestamp_format=None)
                ], source_table=UnavailableSourceTable(table='TRANS'),
                pkey='TRANS_ID', time_col=None, end_time_col=None)
        }, col_groups=[
            ColumnKeyGroup(
                columns=(ColumnKey(table_name='ACCOUNT', col_name='ACCOUNT_ID'),
                         ColumnKey(table_name='DISP', col_name='ACCOUNT_ID'),
                         ColumnKey(table_name='LOAN', col_name='ACCOUNT_ID'),
                         ColumnKey(table_name='ORDERS', col_name='ACCOUNT_ID'),
                         ColumnKey(table_name='TRANS', col_name='ACCOUNT_ID'))),
            ColumnKeyGroup(columns=(
                ColumnKey(table_name='ACCOUNT', col_name='DISTRICT_ID'),
                ColumnKey(table_name='CLIENT', col_name='DISTRICT_ID'),
                ColumnKey(table_name='DISTRICT', col_name='DISTRICT_ID'))),
            ColumnKeyGroup(
                columns=(ColumnKey(table_name='CARD', col_name='DISP_ID'),
                         ColumnKey(table_name='DISP', col_name='DISP_ID'))),
            ColumnKeyGroup(
                columns=(ColumnKey(table_name='CLIENT', col_name='CLIENT_ID'),
                         ColumnKey(table_name='DISP', col_name='CLIENT_ID'))),
        ])


def test_read_parquet_local_unchanged(tmp_path):
    # Local paths keep the native read (no cloud filesystem).
    path = str(tmp_path / 'x.parquet')
    pd.DataFrame({'a': [1, 2, 3]}).to_parquet(path)
    out = read_parquet(path, cols=['a'])
    assert list(out['a']) == [1, 2, 3]


def test_read_parquet_routes_s3_through_arrow_cloud_filesystem(monkeypatch):
    # S3 reads route through the credentialed choke point with a
    # scheme-stripped path handed to the native reader.
    import pyarrow as pa
    from pyarrow import parquet

    import io_utils.pyarrow as io_pyarrow

    seen = {}
    sentinel_fs = object()

    def fake_acf(path):
        seen['fs_path'] = path
        return sentinel_fs

    def fake_read_table(path, columns=None, filesystem=None):
        seen['path'] = path
        seen['columns'] = columns
        seen['filesystem'] = filesystem
        return pa.table({'a': [1, 2]})

    monkeypatch.setattr(io_pyarrow, 'arrow_cloud_filesystem', fake_acf)
    monkeypatch.setattr(parquet, 'read_table', fake_read_table)

    out = read_parquet('s3://bucket/key/f.parquet', cols=['a'])

    assert seen['fs_path'] == 's3://bucket/key/f.parquet'
    assert seen['path'] == 'bucket/key/f.parquet'
    assert seen['columns'] == ['a']
    assert seen['filesystem'] is sentinel_fs
    assert list(out['a']) == [1, 2]
