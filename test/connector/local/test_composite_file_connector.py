# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import random
from dataclasses import asdict
from unittest.mock import MagicMock

import pandas as pd
import pytest
from kumo_core.types import ColStatType
from kumo_core.types.conversions import dtype_to_pandas
from kumoapi.typing import Dtype, Stype
from pydantic import SecretStr

from io_utils.fsutil.credentials import credentials_for_provider
from kumo.connector.composite_file_connector import CompositeFileConnector
from kumo.connector.connector import Connector, ConnectorError
from kumo.graph.graph import Graph
from kumo.stats import compute_col_stats_python
from kumo.validation.validation import ValidationException
from rest.config.credentials import (
    AWSFederatedCredentials,
    AzureFederatedCredentials,
    GoogleCloudCredentials,
)
from rest.config.data_backend import ResolvedConnectorSpec
from rest.config.identity import validate_resolved_connector
from statedb.config.common import ConnectorID


def test_mix_file_connector():
    root_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    connector = CompositeFileConnector(
        'test_connector', root_dir=osp.join(root_dir, '..', 'csv_parquet_data'))
    table_names = connector.list_tables()
    assert len(table_names) == 3

    _, typ = connector.get_path_and_type('ACCOUNT')
    assert typ == 'parquet'
    _, typ = connector.get_path_and_type('CLIENT')
    assert typ == 'parquet'
    _, typ = connector.get_path_and_type('CARD')
    assert typ == 'csv'


def test_listed_but_unreadable_file_error(monkeypatch):
    root_dir = 's3://public-bucket/card_fraud_sample'
    connector = CompositeFileConnector('test_connector', root_dir=root_dir)

    monkeypatch.setattr('kumo.connector.composite_file_connector.fsutil.isdir',
                        lambda path: False)
    monkeypatch.setattr('kumo.connector.composite_file_connector.fsutil.exists',
                        lambda path: False)
    monkeypatch.setattr(
        'kumo.connector.composite_file_connector.fsutil.listdir',
        lambda path, max_items=None: [
            f'{root_dir}/account_table.parquet',
            f'{root_dir}/card_table.parquet',
        ][:max_items])

    with pytest.raises(ConnectorError, match='s3:GetObject'):
        connector.get_path_and_type('account_table')


@pytest.mark.parametrize('root_dir', [
    'gs://some-bucket/exports',
    'abfss://container@account.dfs.core.windows.net/exports',
    's3://locked-down-bucket/exports',
])
def test_missing_table_region_hint_is_best_effort(monkeypatch, root_dir):
    connector = CompositeFileConnector('test_connector', root_dir=root_dir)
    monkeypatch.setattr('kumo.connector.composite_file_connector.fsutil.isdir',
                        lambda path: False)
    monkeypatch.setattr('kumo.connector.composite_file_connector.fsutil.exists',
                        lambda path: False)
    monkeypatch.setattr(
        'kumo.connector.composite_file_connector.fsutil.listdir',
        lambda path, max_items=None: [])
    region_lookup = MagicMock(side_effect=ValueError('region unavailable'))
    monkeypatch.setattr(
        'kumo.connector.composite_file_connector.fsutil.s3_filesystem.'
        'get_bucket_aws_region', region_lookup)

    with pytest.raises(FileNotFoundError, match='does not exist in path'):
        connector.get_path_and_type('account_table')

    if root_dir.startswith('s3://'):
        region_lookup.assert_called_once_with('locked-down-bucket')
    else:
        region_lookup.assert_not_called()


@pytest.mark.parametrize(
    ("relative_name", "file_type"),
    [
        ("2026/07/events.parquet", "parquet"),
        ("exports/users.csv", "csv"),
    ],
)
def test_direct_connector_relative_file(relative_name, file_type, monkeypatch):
    root_dir = 's3://customer-bucket/project-prefix'
    expected_path = f'{root_dir}/{relative_name}'
    connector = CompositeFileConnector('test_connector', root_dir=root_dir)

    monkeypatch.setattr(
        'kumo.connector.composite_file_connector.fsutil.isdir',
        lambda path: False,
    )
    monkeypatch.setattr(
        'kumo.connector.composite_file_connector.fsutil.exists',
        lambda path: path == expected_path,
    )

    assert connector._get_path(relative_name) == expected_path
    assert connector.get_path_and_type(relative_name) == (
        expected_path,
        file_type,
    )


@pytest.mark.parametrize(
    ('root_dir', 'provider', 'credentials'),
    [
        (
            's3://company-data/empty/events',
            's3',
            AWSFederatedCredentials(
                access_key_id=SecretStr('temporary-access-key'),
                secret_access_key=SecretStr('temporary-secret-key'),
                session_token=SecretStr('temporary-session-token'),
            ),
        ),
        (
            'gs://company-data/empty/events',
            'gcs',
            GoogleCloudCredentials(
                provider='gcs',
                credential=MagicMock(name='gcs-credential'),
                project='workload-project',
            ),
        ),
        (
            'abfss://curated@companylake.dfs.core.windows.net/empty/events',
            'adls',
            AzureFederatedCredentials(
                credential=MagicMock(name='azure-credential'), ),
        ),
    ],
)
def test_identity_file_validation_lists_exact_empty_root_once_without_writes(
        root_dir, provider, credentials, monkeypatch):
    """The identity probe is scoped, exact-target, bounded, and read-only."""
    connector = CompositeFileConnector('identity_probe', root_dir=root_dir)
    calls = []
    existence_checks = []

    def list_exact_root(*, path, max_items):
        scope = credentials_for_provider(provider)
        assert scope is not None
        assert scope.credentials == credentials.as_dict()
        calls.append((path, max_items))
        return []  # Empty prefixes are valid connector targets.

    monkeypatch.setattr(
        'kumo.connector.composite_file_connector.fsutil.listdir',
        list_exact_root,
    )

    def exists_exact_root(path):
        existence_checks.append(path)
        return True

    monkeypatch.setattr(
        'kumo.connector.composite_file_connector.fsutil.exists',
        exists_exact_root,
    )
    open_probe = MagicMock(
        side_effect=AssertionError('file existence probe must not run'))
    monkeypatch.setattr(CompositeFileConnector, '_open', open_probe)
    writes = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    for name, write in zip(
        ('touch', 'write_text', 'write_binary', 'makedirs'),
            writes,
            strict=True,
    ):
        monkeypatch.setattr(
            f'kumo.connector.composite_file_connector.fsutil.{name}',
            write,
            raising=False,
        )

    validate_resolved_connector(
        ResolvedConnectorSpec(
            connector_id=ConnectorID.generate(),
            connector_config=connector.config(),
            credentials=credentials,
        ))

    assert calls == [(root_dir, 1)]
    assert existence_checks == ([root_dir] if provider == 'adls' else [])
    open_probe.assert_not_called()
    for write in writes:
        write.assert_not_called()


def test_identity_file_validation_fails_closed_on_list_denial(monkeypatch):
    connector = CompositeFileConnector('identity_probe',
                                       root_dir='s3://company-data/restricted')
    credentials = AWSFederatedCredentials(
        access_key_id=SecretStr('temporary-access-key'),
        secret_access_key=SecretStr('temporary-secret-key'),
        session_token=SecretStr('temporary-session-token'),
    )

    def denied(**_kwargs):
        raise PermissionError('provider denied exact-prefix listing')

    monkeypatch.setattr(
        'kumo.connector.composite_file_connector.fsutil.listdir', denied)

    with pytest.raises(PermissionError, match='denied'):
        validate_resolved_connector(
            ResolvedConnectorSpec(
                connector_id=ConnectorID.generate(),
                connector_config=connector.config(),
                credentials=credentials,
            ))


def test_identity_adls_validation_rejects_missing_exact_target(monkeypatch):
    root_dir = (
        'abfss://curated@companylake.dfs.core.windows.net/missing/events')
    connector = CompositeFileConnector('identity_probe', root_dir=root_dir)
    credentials = AzureFederatedCredentials(credential=MagicMock(
        name='azure-credential'))
    listdir = MagicMock()
    monkeypatch.setattr(
        'kumo.connector.composite_file_connector.fsutil.exists',
        lambda path: False,
    )
    monkeypatch.setattr(
        'kumo.connector.composite_file_connector.fsutil.listdir', listdir)

    with pytest.raises(ConnectorError, match='ADLS.*unavailable'):
        validate_resolved_connector(
            ResolvedConnectorSpec(
                connector_id=ConnectorID.generate(),
                connector_config=connector.config(),
                credentials=credentials,
            ))

    listdir.assert_not_called()


def test_csv_connector_config():
    connector = CompositeFileConnector('test_connector', root_dir='my_test_dir')

    assert asdict(connector.config()) == {
        '_target_':
        'kumo.connector.composite_file_connector.CompositeFileConnector',
        'name': 'test_connector',
        'root_dir': 'my_test_dir',
    }
    assert connector == Connector.from_config(connector.config())


def test_parquet_connector_load():
    root_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'FINANCIAL')
    connector = CompositeFileConnector('parquet', root_dir=root_dir)

    pkey = 'TRANS_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('DATE', Stype.timestamp, Dtype.date),
        ('TYPE', Stype.categorical, Dtype.string),
        ('OPERATION', Stype.categorical, Dtype.string),
        ('AMOUNT', Stype.numerical, Dtype.float),
    ]
    df = connector.read_table('TRANS', pkey, cols)
    assert len(df.columns) == len(cols)
    assert df.columns.values.tolist() == [col[0] for col in cols]
    assert df.dtypes.values.tolist() == [dtype_to_pandas(col[2]) for col in cols]

    df = connector.read_table('TRANS', pkey, cols, num_rows=1234)
    assert df.shape[0] == 1234

    df = connector.read_table_sample('TRANS', pkey, cols, num_rows=12)
    assert df.shape[0] == 12


def test_load_csv_sample():
    root_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'FINANCIAL')
    connector = CompositeFileConnector('csv', root_dir=root_dir)

    pkey = 'TRANS_ID'
    cols = [
        ('ACCOUNT_ID', Stype.ID, Dtype.int),
        ('DATE', Stype.timestamp, Dtype.date),
        ('TYPE', Stype.categorical, Dtype.string),
        ('OPERATION', Stype.categorical, Dtype.string),
        ('AMOUNT', Stype.numerical, Dtype.float),
    ]

    df = connector.read_table_sample('TRANS', pkey, cols, num_rows=12)

    assert df.shape[0] == 12
    assert len(df.columns) == len(cols)
    assert df.columns.values.tolist() == [col[0] for col in cols]
    assert df.dtypes.values.tolist() == [dtype_to_pandas(col[2]) for col in cols]

    df = connector.read_table('TRANS', pkey, cols, num_rows=1234)
    assert df.shape[0] == 1234

    random.seed(10)
    frac = 0.01
    df = connector.read_table_sample('TRANS', pkey, cols, frac=frac)
    assert df.shape[0] == 50


def test_csv_connector_list_tables():
    root_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'FINANCIAL')
    connector = CompositeFileConnector('csv', root_dir=root_dir)

    table_metadata_configs = connector.get_table_metadata(ignore_errors=True)
    assert len(table_metadata_configs) == 15

    root_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'shard_test')
    connector = CompositeFileConnector('csv', root_dir=root_dir)
    table_metadata_configs = connector.get_table_metadata()
    assert len(table_metadata_configs) == 2
    table_names = {t.source_name for t in table_metadata_configs}
    assert table_names == {'full', 'sharded_csv'}


def test_nested_table_boundaries_and_validation(tmp_path):
    root = tmp_path / 'datasets'
    files = [
        'namespace/orders/part-0.parquet',
        'warehouse/events/year=2025/month=01/part-0.parquet',
        'overlap/part-0.parquet',
        'overlap/child/part-1.parquet',
        'mixed/a.csv',
        'mixed/b.parquet',
        '_temporary/ignored.parquet',
    ]
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    connector = CompositeFileConnector('nested', root_dir=str(root))

    assert connector.list_tables() == [
        'namespace/orders',
        'overlap/child',
        'warehouse/events',
    ]
    with pytest.raises(ValidationException, match='key=value'):
        connector.get_path_and_type('overlap')
    with pytest.raises(ValidationException, match='mix CSV and Parquet'):
        connector.get_path_and_type('mixed')
    with pytest.raises(ValidationException, match='namespace/orders'):
        connector.get_path_and_type('namespace')


def test_list_tables_bounds_discovery_without_limiting_table_files(
        tmp_path, monkeypatch):
    root = tmp_path / 'datasets'
    root.mkdir()
    (root / 'a.csv').touch()
    (root / 'b.csv').touch()
    connector = CompositeFileConnector('nested', root_dir=str(root))
    monkeypatch.setattr(
        'kumo_core.sources.layout.MAX_SCANNED_ENTRIES', 2)

    assert connector.list_tables() == ['a', 'b']
    (root / 'c.csv').touch()
    with pytest.raises(ConnectorError, match='more than 2'):
        connector.list_tables()

    large_table = root / 'large'
    large_table.mkdir()
    for part in range(201):
        (large_table / f'part-{part}.parquet').touch()
    assert connector.get_path_and_type('large') == (f'{large_table}/',
                                                    'parquet')


def test_parquet_connector_list_tables():
    root_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'FINANCIAL')
    connector = CompositeFileConnector('parquet', root_dir=root_dir)

    table_metadata_configs = connector.get_table_metadata(ignore_errors=True)
    expect_names = [
        'ACCOUNT',
        'ACCOUNT_NA',
        'CARD',
        'CLIENT',
        'CLIENT_INDEXED',
        'DISP',
        'DISTRICT',
        'LOAN',
        'LOAN_TRAIN_SCHEMA',
        'ORDERS',
        'PREDICTION',
        'TRAINING_BINARY',
        'TRAINING_CLASSIFICATION',
        'TRAINING_REGRESSION',
        'TRANS',
        'TRANS_SMALL',
    ]
    assert len(table_metadata_configs) == len(expect_names)
    names = sorted(table.table_name for table in table_metadata_configs)
    assert names == expect_names

    root_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'TEST')
    connector = CompositeFileConnector('parquet', root_dir=root_dir)
    table_metadata_configs = connector.get_table_metadata()
    assert len(table_metadata_configs) == 4
    assert 'TRANS' in table_metadata_configs[0].table_name
    assert 'TRANS' in table_metadata_configs[1].table_name
    assert 'TRANS' in table_metadata_configs[2].table_name
    assert 'TRANS' in table_metadata_configs[3].table_name


def test_csv_col_types_infer_and_cast():
    root_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'movielens')
    connector = CompositeFileConnector('csv', root_dir=root_dir)
    table_metadata_configs = connector.get_table_metadata()

    graph = Graph(name='test_graph')
    graph.connect(connector)
    graph.add('tag_small')
    graph.add('tags_epoch_small')
    graph.add('synthetic_table')

    expected_dtypes = {
        'tag_small': {
            'userId': Dtype.int64,
            'movieId': Dtype.int64,
            'tag': Dtype.string,
            'timestamp': Dtype.time
        },
        'tags_epoch_small': {
            'userId': Dtype.int64,
            'movieId': Dtype.int64,
            'tag': Dtype.string,
            'timestamp': Dtype.time
        },
        'synthetic_table': {
            'index': Dtype.int64,
            'textcol': Dtype.string,
            'integer': Dtype.int64,
            'multicat1': Dtype.string,
            'lists': Dtype.string,
            'date': Dtype.date,
        }
    }
    expected_stypes = {
        'tag_small': {
            'userId': Stype.ID,
            'movieId': Stype.ID,
            'tag': Stype.categorical,
            'timestamp': Stype.timestamp
        },
        'tags_epoch_small': {
            'userId': Stype.ID,
            'movieId': Stype.ID,
            'tag': Stype.categorical,
            'timestamp': Stype.timestamp
        },
        'synthetic_table': {
            'index': Stype.numerical,
            'textcol': Stype.text,
            'integer': Stype.numerical,
            'multicat1': Stype.multicategorical,
            'lists': Stype.text,
            'date': Stype.timestamp,
        }
    }
    for table in table_metadata_configs:
        # test the two versions of the tag table
        if 'tag' in table.source_name or 'synthetic' in table.source_name:
            for col in table.cols:
                graph[table.name].add_col(col.name, dtype=col.dtype)
                assert col.dtype == expected_dtypes[table.source_name][col.name]

            graph[table.name].infer_stypes()
            # infer timestamp unit:
            graph[table.name].infer_dtypes_from_stypes()

            # Kumo-owned col-type infer/cast assertions. The legacy tail that
            # drove a removed Backend-ABC ingest path (raise sentinel) was
            # dropped; the surviving load-bearing surface is the local
            # dtype + stype inference above.
            for col in graph[table.name].cols:
                assert col.stype == expected_stypes[table.source_name][col.name]


def test_parquet_name_inference_requires_parseable_timestamp_values(tmp_path):
    root_dir = tmp_path / 'files'
    table_dir = root_dir / 'events'
    table_dir.mkdir(parents=True)
    valid = [f'2026-01-{day:02d}' for day in range(1, 21)]
    pd.DataFrame({
        'CONSUMER_TEMPLATE_DATE': [
            f'{i}_hub_notification_{date}'
            for i, date in enumerate(valid)
        ],
        'ORDER_DATE': valid,
        'AT_THRESHOLD_DATE': valid[:18] + ['invalid'] * 2,
        'ABOVE_THRESHOLD_DATE': valid[:17] + ['invalid'] * 3,
        'MOSTLY_NULL_VALID_DATE': pd.Series(
            [pd.NA] * 18 + valid[:2], dtype='string'),
        'MOSTLY_NULL_AT_THRESHOLD_DATE': pd.Series(
            [pd.NA] * 10 + valid[:9] + ['invalid'], dtype='string'),
        'MOSTLY_NULL_INVALID_DATE': pd.Series(
            [pd.NA] * 18 + [valid[0], 'invalid'], dtype='string'),
        'ALL_NULL_DATE': pd.Series([pd.NA] * 20, dtype='string'),
    }).to_parquet(table_dir / 'part-000.parquet', index=False)

    connector = CompositeFileConnector('files', root_dir=str(root_dir))
    [metadata] = connector.get_table_metadata(['events'], skip_cache=True)
    dtypes = {col.name: col.dtype for col in metadata.cols}

    assert dtypes['CONSUMER_TEMPLATE_DATE'] == Dtype.string
    assert dtypes['ORDER_DATE'] == Dtype.date
    assert dtypes['AT_THRESHOLD_DATE'] == Dtype.date
    assert dtypes['ABOVE_THRESHOLD_DATE'] == Dtype.string
    assert dtypes['MOSTLY_NULL_VALID_DATE'] == Dtype.date
    assert dtypes['MOSTLY_NULL_AT_THRESHOLD_DATE'] == Dtype.date
    assert dtypes['MOSTLY_NULL_INVALID_DATE'] == Dtype.string
    assert dtypes['ALL_NULL_DATE'] == Dtype.string


def test_parquet_name_inference_rejects_day_less_integer_dates(tmp_path):
    # A column NAME matching a date token must not promote a bare year (%Y) or
    # year-month (%Y%m) integer column: those carry no day, so treating them as
    # dates would silently invent one. An integer that encodes a full calendar
    # date (YYYYMMDD) still promotes, since it needs nothing invented.
    root_dir = tmp_path / 'files'
    table_dir = root_dir / 'events'
    table_dir.mkdir(parents=True)
    pd.DataFrame({
        'FISCAL_YEAR_DATE': list(range(1990, 2010)),
        'REPORT_MONTH_DATE': ([200001 + m for m in range(12)]
                              + [200101 + m for m in range(8)]),
        'EVENT_FULL_DATE': [20200101 + day for day in range(20)],
        'BILLING_YEAR_DATE': pd.Series(
            [str(year) for year in range(1990, 2010)], dtype='string'),
        'STMT_MONTH_DATE': pd.Series(
            [f'{200001 + m}' for m in range(12)]
            + [f'{200101 + m}' for m in range(8)], dtype='string'),
    }).to_parquet(table_dir / 'part-000.parquet', index=False)

    connector = CompositeFileConnector('files', root_dir=str(root_dir))
    [metadata] = connector.get_table_metadata(['events'], skip_cache=True)
    dtypes = {col.name: col.dtype for col in metadata.cols}

    # Day-less integers keep their physical integer dtype (not promoted).
    assert dtypes['FISCAL_YEAR_DATE'] == Dtype.int64
    assert dtypes['REPORT_MONTH_DATE'] == Dtype.int64
    # The same guard applies to bare year / year-month string columns.
    assert dtypes['BILLING_YEAR_DATE'] == Dtype.string
    assert dtypes['STMT_MONTH_DATE'] == Dtype.string
    # A full encoded date still promotes -- the guard is specific to day-less.
    assert dtypes['EVENT_FULL_DATE'] == Dtype.date


def test_csv_num_rows():
    root_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'FINANCIAL')
    connector = CompositeFileConnector('csv', root_dir=root_dir)

    # single file
    assert connector._get_table_num_rows('LOAN') == 682

    # directory with multiple files
    root_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root_dir = osp.join(root_dir, '..', '..', 'test', 'csv_data', 'shard_test')

    shard_connector = CompositeFileConnector('csv', root_dir, na_values='?')
    assert shard_connector._get_table_num_rows('sharded_csv') == 77


@pytest.mark.xfail(
    reason="dg-bug: DG-native stype inference returns multicategorical "
    "for the `lists` stringlist column on synthetic_types_variety_2 "
    "where the legacy backend inferred sequence. "
    "TODO(item-12): reconcile stype inference for stringlist columns.",
    strict=False,
)
def test_parquet_col_types_infer_and_cast():
    root = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), '../parquet_data',
                    'SYNTHETIC')
    connector = CompositeFileConnector('parquet', root_dir=root)
    table_metadata_configs = connector.get_table_metadata()

    graph = Graph(name='test_graph')
    graph.connect(connector)

    table_name_1 = 'synthetic_types_variety_1'
    table_name_2 = 'synthetic_types_variety_2'

    graph.add(table_name_1)
    graph.add(table_name_2)

    expected_dtypes = {
        f'parquet.{table_name_2}': {
            'ID': Dtype.int32,
            'integer': Dtype.int64,
            'uinteger': Dtype.int64,
            'integer8': Dtype.int32,
            'duration': Dtype.timedelta,  # parquet type = duration[ns]
            'seconds': Dtype.string,
            'date': Dtype.date,
            'lists': Dtype.stringlist,
            'multicat1': Dtype.string,
            'multicat2': Dtype.string
        },
        # DAT_1 and TIME_1 have string types from pyarrow since they are not
        # time since epoch formatted - date formats like DAT_1 can be manually
        # changed to a a timestamp stype but are not automatically assumed
        f'parquet.{table_name_1}': {
            'NUM_1': Dtype.int64,
            'CHAR_1': Dtype.string,
            'CHAR_2': Dtype.string,
            'CHAR_3': Dtype.string,
            'BIN_1': Dtype.string,
            'BOOL': Dtype.bool,
            'DAT_1': Dtype.string,
            'DAT_2': Dtype.date,
            'TIME_1': Dtype.time,
            'TIME_2': Dtype.date,
            'TIME_3': Dtype.date,
            'TIME_4': Dtype.date,
            'TIME_5': Dtype.date,
        }
    }
    expected_stypes = {
        f'parquet.{table_name_2}': {
            'ID': Stype.ID,
            'integer': Stype.numerical,
            'uinteger': Stype.numerical,
            'integer8': Stype.categorical,
            'duration': Stype.numerical,
            'seconds': Stype.multicategorical,
            'date': Stype.timestamp,
            'lists': Stype.sequence,
            'multicat1': Stype.multicategorical,
            'multicat2': Stype.multicategorical
        },
        f'parquet.{table_name_1}': {
            'NUM_1': Stype.numerical,
            'CHAR_1': Stype.categorical,
            'CHAR_2': Stype.categorical,
            'CHAR_3': Stype.categorical,
            'BIN_1': Stype.categorical,
            'BOOL': Stype.categorical,
            'DAT_1': Stype.categorical,
            'DAT_2': Stype.timestamp,
            'TIME_1': Stype.timestamp,
            'TIME_2': Stype.timestamp,
            'TIME_3': Stype.timestamp,
            'TIME_4': Stype.timestamp,
            'TIME_5': Stype.timestamp,
        }
    }

    table_dict = {table.name: table for table in table_metadata_configs}

    table = table_dict[f'parquet.{table_name_2}']

    for col in table.cols:
        if col.dtype != Dtype.unsupported:
            graph[table.name].add_col(col.name, dtype=col.dtype)
        assert col.dtype == expected_dtypes[table.name][col.name]

    graph[table.name].infer_stypes()

    for col in graph[table.name].cols:
        assert col.stype == expected_stypes[table.name][col.name]

    table = table_dict[f'parquet.{table_name_1}']
    for col in table.cols:
        graph[table.name].add_col(col.name, dtype=col.dtype)
        assert col.dtype == expected_dtypes[table.name][col.name]

    graph[table.name].infer_stypes()

    for col in graph[table.name].cols:
        assert col.stype == expected_stypes[table.name][col.name]

    stats_dict_1 = {
        col: [ColStatType.NUM_NA]
        for col in graph[table_name_1].metadata_config().all_cols()
    }

    stats_dict_2 = {
        col: [ColStatType.NUM_NA]
        for col in graph[table_name_2].metadata_config().all_cols()
    }

    # read table sample checks if the data can be cast to the assigned types
    # without failures
    df = connector.read_table_sample(
        table_name_1,
        cols=[col.base_config() for col in graph[table_name_1].cols])
    assert df.shape[0] == 100
    assert df.shape[1] == 13

    for col, stats in stats_dict_1.items():
        stats_result = compute_col_stats_python(df[col.name], stats)
        assert stats_result[ColStatType.NUM_NA].value == 0

    df = connector.read_table_sample(
        table_name_2,
        cols=[col.base_config() for col in graph[table_name_2].cols])
    assert df.shape[0] == 10000
    assert df.shape[1] == 10

    for col, stats in stats_dict_2.items():
        stats_result = compute_col_stats_python(df[col.name], stats)
        assert stats_result[ColStatType.NUM_NA].value == 0
