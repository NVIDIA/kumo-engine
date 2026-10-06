# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import datetime
from typing import get_args

import kumoapi.source_table as source_table_api
import kumoapi.table as table_api
import pytest
from kumo_core.types import ColStatType
from kumo_core.types.schema import TS_FORMAT_METADATA
from kumoapi.source_table import FileType
from kumoapi.typing import Dtype, Stype

from common_utils.error import PublicException
from kumo.stats.stats import Stats
from kumo.validation import ValidationException
from rest.config import table_v2
from rest.config.connector import FileConnectorResourceConfig
from rest.config.table import _get_date_hist
from rest.config.table_v2 import (
    ADLSSourceTable,
    BigQuerySourceTable,
    Column,
    DatabricksSourceTable,
    DataSourceType,
    GCSSourceTable,
    S3SourceTable,
    SnowflakeSourceTable,
    TableDefinition,
    TimestampUnit,
    UnavailableSourceTable,
)
from statedb.config.common import ConnectorID


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


@pytest.fixture(autouse=True, scope="session")
def temporal_workers_ready():
    yield True


def test_get_date_hist():
    assert _get_date_hist({}) == []
    assert _get_date_hist({
        ColStatType.HISTOGRAM_HALF_MONTH:
        Stats(value=None, approx=False),
    }) == []
    assert _get_date_hist({
        ColStatType.HISTOGRAM_HALF_MONTH:
        Stats(value=[32], approx=False),
    }) == []
    d1 = datetime.date(2000, 12, 1)
    d2 = datetime.date(2000, 12, 15)
    d3 = datetime.date(2000, 12, 16)
    d4 = datetime.date(2000, 12, 31)
    assert _get_date_hist({
        ColStatType.HISTOGRAM_HALF_MONTH:
        Stats(value=[(d1, 45), (d3, 50)], approx=False),
    }) == [(d1, d2, 45), (d3, d4, 50)]


def test_convert_to_v2():
    test_v2_table = TableDefinition(
        cols=(Column(
            'test_col',
            Stype.categorical,
            Dtype.string,
            TimestampUnit('s'),
        ), ),
        source_table=S3SourceTable(
            connector_id=ConnectorID(
                'connector_00000000000000000000000000000001'),
            source={
                'mode': 'relative',
                'path': 'customer.parquet'
            },
            file_type=FileType.PARQUET,
        ),
        pkey='test_col',
    )

    assert len(test_v2_table.cols) == 1
    assert test_v2_table.pkey == 'test_col'
    assert test_v2_table.cols[0].timestamp_format == TimestampUnit('s')
    assert test_v2_table.cols[0].dtype == Dtype.string
    assert test_v2_table.cols[0].stype == Stype.categorical


def test_multicategorical_delimiter_survives_all_v2_column_conversions():
    table = TableDefinition(
        cols=(Column(
            name='tags',
            stype=Stype.multicategorical,
            dtype=Dtype.string,
            multicategorical_delimiter=';',
        ), ),
        source_table=S3SourceTable(
            connector_id='connector_' + '1' * 32,
            source={
                'mode': 'relative',
                'path': 'items.parquet'
            },
            file_type=FileType.PARQUET,
        ),
    )

    reconstructed = TableDefinition(**dataclasses.asdict(table))
    feature = reconstructed.cols[0].to_feature_column()
    base_config = reconstructed.col_to_base_config('tags')

    assert reconstructed.cols[0].multicategorical_delimiter == ';'
    assert feature.multicategorical_delimiter == ';'
    assert base_config is not None
    assert base_config.multicategorical_delimiter == ';'


@pytest.mark.parametrize('timestamp_format', ['', '   '])
def test_blank_timestamp_format_is_absent_in_v2_column_conversions(
        timestamp_format):
    col = Column(
        name='customer_id',
        stype=Stype.ID,
        dtype=Dtype.string,
        timestamp_format=timestamp_format,
    )
    table = TableDefinition(
        cols=(col, ),
        source_table=S3SourceTable(
            connector_id='connector_' + '1' * 32,
            source={
                'mode': 'relative',
                'path': 'customers.parquet'
            },
            file_type=FileType.PARQUET,
        ),
        pkey='customer_id',
    )

    feature = table.cols[0].to_feature_column()
    base_config = table.col_to_base_config('customer_id')
    no_format_col = dataclasses.replace(col, timestamp_format=None)

    assert col.timestamp_format is None
    assert table.cols[0].timestamp_format is None
    assert feature.metadata == {}
    assert base_config is not None
    assert base_config.metadata == {}
    assert col.fpid == no_format_col.fpid


def test_real_timestamp_format_survives_v2_column_conversions():
    col = Column(
        name='event_time',
        stype=Stype.timestamp,
        dtype=Dtype.string,
        timestamp_format='%Y-%m-%d',
    )
    table = TableDefinition(
        cols=(col, ),
        source_table=S3SourceTable(
            connector_id='connector_' + '1' * 32,
            source={
                'mode': 'relative',
                'path': 'events.parquet'
            },
            file_type=FileType.PARQUET,
        ),
        pkey='event_time',
    )

    feature = table.cols[0].to_feature_column()
    base_config = table.col_to_base_config('event_time')

    assert feature.metadata == {TS_FORMAT_METADATA: '%Y-%m-%d'}
    assert base_config is not None
    assert base_config.metadata == {TS_FORMAT_METADATA: '%Y-%m-%d'}


def test_delimiter_validation_accepts_extended_kumo_enums():
    column = Column(
        name='tags',
        stype=Stype.multicategorical,
        dtype=Dtype.string,
        multicategorical_delimiter='|',
    )

    assert column.multicategorical_delimiter == '|'


@pytest.mark.parametrize(
    ('stype', 'dtype'),
    [
        (Stype.categorical, Dtype.string),
        (Stype.multicategorical, Dtype.stringlist),
    ],
)
def test_delimiter_validation_rejects_incompatible_extended_kumo_enums(
        stype, dtype):
    with pytest.raises(ValueError, match='multicategorical_delimiter'):
        Column(
            name='tags',
            stype=stype,
            dtype=dtype,
            multicategorical_delimiter=',',
        )


def test_s3_source_table_name_is_connector_relative():
    s3_source_table = S3SourceTable(
        connector_id='connector-s3',
        source={
            'mode': 'relative',
            'path': 'customerltv_mini_integ_test.parquet'
        },
        file_type=FileType.PARQUET,
    )
    assert s3_source_table.table_name == 'customerltv_mini_integ_test.parquet'
    s3_source_table = S3SourceTable(
        connector_id='connector-s3',
        source={
            'mode': 'relative',
            'path': 'nested/customerltv_mini_integ_test'
        },
        file_type=FileType.PARQUET,
    )
    assert s3_source_table.table_name == ('nested/customerltv_mini_integ_test')


@pytest.mark.parametrize(
    ("source_cls", "api_source_cls"),
    [
        (GCSSourceTable, source_table_api.GCSSourceTable),
        (ADLSSourceTable, source_table_api.ADLSSourceTable),
    ],
)
def test_file_source_table_preserves_relative_metadata(source_cls,
                                                       api_source_cls):
    source_table = source_cls(
        connector_id="connector-project-owned",
        source_table_name="minimal_hnm/USERS.csv",
        file_type=FileType.CSV,
    )
    assert source_table.connector_id == "connector-project-owned"
    assert source_table.source_table_name == 'minimal_hnm/USERS.csv'
    assert source_table.file_type == FileType.CSV

    table_def = TableDefinition(
        cols=(),
        source_table=api_source_cls(
            connector_id="connector-project-owned",
            source_table_name="minimal_hnm/USERS.csv",
            file_type=source_table_api.FileType.CSV,
        ),
    )
    assert table_def.source_table.connector_id == source_table.connector_id
    assert table_def.source_table.source_table_name == ('minimal_hnm/USERS.csv')
    assert table_def.source_table.file_type == FileType.CSV


def test_s3_source_table_preserves_discriminated_relative_source():
    source_table = S3SourceTable(
        connector_id="connector-project-owned",
        source={
            'mode': 'relative',
            'path': "minimal_hnm/USERS.csv",
        },
        file_type=FileType.CSV,
    )
    assert source_table.connector_id == "connector-project-owned"
    assert source_table.source.path == 'minimal_hnm/USERS.csv'
    assert source_table.file_type == FileType.CSV

    table_def = TableDefinition(
        cols=(),
        source_table=source_table_api.S3SourceTable(
            connector_id="connector-project-owned",
            source={
                'mode': 'relative',
                'path': "minimal_hnm/USERS.csv",
            },
            file_type=source_table_api.FileType.CSV,
        ),
    )
    assert table_def.source_table.connector_id == source_table.connector_id
    assert table_def.source_table.source.path == 'minimal_hnm/USERS.csv'
    assert table_def.source_table.file_type == FileType.CSV


def _make_v2_cols() -> tuple:
    return (
        Column(name='id', stype=Stype.ID, dtype=Dtype.string,
               timestamp_format=None),
        Column(name='ts', stype=Stype.timestamp, dtype=Dtype.time,
               timestamp_format=table_api.TimestampUnit.SECOND),
        Column(name='val', stype=Stype.numerical, dtype=Dtype.float32,
               timestamp_format=None),
    )


def _s3_named() -> TableDefinition:
    return TableDefinition(
        cols=_make_v2_cols(),
        source_table=S3SourceTable(
            file_type=FileType.PARQUET,
            connector_id='connector_' + '1' * 32,
            source={
                'mode': 'relative',
                'path': 'transactions'
            },
        ),
        pkey='id',
        time_col='ts',
    )


def _gcs_named() -> TableDefinition:
    return TableDefinition(
        cols=_make_v2_cols(),
        source_table=GCSSourceTable(
            file_type=FileType.PARQUET,
            connector_id='connector_' + '2' * 32,
            source_table_name='transactions',
        ),
        pkey='id',
        time_col='ts',
    )


def _snowflake() -> TableDefinition:
    return TableDefinition(
        cols=_make_v2_cols(),
        source_table=SnowflakeSourceTable(
            snowflake_connector_id='my_snowflake',
            database='KUMO',
            schema_name='PUBLIC',
            table='transactions',
            data_source_type=DataSourceType.SNOWFLAKE,
        ),
        pkey='id',
        time_col='ts',
    )


def _databricks() -> TableDefinition:
    return TableDefinition(
        cols=_make_v2_cols(),
        source_table=DatabricksSourceTable(
            databricks_connector_id='my_databricks',
            table='hive_metastore.default.transactions',
        ),
        pkey='id',
        time_col='ts',
    )


def _bigquery() -> TableDefinition:
    return TableDefinition(
        cols=_make_v2_cols(),
        source_table=BigQuerySourceTable(
            bigquery_connector_id='my_bq',
            table_name='transactions',
            project_id='my_project',
            dataset_id='my_dataset',
        ),
        pkey='id',
        time_col='ts',
    )


def _unavailable() -> TableDefinition:
    return TableDefinition(
        cols=_make_v2_cols(),
        source_table=UnavailableSourceTable(table='client_side_table'),
        pkey='id',
        time_col='ts',
    )


def test_unavailable_source_table_has_stable_fpid_and_no_connector():
    table_def = _unavailable()

    assert table_def.source_table.fpid
    assert table_def.fpid
    assert table_def.source_table.table_location == 'client_side_table'
    assert not table_def.source_table.require_connector(
        FileConnectorResourceConfig(name='internal',
                                    root_dir='/tmp/client_side_table'))
    assert TableDefinition(
        **dataclasses.asdict(table_def)).source_table.fpid == (
            table_def.source_table.fpid)


def test_table_def_primary_key_col_returns_none_when_pkey_missing():
    td = TableDefinition(
        cols=_make_v2_cols(),
        source_table=S3SourceTable(
            file_type=FileType.PARQUET,
            connector_id='c',
            source={
                'mode': 'relative',
                'path': 't'
            },
        ),
        # pkey omitted -> None
    )
    assert td.primary_key_col() is None


def test_table_def_time_col_returns_none_when_unset():
    td = TableDefinition(
        cols=_make_v2_cols(),
        source_table=S3SourceTable(
            file_type=FileType.PARQUET,
            connector_id='c',
            source={
                'mode': 'relative',
                'path': 't'
            },
        ),
        pkey='id',
        # time_col / end_time_col omitted
    )
    assert td.time_column() is None
    assert td.end_time_column() is None


def test_table_def_primary_key_col_raises_when_pkey_missing_from_cols():
    # `primary_key_col()` is a bound method, so we exercise it on a stub
    # that has the same shape (`pkey`, `cols`) without constructing a full
    # TableDefinition.
    #
    # A pkey/time_col/end_time_col that names a column not present in `cols`
    # is a pure user-input error surfaced by the graph-build data-backend
    # activities. It must raise an allowlisted `ValidationException`
    # (a `PublicException`), NOT a bare `RuntimeError`: a non-allowlisted
    # type is masked by the activity's credential guard as a generic
    # "contact Kumo support" InternalError (finding 013). Reverting any of
    # these raises to `RuntimeError` regresses the finding and fails here.
    class _Stub:
        pkey = 'ghost'
        cols = (Column(name='real', stype=Stype.numerical,
                       dtype=Dtype.float32), )

    with pytest.raises(ValidationException) as excinfo:
        TableDefinition.primary_key_col(_Stub())
    assert 'ghost' in str(excinfo.value)
    assert isinstance(excinfo.value, PublicException)
    assert not isinstance(excinfo.value, RuntimeError)


def test_table_def_time_cols_raise_validation_when_missing_from_cols():
    # Same finding-013 contract for `time_column()` and `end_time_column()`:
    # a named time/end-time column not present in `cols` must raise the
    # allowlisted `ValidationException`, never a bare `RuntimeError`.
    real_col = Column(name='real', stype=Stype.numerical, dtype=Dtype.float32)

    class _TimeStub:
        time_col = 'ghost'
        cols = (real_col, )

    class _EndTimeStub:
        end_time_col = 'ghost'
        cols = (real_col, )

    with pytest.raises(ValidationException) as time_err:
        TableDefinition.time_column(_TimeStub())
    with pytest.raises(ValidationException) as end_err:
        TableDefinition.end_time_column(_EndTimeStub())

    for excinfo in (time_err, end_err):
        assert 'ghost' in str(excinfo.value)
        assert isinstance(excinfo.value, PublicException)
        assert not isinstance(excinfo.value, RuntimeError)


def test_numerical_stats_without_quantiles_names_what_is_missing():
    r"""A numerical column can arrive correctly typed with the numerical
    statistics themselves absent. Indexing the quantile list unguarded turned
    that into a bare ``IndexError`` several frames from the cause
    (kumo-ai/kumo#30917).
    """
    from rest.config.table import ColumnStats

    counting_stats_only = {
        ColStatType.NUM_NA: Stats(value=0, approx=True, num_entries=10000),
        ColStatType.COUNT: Stats(value=10000, approx=True, num_entries=10000),
        ColStatType.NUM_UNIQUE: Stats(value=9893, approx=True,
                                      num_entries=10000),
    }

    with pytest.raises(ValueError, match='quartile statistics'):
        ColumnStats.from_stats_dict(Stype.numerical, counting_stats_only, 10000,
                                    Dtype.float64)


def test_every_shared_source_table_type_has_a_mirror():
    mirrored = {cls.__name__ for cls in table_v2._SOURCE_TABLE_TYPES}

    unmirrored = [
        member.__name__
        for member in get_args(source_table_api.SourceTableType)
        if member.__name__ not in mirrored
    ]

    assert unmirrored == []


def test_an_unmirrored_source_table_type_is_refused_rather_than_stored():
    unmirrored = source_table_api.LocalFileSourceTable(path='/tmp/rows.parquet')

    with pytest.raises(ValueError, match='LocalFileSourceTable'):
        TableDefinition._validate_source_table(unmirrored)
