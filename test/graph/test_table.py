# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from kumo_core.types import ColStatType
from kumo_core.types.schema import TS_FORMAT_METADATA
from kumoapi.typing import Dtype, Stype

from kumo.connector.csv_connector import CSVConnector
from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.column import Column, EncoderGenerator, ForeignKey, PrimaryKey
from kumo.graph.graph import Graph
from kumo.graph.table import StypeInferralStats, Table
from kumo.stats.compute_stats_python import compute_col_stats_python
from statedb.utils.errors import UnsupportedInDiskgraphError

TEST_TABLE_PARQUET_NAME = "TMP_TEST_TABLE"
TEST_TABLE_PARQUET_FILE = TEST_TABLE_PARQUET_NAME + ".parquet"


def get_cur_dir_path(file_name=TEST_TABLE_PARQUET_FILE):
    root_dir = osp.dirname(osp.realpath(__file__))
    file_path = osp.join(root_dir, file_name)
    return file_path, root_dir


def teardown_function():
    file_path, _ = get_cur_dir_path(TEST_TABLE_PARQUET_FILE)
    if os.path.exists(file_path):
        os.remove(file_path)


def test_table():
    table = Table(table_name='LOAN', data_source='csv', pkey='LOAN_ID')
    assert table.table_name == 'LOAN'
    assert str(table) == ('Table(\n'
                          '  name=csv.LOAN,\n'
                          '  data_source=csv,\n'
                          '  pkey=LOAN_ID,\n'
                          '  fkeys=[],\n'
                          '  cols=[],\n'
                          '  time_col=???,\n'
                          '  end_time_col=???,\n'
                          ')')

    assert len(table) == 1
    assert table.num_fkeys == 0
    assert table.num_cols == 0
    assert table.degree == 0

    assert 'LOAN_ID' in table
    assert table.has_pkey()
    assert table.is_pkey('LOAN_ID')

    assert not table.has_col('AMOUNT')
    assert not table.has_fkey('ACCOUNT_ID')
    assert 'AMOUNT' not in table and 'ACCOUNT_ID' not in table

    assert table.pkey.name == 'LOAN_ID'
    assert table.pkey.dtype is None
    assert table['LOAN_ID'] == table.pkey

    table.add_fkey('ACCOUNT_ID', dtype=Dtype.int, dst_table='ACCOUNT')
    table.add_col('AMOUNT', dtype=Dtype.float)
    table.add_col('STATUS', dtype=Dtype.string)
    table.set_time_col('TIME')
    assert table.time_col.dtype is Dtype.date
    table.set_time_col('TIME', dtype=Dtype.date)
    assert table.time_col.dtype == Dtype.date
    with pytest.raises(ValueError):
        table.set_time_col('TIME', dtype=None)
    with pytest.raises(ValueError):
        table.set_time_col('TIME', dtype=Dtype.int)
    table.set_end_time_col('END_TIME', dtype=Dtype.date)
    assert table.time_col.dtype == Dtype.date
    with pytest.raises(ValueError):
        table.set_end_time_col('END_TIME', dtype=None)
    with pytest.raises(ValueError):
        table.set_end_time_col('END_TIME', dtype=Dtype.int)

    assert str(table) == ('Table(\n'
                          '  name=csv.LOAN,\n'
                          '  data_source=csv,\n'
                          '  pkey=LOAN_ID,\n'
                          '  fkeys=[ACCOUNT_ID],\n'
                          '  cols=[AMOUNT, STATUS],\n'
                          '  time_col=TIME,\n'
                          '  end_time_col=END_TIME,\n'
                          ')')

    assert len(table) == 4
    assert table.num_fkeys == 1
    assert table.num_cols == 2
    assert table.degree == 1

    assert 'ACCOUNT_ID' in table
    assert table.has_fkey('ACCOUNT_ID')
    assert table.fkey('ACCOUNT_ID').name == 'ACCOUNT_ID'
    assert table.fkey('ACCOUNT_ID').dtype == Dtype.int
    assert table.fkey('ACCOUNT_ID').dst_table == 'ACCOUNT'
    assert table['ACCOUNT_ID'] == table.fkey('ACCOUNT_ID')
    assert len(table.fkeys) == 1
    assert table.fkeys[0] == table.fkey('ACCOUNT_ID')
    assert table.fkey_names == ['ACCOUNT_ID']

    assert 'AMOUNT' in table
    assert table.has_col('AMOUNT')
    assert table.col('AMOUNT').name == 'AMOUNT'
    assert table.col('AMOUNT').dtype == Dtype.float
    assert table['AMOUNT'] == table.col('AMOUNT')
    assert 'STATUS' in table
    assert table.has_col('STATUS')
    assert table.col('STATUS').name == 'STATUS'
    assert table.col('STATUS').dtype == Dtype.string
    assert table['STATUS'] == table.col('STATUS')
    assert len(table.cols) == 2
    assert table.cols[0] == table.col('AMOUNT')
    assert table.cols[1] == table.col('STATUS')
    assert table.col_names == ['AMOUNT', 'STATUS']

    table.remove_fkey('ACCOUNT_ID')
    assert len(table) == 3
    assert table.num_fkeys == 0

    table.remove_col('AMOUNT')
    assert len(table) == 2
    assert table.num_cols == 1

    del table['STATUS']
    assert len(table) == 1
    assert table.num_cols == 0


def test_infer_encoders_error_message_includes_inner_exception() -> None:
    table = Table(
        table_name='USERS',
        data_source='csv',
        cols=[
            Column(name='COUNTRY', dtype=Dtype.string, stype=Stype.categorical)
        ],
    )
    encoder_stats = MagicMock(col_stats={'COUNTRY': {}})
    missing_stat = ValueError(
        'Column Stat ColStatType.CATEGORY_COUNTS is required for encoder '
        'inferral but was not provided.')

    with patch.object(EncoderGenerator, 'get_encoder',
                      side_effect=missing_stat):
        with pytest.raises(RuntimeError) as excinfo:
            table.infer_encoders(encoder_stats=encoder_stats)

    assert repr(missing_stat) in str(excinfo.value)
    assert excinfo.value.__cause__ is missing_stat


def test_table_set_end_time_col():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'csv_data',
                        'COUPON_PURCHASE_PREDICTION')

    graph = Graph(name='COUPON_PURCHASE_PREDICTION')
    graph.connect(CSVConnector('csv', root_dir, na_values='?'))
    graph.add('COUPON_DETAIL', pkey='COUPON_ID_HASH')
    graph['COUPON_DETAIL'].infer_metadata()
    graph['COUPON_DETAIL'].set_time_col('DISPFROM')
    graph['COUPON_DETAIL'].set_end_time_col('DISPEND')
    graph.add('COUPON_LIST')
    graph['COUPON_LIST'].add_fkey('COUPON_ID_HASH')
    graph.infer_metadata()
    graph.infer_encoders()
    assert graph['COUPON_DETAIL'].end_time_col.name == 'DISPEND'
    assert 'DISPEND' not in graph['COUPON_DETAIL'].cols

    # Check if the table config is correctly created
    table_config = graph['COUPON_DETAIL'].config()
    assert table_config.end_time_col.name == 'DISPEND'
    table = Table.from_config(table_config)
    assert table.end_time_col.name == 'DISPEND'


def test_table_move_cols():
    table = Table(table_name='LOAN', data_source='csv', pkey='LOAN_ID')
    assert table.pkey_name == 'LOAN_ID'
    assert table.fkey_names == []
    assert table.col_names == []

    table.add_col('ACCOUNT_ID')
    assert table.pkey_name == 'LOAN_ID'
    assert table.fkey_names == []
    assert table.col_names == ['ACCOUNT_ID']

    table.add_fkey('ACCOUNT_ID')
    assert table.pkey_name == 'LOAN_ID'
    assert table.fkey_names == ['ACCOUNT_ID']
    assert table.col_names == []

    table.add_fkey('LOAN_ID')
    assert table.pkey_name is None
    assert table.fkey_names == ['ACCOUNT_ID', 'LOAN_ID']
    assert table.col_names == []


def test_table_modified_stypes():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'csv_data', 'FINANCIAL')

    graph = Graph(name='FINANCIAL')
    graph.connect(CSVConnector('csv', root_dir, na_values='?'))

    graph.add('ACCOUNT', pkey='ACCOUNT_ID')
    # force the `DATE` column to be a `string` so it can be updated
    # later by editing `stype`
    graph['ACCOUNT'].add_col('DATE', dtype=Dtype.string)
    graph['ACCOUNT'].infer_metadata()

    assert graph['ACCOUNT', 'DATE'].dtype == Dtype.string
    assert graph['ACCOUNT', 'DATE'].stype == Stype.categorical

    assert graph['ACCOUNT', 'DISTRICT_ID'].dtype == Dtype.int64
    assert graph['ACCOUNT', 'DISTRICT_ID'].stype == Stype.ID

    # test `infer_dtypes_from_stypes`
    graph['ACCOUNT', 'DATE'].stype = Stype.timestamp
    graph['ACCOUNT', 'DISTRICT_ID'].stype = Stype.categorical

    graph['ACCOUNT'].infer_dtypes_from_stypes()

    assert graph['ACCOUNT', 'DATE'].stype == Stype.timestamp
    assert graph['ACCOUNT', 'DATE'].dtype == Dtype.time

    assert graph['ACCOUNT', 'DISTRICT_ID'].stype == Stype.categorical
    assert graph['ACCOUNT', 'DISTRICT_ID'].dtype == Dtype.int64

    # test `edit_stypes`
    cols_to_stypes = {'DISTRICT_ID': Stype.ID}
    graph['ACCOUNT'].edit_stypes(cols_to_stypes)

    assert graph['ACCOUNT', 'DISTRICT_ID'].stype == Stype.ID
    assert graph['ACCOUNT', 'DISTRICT_ID'].dtype == Dtype.int64

    graph.add('LOAN', pkey='LOAN_ID')

    # force duration to be a `date` so it can be updated later
    graph['LOAN'].add_col('DATE', stype=Stype.timestamp, dtype=Dtype.int64)
    graph['LOAN'].infer_metadata()
    graph['LOAN'].infer_dtypes_from_stypes()
    assert graph['LOAN', 'DATE'].dtype == Dtype.time

    cols_to_stypes = {'DATE': Stype.numerical}
    graph['LOAN'].edit_stypes(cols_to_stypes)
    graph['LOAN'].infer_dtypes_from_stypes()
    assert graph['LOAN', 'DATE'].stype == Stype.numerical
    assert graph['LOAN', 'DATE'].dtype == Dtype.int64


def _ts_table(tmp_path, df, cols):
    r"""Build a `Table` over a temp parquet so `infer_dtypes_from_stypes` can
    read a real sample through a `ParquetConnector`."""
    df.to_parquet(osp.join(str(tmp_path), 'T.parquet'))
    table = Table(table_name='T', data_source='', source_name='T', cols=cols,
                  data_version=None)
    table.connector = ParquetConnector(name='test', root_dir=str(tmp_path))
    return table


def test_infer_dtypes_epoch_int_sets_timestamp_unit(tmp_path):
    # An epoch-int `timestamp` column resolves to a UNIT hint: the detector
    # writes `timestamp_unit` and leaves `metadata['TS_FORMAT']` absent.
    df = pd.DataFrame({
        'EPOCH_MS':
        pd.Series([1659304801388, 1659304901388, 1659305001388], dtype='int64')
    })
    col = Column('EPOCH_MS', dtype=Dtype.int64, stype=Stype.timestamp)
    table = _ts_table(tmp_path, df, [col])
    table.infer_dtypes_from_stypes()
    resolved = table['EPOCH_MS']
    assert resolved.timestamp_unit == 'ms'
    assert TS_FORMAT_METADATA not in resolved.metadata
    assert resolved.dtype == Dtype.time


def test_infer_dtypes_encoded_date_sets_ts_format(tmp_path):
    # An encoded-date (YYYYMMDD) `timestamp` column resolves to a FORMAT hint:
    # the detector writes `metadata['TS_FORMAT']` and leaves `timestamp_unit`
    # None (a unit here would corrupt the column as epoch seconds).
    df = pd.DataFrame(
        {'ED': pd.Series([20230408, 20240115, 20250301], dtype='int64')})
    col = Column('ED', dtype=Dtype.int64, stype=Stype.timestamp)
    table = _ts_table(tmp_path, df, [col])
    table.infer_dtypes_from_stypes()
    resolved = table['ED']
    assert resolved.metadata.get(TS_FORMAT_METADATA) == '%Y%m%d'
    assert resolved.timestamp_unit is None
    assert resolved.dtype == Dtype.time


def test_infer_dtypes_already_hinted_columns_unchanged(tmp_path):
    # The gate leaves a column that already carries a unit OR a format hint
    # untouched; the detector never re-infers a hinted column.
    df = pd.DataFrame({
        'U':
        pd.Series([1, 2, 3], dtype='int64'),
        'F':
        pd.Series([20230408, 20240115, 20250301], dtype='int64'),
    })
    ucol = Column('U', dtype=Dtype.int64, stype=Stype.timestamp)
    ucol.timestamp_unit = 'ns'
    fcol = Column('F', dtype=Dtype.int64, stype=Stype.timestamp)
    fcol.metadata[TS_FORMAT_METADATA] = '%d-%b-%Y'
    table = _ts_table(tmp_path, df, [ucol, fcol])
    table.infer_dtypes_from_stypes()
    # Pre-set unit preserved, no format added.
    assert table['U'].timestamp_unit == 'ns'
    assert TS_FORMAT_METADATA not in table['U'].metadata
    # Pre-set format preserved, no unit added, even though the raw values would
    # otherwise infer '%Y%m%d'.
    assert table['F'].metadata.get(TS_FORMAT_METADATA) == '%d-%b-%Y'
    assert table['F'].timestamp_unit is None


def test_table_unsupported_dtypes():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'TEST')
    connector = ParquetConnector('parquet', root_dir=root_dir)
    metadata = connector.get_table_metadata(["TRANS4"])[0]
    assert metadata.cols[0].dtype == Dtype.int64
    assert metadata.cols[1].dtype == Dtype.unsupported  # null type
    assert metadata.cols[2].dtype == Dtype.int64
    assert metadata.cols[3].dtype == Dtype.intlist  # list type

    graph = Graph(name='TEST')
    graph.connect(connector)

    graph.add('TRANS4', pkey='TRANS_ID')
    graph['TRANS4'].add_col('ACCOUNT_ID')
    graph['TRANS4'].add_col('AMOUNT')
    graph['TRANS4'].add_col('TRANS_ID')
    graph['TRANS4'].add_col('UNSUPPORTED')
    graph['TRANS4'].infer_metadata()

    assert graph['TRANS4', 'ACCOUNT_ID'].dtype == Dtype.int64
    assert graph['TRANS4', 'ACCOUNT_ID'].stype == Stype.ID
    assert graph['TRANS4', 'AMOUNT'].dtype == Dtype.unsupported
    assert graph['TRANS4', 'AMOUNT'].stype == Stype.unsupported
    assert graph['TRANS4', 'TRANS_ID'].dtype == Dtype.int64
    assert graph['TRANS4', 'TRANS_ID'].stype == Stype.ID
    assert graph['TRANS4', 'UNSUPPORTED'].dtype == Dtype.intlist
    assert graph['TRANS4', 'UNSUPPORTED'].stype == Stype.multicategorical


def test_table_add_pkey_fkey_as_feature():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'TEST')

    graph = Graph(name='TEST')
    graph.connect(ParquetConnector('parquet', root_dir))

    graph.add('TRANS4', pkey='TRANS_ID', fkeys=['ACCOUNT_ID'])
    graph['TRANS4'].add_col('ACCOUNT_ID')
    graph['TRANS4'].add_col('TRANS_ID')
    graph['TRANS4'].infer_metadata()

    # On DiskGraph, calling add_col() on a column that was already declared
    # as a pkey/fkey strips the key designation in favour of the feature
    # column (see Table.add_col logger.info "Removing it in order to add it
    # as a feature column"). The dtype/stype should still resolve from the
    # underlying data.
    assert graph['TRANS4', 'ACCOUNT_ID'].dtype == Dtype.int64
    assert graph['TRANS4', 'ACCOUNT_ID'].stype == Stype.ID
    assert graph['TRANS4', 'TRANS_ID'].dtype == Dtype.int64
    assert graph['TRANS4', 'TRANS_ID'].stype == Stype.ID


def test_infer_stype_without_stats():
    table = Table(
        table_name='LOAN', data_source='csv', pkey='LOAN_ID', cols=[
            Column('STRING_ID', dtype=Dtype.string),
            Column('STRING', dtype=Dtype.string),
            Column('INT_VALUE', dtype=Dtype.int64),
            Column('VECTOR', dtype=Dtype.floatlist),
            Column('VECTOR_INT', dtype=Dtype.intlist),
            Column('TIMESTAMP', dtype=Dtype.time),
            Column('timestamp_int', dtype=Dtype.int)
        ])

    table.infer_stypes(require_stats=False)

    assert table['STRING_ID'].stype == Stype.ID
    assert table['STRING'].stype == Stype.categorical
    assert table['INT_VALUE'].stype == Stype.numerical
    assert table['VECTOR'].stype == Stype.sequence
    assert table['TIMESTAMP'].stype == Stype.timestamp
    assert table['timestamp_int'].stype == Stype.numerical
    assert table['VECTOR_INT'].stype == Stype.multicategorical


def test_infer_stype_threshold():
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Generate a test DataFrame
    file_path, root_dir = get_cur_dir_path(TEST_TABLE_PARQUET_FILE)
    int_col_no_duplicates = "int_col_with_all_unique"
    int_col_with_duplicates = "int_col_with_duplicates"

    df = pd.DataFrame({
        int_col_no_duplicates: [1, 2, 3, 4, 5],
        int_col_with_duplicates: [1, 1, 2, 2, 3],
    })

    table = pa.Table.from_pandas(df)
    # Write it to Parquet file
    pq.write_table(table, file_path)

    graph = Graph(name='TEST')
    graph.connect(ParquetConnector('parquet', root_dir))

    graph.add(TEST_TABLE_PARQUET_NAME)
    graph[TEST_TABLE_PARQUET_NAME].add_col(int_col_no_duplicates)
    graph[TEST_TABLE_PARQUET_NAME].add_col(int_col_with_duplicates)

    graph.infer_metadata()

    assert graph[TEST_TABLE_PARQUET_NAME,
                 int_col_no_duplicates].dtype == Dtype.int64
    assert graph[TEST_TABLE_PARQUET_NAME,
                 int_col_no_duplicates].stype == Stype.numerical
    assert graph[TEST_TABLE_PARQUET_NAME,
                 int_col_with_duplicates].dtype == Dtype.int64
    assert graph[TEST_TABLE_PARQUET_NAME,
                 int_col_with_duplicates].stype == Stype.categorical


def test_infer_dtypes_from_stypes_timedelta_writes_unit_not_metadata():
    """Regression: timedelta-source columns persist the unit via the
    typed ``BaseColumnConfig.timestamp_unit`` field, not via
    ``metadata[TS_FORMAT]``. The legacy code wrote the unit into the
    metadata dict as an out-of-band signal for Spark; the typed field
    is the proper destination now that Spark is gone.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from kumo_core.types.schema import TS_FORMAT_METADATA

    file_path, root_dir = get_cur_dir_path(TEST_TABLE_PARQUET_FILE)

    # timedelta64[ns] is the only timedelta unit pandas/pyarrow round-trip
    # through parquet without dtype-narrowing surprises.
    df = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'DURATION': pd.to_timedelta([1, 2, 3, 4, 5], unit='s'),
    })

    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)

    graph = Graph(name='TEST')
    graph.connect(ParquetConnector('parquet', root_dir))

    graph.add(TEST_TABLE_PARQUET_NAME, pkey='ID')
    graph[TEST_TABLE_PARQUET_NAME].add_col('DURATION')
    graph[TEST_TABLE_PARQUET_NAME].infer_metadata()

    # Force stype=timestamp so infer_dtypes_from_stypes enters the
    # timestamp branch (the branch under test).
    graph[TEST_TABLE_PARQUET_NAME, 'DURATION'].stype = Stype.timestamp
    graph[TEST_TABLE_PARQUET_NAME].infer_dtypes_from_stypes()

    col = graph[TEST_TABLE_PARQUET_NAME, 'DURATION']
    # timestamp_unit is set on the typed field. pandas upgrades any
    # timedelta input to ns on parquet round-trip.
    assert col.timestamp_unit == 'ns', (
        f"expected timestamp_unit='ns', got {col.timestamp_unit!r}")
    # ...and metadata[TS_FORMAT] is NOT used as a smuggling channel.
    assert TS_FORMAT_METADATA not in col.metadata, (
        f"expected no metadata[TS_FORMAT] write, got "
        f"{col.metadata.get(TS_FORMAT_METADATA)!r}")


def test_id_columns_are_set_correctly():
    table = Table(
        table_name='LOAN', data_source='csv', pkey='LOAN_ID',
        cols=[Column('STRING_ID', dtype=Dtype.string)], fkeys=[
            ForeignKey(
                'ACCOUNT_ID',
                dst_table="ACCOUNT",
                dst_key="ACCOUNT_ID",
                dtype=Dtype.float,
            )
        ])

    table.compute_col_stats = MagicMock()
    table.infer_stypes()

    assert table['STRING_ID'].stype == Stype.ID
    assert table['LOAN_ID'].stype == Stype.ID
    assert table['ACCOUNT_ID'].stype == Stype.ID


def test_table_names():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'SYNTHETIC')

    graph = Graph(name='test_graph')
    connector = ParquetConnector('parquet', root_dir)
    graph.connect(connector)

    graph.add(name='small_t1', pkey='c0', fkeys=[('count')])
    graph.add(name='small_t2', pkey='count', fkeys=[('c0', 'c1')],
              source_name='small_t2')
    graph.infer_metadata()

    meta_table = connector.get_table_metadata(['small_t2'])[0]

    # from source config tests handling/validation of source names
    Table.from_source_config(meta_config=meta_table, name="placeholder_name",
                             data_source=meta_table.data_source,
                             connector=connector)

    graph = Graph(name='test_graph')
    graph.connect(ParquetConnector('parquet', root_dir))

    t2_char_test_name = 'small-t1'
    graph.add(name='small_t1', pkey='c0', fkeys=[('count')])
    graph.add(name='small_t2', pkey='count', fkeys=[('c0', 'c1')],
              source_name=t2_char_test_name)
    graph.infer_metadata()

    meta_table = connector.get_table_metadata([t2_char_test_name])[0]

    # from source config tests handling/validation of source names
    Table.from_source_config(meta_config=meta_table, name="placeholder_name",
                             data_source=meta_table.data_source,
                             connector=connector)


def test_source_table_name_rejects_parent_traversal():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'SYNTHETIC')
    graph = Graph(name='test_graph')
    graph.connect(ParquetConnector('parquet', root_dir))

    t2_char_test_name = '../small_t2'
    graph.add(name='small_t1', pkey='c0', fkeys=[('count')])
    with pytest.raises(ValueError, match='cannot contain.*parent path'):
        graph.add(name='small_t2', pkey='count', fkeys=[('c0', 'c1')],
                  source_name=t2_char_test_name)


def test_table_accepts_rootless_cloud_source_name():
    source_name = 'gs://company-data/nested/events.parquet'
    table = Table(table_name='events', data_source='cloud',
                  source_name=source_name)

    assert table.source_name == source_name


def test_table_list_types():
    # Kumo-owned list-dtype/stype inference (``Table.infer_stypes`` /
    # ``infer_dtypes_from_stypes``) over LOCAL parquet. The remote-connector
    # (BigQuery + Databricks) seed loop the legacy test ran was dropped: it
    # only re-exercised the same Kumo inference against credentialed remote
    # sources, which the unit harness has no access to. The parquet block is
    # the load-bearing surface (the DG round-trip of the resulting dtype is
    # covered upstream in ``ingest_list_features_e2e.rs``; the Kumo-side
    # inference decision is not).
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'movielens_lists')
    parquet_test_connector = ParquetConnector('parquet', root_dir=root_dir)

    # intlist / stringlist both infer to multicategorical under current
    # Kumo schema inference (``Table._infer_stype_from_dtype``). The legacy
    # ``intlist -> sequence`` (no-ALLOW_ARRAY_TARGETS) expectation reflected
    # an inference rule that no longer exists; ``ALLOW_ARRAY_TARGETS`` no
    # longer gates list-stype inference here (it now only affects the
    # BigQuery dtype map). Assert the current Kumo-owned decision.
    list_types = [("int", Dtype.intlist), ("string", Dtype.stringlist)]
    for type_, dtype in list_types:
        table_name = f"movie_{type_}list"
        metadata = parquet_test_connector.get_table_metadata([table_name])[0]
        table = Table.from_source_config(metadata, name=table_name,
                                         data_source=metadata.data_source,
                                         connector=parquet_test_connector)
        table.infer_stypes()
        table.infer_dtypes_from_stypes()

        assert table['genres'].dtype == dtype
        assert table['genres'].stype == Stype.multicategorical
        assert table['genres'].multicategorical_delimiter is None


@pytest.mark.parametrize(
    'raw_values',
    [
        ['08:15:00', '09:30:00', '10:45:00', '11:00:00'],
        ['["red", "blue"]', '["green"]', '["blue"]'],
    ],
)
def test_scalar_strings_without_supported_delimiter_infer_categorical(
        raw_values):
    """Time and JSON-like scalar strings must not become list semantics."""
    values = pd.Series(raw_values, dtype='string')
    computed = compute_col_stats_python(
        values,
        StypeInferralStats.REQUIRED_STATS,
    )
    table = Table(
        table_name='TRANSACTIONS',
        data_source='bigquery',
        cols=[Column('Time', dtype=Dtype.string)],
    )
    table.compute_col_stats = MagicMock(return_value={'Time': computed}, )

    table.infer_stypes()

    assert computed[ColStatType.MULTI_CATEGORIES_SEPARATOR].value is None
    assert table['Time'].dtype == Dtype.string
    assert table['Time'].stype == Stype.categorical
    assert table['Time'].multicategorical_delimiter is None


def test_supported_delimiter_proof_allows_scalar_multicategorical_inference():
    values = pd.Series(
        ['red,blue', 'red', 'blue', 'red,red'],
        dtype='string',
    )
    computed = compute_col_stats_python(
        values,
        StypeInferralStats.REQUIRED_STATS,
    )
    table = Table(
        table_name='ITEMS',
        data_source='bigquery',
        cols=[Column('tags', dtype=Dtype.string)],
    )
    table.compute_col_stats = MagicMock(return_value={'tags': computed})

    table.infer_stypes()

    assert computed[ColStatType.MULTI_CATEGORIES_SEPARATOR].value == ','
    assert table['tags'].stype == Stype.multicategorical
    assert table['tags'].multicategorical_delimiter == ','


def test_declared_scalar_multicategorical_reuses_approx_stats_delimiter():
    values = pd.Series(['red,blue'] * 10 + ['plain'] * 990, dtype='string')
    computed = compute_col_stats_python(
        values,
        StypeInferralStats.REQUIRED_STATS,
    )
    table = Table(
        table_name='ITEMS',
        data_source='bigquery',
        cols=[
            Column(
                'tags',
                stype=Stype.multicategorical,
                dtype=Dtype.string,
            )
        ],
    )
    table.compute_col_stats = MagicMock(return_value={'tags': computed})

    table.infer_stypes()

    table.compute_col_stats.assert_called_once()
    assert table.compute_col_stats.call_args.kwargs['cols'] == [table['tags']]
    assert table['tags'].multicategorical_delimiter == ','


def test_declared_scalar_multicategorical_trusts_explicit_delimiter():
    table = Table(
        table_name='ITEMS',
        data_source='bigquery',
        cols=[
            Column(
                'tags',
                stype=Stype.multicategorical,
                dtype=Dtype.string,
                multicategorical_delimiter='|',
            )
        ],
    )
    table.compute_col_stats = MagicMock()

    table.infer_stypes()

    table.compute_col_stats.assert_not_called()
    assert table['tags'].multicategorical_delimiter == '|'


# Tests below cover the tier-2 fallback for when the ``table_is_finalized=True``
# branch of ``Table.compute_col_stats`` hits a DG cache miss. The autouse
# ``_install_dg_col_stats_fallback`` hook in ``test/conftest.py`` would mask
# this by seeding the cache, so we temporarily unwrap
# ``Table.compute_col_stats`` for the duration of each test.


def _unwrap_compute_col_stats():
    """Return the unwrapped ``Table.compute_col_stats`` from the conftest
    autouse seed wrapper's closure (or the current attr if not wrapped)."""
    fn = Table.compute_col_stats
    closure = getattr(fn, '__closure__', None) or ()
    for cell in closure:
        candidate = cell.cell_contents
        if (callable(candidate)
                and getattr(candidate, '__name__', '') == 'compute_col_stats'
                and candidate is not fn):
            return candidate
    return fn


def _build_pandas_table(source_name: str = "users") -> Table:
    import pandas as pd

    from kumo.connector.pandas_connector import PandasConnector

    df = pd.DataFrame({
        "id":
        list(range(100)),
        "name": [f"user_{i % 7}" if i % 5 else None for i in range(100)],
        "score": [float(i) / 3.0 if i % 4 else None for i in range(100)],
    })
    # ``source_name`` keys the on-disk col-stats cache dir; tests that persist
    # to it pass a unique name so they don't share state across the session.
    connector = PandasConnector({source_name: df})
    table = Table(
        table_name=source_name,
        data_source="pandas",
        pkey=PrimaryKey("id", dtype=Dtype.int64),
        cols=[
            Column("name", dtype=Dtype.string, stype=Stype.text),
            Column("score", dtype=Dtype.float32, stype=Stype.numerical),
        ],
        source_name=source_name,
    )
    table.connector = connector
    return table


def test_compute_col_stats_approx_falls_back_to_connector_on_cache_miss(
        monkeypatch):
    from kumo_core.types import ColStatType

    monkeypatch.setattr(Table, 'compute_col_stats', _unwrap_compute_col_stats())
    table = _build_pandas_table()
    stats = table.compute_col_stats(
        col_stat_types=[ColStatType.NA_FRACTION],
        approx=True,
    )
    # Tier-2 connector fallback must have produced stats. The cache is
    # empty (no DG ingest in this test), so any non-empty result proves
    # the fallback fired.
    assert "name" in stats
    assert "score" in stats
    assert ColStatType.NA_FRACTION in stats["name"]
    assert ColStatType.NA_FRACTION in stats["score"]
    assert stats["name"][ColStatType.NA_FRACTION].approx is True


def test_compute_col_stats_approx_fallback_persists_and_reuses(monkeypatch):
    # The connector-sample fallback now persists its result to the backend
    # cache (restoring the write #28324 dropped), so a second call is served
    # from cache instead of re-sampling the source. Mirrors the REST
    # ``DataBackend.get_min_col_stats`` fallback.
    from kumo_core.types import ColStatType

    monkeypatch.setattr(Table, 'compute_col_stats', _unwrap_compute_col_stats())
    # Unique source so the on-disk cache dir isn't shared with sibling tests.
    table = _build_pandas_table(source_name="users_persist_reuse")

    sample_calls = {"n": 0}
    real_sample = table.connector.compute_col_stats

    def _counting_sample(*args, **kwargs):
        sample_calls["n"] += 1
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(table.connector, 'compute_col_stats', _counting_sample)

    first = table.compute_col_stats(col_stat_types=[ColStatType.NA_FRACTION],
                                    approx=True)
    assert first["name"][ColStatType.NA_FRACTION].approx is True
    assert sample_calls["n"] == 1  # sampled the source once

    second = table.compute_col_stats(col_stat_types=[ColStatType.NA_FRACTION],
                                     approx=True)
    assert second["name"][ColStatType.NA_FRACTION].approx is True
    assert sample_calls["n"] == 1  # served from cache, not re-sampled


def test_compute_col_stats_exact_wins_over_persisted_sample(monkeypatch):
    # Safety guard for the fallback persist: once an exact stat exists (as DG
    # ingest would write), the previously persisted approx sample must NOT be
    # served. Exact wins on read, so a stale sample can't survive ingestion.
    from kumo_core.types import ColStatType

    from kumo.data import get_active_cache
    from kumo.stats import Stats

    monkeypatch.setattr(Table, 'compute_col_stats', _unwrap_compute_col_stats())
    table = _build_pandas_table(source_name="users_exact_wins")
    data_version = table.connector.get_data_version(table.source_name)
    fingerprint = table.fingerprint()
    name_cfg = table.cols[0].base_config()

    # Fallback persists an approx NA_FRACTION...
    table.compute_col_stats(col_stat_types=[ColStatType.NA_FRACTION],
                            approx=True)
    # ...then an exact NA_FRACTION lands for the same key.
    cache = get_active_cache()
    cache.add_col_stats(
        fingerprint, {
            name_cfg: {
                ColStatType.NA_FRACTION:
                Stats(value=0.0, approx=False, data_version=data_version),
            },
        })
    cache.save_col_stats_cache(fingerprint, data_version)

    got = cache.get_col_stats(fingerprint,
                              {name_cfg: [ColStatType.NA_FRACTION]},
                              data_version, approx=True)
    assert got[name_cfg][ColStatType.NA_FRACTION].approx is False
    assert got[name_cfg][ColStatType.NA_FRACTION].value == 0.0


def test_compute_col_stats_exact_still_raises_without_cache(monkeypatch):
    from kumo_core.types import ColStatType

    monkeypatch.setattr(Table, 'compute_col_stats', _unwrap_compute_col_stats())
    table = _build_pandas_table()
    with pytest.raises(
            UnsupportedInDiskgraphError,
            match="Requested col stats missing from the DiskGraph-"
            "populated cache",
    ):
        table.compute_col_stats(
            col_stat_types=[ColStatType.NA_FRACTION],
            approx=False,
        )


def _build_pandas_table_with_time_col() -> Table:
    import pandas as pd

    from kumo.connector.pandas_connector import PandasConnector

    # Date range chosen so MIN/MAX are unambiguous and stable across runs.
    df = pd.DataFrame({
        "id":
        list(range(60)),
        "ts":
        pd.to_datetime([f"2024-01-{(i % 28) + 1:02d}" for i in range(60)]),
        "score": [float(i) / 3.0 if i % 4 else None for i in range(60)],
    })
    connector = PandasConnector({"events": df})
    table = Table(
        table_name="events",
        data_source="pandas",
        pkey=PrimaryKey("id", dtype=Dtype.int64),
        cols=[
            Column("score", dtype=Dtype.float32, stype=Stype.numerical),
        ],
        source_name="events",
    )
    table.connector = connector
    # Mirrors the production pquery setup: a TimeColumn with date dtype that
    # `_get_target_time_range` later queries for MIN/MAX.
    table.set_time_col("ts", dtype=Dtype.date)
    return table


def test_compute_col_stats_min_max_approx_no_cache(monkeypatch):
    # Mirrors the `_get_target_time_range` call shape in
    # kumo/pquery/generator_utils.py: approx MIN/MAX on the time column with an
    # empty DG cache. The tier-2 fallback must serve this without raising
    # UnsupportedInDiskgraphError (which would break pquery validation).
    from kumo_core.types import ColStatType

    monkeypatch.setattr(Table, 'compute_col_stats', _unwrap_compute_col_stats())
    table = _build_pandas_table_with_time_col()
    stats = table.compute_col_stats(
        [table.time_col],
        [ColStatType.MIN, ColStatType.MAX],
        approx=True,
    )
    assert table.time_col.name in stats
    by_stat = stats[table.time_col.name]
    assert ColStatType.MIN in by_stat
    assert ColStatType.MAX in by_stat
    assert by_stat[ColStatType.MIN].value is not None
    assert by_stat[ColStatType.MAX].value is not None
    assert by_stat[ColStatType.MIN].approx is True
    assert by_stat[ColStatType.MAX].approx is True


def test_compute_col_stats_partial_cache_approx_falls_back_for_missing_stats(
        monkeypatch):
    # Legacy-cache scenario: DG ingest populated *some* col stats for this
    # table but not the full set the caller asks for. The tier-2 fallback
    # must fill in the missing entries from the connector without
    # clobbering the cache-resident ones.
    from kumo_core.types import ColStatType
    from kumo_core.types.schema import BaseColumnConfig

    from kumo.data import get_active_cache
    from kumo.stats import Stats

    monkeypatch.setattr(Table, 'compute_col_stats', _unwrap_compute_col_stats())
    table = _build_pandas_table()
    data_version = table.connector.get_data_version(table.source_name)
    fingerprint = table.partial_fingerprint()

    # Seed *only* COUNT for each data column with a sentinel value so the
    # assertion below can distinguish cache-hit (sentinel) from
    # tier-2-recompute (would yield the real row count). COUNT is chosen
    # because it passes both `ColStatType.supports(dtype)` AND
    # `supports_stype(stype)` for the text and numerical columns this
    # helper builds — NUM_UNIQUE fails the stype gate for both stypes and
    # would be silently dropped from `col_stats_dict` before lookup. The
    # cache lookup key embeds (col_cfg.stype, col_cfg.dtype), so mirror
    # the shape `compute_col_stats` builds from `col.base_config()`.
    sentinel_by_col = {"name": 4242, "score": 9999}
    seed: dict[BaseColumnConfig, dict[ColStatType, Stats]] = {}
    for col in table.cols:
        cfg = col.base_config()
        seed[cfg] = {
            ColStatType.COUNT:
            Stats(value=sentinel_by_col[col.name], approx=False,
                  data_version=data_version),
        }
    cache = get_active_cache()
    assert cache is not None
    cache.add_col_stats(fingerprint, seed)
    cache.save_col_stats_cache(fingerprint, data_version)

    stats = table.compute_col_stats(
        col_stat_types=[ColStatType.COUNT, ColStatType.NA_FRACTION],
        approx=True,
    )

    for col_name, sentinel in sentinel_by_col.items():
        assert col_name in stats, f"missing {col_name} in {stats}"
        by_stat = stats[col_name]
        # COUNT came from the seeded cache entry — sentinel preserved.
        assert ColStatType.COUNT in by_stat
        assert by_stat[ColStatType.COUNT].value == sentinel
        # NA_FRACTION came from the tier-2 connector path — approx=True.
        assert ColStatType.NA_FRACTION in by_stat
        assert by_stat[ColStatType.NA_FRACTION].approx is True


def test_compute_col_stats_connector_failure_still_raises_with_remaining_missing(
        monkeypatch):
    # Tier-2 failure path: the connector raises mid-fallback. The cache-miss
    # error must still surface with an accurate Missing: list (the stat type
    # the caller requested), not a stale snapshot from before the failed
    # tier-2 attempt.
    from kumo_core.types import ColStatType

    monkeypatch.setattr(Table, 'compute_col_stats', _unwrap_compute_col_stats())
    table = _build_pandas_table()

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated tier-2 failure")

    monkeypatch.setattr(table.connector, "compute_col_stats", _raise)

    with pytest.raises(UnsupportedInDiskgraphError) as excinfo:
        table.compute_col_stats(
            col_stat_types=[ColStatType.NA_FRACTION],
            approx=True,
        )
    msg = str(excinfo.value)
    assert "Requested col stats missing from the DiskGraph-populated cache" \
        in msg
    assert "NA_FRACTION" in msg
