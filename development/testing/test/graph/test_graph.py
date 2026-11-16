# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging as sys_logging
import os
import os.path as osp
import shutil
from copy import deepcopy
from io import BytesIO

import pandas as pd
import pytest
from kumo_core.types import ColStatType
from kumoapi.encoder import Datetime, Index, Null, Numerical
from kumoapi.typing import Dtype, Stype

from io_utils import fsutil
from io_utils.json import load_json
from kumo.config import GraphConfig
from kumo.connector.csv_connector import CSVConnector
from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.column import Column, ForeignKey, PrimaryKey
from kumo.graph.graph import Graph
from kumo.graph.table import Table
from kumo.home import get_home_dir
from kumo.testing import withPackage
from kumo.testing.data import connect_financial_csv, financial_csv_dir
from kumo.validation.encoder_validation import (
    ColumnNotInTableError,
    TableNotInGraphError,
)
from kumo.validation.validation import ValidationException

TEST_GRAPH_NAME = 'kumo_test_graph'
TEST_PERSIST_DIR = 'test_persist_dir'
TEST_GRAPH_STATS_FILE = "graph_stats.json"


def _financial_test_graph(name: str = TEST_GRAPH_NAME) -> Graph:
    graph = Graph(name=name)
    connect_financial_csv(graph)
    return graph


def teardown_function():
    folder = osp.join(get_home_dir(), 'graphs', TEST_GRAPH_NAME)
    shutil.rmtree(folder, ignore_errors=True)

    persist_folder = osp.join(get_home_dir(), TEST_PERSIST_DIR)
    shutil.rmtree(persist_folder, ignore_errors=True)


def test_graph_ambiguous_access_resolves_to_column(tmp_path):
    # If a table name is the same as a data source, we may get ambiguous access
    # Example: Data source "anime", tables "anime" and "rating", column
    # "anime.rating" exists but access through __getitem__ is ambiguous
    pd.DataFrame({
        'rating': [1]
    }).to_csv(
        tmp_path / 'anime.csv',
        index=False,
    )
    pd.DataFrame({
        'id': [1]
    }).to_csv(
        tmp_path / 'rating.csv',
        index=False,
    )
    graph = Graph(TEST_GRAPH_NAME)
    graph.connect(CSVConnector('anime', str(tmp_path)))
    graph.add(name='anime', data_source='anime', cols=['rating'])
    graph.add(name='rating', data_source='anime')

    # Make sure ambiguous access resolves to a column (anime.rating)
    # is both a valid table as well as a valid column
    assert (graph['anime.rating'] == graph.column_from_fqn('anime.rating'))
    assert isinstance(graph['anime.rating'], Column)

    # Make sure unambiguous access to a table works
    assert graph['anime.anime'] is graph.table('anime')
    assert isinstance(graph['anime.anime'], Table)

    # Make sure the tables are still accessible on their own
    assert graph['rating'] is graph.table('rating')
    assert isinstance(graph['rating'], Table)
    assert graph['anime'] is graph.table('anime')
    assert isinstance(graph['anime'], Table)


def test_graph_with_fully_qualified_names():
    graph = Graph(TEST_GRAPH_NAME)

    graph.connect(CSVConnector('csv1', '.'))
    graph.connect(CSVConnector('csv2', '.'))

    graph.add('LOAN', 'csv1', pkey='LOAN_ID', fkeys=['ACCOUNT_ID'])
    graph.add('LOAN', 'csv2', pkey='LOAN_ID', fkeys=['ACCOUNT_ID'])
    assert len(graph) == 2

    assert graph.table_names == ['csv1.LOAN', 'csv2.LOAN']

    with pytest.raises(ValidationException):
        graph['LOAN']

    assert graph['csv1.LOAN'].table_name == 'LOAN'
    assert graph['csv1.LOAN'].data_source == 'csv1'

    assert graph['csv2.LOAN'].table_name == 'LOAN'
    assert graph['csv2.LOAN'].data_source == 'csv2'

    del graph['csv1', 'LOAN']
    assert len(graph) == 1

    assert graph['LOAN'].table_name == 'LOAN'
    assert graph['LOAN'].data_source == 'csv2'


def test_graph_add_from_config():
    root_dir = financial_csv_dir()
    path = osp.join(root_dir, 'metadata.json')

    graph = _financial_test_graph('FINANCIAL')
    graph.add_from_config(path, data_source='csv')
    graph.infer_metadata()
    graph.infer_encoders()
    graph.validate()

    assert str(graph) == ('Graph(\n'
                          '  name=FINANCIAL,\n'
                          '  data_sources=[csv],\n'
                          '  tables=[csv.ACCOUNT, csv.CARD, csv.CLIENT,'
                          ' csv.DISP, csv.DISTRICT, csv.LOAN, csv.ORDERS,'
                          ' csv.TRANS],\n'
                          ')')

    # Test `Graph` functions:
    assert len(graph['LOAN'].fkeys) == 1
    assert graph['LOAN'].fkeys[0].dst_table == 'csv.ACCOUNT'
    assert len(graph['TRANS'].fkeys) == 1
    assert graph['TRANS'].fkeys[0].dst_table == 'csv.ACCOUNT'
    assert len(graph['ORDERS'].fkeys) == 1
    assert graph['ORDERS'].fkeys[0].dst_table == 'csv.ACCOUNT'
    assert len(graph['DISP'].fkeys) == 2
    assert graph['DISP'].fkeys[0].dst_table == 'csv.CLIENT'
    assert graph['DISP'].fkeys[1].dst_table == 'csv.ACCOUNT'
    assert len(graph['ACCOUNT'].fkeys) == 1
    assert graph['ACCOUNT'].fkeys[0].dst_table == 'csv.DISTRICT'
    assert len(graph['CARD'].fkeys) == 1
    assert graph['CARD'].fkeys[0].dst_table == 'csv.DISP'


def test_graph_infer_metadata(create_graph):
    graph = create_graph(name='HOME_CREDIT_DEFAULT_RISK')
    graph.infer_metadata()

    assert len(graph) == 7
    assert len(graph['APPLICATION'].cols) == 121
    # AMT_ANNUITY is empty but not dropped
    assert len(graph['BUREAU'].cols) == 15
    assert len(graph['BUREAU_BALANCE'].cols) == 2
    assert len(graph['PREVIOUS_APPLICATION'].cols) == 35
    assert len(graph['POS_CASH_BALANCE'].cols) == 6
    assert len(graph['CREDIT_CARD_BALANCE'].cols) == 21
    assert len(graph['INSTALLMENTS_PAYMENTS'].cols) == 6


def test_graph_infer_metadata_include_exclude_cols(create_graph):
    graph = create_graph(name='HOME_CREDIT_DEFAULT_RISK')

    # Compared to the test_graph_infer_metadata, this test:
    # Exclude 2 columns in POS_CASH_BALANCE table, 1 column in BUREAU table.
    graph.infer_metadata(
        exclude_cols={
            'BUREAU': {'CREDIT_ACTIVE'},
            'CREDIT_CARD_BALANCE': {'AMT_BALANCE', 'MONTHS_BALANCE'}
        })

    assert len(graph) == 7
    assert len(graph['APPLICATION'].cols) == 121
    # AMT_ANNUITY is empty but not dropped
    assert len(graph['BUREAU'].cols) == 14
    assert len(graph['BUREAU_BALANCE'].cols) == 2
    assert len(graph['PREVIOUS_APPLICATION'].cols) == 35
    assert len(graph['POS_CASH_BALANCE'].cols) == 6
    assert len(graph['CREDIT_CARD_BALANCE'].cols) == 19
    assert len(graph['INSTALLMENTS_PAYMENTS'].cols) == 6


def test_graph_uninferred_links(caplog):
    graph = _financial_test_graph('FINANCIAL')

    graph.add('LOAN', 'csv', pkey='LOAN_ID', fkeys=['TRANS_ID'],
              cols=['ACCOUNT_ID'])
    graph.add('ACCOUNT', 'csv', pkey='ACCOUNT_ID')

    # table has uninferred fkey:
    with caplog.at_level(sys_logging.WARNING):
        graph.infer_links()
    assert ("Cannot infer destination table for") in caplog.text
    assert graph['LOAN']['TRANS_ID'].dst_table is None


def test_graph_infer_encoders_from_csv(create_graph):
    graph = create_graph(name='FINANCIAL')
    graph.infer_metadata()
    graph.infer_encoders()

    assert isinstance(graph['LOAN', 'DATE'].encoder, Datetime)
    assert isinstance(graph['LOAN', 'AMOUNT'].encoder, Numerical)
    assert isinstance(graph['LOAN', 'DURATION'].encoder, Index)
    assert isinstance(graph['LOAN', 'PAYMENTS'].encoder, Numerical)
    assert isinstance(graph['LOAN', 'STATUS'].encoder, Index)


def test_graph_validate_ignore_encoders(create_graph):
    graph = create_graph(name='FINANCIAL')
    graph.infer_metadata()

    # Should raise error if encoders are missing
    with pytest.raises(ValueError, match="has no encoder"):
        graph.validate(raise_on_error=True)

    # Should pass if ignore encoders
    graph.validate(raise_on_error=True, ignore_encoders=True)

    # Should pass if encoders are present
    graph.infer_encoders()
    graph.validate(raise_on_error=True)


def test_graph_indexing():
    graph = Graph(name='FINANCIAL')
    connector = CSVConnector('csv', ".")
    graph.connect(connector)
    graph.add('TABLE', 'csv', 'PKEY', fkeys=['FKEY'], cols=['COL1', 'COL2'])

    assert isinstance(graph['TABLE'], Table)
    assert isinstance(graph['csv.TABLE'], Table)
    assert isinstance(graph[('csv', 'TABLE')], Table)
    assert isinstance(graph[('TABLE', )], Table)

    assert isinstance(graph['TABLE', 'PKEY'], PrimaryKey)
    assert isinstance(graph['TABLE.PKEY'], PrimaryKey)
    assert isinstance(graph['csv.TABLE', 'PKEY'], PrimaryKey)
    assert isinstance(graph[('csv', 'TABLE'), 'PKEY'], PrimaryKey)
    assert isinstance(graph['csv.TABLE.PKEY'], PrimaryKey)

    assert isinstance(graph['TABLE', 'FKEY'], ForeignKey)
    assert isinstance(graph['TABLE.FKEY'], ForeignKey)
    assert isinstance(graph['csv.TABLE', 'FKEY'], ForeignKey)
    assert isinstance(graph[('csv', 'TABLE'), 'FKEY'], ForeignKey)
    assert isinstance(graph['csv.TABLE.FKEY'], ForeignKey)

    assert isinstance(graph['TABLE', 'COL1'], Column)
    assert isinstance(graph['TABLE.COL1'], Column)
    assert isinstance(graph['csv.TABLE', 'COL1'], Column)
    assert isinstance(graph[('csv', 'TABLE'), 'COL1'], Column)
    assert isinstance(graph['csv.TABLE.COL1'], Column)

    table = graph['TABLE', ['FKEY', 'COL2']]
    assert isinstance(table, Table)
    assert table.has_pkey()
    assert table.pkey.name == 'PKEY'
    assert table.fkey_names == ['FKEY']
    assert table.col_names == ['COL2']

    table = graph['TABLE', ['COL1']]
    assert isinstance(table, Table)
    assert table.has_pkey()
    assert table.pkey.name == 'PKEY'
    assert table.num_fkeys == 0
    assert table.col_names == ['COL1']


def test_graph_link_with_multiple_data_sources(create_graph):
    graph = create_graph(name='FINANCIAL')
    graph.connect(ParquetConnector('parquet', '/test/path'))
    graph.add(name='ACCOUNT', data_source='parquet')
    graph.link('ACCOUNT', 'DISTRICT_ID', 'DISTRICT', 'DISTRICT_ID',
               src_data_source='parquet')
    table = graph.table('ACCOUNT', 'parquet')

    assert len(table.fkeys) == 1
    fkey = table.fkeys[0]
    assert fkey.dst_table == 'csv.DISTRICT'
    assert fkey.dst_key == 'DISTRICT_ID'


def test_graph_link_with_data_source_conflict(create_graph):
    graph = create_graph(name='FINANCIAL')
    graph.connect(ParquetConnector('parquet', '/test/path'))
    graph.add(name='ACCOUNT', data_source='parquet')
    with pytest.raises(ValidationException):
        graph.link('ACCOUNT', 'DISTRICT_ID', 'DISTRICT', 'DISTRICT_ID')


def test_graph_link_with_incorrect_table(create_graph):
    graph = create_graph(name='FINANCIAL')
    graph.connect(ParquetConnector('parquet', '/test/path'))
    graph.add(name='ACCOUNT', data_source='parquet')
    with pytest.raises(KeyError):
        graph.link('INCORRECT_TABLE', 'DISTRICT_ID', 'DISTRICT', 'DISTRICT_ID')


def test_graph_link_with_incorrect_data_source(create_graph):
    graph = create_graph(name='FINANCIAL')
    graph.connect(ParquetConnector('parquet', '/test/path'))
    graph.add(name='ACCOUNT', data_source='parquet')
    with pytest.raises(ValidationException,
                       match="incorrect_data_source does not exist"):
        graph.link('ACCOUNT', 'DISTRICT_ID', 'DISTRICT', 'DISTRICT_ID',
                   src_data_source='incorrect_data_source')


@withPackage('graphviz', executables=('dot', ))
def test_graph_visualize_via_graphviz(create_graph):
    graph = create_graph(name='FINANCIAL', infer=True)

    filepath = osp.join('.', 'metagraph.pdf')

    assert not osp.exists(filepath)
    graph.visualize(filepath)
    assert osp.exists(filepath)
    os.remove(filepath)

    assert not osp.exists(filepath)
    graph.visualize(filepath)
    assert osp.exists(filepath)
    os.remove(filepath)

    buf = BytesIO()
    graph.visualize(buf)
    assert len(buf.getvalue()) > 0


def test_fully_qualified_table_names():
    name = 'name'
    data_source = 'csv'

    # Test graph
    graph = Graph()

    # Add basic table
    graph._data_source_dict[data_source] = 'c"onnector_impl'
    fqn = graph.fully_qualified_table_name(name, data_source)
    assert fqn == f"{data_source}.{name}"
    graph._tables[fqn] = Table(table_name=name, data_source=data_source)

    # Basic tests
    expected_output = f"{data_source}.{name}"

    # * name={name}, data_source={data_source}
    assert graph.fully_qualified_table_name(
        name=name, data_source=data_source) == expected_output

    # * name=({data_source}, {name}), data_source=None
    assert graph.fully_qualified_table_name(name=(data_source, name),
                                            data_source=None) == expected_output

    # * name={data_source}.{name}, data_source=None
    assert graph.fully_qualified_table_name(name=f'{data_source}.{name}',
                                            data_source=None) == expected_output

    # * name={name}, data_source=None (unique name)
    assert graph.fully_qualified_table_name(name=name,
                                            data_source=None) == expected_output


def test_graph_has_table():
    graph = Graph()
    graph.connect(CSVConnector('csv', '.'))
    graph.add('LOAN', 'csv')

    assert len(graph) == 1
    assert graph.has_table('LOAN')
    assert not graph.has_table('UNKNOWN')


def test_infer_col_groups_respects_integer_width_direction():
    graph = Graph(name=TEST_GRAPH_NAME)
    connect_financial_csv(graph)
    graph.add("ACCOUNT", "csv", pkey=PrimaryKey(name="ACCOUNT_ID",
                                                dtype=Dtype.int64))
    graph.add(
        "LOAN",
        "csv",
        pkey=PrimaryKey(name="LOAN_ID", dtype=Dtype.int64),
        cols=[Column(name="ACCOUNT_ID", dtype=Dtype.int32)],
    )

    groups = graph.infer_col_groups()
    assert len(groups) == 1
    assert (groups[0].fkeys[0].table == "csv.LOAN"
            and groups[0].fkeys[0].col == "ACCOUNT_ID")

    graph.table("ACCOUNT").pkey.dtype = Dtype.int32
    graph.table("LOAN").cols[0].dtype = Dtype.int64

    assert graph.infer_col_groups() == []


def test_generate_graph_encoders(generate_graph):
    # Generate a medium-sized graph with medium edge density, and feature
    # columns included with 0 rows as N/A
    graph = generate_graph(f'{TEST_GRAPH_NAME}_0', 'MEDIUM', 'MEDIUM', True, 0)
    graph.infer_metadata()
    graph.infer_encoders()
    # Table 0 has column types str, date, time
    assert isinstance(graph['t0', '0'].encoder, Index)
    assert isinstance(graph['t0', '1'].encoder, Datetime)
    assert isinstance(graph['t0', '2'].encoder, Datetime)

    # Table 1 has column types int, bool, float
    assert isinstance(graph['t1', '0'].encoder, Numerical)
    assert isinstance(graph['t1', '1'].encoder, Index)
    assert isinstance(graph['t1', '2'].encoder, Numerical)


def test_graph_col_stats_fallback_seeds_encoder_requirements():
    users = pd.DataFrame({
        'USER_ID': [1, 2, 3],
        'AGE': [20, None, 40],
    })
    graph = Graph(name=f'{TEST_GRAPH_NAME}_stats')
    graph.add('USERS', data_source=users, pkey='USER_ID', cols=['AGE'])

    graph.infer_metadata()
    age = graph['USERS', 'AGE']
    stats = graph['USERS'].compute_col_stats(
        [age],
        [ColStatType.COUNT, ColStatType.NUM_NA],
    )['AGE']

    assert stats[ColStatType.COUNT].value == 3
    assert stats[ColStatType.NUM_NA].value == 1


def test_generate_graph_encoders_low_na(generate_graph):
    # Same graph size and density but with 0.2 fraction of feature entries as
    # N/A
    graph = generate_graph(f'{TEST_GRAPH_NAME}_1', 'MEDIUM', 'MEDIUM', True,
                           0.2)
    graph.infer_metadata()
    graph.infer_encoders()

    # The bool column with N/A values encodes as Index(3)
    assert isinstance(graph['t1', '1'].encoder, Index)


def test_generate_graph_encoders_high_na(generate_graph):
    # Small, low-density graph with high N/A rate so expect null encoding
    graph = generate_graph(f'{TEST_GRAPH_NAME}_2', 'SMALL', 'LOW', True, 0.8)
    graph.infer_metadata()
    graph.infer_encoders()

    assert isinstance(graph['t1', '1'].encoder, Index)


def test_graph_copy_encoders(generate_graph):
    graph = generate_graph(f'{TEST_GRAPH_NAME}_0', 'MEDIUM', 'MEDIUM', True, 0)
    graph.infer_metadata()
    graph.infer_encoders()

    graph2 = deepcopy(graph)
    graph2['t0', '0'].encoder = Null()
    graph2['t0', '1'].encoder = Null()
    graph2['t0', '2'].encoder = Null()
    graph2['t1', '0'].encoder = Null()
    graph2['t1', '1'].encoder = Null()
    graph2['t1', '2'].encoder = Null()

    graph2.copy_encoders(graph)

    # Table 0 has column types str, date, time
    assert isinstance(graph2['t0', '0'].encoder, Index)
    assert isinstance(graph2['t0', '1'].encoder, Datetime)
    assert isinstance(graph2['t0', '2'].encoder, Datetime)

    # Table 1 has column types int, bool, float
    assert isinstance(graph2['t1', '0'].encoder, Numerical)
    assert isinstance(graph2['t1', '1'].encoder, Index)
    assert isinstance(graph2['t1', '2'].encoder, Numerical)


def test_graph_pandas_view(create_graph):
    graph = create_graph(name='FINANCIAL')
    graph.infer_metadata()

    assert graph['LOAN'].pandas().shape == (682, 6)
    assert graph['LOAN'].pandas[:10].shape == (10, 6)
    assert graph['LOAN'].pandas[10:20].shape == (10, 6)
    assert graph['LOAN'].pandas.sample(n=10).shape == (10, 6)

    assert graph['LOAN', 'STATUS'].pandas().shape == (682, )
    assert graph['LOAN', 'STATUS'].pandas[:10].shape == (10, )
    assert graph['LOAN', 'STATUS'].pandas[10:20].shape == (10, )
    assert graph['LOAN', 'STATUS'].pandas.sample(n=10).shape == (10, )

    assert graph['LOAN', ..., :10].shape == (10, 6)
    assert graph['LOAN', ..., 10:20].shape == (10, 6)
    assert graph['LOAN', 'STATUS', :10].shape == (10, )
    assert graph['LOAN', 'STATUS', 10:20].shape == (10, )
    assert graph['LOAN', ['AMOUNT', 'STATUS'], :10].shape == (10, 2)
    assert graph['LOAN', ['AMOUNT', 'STATUS'], 10:20].shape == (10, 2)


def test_graph_resolution_order():
    graph = Graph()
    graph.connect(CSVConnector('csv', '/tmp'))
    graph.add('csv', cols=['csv'])

    assert isinstance(graph['csv'], Table)
    assert isinstance(graph['csv.csv'], Column)
    assert isinstance(graph['csv.csv.csv'], Column)


def test_table_source_name():
    graph = _financial_test_graph()

    graph.add('loan_table', pkey='LOAN_ID', source_name='LOAN')
    graph.add('account_table', pkey='ACCOUNT_ID', source_name='ACCOUNT')
    graph['loan_table'].link('ACCOUNT_ID', dst_table='account_table')

    assert len(graph['loan_table'].pandas()) == 682
    assert len(graph['account_table'].pandas()) == 4500


def test_graph_remove_unused_cols(generate_graph):
    graph = generate_graph(TEST_GRAPH_NAME, size='SMALL', feat_na_frac=0.9)
    # TODO Some dtypes do not support N/A within `generate_graph` yet.
    graph.infer_metadata()
    graph.infer_encoders()
    assert len(graph['t0'].cols) == 3
    assert len(graph['t1'].cols) == 3
    assert len(graph['t2'].cols) == 1

    graph['t0', '0'].encoder = Null()
    graph['t1', '0'].encoder = Null()
    graph['t1', '2'].encoder = Null()
    graph['t2', '0'].encoder = Null()

    graph.remove_unused_cols()
    assert len(graph['t0'].cols) == 2
    assert len(graph['t1'].cols) == 1
    assert len(graph['t2'].cols) == 0


def test_graph_pandas_connector(tmp_path):
    data = [{
        "name": "ted",
        "age": 34,
        "value": 0.10
    }, {
        "name": "barney",
        "age": 35,
        "value": 1.2
    }]
    df = pd.DataFrame.from_dict(data, )

    graph = Graph(name=TEST_GRAPH_NAME)
    assert graph.name == TEST_GRAPH_NAME

    # Save/load round-trip through an explicit temp dir (no dependence on the
    # process-global home dir).
    save_dir = str(tmp_path)
    graph.add('pds_data', data_source=df, pkey="name", cols=['age', 'value'])
    graph.save(destination=save_dir)

    config_path = osp.join(save_dir, 'graphs', TEST_GRAPH_NAME, 'config.json')
    new_graph = Graph.from_config(load_json(config_path, cls=GraphConfig))
    table = new_graph['pds_data']

    df.set_index("name", inplace=True)
    table_df = table.pandas()
    assert len(table_df) == 2
    assert list(table_df['age'].values) == [34, 35]
    assert list(table_df['value'].values) == [0.1, 1.2]


def test_graph_save_destination_does_not_use_global_home_dir(tmp_path):
    # ``save(destination=...)`` must write under the given dir and never
    # consult the process-global home dir, so the call is safe to run on a
    # shared thread pool without ``temp_home_dir``.
    graph = _financial_test_graph()
    dest = str(tmp_path / 'explicit_dest')

    graph.save(destination=dest)

    assert fsutil.exists(
        osp.join(dest, 'graphs', TEST_GRAPH_NAME, 'config.json'))
    # Nothing leaked to the global home dir.
    assert not fsutil.exists(osp.join(get_home_dir(), 'graphs',
                                      TEST_GRAPH_NAME))


@pytest.mark.parametrize("account_dtype,loan_dtype,expect_success", [
    pytest.param(Dtype.int64, Dtype.int32, True,
                 id="safe_int32_fkey_to_int64_pkey"),
    pytest.param(Dtype.int, Dtype.int32, True,
                 id="safe_int32_fkey_to_int_pkey"),
    pytest.param(Dtype.int32, Dtype.int64, False,
                 id="reject_int64_fkey_to_int32_pkey"),
    pytest.param(Dtype.int32, Dtype.int, False,
                 id="reject_int_fkey_to_int32_pkey"),
    (Dtype.string, Dtype.int, False),
    (Dtype.int, Dtype.float, False),
    (Dtype.date, Dtype.time, False),
    (Dtype.string, Dtype.time, False),
    (Dtype.string, Dtype.bool, False),
    (Dtype.int, Dtype.bool, False),
    (Dtype.int, Dtype.time, False),
])
def test_graph_fkey_dtype_validation(account_dtype: Dtype, loan_dtype: Dtype,
                                     expect_success: True):

    graph = _financial_test_graph()

    graph.add(
        'ACCOUNT',
        'csv',
        pkey=PrimaryKey(
            'ACCOUNT_ID',
            dtype=account_dtype,
        ),
    )
    graph.add(
        'LOAN',
        'csv',
        pkey='LOAN_ID',
        fkeys=[
            ForeignKey(
                'ACCOUNT_ID',
                dtype=loan_dtype,
                dst_table='csv.ACCOUNT',
            )
        ],
    )

    # Float/double keys are rejected outright; no silent int<->float link.
    has_float_key = account_dtype.is_float() or loan_dtype.is_float()

    if expect_success and not has_float_key:
        assert graph.validate()
    elif has_float_key:
        with pytest.raises(
                Exception, match="float/double key columns "
                "are not supported"):
            graph.validate()
    else:
        with pytest.raises(Exception, match="Links are only allowed between "
                           "matching"):
            graph.validate()


def test_graph_float_fkey_is_rejected():
    # A float/double foreign key is dropped at ingest; validation must reject it.
    graph = _financial_test_graph()
    graph.add(
        'ACCOUNT',
        'csv',
        pkey=PrimaryKey('ACCOUNT_ID', dtype=Dtype.int),
    )
    graph.add(
        'LOAN',
        'csv',
        pkey='LOAN_ID',
        fkeys=[
            ForeignKey(
                'ACCOUNT_ID',
                dtype=Dtype.float,
                dst_table='csv.ACCOUNT',
            )
        ],
    )

    with pytest.raises(ValueError, match="float/double key columns are not "
                       "supported"):
        graph.validate()

    # The error must name the column and tell the user to set its dtype to int.
    with pytest.raises(ValueError, match="set the dtype of 'ACCOUNT_ID' to "
                       "'int'"):
        graph.validate()

    # Soft validation still surfaces the problem without raising.
    assert not graph.validate(raise_on_error=False)


@pytest.mark.parametrize("float_dtype",
                         [Dtype.float, Dtype.float32, Dtype.float64])
def test_graph_float_pkey_is_rejected(float_dtype: Dtype):
    # A float/double primary key yields zero nodes at ingest; reject it.
    graph = _financial_test_graph()
    graph.add(
        'ACCOUNT',
        'csv',
        pkey=PrimaryKey('ACCOUNT_ID', dtype=float_dtype),
    )

    with pytest.raises(ValueError, match="float/double key columns are not "
                       "supported"):
        graph.validate()

    with pytest.raises(ValueError, match="set the dtype of 'ACCOUNT_ID' to "
                       "'int'"):
        graph.validate()


def test_graph_int_and_string_keys_pass():
    # Integer and string keys are valid and must not trip the float-key guard.
    graph = _financial_test_graph()
    graph.add(
        'ACCOUNT',
        'csv',
        pkey=PrimaryKey('ACCOUNT_ID', dtype=Dtype.int64),
    )
    graph.add(
        'LOAN',
        'csv',
        pkey=PrimaryKey('LOAN_ID', dtype=Dtype.string),
        fkeys=[
            ForeignKey(
                'ACCOUNT_ID',
                dtype=Dtype.int64,
                dst_table='csv.ACCOUNT',
            )
        ],
    )

    assert graph.validate()


def test_graph_non_key_float_column_passes():
    # A non-key float column (a feature) must not trip the float-key guard.
    graph = _financial_test_graph()
    graph.add(
        'ACCOUNT',
        'csv',
        pkey=PrimaryKey('ACCOUNT_ID', dtype=Dtype.int64),
        cols=[
            Column(name='BALANCE', dtype=Dtype.float64, stype=Stype.numerical),
        ],
    )

    assert graph.validate(ignore_encoders=True)


def test_graph_with_no_tables_is_rejected():
    graph = Graph(name=TEST_GRAPH_NAME)

    with pytest.raises(ValueError, match="must contain at least one table"):
        graph.validate()

    # Soft validation surfaces the problem without raising.
    with pytest.warns(UserWarning, match="must contain at least one table"):
        assert not graph.validate(raise_on_error=False)


def test_verify_edge_type(create_graph):
    graph = create_graph(name='FINANCIAL')

    assert graph._verify_edge_type('DISP_ID') == ('csv.CARD', 'DISP_ID',
                                                  'csv.DISP')
    assert graph._verify_edge_type(
        ('csv.LOAN', 'csv.ACCOUNT')) == ('csv.LOAN', 'ACCOUNT_ID',
                                         'csv.ACCOUNT')
    assert graph._verify_edge_type(
        ('csv.ACCOUNT', 'csv.LOAN')) == ('csv.ACCOUNT', 'rev_ACCOUNT_ID',
                                         'csv.LOAN')
    assert graph._verify_edge_type(
        ('csv.TRANS', 'ACCOUNT_ID',
         'csv.ACCOUNT')) == ('csv.TRANS', 'ACCOUNT_ID', 'csv.ACCOUNT')

    with pytest.raises(ValueError, match=r"Ambiguous .*"):
        graph._verify_edge_type(('ACCOUNT_ID'))

    with pytest.raises(ValueError, match=r"Unknown .*"):
        graph._verify_edge_type(("START", "RELATION", "END"))


def test_infer_col_groups():

    graph = _financial_test_graph()

    graph.add(
        'ACCOUNT', 'csv', pkey=PrimaryKey(name='ACCOUNT_ID', dtype=Dtype.int64),
        cols=[
            Column(name='ACC_COL1', dtype=Dtype.int16),
            Column(name='ACC_COL2', dtype=Dtype.int16)
        ])
    # Set pkey(LOAN) = 'iD'
    graph.add(
        'LOAN',
        'csv',
        pkey=PrimaryKey('iD', dtype=Dtype.int16),
        # candidate fkey with dtype=float (valid)
        cols=[
            Column(name='ACCOUNT_ID', dtype=Dtype.int64),
            Column(name='LOAN_COL2', dtype=Dtype.int16)
        ])

    groups = graph.infer_col_groups()
    assert groups[0].fkeys[0].table == 'csv.LOAN' and groups[0].fkeys[
        0].col == 'ACCOUNT_ID'

    # Change the dtype of ACCOUNT_ID to bool (invalid) in LOAN table.
    graph.table('LOAN').cols[0].dtype = Dtype.bool

    # We should get zero link groups
    groups = graph.infer_col_groups()
    assert groups == []

    # Revert it back to valid dtype
    graph.table('LOAN').cols[0].dtype = Dtype.int64

    # Add a table TRANS of different connector.
    # Also tests a case where it links to pkey(LOAN)='iD'
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', 'parquet_data', 'TEST')
    parq_connector = ParquetConnector('parquet', root_dir=root_dir)
    graph.connect(parq_connector)

    graph.add(
        'TRANS',
        'parquet',
        pkey=PrimaryKey('TRANS_ID', dtype=Dtype.int64),
        # fkey LOAN_ID should link to pkey = 'iD'(Case insensitve match)
        cols=[
            Column(name='LOAN_ID', dtype=Dtype.int16),
            Column(name='TRANS_COL2', dtype=Dtype.int16)
        ])

    groups = graph.infer_col_groups()
    assert groups[1].fkeys[0].table == 'parquet.TRANS' and groups[1].fkeys[
        0].col == 'LOAN_ID'

    # Remove all the pkeys and expect an empty group.
    graph.table('ACCOUNT').remove_pkey()
    graph.table('LOAN').remove_pkey()
    graph.table('TRANS').remove_pkey()

    groups = graph.infer_col_groups()
    assert groups == []

    # Add a candidate fkey name as fkey to LOAN table and expect it
    # to be in the group.
    graph.table('ACCOUNT').set_pkey('ID')
    graph.table('LOAN').add_fkey('ACCOUNT_ID')
    groups = graph.infer_col_groups()
    assert (groups[0].fkeys[0].table == 'csv.LOAN'
            and groups[0].fkeys[0].col == 'ACCOUNT_ID')

    # Rename table ACCOUNT to ACCOUNTS and expect its pkey as a fkey in
    # LOAN table to be in the group.
    # Tests inflect.singular_noun()
    graph.remove('ACCOUNT')
    graph.add('ACCOUNTS', 'csv', pkey=PrimaryKey(name='ID', dtype=Dtype.int64))
    groups = graph.infer_col_groups()
    assert (groups[0].fkeys[0].table == 'csv.LOAN'
            and groups[0].fkeys[0].col == 'ACCOUNT_ID')


def test_encoder_override_with_incorrect_table_name(create_graph):
    graph: Graph = create_graph(name='FINANCIAL')
    graph.infer_links().infer_metadata().infer_encoders()

    with pytest.raises(TableNotInGraphError) as e:
        graph.apply_encoder_overrides({'csv.ACCOUNTSSSSSS.col': Index})
    assert 'ACCOUNTSSSSSS' in e.value.title_and_message()


def test_encoder_override_with_incorrect_column_name(create_graph):
    graph: Graph = create_graph(name='FINANCIAL')
    graph.infer_links().infer_metadata().infer_encoders()

    with pytest.raises(ColumnNotInTableError) as e:
        graph.apply_encoder_overrides({'csv.ACCOUNT.col_298': Index})
    assert 'col_298' in e.value.title_and_message()
