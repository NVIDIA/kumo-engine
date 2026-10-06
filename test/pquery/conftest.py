# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import os.path as osp
import pathlib
import uuid
from typing import Callable

import pandas as pd
import pytest
from kumoapi.typing import Dtype, Stype

from kumo.connector.csv_connector import CSVConnector
from kumo.connector.parquet_connector import ParquetConnector
from kumo.connector.snowflake_connector import SnowflakeConnector
from kumo.graph.graph import Graph
from kumo.testing.snowflake import withSnowflakeCredentials


@pytest.fixture(scope='module')
@withSnowflakeCredentials
def snowflake_connector() -> SnowflakeConnector:
    connector = SnowflakeConnector('snowflake',
                                   account='xva19026',
                                   warehouse='WH_XS',
                                   database='KUMO',
                                   schema_name='FINANCIAL')
    return connector


# Node Prediction Graphs ######################################################


@pytest.fixture()
def create_invalid_static_node_pred_graph(tmp_path: pathlib.Path) -> Callable:
    return functools.partial(static_node_pred_invalid_test_graph, tmp_path)


TEST_GRAPH_NAME = 'kumo_test_graph'


def static_node_pred_invalid_test_graph(
    tmp_path: pathlib.Path,
    mode: str = 'duplicate',
    name: str = TEST_GRAPH_NAME,
    user_has_pkey: bool = True,
    target_key_has_nones: bool = False,
    target_key_has_empty_strings: bool = False,
) -> Graph:
    r"""This graph should raise an error when used to test static node
    prediction queries.

    Args:
        mode (str): 'duplicate' or 'time'. If 'duplicate' is selected, the
            target table will have duplicate rows. If 'time' is selected, the
            target table will have a time column and duplicate rows.
        user_has_pkey (bool): If `True`, `USERS` table will contain the pkey
            and `TARGET` fkey. Otherwise, vice versa.
        target_key_has_nones (bool): If `True`, `USERS_TARGET.USER_ID` column
            will contain `N/A` values.
        target_key_has_empty_strings (bool): If `True`, `USERS_TARGET.STATUS`
            column will contain empty string values.
    """
    assert mode in ['duplicate', 'time']

    user_ids = [0, 1, 2, pd.NA] if target_key_has_nones else [0, 1, 2, 3]
    entity_df = pd.DataFrame({
        'USER_ID': [0, 1, 2, 3],
        'DATE': ['1990-01-01', '1990-01-02', '1990-01-03', '1990-01-04'],
        'PAYMENTS': [10.0, 20.0, 40.0, 80.0],
    })

    status_values = ['A', 'B', pd.NA, 'B']
    if target_key_has_empty_strings:
        status_values = ['A', 'B', '', 'B']
    target_df = pd.DataFrame({
        'USER_ID':
        user_ids * 2,
        'STATUS':
        status_values * 2,
        'DATE': ['1990-01-01', '1990-01-02', '1990-01-03', '1990-01-04'] * 2,
    })
    if mode == 'duplicate':
        target_df = target_df.drop(columns=['DATE'])

    entity_df = entity_df.set_index('USER_ID')
    target_df = target_df.set_index('USER_ID')

    graph = Graph(name=name)
    if user_has_pkey:
        add_df(graph, entity_df, 'USERS', tmp_path, pkey='USER_ID',
               cols=['PAYMENTS'], time_col='DATE')
    else:
        add_df(graph, entity_df, 'USERS', tmp_path, fkeys=['USER_ID'],
               cols=['PAYMENTS'], time_col='DATE')
    if mode == 'duplicate':
        if user_has_pkey:
            add_df(graph, target_df, 'USERS_TARGET', tmp_path,
                   fkeys=['USER_ID'], cols=['STATUS'])
        else:
            add_df(graph, target_df, 'USERS_TARGET', tmp_path, pkey='USER_ID',
                   cols=['STATUS'])
    if mode == 'time':
        if user_has_pkey:
            add_df(graph, target_df, 'USERS_TARGET', tmp_path,
                   fkeys=['USER_ID'], cols=['STATUS'], time_col='DATE')
        else:
            add_df(graph, target_df, 'USERS_TARGET', tmp_path, pkey='USER_ID',
                   cols=['STATUS'], time_col='DATE')

    return graph.infer_links().infer_metadata().infer_encoders()


@pytest.fixture(scope='function')
def candidate_test_graph(tempdir) -> Graph:
    entity_df = pd.DataFrame({
        'USER_ID': range(2),
        'AGE': [17, 25],
        'DUMMY_ID': range(2),
        'DATE': pd.to_datetime(['1/1/2018', '1/10/2018']),
    })
    new_entity_df = pd.DataFrame({
        'USER_ID': range(2, 4),
        'AGE': [18, 26],
        'DUMMY_ID': range(2)
    })
    new2_entity_df = pd.DataFrame({
        'USER_ID': [0, 1] * 2,
        'FUTURE_SPEND': [2.0, 5.0] * 2,
    })
    na_entity_df = pd.DataFrame({
        'USER_ID': [0, 1] * 2,
        'FUTURE_SPEND': [None, 3.0] * 2,
    })
    str_agg_entity_df = pd.DataFrame({
        'USER_ID': [0, 1] * 2,
        'FUTURE_SPEND': ['a', 'b'] * 2,
    })
    wrong_pkey_entity_df = pd.DataFrame({
        'USER_ID': [2, 1] * 2,
        'FUTURE_SPEND': [2.0, 3.0] * 2,
    })
    binary_entity_df = pd.DataFrame({
        'USER_ID': [0, 1] * 2,
        'FUTURE_SPEND': [True, False] * 2,
    })

    dummy_df = pd.DataFrame({'DUMMY_ID': range(2), 'IQ': range(10, 12)})
    dates = pd.Series(pd.date_range(start='1/1/2018', end='1/30/2018'))
    dates = pd.concat([dates, dates], axis=0, ignore_index=True)
    types = ['A', 'B', 'C'] * 20
    target_df = pd.DataFrame({
        'TRANS_ID': range(len(dates)),
        'USER_ID': [0] * 30 + [1] * 30,
        'AMOUNT': [1.0] * len(dates),
        'DATE': dates,
        'TYPE': types
    })
    # Save tables to csv files:
    tables_dir = 'tables-' + uuid.uuid4().hex[:16]
    tables_path = osp.join(tempdir.name, tables_dir)
    os.makedirs(tables_path)
    tables_names = [
        'USERS',
        'TRANS',
        'DUMMIES',
        'NEW_USERS',
        'FUTURE_AGG',
        'FUTURE_AGG_NA',
        'FUTURE_AGG_STR',
        'FUTURE_AGG_WRONG_PKEY',
        'FUTURE_AGG_BINARY',
    ]
    tables = [
        entity_df,
        target_df,
        dummy_df,
        new_entity_df,
        new2_entity_df,
        na_entity_df,
        str_agg_entity_df,
        wrong_pkey_entity_df,
        binary_entity_df,
    ]
    for table_name, table in zip(tables_names, tables):
        table_path = os.path.join(tables_path, table_name + '.csv')
        table.to_csv(table_path, index=False)

    graph = Graph('test')
    graph.connect(CSVConnector('csv',
                               root_dir=tables_path,
                               na_values='?'))
    graph.add('USERS', pkey='USER_ID')
    graph.add('DUMMIES', pkey='DUMMY_ID')
    graph.add('TRANS', pkey='TRANS_ID')
    graph.link('TRANS', 'USER_ID', 'USERS', 'USER_ID')
    graph.link('USERS', 'DUMMY_ID', 'DUMMIES', 'DUMMY_ID')
    graph.infer_metadata()
    graph.infer_encoders()
    graph['TRANS'].set_time_col('DATE')
    graph['USERS'].set_time_col('DATE')
    return graph


# Link Prediction Graphs ######################################################


@pytest.fixture()
def create_lp_graph_invalid_targets() -> Callable:
    return lp_test_graph_invalid_targets


def lp_test_graph_invalid_targets(
    root_dir: str = None,
    num_entities: int = 2,
    set_time_col: bool = True,
) -> Graph:
    """Graph used to test link prediction queries. Graph has
    tables `USERS`, `PRODUCTS` and `TRANS`. `TRANS` has foreign
    keys to `USERS` and `PRODUCTS`. Further, `TRANS` has a few invalid foreign
    keys to `PRODUCTS`, i.e. `NaN`s and foreign keys without a matching
    primary key.

    Args:
        root_dir: Directory where csv files of tables will be saved.
        num_entities: Number of users in `USERS` table.
        set_time_col: If `True` will set time_col in `TRANS` table.
    """
    # TODO: Verify that we are able to drop these tables in static link pred
    tables_dir = 'tables-' + uuid.uuid4().hex[:16]
    tables_path = os.path.join(root_dir, tables_dir)
    os.makedirs(tables_path)

    entity1_path = os.path.join(tables_path, 'USERS.csv')
    entity2_path = os.path.join(tables_path, 'PRODUCTS.csv')
    trans_path = os.path.join(tables_path, 'TRANS.csv')
    entity2_extra_connect_path = os.path.join(tables_path,
                                              'PRODUCT_BRANDS.csv')

    # Create user entity table
    ages = list(range(17, 17 + (num_entities) * 8, 8))
    user_ids = list(range(num_entities))
    pd.DataFrame({
        'USER_ID': user_ids,
        'AGE': ages
    }).to_csv(entity1_path, index=False)

    # Create product entity table
    pd.DataFrame({
        'PRODUCT_ID': range(2),
        'BRAND': ['PATAGONIA', 'KUMO']
    }).to_csv(entity2_path, index=False)

    # Create product feature table
    pd.DataFrame({
        'BRAND': ['PATAGONIA', 'KUMO', 'PATAGONIA', 'PATAGONIA', 'KUMO'],
        'BRAND_VALUE': [5, 5, 10, 20, 5],
        'DATE':
        ['2018-01-01', '2018-01-01', '2018-01-10', '2018-01-20', '2018-01-15']
    }).to_csv(entity2_extra_connect_path, index=False)

    # Create Transaction table with some invalid product_ids:
    dates = pd.Series(pd.date_range(start='1/1/2018', end='1/30/2018'))
    dates = pd.concat([dates] * num_entities, axis=0, ignore_index=True)
    user_ids = []
    product_ids = []
    amounts = []
    for i in range(num_entities):
        user_ids += [i] * 30
        product_ids += [0] * 10 + [1] * 10 + [None] * 5 + [100] * 5
        amounts += [1.0, 2.0, 3.0] * 10

    pd.DataFrame({
        'TRANS_ID': range(len(dates)),
        'USER_ID': user_ids,
        'PRODUCT_ID': product_ids,
        'AMOUNT': amounts,
        'DATE': dates,
    }).to_csv(trans_path, index=False)

    graph = Graph('test')
    graph.connect(CSVConnector('csv',
                               root_dir=tables_path,
                               na_values='?'))

    graph.add(name='USERS', pkey='USER_ID', cols=['AGE'])
    graph.add(name='PRODUCTS', pkey='PRODUCT_ID', cols=['BRAND'])
    graph.add(name='PRODUCT_BRANDS', pkey='BRAND', cols=['BRAND_VALUE'])
    graph.add(
        name='TRANS',
        fkeys=[
            dict(name='USER_ID', dst_table='USERS', dtype=Dtype.int),
            dict(name='PRODUCT_ID', dst_table='PRODUCTS', dtype=Dtype.int),
        ],
        cols=['USER_ID', 'PRODUCT_ID', 'AMOUNT'],
    )

    graph.infer_links()
    graph = graph.infer_metadata()
    graph['TRANS', 'AMOUNT'].stype = Stype.numerical
    graph['TRANS', 'AMOUNT'].dtype = Dtype.float
    graph = graph.infer_encoders()
    if set_time_col:
        graph['TRANS'].set_time_col('DATE')

    return graph


@pytest.fixture(scope='function')
def link_pred_same_node_test_graph(tmp_path: pathlib.Path) -> Graph:
    """Graph used to test link prediction queries. Graph has
    tables `USERS`, and `TRANS`. `TRANS` has two foreign key to
    `USERS`. """

    entity_df = pd.DataFrame({'USER_ID': range(3), 'AGE': [17, 25, 33]})
    dates = pd.Series(pd.date_range(start='1/1/2018', end='1/30/2018'))
    dates = pd.concat([dates, dates, dates], axis=0, ignore_index=True)
    target_df = pd.DataFrame({
        'TRANS_ID':
        range(len(dates)),
        'SENDER_ID': [0] * 30 + [1] * 30 + [2] * 30,
        'RECEIVER_ID':
        [0] * 15 + [1] * 15 + [0] * 15 + [4] * 15 + [0] * 15 + [1] * 15,
        'AMOUNT': [1.0, 2.0, 3.0] * 30,
        'DATE':
        dates,
    })

    graph = Graph('test')
    add_df(graph, entity_df, 'USERS', tmp_path, pkey='USER_ID')
    add_df(graph, target_df, 'TRANS', tmp_path, pkey='TRANS_ID')
    graph.link('TRANS', 'SENDER_ID', 'USERS', 'USER_ID')
    graph.link('TRANS', 'RECEIVER_ID', 'USERS', 'USER_ID')
    graph.infer_metadata()
    graph.infer_encoders()
    graph['TRANS'].set_time_col('DATE')
    return graph


@pytest.fixture(scope='function')
def graph_with_entity_dates(tmp_path: pathlib.Path) -> Callable:
    return functools.partial(_graph_with_entity_dates, tmp_path)


def _graph_with_entity_dates(
    tmp_path: pathlib.Path,
    with_start_date_users: bool = True,
    with_end_date_users: bool = True,
    with_start_date_products: bool = True,
    with_end_date_products: bool = True,
    add_na_pkeys: bool = False,
    add_na_timestamps: bool = False,
    add_year_ago_data: bool = False,
) -> Graph:
    """
    Graph used to test queries with temporal aggregations.
    Contains entity table `USERS`, transactions table `TRANS` and
    product table `PRODUCTS`.
    Args:
        with_start_date_users: If `True` will set time_col in `USERS` table.
        with_end_date_users: If `True` will set end_time_col in `USERS` table.
        with_start_date_products: If `True` will set time_col in `PRODUCTS`
            table.
        with_end_date_products: If `True` will set end_time_col in `PRODUCTS`
            table.
        add_na_pkeys: If `True`, tables will additionally include rows with
            pd.NA primary keys.
        add_na_timestamps: If `True`, tables will additionally include rows
            with pd.NA timestamps.
    """
    entity_df = pd.DataFrame({
        'USER_ID': range(3),
        'AGE': [17, 25, 40],
        'JOINED': ['1/1/2016', '1/21/2018', '1/1/2018'],
        'LEFT': ['1/15/2018', '1/27/2018', 'N/A'],
    })
    product_df = pd.DataFrame({
        'PRODUCT_ID':
        range(5, 8),
        'BRAND': ['AA', 'BB', 'CC'],
        'CREATED': ['1/1/2018', '1/10/2018', '1/1/2018'],
        'DISCONTINUED': ['1/10/2018', '1/27/2018', 'N/A'],
    })
    if add_na_pkeys:
        entity_df.loc[len(entity_df)] = [pd.NA, 33, '1/2/2018', 'N/A']
        entity_df.loc[len(entity_df)] = [pd.NA, 34, '1/1/2018', '1/20/2018']
        product_df.loc[len(product_df)] = [pd.NA, 'AA', '1/1/2018', 'N/A']
    if add_na_timestamps:
        entity_df.loc[len(entity_df)] = [3, 33, 'N/A', 'N/A']
        entity_df.loc[len(entity_df)] = [4, 33, 'N/A', '1/20/2018']
        product_df.loc[len(product_df)] = [8, 'BB', 'N/A', '1/20/2018']
        product_df.loc[len(product_df)] = [9, 'BB', 'N/A', 'N/A']
    dates = pd.Series(pd.date_range(start='1/1/2018', end='1/30/2018'))
    dates = pd.concat([dates, dates, dates], axis=0, ignore_index=True)
    types = ['A', 'B', 'C'] * 30
    target_df = pd.DataFrame({
        'TRANS_ID': range(len(dates)),
        'USER_ID': [0] * 30 + [1] * 30 + [2] * 30,
        'AMOUNT': [1.0] * len(dates),
        'PRODUCT_ID': [5, 6, 7] * 30,
        'DATE': dates,
        'TYPE': types
    })
    if add_year_ago_data:
        # Duplicate the data to exactly a year ago (364 days)
        year_ago_data = target_df.copy()
        year_ago_data['DATE'] = year_ago_data['DATE'] - pd.Timedelta(days=364)
        year_ago_data['AMOUNT'] = year_ago_data['AMOUNT']
        year_ago_data['TRANS_ID'] = year_ago_data['TRANS_ID'] + len(dates)
        target_df = pd.concat([target_df, year_ago_data], axis=0,
                              ignore_index=True)

    if add_na_pkeys:
        target_df.loc[len(target_df)] = [
            pd.NA, 0, 1.0, 5, pd.Timestamp('1/10/2018'), 'A'
        ]
        target_df.loc[len(target_df)] = [
            len(dates), pd.NA, 1.0, 5,
            pd.Timestamp('1/10/2018'), 'A'
        ]
        target_df.loc[len(target_df)] = [
            len(dates) + 1, 1, 1.0, pd.NA,
            pd.Timestamp('1/10/2018'), 'A'
        ]
    if add_na_timestamps:
        target_df.loc[len(target_df)] = [len(target_df), 0, 1.0, 5, pd.NA, 'A']
    graph = Graph('test')
    add_df(graph, entity_df, 'USERS', tmp_path, pkey='USER_ID')
    add_df(graph, target_df, 'TRANS', tmp_path, pkey='TRANS_ID')
    add_df(graph, product_df, 'PRODUCTS', tmp_path, pkey='PRODUCT_ID')
    graph.link('TRANS', 'USER_ID', 'USERS', 'USER_ID')
    graph.link('TRANS', 'PRODUCT_ID', 'PRODUCTS', 'PRODUCT_ID')
    graph.infer_metadata()
    graph.infer_encoders()
    graph['TRANS'].set_time_col('DATE')
    if with_start_date_users:
        graph['USERS'].set_time_col('JOINED')
    if with_end_date_users:
        graph['USERS'].set_end_time_col('LEFT')
    if with_start_date_products:
        graph['PRODUCTS'].set_time_col('CREATED')
    if with_end_date_products:
        graph['PRODUCTS'].set_end_time_col('DISCONTINUED')
    return graph


# H&M integration test graph ##################################################


@pytest.fixture()
def create_hm_integ_test_graph() -> Callable:
    return hm_integ_test_graph


def hm_integ_test_graph() -> Graph:
    ARTICLES = 'articles'
    CUSTOMERS = 'customers'
    TRANSACTIONS = 'transactions'

    root_dir = 's3://kumo-public-datasets/hnm_integ_test'
    connector = ParquetConnector(name='hnm_integ_test', root_dir=root_dir)

    graph = Graph(name='H&M')
    graph.connect(connector)

    graph.add(ARTICLES, pkey='article_id')
    graph.add(CUSTOMERS, pkey='customer_id')
    graph.add(TRANSACTIONS, fkeys=['article_id', 'customer_id'])

    graph[TRANSACTIONS].set_time_col('t_dat')
    graph[TRANSACTIONS].add_col('t_dat', stype=Stype.timestamp, dtype=Dtype.date)
    graph[TRANSACTIONS, 't_dat'].metadata = {'TS_FORMAT': 'yyyy-MM-dd'}

    graph.infer_metadata()
    graph.infer_links()
    # graph.infer_encoders()

    return graph


# Utilities ###################################################################


@pytest.fixture(scope='module')
def add_df_to_graph() -> Callable:
    return add_df


def add_df(
    graph: Graph,
    df: pd.DataFrame,
    name: str,
    tmp_path: pathlib.Path,
    **kwargs,
):
    path: str = osp.join(tmp_path.resolve().as_posix(), 'parquet')
    os.makedirs(path, exist_ok=True)
    # Use int96 timestamps for PyArrow 14 compatibility
    df.to_parquet(osp.join(path, f'{name}.parquet'),
                  use_deprecated_int96_timestamps=True)

    # The data source is named 'parquet_source' and not 'parquet'
    # to avoid conflict with other fixtures that are using the same
    # data source name and table names (e.g. 'USERS').
    if not graph.has_data_source('parquet_source'):
        graph.connect(ParquetConnector('parquet_source', path))

    graph.add(name, **kwargs)
    return graph
