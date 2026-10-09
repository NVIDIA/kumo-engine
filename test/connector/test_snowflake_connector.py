# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from diskgraph import MULTICAT_DELIMITERS
from kumo_core.types import ColStatType
from kumoapi.typing import Dtype, Stype

import kumo.connector.snowflake_connector as snowflake_connector
from io_utils.secrets import get_secrets
from kumo.config import BaseColumnConfig
from kumo.connector.connector import (
    ConnectorError,
    SourceTableUnavailableError,
)
from kumo.connector.snowflake_connector import SnowflakeConnector, _quote_id
from kumo.graph.graph import Graph
from kumo.stats import compute_col_stats_python
from kumo.testing import withSnowflakeCredentials
from kumo.testing.snowflake import (
    SnowflakeTableConnector,
    setup_full_snowflake_session,
)
from statedb.config.typing import PANDAS_TO_SQL

_SAFE_SNOWFLAKE_PROVIDER_ERROR = 'Snowflake could not complete the operation'


def _write_df_to_snowflake(connector, table_name, df, replace=False):
    r"""Test helper: write a pandas DataFrame to a Snowflake table."""
    from snowflake.connector.pandas_tools import write_pandas

    if len(df.columns) == 0:
        raise ValueError(f"Cannot write to table {table_name} as the "
                         "data frame has no columns")

    df.convert_dtypes()

    if replace:
        query = f'CREATE OR REPLACE TABLE {_quote_id(table_name)} ('
    else:
        query = f'CREATE TABLE IF NOT EXISTS {_quote_id(table_name)} ('

    for col in df:
        if (len(df[col]) > 0 and isinstance(df[col].iloc[0], np.ndarray)
                and df[col].iloc[0].ndim == 1):
            query += f"{_quote_id(col)} ARRAY, "
        else:
            query += f"{_quote_id(col)} {PANDAS_TO_SQL[df[col].dtype]}, "

    query = query[:-2] + ')'

    for col in df.select_dtypes(include='datetime64[ns]').columns:
        df[col] = df[col].astype(str).replace({"NaT": None})

    connector.execute_meta_query(connector.context, query)

    if len(df) == 0:
        return

    write_pandas(connector.context, df, table_name)


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_credentials_as_params():
    credentials = {
        'user': os.getenv(snowflake_connector._ENV_SNOWFLAKE_USER),
        'private_key': os.getenv(snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY)
    }

    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='FINANCIAL')
    connector.open(credentials=credentials)
    assert connector.read_table('LOAN') is not None
    connector.close()

    with SnowflakeConnector('snowflake', account='xva19026', warehouse='WH_XS',
                            database='KUMO', schema_name='FINANCIAL',
                            credentials=credentials) as connector:
        assert connector.read_table('LOAN') is not None


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_wrong_credentials():

    with pytest.raises(ConnectorError, match=_SAFE_SNOWFLAKE_PROVIDER_ERROR):
        credentials = {'user': 'a', 'password': '1'}
        connector = SnowflakeConnector('snowflake', account='xva19026',
                                       warehouse='WH_XS', database='KUMO',
                                       schema_name='FINANCIAL')
        connector.open(credentials=credentials)

    with pytest.raises(
            ConnectorError,
            match='Error in connecting to the provided Snowflake account.'):
        credentials = {
            'user': os.getenv(snowflake_connector._ENV_SNOWFLAKE_USER),
            'private_key':
            os.getenv(snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY)
        }
        connector = SnowflakeConnector('snowflake', account='wrongaccount',
                                       warehouse='WH_XS', database='KUMO',
                                       schema_name='HM')
        connector.open(credentials=credentials)

    with pytest.raises(ConnectorError, match=_SAFE_SNOWFLAKE_PROVIDER_ERROR):
        credentials = {
            'user': os.getenv(snowflake_connector._ENV_SNOWFLAKE_USER),
            'private_key':
            os.getenv(snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY)
        }
        connector = SnowflakeConnector('snowflake', account='xva19026',
                                       warehouse='WH_XS', database='KUMO_notdb',
                                       schema_name='HM')
        connector.open(credentials=credentials)
        connector._get_table_metadata()
        connector.close()

    with pytest.raises(ConnectorError, match=_SAFE_SNOWFLAKE_PROVIDER_ERROR):
        credentials = {
            'user': os.getenv(snowflake_connector._ENV_SNOWFLAKE_USER),
            'private_key':
            os.getenv(snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY)
        }
        connector = SnowflakeConnector('snowflake', account='xva19026',
                                       warehouse='WH_XS', database='KUMO',
                                       schema_name='NOTREALSCHEMA')
        connector.open(credentials=credentials)
        connector._get_table_metadata()
        connector.close()


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_read_table():
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='FINANCIAL')

    df = connector.read_table('LOAN', pkey='LOAN_ID')
    assert df.shape == (682, 6)

    chunk = connector.read_table('LOAN', pkey='LOAN_ID', num_rows=5)
    assert df[:5].equals(chunk)

    chunk = connector.read_table('LOAN', pkey='LOAN_ID', num_rows=3, offset=2)
    assert df[2:5].equals(chunk)


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_get_table_metadata():
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='FINANCIAL_PYTEST')

    table_metadata_config = connector.get_table_metadata()
    assert len(table_metadata_config) >= 8  # Including train/pred tables

    # TODO: Add more tests
    for config in table_metadata_config:
        if config.table_name == 'ACCOUNT':
            assert config.pkey.name == 'ACCOUNT_ID'
            assert len(config.fkeys) == 1
            assert config.fkeys[0].name == 'DISTRICT_ID'
            assert len(config.cols) == 2

    # ACCOUNT_VIEW is a known view in the FINANCIAL schema.
    assert any(table.table_name == "ACCOUNT_VIEW"
               for table in table_metadata_config)
    assert connector.is_table_view("ACCOUNT_VIEW")
    # and a materialized view:
    assert connector.is_table_view("TRANS_VIEW_MAT")
    with pytest.raises(SourceTableUnavailableError,
                       match="Source table 'FAKETABLE'"):
        connector.is_table_view("FAKETABLE")

    assert set(table.table_name for table in table_metadata_config) >= set(
        connector.get_tables_info(connector.context))

    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='HM_SAMPLED')

    table_metadata_config = connector.get_table_metadata()
    names = [table.table_name for table in table_metadata_config]
    # TEST_TABLE exists but lacks permissions
    assert "TEST_TABLE" not in names
    assert "CUSTOMERS" in names


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_get_exact_col_stats():
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='FINANCIAL')

    out = connector._get_exact_col_stats(
        'LOAN',
        {
            BaseColumnConfig.cast('AMOUNT'): [
                ColStatType.MEAN,
                ColStatType.MEDIAN,
                ColStatType.MIN,
                ColStatType.MAX,
                ColStatType.STD,
            ],
            BaseColumnConfig.cast('STATUS'): [
                ColStatType.NUM_UNIQUE,
                ColStatType.UNIQUE_FRACTION,
                ColStatType.CATEGORY_COUNTS,
            ],
        },
    )
    assert len(out) == 2

    stats = out[BaseColumnConfig.cast('AMOUNT')]
    assert len(stats) == 5
    assert stats[ColStatType.MEAN].value == pytest.approx(151410.1)
    assert stats[ColStatType.MEAN].num_entries == 682
    assert stats[ColStatType.MEDIAN].value == 116928
    assert stats[ColStatType.MEDIAN].num_entries == 682
    assert stats[ColStatType.MIN].value == 4980
    assert stats[ColStatType.MIN].num_entries == 682
    assert stats[ColStatType.MAX].value == 590820
    assert stats[ColStatType.MAX].num_entries == 682
    assert stats[ColStatType.STD].value == pytest.approx(113372.4)
    assert stats[ColStatType.STD].num_entries == 682

    stats = out[BaseColumnConfig.cast('STATUS')]
    assert len(stats) == 3
    assert stats[ColStatType.NUM_UNIQUE].value == 4
    assert stats[ColStatType.NUM_UNIQUE].num_entries == 682
    assert stats[ColStatType.UNIQUE_FRACTION].value == pytest.approx(
        0.0058, abs=1e-2)
    assert stats[ColStatType.UNIQUE_FRACTION].num_entries == 682
    assert stats[ColStatType.CATEGORY_COUNTS].value == [('C', 403), ('A', 203),
                                                        ('D', 45), ('B', 31)]
    assert stats[ColStatType.CATEGORY_COUNTS].num_entries == 682

    out = connector._get_exact_col_stats(
        'DISTRICT',
        {
            BaseColumnConfig.cast('A12'): [
                ColStatType.NUM_NA,
                ColStatType.NA_FRACTION,
            ],
        },
    )
    assert len(out) == 1

    stats = out[BaseColumnConfig.cast('A12')]
    assert stats[ColStatType.NUM_NA].value == 1
    assert stats[ColStatType.NUM_NA].num_entries == 77
    assert stats[ColStatType.NA_FRACTION].value == pytest.approx(1 / 77, 0.001)
    assert stats[ColStatType.NA_FRACTION].num_entries == 77


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_data_version():
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='FINANCIAL')

    # _get_data_version issues a Snowflake metadata query. Testing
    # data version for only two tables to ensure the test isn't slow.
    table_data_versions = {
        'TRANS': 1737876795,
        'ORDERS': 1737878718,
    }

    for table_name, version in table_data_versions.items():
        # get table data versions
        assert connector._get_data_version(table_name) == str(
            table_data_versions[table_name])

    timestamp = datetime.now(timezone.utc).replace(minute=0, second=0,
                                                   microsecond=0)
    current_hour = str(timestamp.timestamp()).split('.')[0]
    assert int(
        connector._get_data_version('TEST_TMP_VIEW')) >= int(current_hour)


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_lower_case_data_version():
    connector_uppercase = SnowflakeConnector('snowflake', account='xva19026',
                                             warehouse='WH_XS', database='KUMO',
                                             schema_name='FINANCIAL')

    connector_lowercase = SnowflakeConnector('snowflake', account='xva19026',
                                             warehouse='WH_XS', database='kumo',
                                             schema_name='financial')

    # _get_data_version issues a Snowflake metadata query.
    tables = ['TRANS', 'ORDERS']

    for table in tables:
        # get table data versions
        assert (connector_uppercase._get_data_version(table) ==
                connector_lowercase._get_data_version(table))


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_table_info():
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='FINANCIAL')

    # num rows
    assert connector._get_table_num_rows('ACCOUNT') == 4500
    assert connector._get_table_num_rows('CARD') == 892
    assert connector._get_table_num_rows('CLIENT') == 5369
    assert connector._get_table_num_rows('DISP') == 5369
    assert connector._get_table_num_rows('DISTRICT') == 77
    assert connector._get_table_num_rows('LOAN') == 682
    assert connector._get_table_num_rows('ORDERS') == 6471

    # table size (in bytes)
    assert connector._get_table_source_size('ACCOUNT') == 43008
    assert connector._get_table_source_size('CARD') == 10752
    assert connector._get_table_source_size('CLIENT') == 52736
    assert connector._get_table_source_size('DISP') == 35328
    assert connector._get_table_source_size('DISTRICT') == 8704
    assert connector._get_table_source_size('LOAN') == 13824
    assert connector._get_table_source_size('ORDERS') == 107520
    assert connector._get_table_source_size('NONTABLE') == 0

    assert connector._query_info_table('ACCOUNT', 'BYTES') == 43008
    assert connector._query_info_table('CARD', 'BYTES') == 10752
    assert connector._query_info_table('CLIENT', 'BYTES') == 52736
    assert connector._query_info_table('DISP', 'BYTES') == 35328
    assert connector._query_info_table('DISTRICT', 'BYTES') == 8704
    assert connector._query_info_table('LOAN', 'BYTES') == 13824
    assert connector._query_info_table('ORDERS', 'BYTES') == 107520
    assert connector._query_info_table('NONTABLE', 'BYTES') == 0


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_overwrite_table(
    writable_snowflake_table: SnowflakeTableConnector, ):
    name = writable_snowflake_table.table_name
    overwritten_name = f'{name}_overwritten'
    connector = writable_snowflake_table.connector

    nrows = 2
    pred_df = pd.DataFrame({
        'LOAN_ID': range(nrows),
        'EMBEDDING': [np.array(range(5))] * nrows,
        'A_PROB': [0.51] * nrows
    })

    _write_df_to_snowflake(connector, name, pred_df, replace=True)
    connector.drop_table_if_exists(overwritten_name)
    connector.overwrite_table(name, overwritten_name)
    uploaded_df = connector.read_table(overwritten_name)
    uploaded_df['EMBEDDING'] = uploaded_df['EMBEDDING'].apply(json.loads).apply(
        np.array)
    assert str(uploaded_df) == str(pred_df)

    pred_df = pd.DataFrame({
        'LOAN_ID': range(nrows),
        'EMBEDDING': [np.array(range(3))] * nrows,
        'A_PROB': [0.22] * nrows
    })
    _write_df_to_snowflake(connector, name, pred_df, replace=True)
    connector.overwrite_table(name, overwritten_name)
    uploaded_df = connector.read_table(overwritten_name)
    uploaded_df['EMBEDDING'] = uploaded_df['EMBEDDING'].apply(json.loads).apply(
        np.array)
    assert str(uploaded_df) == str(pred_df)


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_array_type():
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='TEST_SNOWFLAKE_TYPE')
    metadata = connector.get_table_metadata(['TEST_ARRAY'])[0]
    assert (metadata.pkey.name == 'ID' or metadata.pkey.name == 'TARGET')
    # There are others two columns in the table apart from the primary key.
    assert len(metadata.cols) == 2
    for col in metadata.cols:
        if col.name == 'ARR':
            assert col.dtype == Dtype.string


@pytest.mark.snowflake
@withSnowflakeCredentials
# TODO(jinu) use snowflake dataset used in
# test/rest/integ/test_data_variety/
def test_snowflake_col_types_infer_and_cast():

    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='SCHEMA_TEST')
    table_metadata_configs = connector.get_table_metadata()

    graph = Graph(name='test_graph')
    graph.connect(connector)
    table_name_1 = 'TYPES_TEST'
    table_name_2 = 'TYPES_TEST_2'
    table_name_3 = 'TEST_ID_FLOATS'
    graph.add(table_name_1)
    graph.add(table_name_2)
    graph.add(table_name_3)

    expected_dtypes = {
        f'snowflake.{table_name_1}': {
            'NUM_1': Dtype.int64,
            'CHAR_1': Dtype.string,
            'CHAR_2': Dtype.string,
            'CHAR_3': Dtype.string,
            'BIN_1': Dtype.binary,
            'BOOL': Dtype.bool,
            'DAT_1': Dtype.date,
            'DAT_2': Dtype.time,
            'TIME_1': Dtype.unsupported,
            'TIME_2': Dtype.time,
            'TIME_3': Dtype.time,
            'TIME_4': Dtype.time,
            'TIME_5': Dtype.time,
        },
        f'snowflake.{table_name_2}': {
            'ID': Dtype.int64,
            'INTEGER': Dtype.int64,
            'UINTEGER': Dtype.int64,
            'INTEGER8': Dtype.int64,
            'DATE': Dtype.date,
            'LISTS': Dtype.string,
            'MULTICAT1': Dtype.string,
            'MULTICAT2': Dtype.string
        },
        f'snowflake.{table_name_3}': {
            'FLOAT_ID': Dtype.float64,
            'FLOAT_2_ID': Dtype.float64
        }
    }
    expected_stypes = {
        f'snowflake.{table_name_1}': {
            'NUM_1': Stype.numerical,
            'CHAR_1': Stype.categorical,
            'CHAR_2': Stype.categorical,
            'CHAR_3': Stype.categorical,
            'BIN_1': Stype.categorical,
            'BOOL': Stype.categorical,
            'DAT_1': Stype.timestamp,
            'DAT_2': Stype.timestamp,
            'TIME_1': Stype.unsupported,
            'TIME_2': Stype.timestamp,
            'TIME_3': Stype.timestamp,
            'TIME_4': Stype.timestamp,
            'TIME_5': Stype.timestamp,
        },
        f'snowflake.{table_name_2}': {
            'ID': Stype.ID,
            'INTEGER': Stype.numerical,
            'UINTEGER': Stype.numerical,
            'INTEGER8': Stype.categorical,
            'DATE': Stype.timestamp,
            # LISTS contains bracketed JSON-like strings, not a supported
            # delimiter-encoded scalar multicategorical column.
            'LISTS': Stype.text,
            'MULTICAT1': Stype.multicategorical,
            'MULTICAT2': Stype.multicategorical
        },
        f'snowflake.{table_name_3}': {
            'FLOAT_ID': Stype.numerical,
            'FLOAT_2_ID': Stype.numerical
        }
    }

    table_dict = {table.name: table for table in table_metadata_configs}
    for table_name in [table_name_1, table_name_2, table_name_3]:
        table = table_dict[f'snowflake.{table_name}']
        for col in table.cols:
            graph[table.name].add_col(col.name, dtype=col.dtype)
            assert col.dtype == expected_dtypes[table.name][col.name]

        graph[table.name].infer_stypes()

        for col in graph[table.name].cols:
            assert col.stype == expected_stypes[table.name][col.name]
            if col.name in ('MULTICAT1', 'MULTICAT2'):
                assert (col.multicategorical_delimiter in MULTICAT_DELIMITERS)
            else:
                assert col.multicategorical_delimiter is None

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
    assert df.shape[1] == 8

    for col, stats in stats_dict_2.items():
        stats_result = compute_col_stats_python(df[col.name], stats)
        assert stats_result[ColStatType.NUM_NA].value == 0


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_table_exists(
        snowflake_financial_connector: SnowflakeConnector):
    connector = snowflake_financial_connector
    assert connector.table_exists("ACCOUNT") is True
    assert connector.table_exists("DOES_NOT_EXIST") is False


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_read_sample():
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='SCHEMA_TEST')
    table_metadata_configs = connector.get_table_metadata()

    expected_shapes = {
        'COL_NAMES_TEST': (10, 7),
        'EMPTY_TEST': (0, 4),
        'MOVIELENS_TAG': (10, 4),
        'TYPES_TEST': (10, 13),
        'TYPES_TEST_2': (10, 8),
        'TEST_VIEW': (10, 2),
        'TEST_VIEW_2': (10, 2)
    }

    for table_config in table_metadata_configs:
        if table_config.source_name == 'EMPTY_TEST':
            with pytest.raises(
                    ConnectorError,
                    match="Failed to read any rows from table 'EMPTY_TEST'"):
                df = connector.read_table_sample(table_config.source_name,
                                                 num_rows=10)
        elif table_config.source_name in expected_shapes.keys():
            df = connector.read_table_sample(table_config.source_name,
                                             num_rows=10)
            assert expected_shapes[table_config.source_name] == df.shape


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_read_snowpark(tempdir, monkeypatch):
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='FINANCIAL')

    # ``setup_full_snowflake_session`` routes through the module-level
    # cache in :mod:`kumo.connector.snowflake_session`
    # (``get_or_build_session``) so ``connector._get_snowpark_df`` reads
    # the same session via the same connector key.
    del monkeypatch  # kept in signature for fixture wiring
    session = setup_full_snowflake_session(tempdir, snow_connector=connector)

    table_name = f'TEST_TABLE_{uuid.uuid4()}'.upper().replace('-', '_')
    session.sql(f"DROP TABLE IF EXISTS {table_name}").collect()
    data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    table_df = session.create_dataframe(data_df)
    table_df.write.save_as_table(table_name)
    version = connector.get_data_version(table_name, skip_cache=True)
    table_df = connector._get_snowpark_df(table_name, version)
    # Wait to get a new timestamp
    time.sleep(10)
    # Append data to the table
    table_df.write.save_as_table(table_name, mode='append')
    new_version = connector.get_data_version(table_name, skip_cache=True)
    new_table_df = connector._get_snowpark_df(table_name, new_version)
    assert new_table_df.count() == 6
    table_df = connector._get_snowpark_df(table_name, version)
    assert table_df.count() == 3

    session.sql(f"DROP TABLE IF EXISTS {table_name}").collect()


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_vector_type():
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='TEST_SNOWFLAKE_TYPE')

    table_metadata = connector.get_table_metadata(['TEST_VECTORS'])[0]
    for col in table_metadata.cols:
        assert col.dtype == Dtype.unsupported


@pytest.mark.snowflake
def test_keypair_snowflake_credentials(tempdir, monkeypatch,
                                       valid_fake_private_key):
    secrets = get_secrets("xva19026.snowflakecomputing.com")
    strkey = secrets['kumo_secretkey']

    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='TEST_SNOWFLAKE_TYPE')
    # internal logic defaults to env var user-pass auth if set so temporarily
    # unset these:
    monkeypatch.delenv(snowflake_connector._ENV_SNOWFLAKE_USER, raising=False)
    monkeypatch.delenv(snowflake_connector._ENV_SNOWFLAKE_PASSWORD,
                       raising=False)

    creds = {'user': 'mfatest', 'private_key': strkey}
    connector.set_credentials(creds)

    assert 'private_key' in connector._credentials
    assert 'password' not in connector._credentials
    connector.open(creds)
    assert len(connector.list_tables()) >= 5

    # verify that snowpark can be used with a keypair auth connection.
    session = setup_full_snowflake_session(tempdir, snow_connector=connector)

    table_name = f'TEST_TABLE_{uuid.uuid4()}'.upper().replace('-', '_')
    data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    table_df = session.create_dataframe(data_df)
    table_df.write.save_as_table(table_name)

    # test connector again and verify the results of using snowpark APIs:
    version = connector.get_data_version(table_name, skip_cache=True)
    table_df = connector._get_snowpark_df(table_name, version)

    assert table_df.count() == 3

    session.sql(f"DROP TABLE IF EXISTS {table_name}").collect()

    # invalid key:
    creds = {'user': 'mfatest', 'private_key': 'fake_pkey'}
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='TEST_SNOWFLAKE_TYPE')
    connector.set_credentials(creds)
    assert 'private_key' in connector._credentials
    assert 'password' not in connector._credentials
    with pytest.raises(ConnectorError,
                       match='invalid or in an unsupported format'):
        connector.open(creds)

    creds = {'user': 'mfatest', 'private_key': valid_fake_private_key}
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='TEST_SNOWFLAKE_TYPE')
    connector.set_credentials(creds)
    with pytest.raises(ConnectorError, match=_SAFE_SNOWFLAKE_PROVIDER_ERROR):
        connector.open(creds)

    encrypted_key = secrets['kumo_secretkey_encrypted']
    passphrase = secrets['kumo_secretkey_encrypted_passphrase']
    creds = {
        'user': 'mfatest',
        'private_key': encrypted_key,
        'private_key_passphrase': passphrase
    }
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='TEST_SNOWFLAKE_TYPE')
    connector.set_credentials(creds)
    connector.open(creds)
    assert len(connector.list_tables()) >= 5


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_snowflake_sample_rows_with_timestamps(
        writable_snowflake_table: SnowflakeTableConnector, monkeypatch):
    name = writable_snowflake_table.table_name
    connector = writable_snowflake_table.connector
    connector.execute_meta_query(
        connector.context, f"""
        CREATE OR REPLACE TABLE {name} (
            id INT AUTOINCREMENT,
            timestamp_column TIMESTAMP_NTZ(9)
        );
    """)

    connector.execute_meta_query(
        connector.context, f"""
        INSERT INTO {name} (timestamp_column) VALUES
            ('2012-01-01 00:00:00'::TIMESTAMP_NTZ),
            ('2011-01-02 00:00:00'::TIMESTAMP_NTZ);
    """)
    df = connector.read_table_sample(name, num_rows=10, skip_cache=True)
    assert len(df) == 2
    assert set(df.columns) == set(['ID', 'TIMESTAMP_COLUMN'])
    assert df.dtypes['ID'] == 'int8'
    assert df.dtypes['TIMESTAMP_COLUMN'] == 'datetime64[ns]'

    connector.execute_meta_query(
        connector.context, f"""
        INSERT INTO {name} (timestamp_column) VALUES
            ('3122-01-01 00:00:00'::TIMESTAMP_NTZ),
            ('0011-01-02 00:00:00'::TIMESTAMP_NTZ);
    """)

    df = connector.read_table_sample(name, num_rows=10, skip_cache=True)
    assert len(df) == 4
    assert set(df.columns) == set(['ID', 'TIMESTAMP_COLUMN'])
    assert df.dtypes['ID'] == 'int8'
    assert df.dtypes['TIMESTAMP_COLUMN'] == 'object'

    connector.execute_meta_query(
        connector.context, f"""
        INSERT INTO {name} (timestamp_column)
            SELECT TO_TIMESTAMP('2001-01-01T00:00:59') as d1
                from table(generator(rowcount => 100000))
            UNION ALL SELECT TO_TIMESTAMP('9999-01-01T00:00:59') as d1
    """)
    monkeypatch.setattr('kumo.connector.connector.NUM_SAMPLE_ROWS', 200000)
    df = connector.read_table_sample(name, num_rows=200000, skip_cache=True)
    assert len(df) == 100005
    assert df.dtypes['TIMESTAMP_COLUMN'] == 'object'
    assert datetime(9999, 1, 1, 0, 0, 59) in df['TIMESTAMP_COLUMN'].tolist()


class TestSnowflakeAccessCheck:

    _credentials = {'user': 'test-user', 'password': 'test-password'}

    @pytest.mark.snowflake
    @withSnowflakeCredentials
    def test_successful_access(self):

        connector = SnowflakeConnector('snowflake', account='xva19026',
                                       warehouse='WH_XS', database='KUMO',
                                       schema_name='FINANCIAL')
        connector.open()

        connector.close()

    @pytest.mark.snowflake
    @withSnowflakeCredentials
    def test_invalid_account(self):

        # Not checking for a specific error message because the
        # message seems to differ between GitHub and my dev VM.
        with pytest.raises(ConnectorError):
            connector = SnowflakeConnector('snowflake', account='mxyzptlk',
                                           warehouse='WH_XS', database='KUMO',
                                           schema_name='FINANCIAL')
            connector.open()

            connector.close()

    @pytest.mark.snowflake
    @withSnowflakeCredentials
    def test_invalid_warehouse(self):

        with pytest.raises(ConnectorError,
                           match=_SAFE_SNOWFLAKE_PROVIDER_ERROR):
            connector = SnowflakeConnector('snowflake', account='xva19026',
                                           warehouse='MXYZPTLK',
                                           database='KUMO',
                                           schema_name='FINANCIAL')
            connector.open()

            connector.close()

    @pytest.mark.snowflake
    @withSnowflakeCredentials
    def test_invalid_db(self):

        with pytest.raises(ConnectorError,
                           match=_SAFE_SNOWFLAKE_PROVIDER_ERROR):
            connector = SnowflakeConnector('snowflake', account='xva19026',
                                           warehouse='WH_XS',
                                           database='MXYZPTLK',
                                           schema_name='FINANCIAL')
            connector.open()

            connector.close()

    @pytest.mark.snowflake
    @withSnowflakeCredentials
    def test_invalid_schema(self):

        with pytest.raises(ConnectorError,
                           match=_SAFE_SNOWFLAKE_PROVIDER_ERROR):
            connector = SnowflakeConnector('snowflake', account='xva19026',
                                           warehouse='WH_XS', database='KUMO',
                                           schema_name='MXYZPTLK')
            connector.open()

            connector.close()


_TEST_STAGING_PATH = 's3://kumo-preprod-dataplane-bucket/snowflake-integ-test/'


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_write_artifacts_to_table(writable_snowflake_table,
                                  snowflake_financial_connector):
    """Seed a table, export to S3, then write_artifacts_to_table back."""
    from io_utils import fsutil

    dest_name = writable_snowflake_table.table_name
    connector = writable_snowflake_table.connector

    artifact_path = (_TEST_STAGING_PATH + f'artifacts_{uuid.uuid4().hex[:8]}/')

    # Seed a small DataFrame into Snowflake
    nrows = 5
    pred_df = pd.DataFrame({
        'LOAN_ID': range(nrows),
        'A_PROB': [0.51] * nrows,
    })
    _write_df_to_snowflake(connector, dest_name, pred_df, replace=True)
    uploaded_df = connector.read_table(dest_name)
    assert len(uploaded_df) == nrows

    from io_utils.fsutil.credentials import credential_scope

    credentials = snowflake_connector.get_temporary_aws_credential(
        artifact_path)
    with credential_scope(
        {
            'provider': 's3',
            'aws_access_key_id': credentials['AccessKeyId'],
            'aws_secret_access_key': credentials['SecretAccessKey'],
            'aws_session_token': credentials['SessionToken'],
        }, require_explicit=True):
        source_connector = snowflake_financial_connector
        source_connector.open()
        try:
            source_connector.export_table_read_only(
                'LOAN', artifact_path, cols={
                    'LOAN_ID': Dtype.int,
                    'ACCOUNT_ID': Dtype.int
                })
            connector.write_artifacts_to_table(dest_name, artifact_path,
                                               replace=True)
            result_df = connector.read_table(dest_name)
            assert len(result_df) > nrows
        finally:
            fsutil.rm(artifact_path, ignore_errors=True)


@pytest.mark.snowflake
@withSnowflakeCredentials
def test_export_and_download(snowflake_financial_connector, tmp_path):
    """Stages LOAN, then copies Kumo-owned parquet locally."""
    from io_utils import fsutil
    from kumo.connector.connector import ExportMetrics
    from kumo.data import MaterializationStage
    from kumo.data.utils import get_materialize_path

    connector = snowflake_financial_connector
    connector.open()

    # Unique fingerprint id isolates this test run within the shared stage.
    test_fpid = f'test.loan.{uuid.uuid4().hex[:8]}'
    data_version = '1'
    table_fp = Mock()
    table_fp.ingest_path_fpid = test_fpid
    table_fp.cols = {'LOAN_ID': Dtype.int, 'ACCOUNT_ID': Dtype.int}

    try:
        metrics = connector.export_table('LOAN', table_fp, data_version)
        assert isinstance(metrics, ExportMetrics)
        assert metrics.num_rows > 0
        assert metrics.source_size > 0
        connector.download('LOAN', table_fp, data_version, str(tmp_path))
        downloaded = list(tmp_path.iterdir())
        assert len(downloaded) > 0
    finally:
        fsutil.rm(
            get_materialize_path(table_fp, data_version,
                                 MaterializationStage.INGEST),
            ignore_errors=True,
        )


@pytest.mark.snowflake
@withSnowflakeCredentials
@pytest.mark.parametrize(
    'artifact_name,source_table',
    [
        (
            'processed_predictions.parquet',
            pa.table({
                'ENTITY_ID': [1, 2],
                'SCORE': [0.25, 0.75],
            }),
        ),
        (
            'processed_embeddings.parquet',
            pa.table({
                'ENTITY_ID': [1, 2],
                'EMBEDDING': [[0.1, 0.2], [0.3, 0.4]],
            }),
        ),
    ],
)
def test_write_artifacts_to_table_from_local_parquet_live(
        writable_snowflake_table, tmp_path, artifact_name, source_table):
    artifact_path = tmp_path / artifact_name
    artifact_path.mkdir()
    pq.write_table(source_table, artifact_path / 'part-00000.parquet')

    connector = writable_snowflake_table.connector
    connector.open()
    connector.write_artifacts_to_table(
        writable_snowflake_table.table_name,
        str(artifact_path),
        replace=True,
    )

    loaded = connector.read_table(writable_snowflake_table.table_name)
    assert len(loaded) == source_table.num_rows
    assert set(loaded.columns) == set(source_table.column_names)
