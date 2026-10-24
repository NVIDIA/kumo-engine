# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from kumo_core.types import ColStatType
from kumo_core.types.schema import BaseColumnConfig
from kumo_databricks.sql import quote_identifier
from kumoapi.typing import Dtype, Stype

import kumo.pquery.AST.ast_node as ast_node
from io_utils import fsutil
from kumo.connector.connector import ConnectorError, SourceTableNotFoundError
from kumo.connector.databricks_connector import DatabricksConnector
from kumo.connector.sql_connector import InternalWriteMode
from kumo.graph.graph import Graph
from kumo.graph.utils import get_full_encoder_stats
from kumo.stats.compute_stats_python import compute_col_stats_python
from kumo.testing.databricks import (
    DatabricksTableConnector,
    withDatabricksOauthCredentials,
)
from statedb.config.typing import PANDAS_TO_SQL

_HOST = os.environ.get('DATABRICKS_HOST', 'workspace.cloud.databricks.com')


# TODO: Use a Databricks volume for seed staging to remove the AWS dependency.
_TEST_SEED_STAGING_PATH = os.environ.get(
    'VOLUME_MOUNT_PATH',
    's3://kumo-preprod-dataplane-bucket/databricks-integ-test-staging/')


def _write_df_to_databricks(connector, table_schema_name, df, replace=False,
                            **kwargs):
    r"""Test helper: write a pandas DataFrame to a Databricks table."""

    schema_name, _ = connector.require_output_table_within_connector_target(
        table_schema_name)
    fqn = connector._fully_qualify_output_table_name(table_schema_name)

    if len(df.columns) == 0:
        raise ValueError(f"Cannot write to table {fqn} as "
                         "the data frame has no columns")

    df.convert_dtypes()
    for col in df.select_dtypes(include='datetime64[ns]').columns:
        df[col] = df[col].astype(str).replace({"NaT": None})

    staging_root_path = os.path.join(_TEST_SEED_STAGING_PATH,
                                     'kumo_prediction_table')
    tmp_staging_folder = os.path.join(staging_root_path,
                                      next(tempfile._get_candidate_names()), '')
    try:
        context = connector.context
        connector._get_df_result(
            context, f'describe schema '
            f'{quote_identifier(schema_name)}')

        if replace:
            query = f'CREATE OR REPLACE TABLE {fqn} ('
            for col in df:
                if (len(df[col]) > 0 and isinstance(df[col].iloc[0], np.ndarray)
                        and df[col].iloc[0].ndim == 1):
                    query += f"{quote_identifier(col)} ARRAY, "
                else:
                    query += (f"{quote_identifier(col)} "
                              f"{PANDAS_TO_SQL[df[col].dtype]}, ")
            query = query[:-2] + ')'
        else:
            query = f'CREATE TABLE IF NOT EXISTS {fqn}'
        connector._get_df_result(context, query)

        if not fsutil.exists(tmp_staging_folder):
            fsutil.makedirs(tmp_staging_folder)
        path = os.path.join(tmp_staging_folder, "prediction.parquet")
        df.to_parquet(path, **kwargs)

        upload_sql = ("COPY INTO {table_name} FROM '{staging_path}' "
                      "FILEFORMAT = PARQUET "
                      "COPY_OPTIONS ('mergeSchema' = 'true')").format(
                          table_name=fqn, staging_path=path)
        connector._get_df_result(context, upload_sql)
    except Exception as e:
        if 'SCHEMA_NOT_FOUND' in str(e):
            raise ValueError(f"Cannot write to table {fqn} "
                             f"since schema {schema_name} does not exist.")
        raise RuntimeError(f"Error writing data to '{fqn}': {str(e)}.") from e
    finally:
        fsutil.rm(tmp_staging_folder, ignore_errors=True)


@withDatabricksOauthCredentials
def test_dbx_non_existent_table_data_version(
        databricks_hm_connector: DatabricksConnector):

    # table's legit - should not fail.
    databricks_hm_connector._get_data_version('customers')

    with pytest.raises(SourceTableNotFoundError) as missing_table_error:
        databricks_hm_connector._get_data_version('NON_EXISTENT_TABLE')
    assert "Source table 'NON_EXISTENT_TABLE'" in str(missing_table_error.value)


@withDatabricksOauthCredentials
def test_wrong_credentials():
    wrong_creds = {'client_id': 'wrong_id', 'client_secret': 'wrong_secret'}
    safe_connection_error = (
        'Databricks could not establish the configured connection')

    # Wrong OAuth credentials
    with pytest.raises(ConnectorError, match=safe_connection_error):
        connector = DatabricksConnector('databricks', host=_HOST,
                                        cluster_id='0626-191119-75431625',
                                        warehouse_id='82d859e4a6b283a0',
                                        catalog='hive_mapped_uc',
                                        schema_name='hm')
        connector.open(credentials=wrong_creds)

    # Wrong host (use PAT — OAuth hangs due to OIDC retry on unreachable host)
    with pytest.raises(ConnectorError, match=safe_connection_error):
        connector = DatabricksConnector('databricks', host='wrong_host',
                                        cluster_id='0626-191119-75431625',
                                        warehouse_id='82d859e4a6b283a0',
                                        catalog='hive_mapped_uc',
                                        schema_name='hm')
        connector.open(credentials={'pat': 'dummy_token'})

    # Wrong warehouse (use PAT — same OIDC retry issue with OAuth)
    with pytest.raises(ConnectorError, match=safe_connection_error):
        connector = DatabricksConnector('databricks', host=_HOST,
                                        cluster_id='0626-191119-75431625',
                                        warehouse_id='wrong_warehouse',
                                        catalog='hive_mapped_uc',
                                        schema_name='hm')
        connector.open(credentials={'pat': 'dummy_token'})

    # Wrong catalog (connects with valid OAuth, fails at query time)
    with pytest.raises(ConnectorError,
                       match='Schema hm does not exist under catalog'):
        connector = DatabricksConnector('databricks', host=_HOST,
                                        cluster_id='0626-191119-75431625',
                                        warehouse_id='82d859e4a6b283a0',
                                        catalog='wrong_catalog',
                                        schema_name='hm')
        connector.open()
        connector.read_table("customers", num_rows=100)
        connector.close()


@withDatabricksOauthCredentials
def test_databricks_read_table(databricks_hm_connector: DatabricksConnector):
    connector = databricks_hm_connector

    df = connector.read_table('customers', pkey='customer_id', num_rows=100)
    assert df.shape == (100, 6)

    chunk = connector.read_table('customers', pkey='customer_id', num_rows=5)
    assert df[:5].equals(chunk)

    chunk = connector.read_table('customers', pkey='customer_id', num_rows=3,
                                 offset=2)
    assert chunk.shape == (3, 6)


@withDatabricksOauthCredentials
def test_databricks_export_query_is_exact_not_an_estimate(
        databricks_hm_connector: DatabricksConnector, tmp_path):
    """A Custom SQL table is staged through the shared reader, so its
    NUM_ROWS is the rows that landed in Kumo parquet -- equal to the query's
    own COUNT(*), and exact rather than an optimizer estimate."""
    import pyarrow.parquet as pq
    from kumo_core.types.schema import SqlTableFingerprint

    from kumo.data import MaterializationStage
    from kumo.data.utils import get_materialize_path

    os.environ.setdefault('KUMO_DATA_DIR', str(tmp_path))
    connector = databricks_hm_connector
    connector.open()
    sql = 'SELECT customer_id FROM customers'
    truth = connector.get_query_num_rows(sql)
    assert truth > 0

    table_fp = SqlTableFingerprint(connector_name='dbx_export_query',
                                   source_name='customers_query',
                                   cols={'customer_id': Dtype.string},
                                   pkey=None, time_col=None, end_time_col=None)
    metrics = connector.export_query(sql, table_fp, 'v1')

    assert metrics.num_rows == truth
    assert not metrics.approx
    staged = get_materialize_path(table_fp, 'v1', MaterializationStage.INGEST)
    parts = [p for p in fsutil.listdir(staged) if p.endswith('.parquet')]
    assert sum(pq.ParquetFile(p).metadata.num_rows for p in parts) == truth
    connector.close()


@withDatabricksOauthCredentials
def test_databricks_get_table_metadata(
        databricks_hm_connector: DatabricksConnector,
        databricks_test_schema_connector: DatabricksConnector):
    connector = databricks_hm_connector

    table_metadata_config = connector.get_table_metadata()
    assert len(table_metadata_config) >= 6  # Including train/pred tables
    articles = next(config for config in table_metadata_config
                    if config.table_name == 'articles')
    assert len(articles.cols) == 25

    table_metadata_config = databricks_test_schema_connector.get_table_metadata(
        skip_cache=True, skip_inferral=True)
    # >= 3: test_schema has empty_table, type_test, type_test_2 plus
    # whatever fixture tables tests have provisioned (e.g. the
    # hm_*_small subset tables used by the diskgraph e2e).
    assert len(table_metadata_config) >= 3


@withDatabricksOauthCredentials
def test_get_exact_col_stats(databricks_hm_connector: DatabricksConnector):
    connector = databricks_hm_connector

    out = connector._get_exact_col_stats(
        'transactions',
        {
            BaseColumnConfig.cast('price'): [
                ColStatType.MEAN,
                ColStatType.MEDIAN,
                ColStatType.MIN,
                ColStatType.MAX,
            ],
            BaseColumnConfig.cast('sales_channel_id'): [
                ColStatType.NUM_UNIQUE,
                ColStatType.CATEGORY_COUNTS,
            ],
        },
    )
    assert len(out) == 2

    stats = out[BaseColumnConfig.cast('price')]
    assert len(stats) == 4
    assert stats[ColStatType.MEAN].value == pytest.approx(0.0278, abs=1e-4)
    assert stats[ColStatType.MEAN].num_entries == 31788324
    assert stats[ColStatType.MEDIAN].value == pytest.approx(0.0254, abs=1e-4)
    assert stats[ColStatType.MEDIAN].num_entries == 31788324
    assert stats[ColStatType.MIN].value == pytest.approx(1.6949e-5, abs=1e-4)
    assert stats[ColStatType.MIN].num_entries == 31788324
    assert stats[ColStatType.MAX].value == pytest.approx(0.5915, abs=1e-4)
    assert stats[ColStatType.MAX].num_entries == 31788324

    stats = out[BaseColumnConfig.cast('sales_channel_id')]
    assert len(stats) == 2
    assert stats[ColStatType.NUM_UNIQUE].value == 2
    assert stats[ColStatType.NUM_UNIQUE].num_entries == 31788324
    assert stats[ColStatType.CATEGORY_COUNTS].value == [(2, 22379862),
                                                        (1, 9408462)]
    assert stats[ColStatType.CATEGORY_COUNTS].num_entries == 31788324


@withDatabricksOauthCredentials
def test_databricks_data_version(databricks_hm_connector: DatabricksConnector):
    connector = databricks_hm_connector

    # _get_data_version issues a Databricks metadata query to return a
    # version ID tracked internally for Databricks Delta tables.
    table_data_versions = {
        'customers': '2_1777913429',
        'articles': '0_1777913450',
        'transactions': '1_1777913534',
    }

    for table_name, version in table_data_versions.items():
        # get table data versions
        assert connector._get_data_version(table_name) == str(version)

    # Views do not have history tracked. Data version is current timestamp.
    timestamp = datetime.now(timezone.utc).replace(minute=0, second=0,
                                                   microsecond=0)
    current_hour = str(timestamp.timestamp()).split('.')[0]
    assert int(
        connector._get_data_version('customer_view')) >= int(current_hour)


@withDatabricksOauthCredentials
def test_databricks_table_info(databricks_hm_connector: DatabricksConnector):
    connector = databricks_hm_connector

    # num rows
    assert connector._get_table_num_rows('customers') == 1371980
    assert connector._get_table_num_rows('articles') == 105542
    assert connector._get_table_num_rows('transactions') == 31788324

    # table size (in bytes)
    assert connector._get_table_source_size('customers') == 167722166
    assert connector._get_table_source_size('articles') == 6273226
    assert connector._get_table_source_size('transactions') == 776368084


@pytest.mark.private_resource('aws')
@withDatabricksOauthCredentials
def test_databricks_write_df_and_write_artifacts_to_table(
        writable_databricks_table: DatabricksTableConnector):
    name = writable_databricks_table.table_name
    connector = writable_databricks_table.connector

    nrows = 2
    pred_df = pd.DataFrame({
        'LOAN_ID': range(nrows),
        'EMBEDDING': [np.array(range(5))] * nrows,
        'A_PROB': [0.51] * nrows
    })
    _write_df_to_databricks(connector, name, pred_df)

    output_table = connector._fully_qualify_output_table_name(name)
    uploaded_df = connector._get_df_result(connector.context,
                                           f'SELECT * FROM {output_table}')
    assert str(uploaded_df) == str(pred_df)
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_path = os.path.join(temp_dir, 'predictions.parquet')
        pd.DataFrame({
            'LOAN_ID': range(99),
            'A_PROB': [0.51] * 99,
        }).to_parquet(artifact_path)
        connector.write_artifacts_to_table(name, artifact_path, True)
    uploaded_df = connector._get_df_result(connector.context,
                                           f'SELECT * FROM {output_table}')
    assert len(uploaded_df) == 99


@withDatabricksOauthCredentials
def test_databricks_col_types_infer_and_cast(
        databricks_test_schema_connector: DatabricksConnector):

    connector = databricks_test_schema_connector
    table_metadata_configs = connector.get_table_metadata()

    graph = Graph(name='test_graph')
    graph.connect(connector)
    table_name_1 = 'type_test'
    table_name_2 = 'type_test_2'
    graph.add(table_name_1)
    graph.add(table_name_2)

    expected_dtypes = {
        f'databricks.{table_name_1}': {
            'NUM_1': Dtype.int64,
            'CHAR_1': Dtype.string,
            'BIN_1': Dtype.binary,
            'DOUB_1': Dtype.float,
            'BOOL': Dtype.bool,
            'DAT_1': Dtype.date,
            'TIME_1': Dtype.time,
            'ARR_1': Dtype.floatlist,
            'ARR_2': Dtype.intlist,
            'ARR_3': Dtype.stringlist,
        },
        f'databricks.{table_name_2}': {
            'ID': Dtype.int64,
            'INT_1': Dtype.int64,
            'SHORT_INT': Dtype.int64,
            'INT_8': Dtype.int64,
            'DAT_1': Dtype.date,
            'MULTICAT_1': Dtype.string,
            'MULTICAT_2': Dtype.string,
        }
    }
    expected_stypes = {
        f'databricks.{table_name_1}': {
            'NUM_1': Stype.numerical,
            'CHAR_1': Stype.categorical,
            'BIN_1': Stype.categorical,
            'DOUB_1': Stype.numerical,
            'BOOL': Stype.categorical,
            'DAT_1': Stype.timestamp,
            'TIME_1': Stype.timestamp,
            'ARR_1': Stype.sequence,
            'ARR_2': Stype.multicategorical,
            'ARR_3': Stype.multicategorical,
        },
        f'databricks.{table_name_2}': {
            'ID': Stype.ID,
            'INT_1': Stype.numerical,
            'SHORT_INT': Stype.numerical,
            'INT_8': Stype.categorical,
            'DAT_1': Stype.timestamp,
            'MULTICAT_1': Stype.multicategorical,
            'MULTICAT_2': Stype.multicategorical
        }
    }

    table_dict = {table.name: table for table in table_metadata_configs}

    table = table_dict[f'databricks.{table_name_2}']
    for col in table.cols:
        graph[table.name].add_col(col.name, dtype=col.dtype)
        assert col.dtype == expected_dtypes[table.name][col.name]

    graph[table.name].infer_stypes()

    for col in graph[table.name].cols:
        assert col.stype == expected_stypes[table.name][col.name]

    table = table_dict[f'databricks.{table_name_1}']
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

    stats_dict_2 = get_full_encoder_stats([
        graph[table_name_1].col('ARR_1'),
    ])

    stats_dict_3 = {
        col: [ColStatType.NUM_NA]
        for col in graph[table_name_2].metadata_config().all_cols()
    }

    # read table sample checks if the data can be cast to the assigned types
    # without failures
    df = connector.read_table_sample(
        table_name_1,
        cols=[col.base_config() for col in graph[table_name_1].cols])
    assert df.shape[0] == 20
    assert df.shape[1] == 10

    for col, stats in stats_dict_1.items():
        stats_result = compute_col_stats_python(df[col.name], stats)
        assert stats_result[ColStatType.NUM_NA].value == 0

    for col, stats in stats_dict_2.items():
        stats_result = compute_col_stats_python(df[col.name], stats)
        assert stats_result[ColStatType.SEQUENCE_MAX_LENGTH].value == 4

    df = connector.read_table_sample(
        table_name_2,
        cols=[col.base_config() for col in graph[table_name_2].cols])
    assert df.shape[0] == 50
    assert df.shape[1] == 7

    for col, stats in stats_dict_3.items():
        stats_result = compute_col_stats_python(df[col.name], stats)
        assert stats_result[ColStatType.NUM_NA].value == 0


@withDatabricksOauthCredentials
def test_databricks_connector_list_types(databricks_test_connector,
                                         monkeypatch):
    monkeypatch.setattr(
        ast_node,
        'ALLOW_ARRAY_TARGETS',
        True,
    )

    metadata = databricks_test_connector.get_table_metadata(
        ["movie_stringlist"])[0]
    genres_col = [col for col in metadata.cols if col.name == 'genres'][0]
    assert genres_col.dtype == Dtype.stringlist

    metadata = databricks_test_connector.get_table_metadata(["movie_intlist"
                                                             ])[0]
    genres_col = [col for col in metadata.cols if col.name == 'genres'][0]
    assert genres_col.dtype == Dtype.intlist

    databricks_test_connector.close()


@withDatabricksOauthCredentials
def test_databricks_read_sample(
        databricks_hm_connector: DatabricksConnector,
        databricks_test_schema_connector: DatabricksConnector):
    connector = databricks_hm_connector

    expected_shapes = {
        'customers': (10, 7),
        'articles': (10, 25),
        'transactions': (10, 5),
    }
    for table, expected_shape in expected_shapes.items():
        df = connector.read_table_sample(table, num_rows=10)
        assert expected_shape == df.shape

    with pytest.raises(ConnectorError,
                       match="Failed to read any rows from table"):
        databricks_test_schema_connector.read_table_sample(
            'empty_table', num_rows=10)


@withDatabricksOauthCredentials
def test_connector_validate_table_schema(
        databricks_hm_connector: DatabricksConnector):
    databricks_hm_connector.check_schema_exists(schema_name='hm')
    with pytest.raises(ConnectorError, match="Schema hm2 does not exist"):
        databricks_hm_connector.check_schema_exists(schema_name='hm2')
    # There is no stable schema in this shared workspace that the test
    # principal is guaranteed to be unable to use, so the no-permission
    # branch is covered by mock-only tests.
    databricks_hm_connector.check_table_write_permissions(
        'predictions', 'hm', InternalWriteMode.APPEND)


@withDatabricksOauthCredentials
def test_databricks_table_info(databricks_otto_connector: DatabricksConnector):
    connector = databricks_otto_connector
    from rest.utils.connector import get_sample_rows

    # num rows
    result = get_sample_rows(connector, None, 100, serialize_to_json=False,
                             source_table_name='events')
    assert (result.columns[1].name == 'ts'
            and result.columns[1].dtype == Dtype.int64)
    assert result.rows[0][1] and result.rows[0][1] != 'NaT'
    result = get_sample_rows(connector, None, 100, serialize_to_json=False,
                             source_table_name='events_float')
    assert (result.columns[6].name == 'ts_decimal'
            and result.columns[6].dtype == Dtype.float)
