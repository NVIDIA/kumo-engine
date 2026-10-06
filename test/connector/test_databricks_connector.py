# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from test.connector.artifact_staging_contract import (
    CancelledByRuntime,
    assert_indexed_parquet_parts,
    local_artifact_dataset,
)
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import databricks.sql.exc as DatabricksErrors
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from databricks.sdk.errors.platform import PermissionDenied
from kumo_core.sources.types import ReadMode
from kumo_core.types import ColStatType
from kumo_core.types.conversions import can_cast_safely
from kumo_core.types.schema import BaseColumnConfig
from kumo_databricks.sql import quote_identifier
from kumoapi.typing import Dtype, Stype
from pydantic import SecretStr
from urllib3.exceptions import HTTPError

import kumo.connector.databricks_connector as databricks_connector
import kumo.pquery.AST.ast_node as ast_node
from io_utils import fsutil
from io_utils.databricks import DatabricksClient, DatabricksConnectionInfo
from io_utils.errors import (
    WarehouseExtendedRetriableError,
    WarehouseRetriableError,
)
from kumo.config.graph import PrimaryKeyConfig
from kumo.connector.connector import (
    Connector,
    ConnectorError,
    SourceTableNotFoundError,
)
from kumo.connector.databricks_connector import (
    DatabricksConnector,
    validate_databricks_auth,
)
from kumo.connector.sql_connector import InternalWriteMode
from kumo.graph.graph import Graph
from kumo.graph.utils import get_full_encoder_stats
from kumo.stats.compute_stats_python import compute_col_stats_python
from kumo.testing.databricks import (
    DatabricksTableConnector,
    disable_databricks_env_credentials,
    withDatabricksOauthCredentials,
)
from statedb.config.typing import PANDAS_TO_SQL
from statedb.utils.errors import (
    ConnectorValidationDiagnosticCode,
    ConnectorValidationDiagnosticError,
)

# Staging path for seeding test data via COPY INTO. Must be OUTSIDE any
# Unity Catalog external location scope, so COPY INTO falls back to the
# workspace instance profile instead of requiring READ FILES grants.
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


@pytest.mark.parametrize('schema_name', [None, '  '])
def test_databricks_connector_requires_schema(schema_name):
    with pytest.raises(ConnectorError, match='schema'):
        DatabricksConnector(
            'databricks',
            host='https://workspace.cloud.databricks.com',
            cluster_id='',
            warehouse_id='warehouse-id',
            catalog='main',
            schema_name=schema_name,
        )


def test_validate_permissions_requires_nonblank_catalog():
    connector = DatabricksConnector(
        'databricks',
        host='https://workspace.cloud.databricks.com',
        cluster_id='',
        warehouse_id='warehouse-id',
        catalog=' ',
        schema_name='sales',
    )
    connector.check_schema_exists = Mock()

    with pytest.raises(ConnectorError, match='catalog'):
        connector.validate_permissions()

    connector.check_schema_exists.assert_not_called()


def test_databricks_connector_config():
    connector = DatabricksConnector('databricks', 'host', 'cluster',
                                    'warehouse', 'catalog', 'schema')

    assert asdict(connector.config()) == {
        '_target_': 'kumo.connector.databricks_connector.DatabricksConnector',
        'name': 'databricks',
        'host': 'https://host',
        'cluster_id': 'cluster',
        'warehouse_id': 'warehouse',
        'catalog': 'catalog',
        'schema_name': 'schema',
    }
    assert connector == Connector.from_config(connector.config())


def test_databricks_sql_connection_uses_configured_schema():
    connector = DatabricksConnector('databricks', 'host', 'cluster',
                                    'warehouse', 'catalog', 'target_schema')
    connection = MagicMock()

    with patch.object(
            DatabricksClient,
            'create_sql_connection',
            return_value=connection,
    ) as create_connection:
        result = connector._open(credentials={'pat': SecretStr('pat-token')})

    assert result is connection
    create_connection.assert_called_once_with(schema='target_schema')


@pytest.mark.parametrize(
    ('provider_text', 'expected_type'),
    [
        ('Max retries exceeded with url: /oidc token=oauth-secret-canary',
         WarehouseExtendedRetriableError),
        ('provider response token=connection-secret-canary',
         WarehouseRetriableError),
    ],
)
def test_databricks_connection_retry_error_redacts_provider_text(
        provider_text, expected_type):
    client = DatabricksClient(
        DatabricksConnectionInfo(
            host='https://example',
            cluster_id='cluster',
            warehouse_id='warehouse',
            catalog='catalog',
            credentials={},
        ))

    with pytest.raises(expected_type) as exc_info:
        client._process_exception_for_oauth_retry(RuntimeError(provider_text))

    assert provider_text not in str(exc_info.value)
    assert 'secret-canary' not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None


def test_databricks_connector_host():
    connector = DatabricksConnector('databricks', 'host', 'cluster',
                                    'warehouse', 'catalog', 'schema')
    assert connector.host == 'https://host'
    connector = DatabricksConnector('databricks', 'https://host/', 'cluster',
                                    'warehouse', 'catalog', 'schema')
    assert connector.host == 'https://host'
    connector = DatabricksConnector('databricks', 'https://host/login.html',
                                    'cluster', 'warehouse', 'catalog', 'schema')
    assert connector.host == 'https://host'
    # Test http scheme is converted to https
    connector = DatabricksConnector('databricks', 'http://host', 'cluster',
                                    'warehouse', 'catalog', 'schema')
    assert connector.host == 'https://host'


def test_select_query():
    connector = DatabricksConnector('databricks', 'host', 'cluster',
                                    'warehouse', 'catalog', 'schema')
    query = connector.get_query(table_name='test',
                                pkey=PrimaryKeyConfig('pkey'),
                                cols=[BaseColumnConfig('col1')])
    assert query == 'SELECT col1, pkey from `schema`.`test`'
    query = connector.get_query(table_name='test',
                                pkey=PrimaryKeyConfig('pkey'))
    assert query == 'SELECT * from `schema`.`test`'
    query = connector.get_query(table_name='test')
    assert query == 'SELECT * from `schema`.`test`'
    query = connector.get_query(table_name='test',
                                pkey=PrimaryKeyConfig('pkey'), cols=[])
    assert query == 'SELECT pkey from `schema`.`test`'
    with pytest.raises(ValueError):
        query = connector.get_query(table_name='test', cols=[])
    query = connector.get_query(table_name='test', limit=100)
    assert query == 'SELECT * from `schema`.`test` LIMIT 100'
    query = connector.get_query(table_name='test', limit=100, offset=10)
    assert query == 'SELECT * from `schema`.`test` LIMIT 100 OFFSET 10'


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


def test_databricks_query(databricks_hm_connector: DatabricksConnector):
    connector = databricks_hm_connector

    assert connector.get_query('customers') == \
        'SELECT * from `hm`.`customers`'
    assert connector.get_query('customers', limit=10) == \
        'SELECT * from `hm`.`customers` LIMIT 10'
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


def test_throw_error_if_no_credentials(
        databricks_hm_connector: DatabricksConnector):
    connector = databricks_hm_connector
    with disable_databricks_env_credentials():
        with pytest.raises(ConnectorError,
                           match=r"No credentials were provided"):
            connector.open()


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


def test_databricks_connection_info():
    with disable_databricks_env_credentials():
        connector = DatabricksConnector('databricks', 'host', 'cluster',
                                        'warehouse', 'catalog', 'schema')
        connection_info = connector._to_databricks_connection_info()
        assert connection_info is not None
        assert connection_info.cluster_id == 'cluster'
        assert connection_info.host == 'https://host'
        assert not connection_info.credentials
        connector.set_credentials({'pat': SecretStr('token')})
        connection_info = connector._to_databricks_connection_info()
        assert connection_info.credentials
        assert connection_info.credentials['pat'].get_secret_value() == 'token'
        connector.set_credentials({
            'client_id': SecretStr('client'),
            'client_secret': SecretStr('token')
        })
        connection_info = connector._to_databricks_connection_info()
        assert connection_info.credentials['client_id'].get_secret_value(
        ) == 'client'
        assert connection_info.credentials['client_secret'].get_secret_value(
        ) == 'token'


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


# validate_databricks_auth tests #############################################

_HOST = os.environ.get('DATABRICKS_HOST', 'workspace.cloud.databricks.com')


@pytest.mark.parametrize(
    ('credentials', 'client_args'),
    [
        (
            {
                'pat': SecretStr('my_token')
            },
            {
                'auth_type': 'pat',
                'token': 'my_token'
            },
        ),
        (
            {
                'client_id': SecretStr('cid'),
                'client_secret': SecretStr('csecret'),
            },
            {
                'auth_type': 'oauth-m2m',
                'client_id': 'cid',
                'client_secret': 'csecret'
            },
        ),
    ],
    ids=['pat', 'oauth'],
)
def test_validate_auth_modes(credentials, client_args):
    with patch('kumo.connector.databricks_connector.WorkspaceClient') as client:
        validate_databricks_auth(_HOST, credentials)

    client.assert_called_once_with(host='https://' + _HOST, **client_args)
    client.return_value.current_user.me.assert_called_once_with()


@pytest.mark.parametrize(
    ('credentials', 'provider_error', 'message'),
    [
        ({}, None, 'No Databricks credentials were provided'),
        ({
            'client_id': SecretStr('cid')
        }, None, 'Databricks OAuth credentials are incomplete'),
        ({
            'pat': SecretStr('bad_token')
        }, Exception('401 Unauthorized'), 'Failed to authenticate'),
        ({
            'pat': SecretStr('bad_token')
        }, PermissionDenied('Invalid access token'),
         'Databricks PAT authentication failed'),
        ({
            'pat': SecretStr('bad_token'),
            'client_id': SecretStr('cid'),
            'client_secret': SecretStr('csecret'),
        }, PermissionDenied('Invalid credentials'),
         'Databricks OAuth authentication failed'),
    ],
    ids=[
        'missing', 'partial_oauth', 'provider', 'denied_pat', 'oauth_precedence'
    ],
)
def test_validate_auth_failures(credentials, provider_error, message):
    with patch('kumo.connector.databricks_connector.WorkspaceClient') as client:
        if provider_error is not None:
            client.return_value.current_user.me.side_effect = provider_error
        with pytest.raises(ConnectorError, match=message):
            validate_databricks_auth(_HOST, credentials)

    if provider_error is None:
        client.assert_not_called()


def _make_unopened_databricks_connector() -> DatabricksConnector:
    """A DatabricksConnector that won't try to open a real warehouse.

    Sad-path tests below patch the methods that would otherwise hit the
    network; construction itself is cheap and doesn't authenticate.
    """
    return DatabricksConnector('databricks', host='https://example',
                               cluster_id='cl', warehouse_id='wh',
                               catalog='kumo_test', schema_name='schema')


def _make_output_databricks_connector(
    credentials: dict[str, SecretStr] | None = None, ) -> DatabricksConnector:
    connector = DatabricksConnector('databricks', host='https://example',
                                    cluster_id='cl', warehouse_id='wh',
                                    catalog='catalog',
                                    schema_name='authorized_schema')
    connector._context = Mock()
    connector._credentials = credentials
    return connector


@pytest.mark.parametrize(
    ('coordinate', 'expected'),
    [
        ('table-name', ('sales-schema', 'table-name')),
        (
            "sales-schema.订单@#$%20?;quote'back\\slash",
            ('sales-schema', "订单@#$%20?;quote'back\\slash"),
        ),
        ('main.sales-schema.tick`name', ('sales-schema', 'tick`name')),
        ('literal__kumo__table', ('sales-schema', 'literal__kumo__table')),
        ('bad name', None),
        ('bad/name', None),
        ('bad\x7f', None),
    ],
)
def test_databricks_table_coordinate_preserves_uc_identifier_rules(
        coordinate, expected):
    connector = DatabricksConnector(
        'databricks',
        host='https://example',
        cluster_id='cl',
        warehouse_id='wh',
        catalog='main',
        schema_name='sales-schema',
    )
    if expected is None:
        with pytest.raises(ConnectorError, match='does not permit'):
            connector._target_table_coordinate(coordinate)
    else:
        assert connector._target_table_coordinate(coordinate) == expected


def test_databricks_output_table_must_use_connector_schema():
    connector = _make_output_databricks_connector()

    assert connector.require_output_table_within_connector_target(
        'AUTHORIZED_SCHEMA__kumo__predictions') == (
            'AUTHORIZED_SCHEMA',
            'predictions',
        )
    with pytest.raises(ConnectorError, match='connector schema'):
        connector.require_output_table_within_connector_target(
            'other_schema__kumo__predictions')
    with pytest.raises(ConnectorError, match='exactly one schema'):
        connector.require_output_table_within_connector_target(
            'catalog__kumo__schema__kumo__predictions')


def test_databricks_actual_output_write_rejects_other_schema_before_io():
    connector = _make_output_databricks_connector()

    with pytest.raises(ConnectorError, match='connector schema'):
        connector.write_artifacts_to_table(
            'other_schema__kumo__predictions',
            's3://internal-artifacts/predictions/',
        )


@pytest.mark.parametrize(
    'artifact_path',
    [
        's3://internal-artifacts/predictions/',
        'abfss://artifacts@account.dfs.core.windows.net/predictions/',
        'gs://internal-artifacts/predictions/',
        '/Volumes/catalog/authorized_schema/external/predictions/',
        '/Volumes/other/authorized_schema/_kumo_staging/artifact-export/job/',
    ],
)
def test_databricks_output_rejects_non_managed_artifacts_before_io(
        artifact_path):
    connector = _make_output_databricks_connector()
    with (
            patch.object(connector, '_get_df_result') as run_sql,
            pytest.raises(ConnectorError),
    ):
        connector.write_artifacts_to_table(
            'authorized_schema__kumo__predictions',
            artifact_path,
        )
    run_sql.assert_not_called()


@pytest.mark.parametrize(
    'embeddings',
    [False, True],
    ids=['predictions', 'embeddings'],
)
def test_databricks_local_artifact_uses_parallel_managed_volume_put(
        tmp_path, embeddings):
    artifact_path = local_artifact_dataset(tmp_path, shards=2,
                                           embeddings=embeddings)
    connector = _make_output_databricks_connector()

    with (
            patch.object(connector, '_get_df_result',
                         return_value=pd.DataFrame()) as run_sql,
            patch.object(connector, '_run_volume_commands') as run_volume,
    ):
        connector.write_artifacts_to_table(
            'authorized_schema__kumo__predictions', str(artifact_path))

    put_call, remove_call = run_volume.call_args_list
    put_commands = put_call.args[0]
    assert put_call.kwargs['worker_count'] == 2
    assert put_call.kwargs['staging_allowed_local_path'] == str(artifact_path)
    destinations = [
        command.partition(' INTO ')[2].removesuffix(' OVERWRITE').strip("'")
        for command in put_commands
    ]
    prefix = assert_indexed_parquet_parts(destinations)
    assert prefix.startswith(
        '/Volumes/catalog/authorized_schema/_kumo_staging/artifact-export/')
    copy_sql = next(call_.args[1] for call_ in run_sql.call_args_list
                    if call_.args[1].startswith('COPY INTO'))
    assert f"FROM '{prefix}'" in copy_sql
    assert remove_call.args[0] == [
        f'REMOVE \'{path}\'' for path in reversed(destinations)
    ]
    queries = [call.args[1] for call in run_sql.call_args_list]
    assert ('DESCRIBE VOLUME '
            '`catalog`.`authorized_schema`.`_kumo_staging`') in queries


def test_databricks_local_artifact_rejects_empty_dataset(tmp_path):
    connector = _make_output_databricks_connector()
    with pytest.raises(ConnectorError, match='contains no Parquet files'):
        connector.write_artifacts_to_table(
            'authorized_schema__kumo__predictions',
            str(tmp_path),
        )


@pytest.mark.parametrize(
    'scenario',
    ['partial_upload', 'load_failure', 'base_exception'],
)
def test_databricks_managed_volume_failure_contract(tmp_path, caplog, scenario):
    artifact_path = local_artifact_dataset(
        tmp_path, shards=2 if scenario == 'partial_upload' else 1)
    connector = _make_output_databricks_connector()
    cancellation = CancelledByRuntime('cancelled')

    def run_sql(_context, query):
        if scenario == 'load_failure' and query.startswith('COPY INTO'):
            raise RuntimeError('copy failed')
        return pd.DataFrame()

    volume_error = {
        'partial_upload': RuntimeError('private provider detail'),
        'load_failure': None,
        'base_exception': cancellation,
    }[scenario]
    expected_type = (CancelledByRuntime
                     if scenario == 'base_exception' else ConnectorError)

    with (
            patch.object(
                connector,
                '_get_df_result',
                side_effect=run_sql,
            ) as run_query,
            patch.object(
                connector,
                '_run_volume_commands',
                side_effect=volume_error,
            ) as run_volume,
            pytest.raises(expected_type) as exc_info,
    ):
        connector.write_artifacts_to_table(
            'authorized_schema__kumo__predictions',
            str(artifact_path),
            replace=(scenario == 'load_failure'),
        )

    if scenario == 'base_exception':
        assert exc_info.value is cancellation
    else:
        assert 'configured Databricks output table' in str(exc_info.value)
    if scenario == 'partial_upload':
        assert 'private provider detail' not in caplog.text
        assert run_volume.call_count == 2
    if scenario == 'load_failure':
        assert not any(
            query.startswith('CREATE OR REPLACE TABLE '
                             '`authorized_schema`.`predictions`')
            for query in (call.args[1] for call in run_query.call_args_list))


def test_databricks_managed_volume_producer_manifest_uses_one_copy():
    connector = _make_output_databricks_connector()
    stage = ('/Volumes/catalog/authorized_schema/_kumo_staging/'
             'artifact-export/job-123')
    files = [f'{stage}/worker-{index:02d}.parquet' for index in range(8)]

    with (
            patch.object(connector, '_get_df_result',
                         return_value=pd.DataFrame()) as run_sql,
            patch.object(connector, '_run_volume_commands') as run_volume,
    ):
        connector.write_artifacts_to_table(
            'authorized_schema__kumo__predictions',
            stage,
            staged_files=files,
        )

    copy_queries = [
        call.args[1] for call in run_sql.call_args_list
        if call.args[1].startswith('COPY INTO')
    ]
    assert len(copy_queries) == 1
    assert f"FROM '{stage}'" in copy_queries[0]
    cleanup_commands = run_volume.call_args.args[0]
    assert cleanup_commands == [f"REMOVE '{path}'" for path in reversed(files)]


@pytest.mark.parametrize(
    ('files', 'message'),
    [
        (None, 'exact uploaded-file manifest'),
        ([
            '/Volumes/catalog/authorized_schema/_kumo_staging/'
            'artifact-export/other/part.parquet'
        ], 'outside its declared managed-volume directory'),
        ([
            '/Volumes/catalog/authorized_schema/_kumo_staging/'
            'artifact-export/job-123/part.csv'
        ], 'not Parquet'),
    ],
)
def test_databricks_managed_volume_manifest_fails_closed(files, message):
    connector = _make_output_databricks_connector()
    stage = ('/Volumes/catalog/authorized_schema/_kumo_staging/'
             'artifact-export/job-123')
    with (
            patch.object(connector, '_get_df_result') as run_sql,
            pytest.raises(ConnectorError, match=message),
    ):
        connector.write_artifacts_to_table(
            'authorized_schema__kumo__predictions',
            stage,
            staged_files=files,
        )
    run_sql.assert_not_called()


def test_databricks_volume_worker_count_is_memory_bounded(
        tmp_path, monkeypatch):
    files = []
    for index in range(6):
        file = tmp_path / f'part-{index}.parquet'
        file.write_bytes(b'x' * 32)
        files.append(file)
    monkeypatch.setenv('DATABRICKS_VOLUME_UPLOAD_CONCURRENCY', '6')
    monkeypatch.setenv('DATABRICKS_VOLUME_UPLOAD_MEMORY_BUDGET', '96')
    connector = _make_output_databricks_connector()
    assert connector._volume_worker_count(files) == 3


def test_databricks_volume_commands_use_persistent_parallel_sessions():
    connector = _make_output_databricks_connector({
        'pat':
        SecretStr('brokered-token'),
    })
    commands = [
        f"PUT 'local-{index}' INTO 'remote-{index}'" for index in range(6)
    ]
    executed: list[str] = []
    connections = []

    class Cursor:
        def execute(self, command):
            executed.append(command)

        def close(self):
            pass

    class Connection:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            connections.append(self)

        def cursor(self):
            return Cursor()

        def close(self):
            pass

    def client_factory(_connection_info):
        return SimpleNamespace(
            create_sql_connection=lambda **kwargs: Connection(kwargs))

    with patch.object(databricks_connector, 'DatabricksClient',
                      side_effect=client_factory):
        connector._run_volume_commands(
            commands,
            worker_count=3,
            staging_allowed_local_path='/tmp/artifacts',
        )

    assert set(executed) == set(commands)
    assert len(connections) == 3
    assert all(
        connection.kwargs == {
            'schema': 'authorized_schema',
            'staging_allowed_local_path': '/tmp/artifacts',
        } for connection in connections)


@pytest.mark.parametrize('exists', [True, False])
def test_databricks_managed_volume_reuses_or_creates_one_schema_volume(exists):
    connector = _make_output_databricks_connector()
    side_effect = ([pd.DataFrame()] if exists else [
        RuntimeError('[VOLUME_DOES_NOT_EXIST]'),
        pd.DataFrame(),
    ])
    with patch.object(
            connector,
            '_get_df_result',
            side_effect=side_effect,
    ) as run_sql:
        connector.ensure_managed_volume()

    queries = [call.args[1] for call in run_sql.call_args_list]
    assert queries[0] == ('DESCRIBE VOLUME '
                          '`catalog`.`authorized_schema`.`_kumo_staging`')
    assert queries[1:] == ([] if exists else [
        'CREATE VOLUME IF NOT EXISTS '
        '`catalog`.`authorized_schema`.`_kumo_staging`'
    ])


@pytest.mark.parametrize('missing', [False, True])
def test_databricks_managed_volume_permission_error_is_actionable(missing):
    connector = _make_output_databricks_connector()
    side_effect = ([
        RuntimeError('[UC_VOLUME_NOT_FOUND]'),
        RuntimeError('PERMISSION_DENIED')
    ] if missing else [RuntimeError('INSUFFICIENT_PRIVILEGES')])

    with patch.object(connector, '_get_df_result', side_effect=side_effect):
        with pytest.raises(
                ConnectorError,
                match=(r'_kumo_staging.*USE CATALOG.*USE SCHEMA.*'
                       r'READ VOLUME.*WRITE VOLUME'),
        ):
            connector.ensure_managed_volume()


def test_databricks_missing_schema_is_not_treated_as_a_missing_volume():
    connector = _make_output_databricks_connector()
    with (
            patch.object(
                connector,
                '_get_df_result',
                side_effect=RuntimeError('[SCHEMA_DOES_NOT_EXIST]'),
            ) as run_sql,
            pytest.raises(RuntimeError, match='SCHEMA_DOES_NOT_EXIST'),
    ):
        connector.ensure_managed_volume()
    run_sql.assert_called_once()


@pytest.mark.parametrize('missing', [False, True])
def test_databricks_clear_managed_volume_file_prefix(missing):
    connector = _make_output_databricks_connector()
    stage = connector.managed_volume_stage('job')
    listing = pd.DataFrame({
        'path': [
            f'dbfs:{stage}/shard-00003-00000000.parquet',
            f'dbfs:{stage}/shard-00003-00000001.parquet',
            f'dbfs:{stage}/shard-00004-00000000.parquet',
            f'dbfs:{stage}/unrelated.txt',
        ],
    })
    result = (RuntimeError('[PATH_NOT_FOUND]') if missing else listing)
    with patch.object(
            connector,
            '_get_df_result',
            side_effect=result if missing else None,
            return_value=None if missing else result,
    ) as run_sql, patch.object(connector, '_cleanup_volume_files') as cleanup:
        connector.clear_managed_volume_file_prefix(stage, 'shard-00003')

    if missing:
        cleanup.assert_not_called()
    else:
        assert run_sql.call_args.args[1] == f"LIST '{stage}'"
        cleanup.assert_called_once_with([
            f'{stage}/shard-00003-00000000.parquet',
            f'{stage}/shard-00003-00000001.parquet',
        ])


def test_databricks_reads_bounded_stats_sample_from_managed_volume():
    connector = _make_output_databricks_connector()
    stage = connector.managed_volume_stage('stats-job')
    sample = pd.DataFrame({
        'ENTITY': [1, 2],
        'SCORE': [0.25, 0.75],
    })
    columns = [
        BaseColumnConfig(name='ENTITY', stype=Stype.ID, dtype=Dtype.int64),
        BaseColumnConfig(name='SCORE', stype=Stype.numerical,
                         dtype=Dtype.float64),
    ]
    with patch.object(connector, '_get_df_result',
                      return_value=sample) as run_sql:
        result = connector.read_managed_volume_parquet_sample(
            stage, columns, num_rows=10_000)

    assert result is sample
    assert run_sql.call_args.args[1] == (
        'SELECT `ENTITY`, `SCORE` '
        f"FROM read_files('{stage}', format => 'parquet') LIMIT 10000")


@pytest.mark.parametrize('sql', [None, 'SELECT order_id AS ID FROM orders'])
def test_databricks_custom_table_export_goes_through_the_shared_reader(sql):
    """A custom training or prediction table is staged to Kumo-owned parquet
    by the same source reader graph ingest uses -- no source spec and no
    credential file."""
    connector = _make_unopened_databricks_connector()
    cols = {'ID': Dtype.int64}

    with (
            patch.object(connector, '_source_reader') as reader,
            patch('kumo.connector.source_bridge.stage_source') as stage,
    ):
        reader.return_value = 'reader'
        connector.export_table_read_only('orders', '/dest', cols, sql)

    assert stage.call_args.args == ('reader', 'orders', '/dest')
    # Read-only is the method's contract, so the mode is pinned and the
    # reader is built without a staging volume: a staged read would
    # `CREATE VOLUME` and `INSERT OVERWRITE DIRECTORY` into the customer's
    # Unity Catalog, which is a write to the source account.
    reader.assert_called_once_with(staged=False)
    assert stage.call_args.kwargs == {
        'columns': cols,
        'sql': sql,
        'mode': ReadMode.STREAMED,
        # Supplied so the reader does not spend a `DESCRIBE HISTORY`
        # deriving a stamp this path discards.
        'version': 'custom-source',
    }


def test_databricks_custom_table_export_resolves_columns_when_unset():
    """``cols=None`` is the warehouse-branch call shape, so the projection is
    the source's own schema -- a table is described, a query is not run."""
    connector = _make_unopened_databricks_connector()
    metadata = SimpleNamespace(
        cols=[SimpleNamespace(name='ID', dtype=Dtype.int64)])

    with (
            patch.object(connector, '_source_reader', return_value='reader'),
            patch.object(connector, 'get_table_metadata',
                         return_value=[metadata]) as table_metadata,
            patch('kumo.connector.source_bridge.stage_source') as stage,
    ):
        connector.export_table_read_only('orders', '/dest')

    table_metadata.assert_called_once_with(['orders'])
    assert stage.call_args.kwargs['columns'] == {'ID': Dtype.int64}


@pytest.mark.parametrize('sql', [None, 'SELECT * FROM weights'])
def test_databricks_custom_table_export_projects_only_readable_columns(sql):
    """A column of a type Kumo has no ``Dtype`` for -- ``STRUCT``, ``MAP``,
    ``VARIANT``, ``INTERVAL`` -- is never selected. Resolving the whole schema
    would decode and persist a column the user never referenced, on the
    warehouse's clock and into Kumo's storage.

    Restores the invariant the retired direct path stated: unselected source
    columns do not affect ingestion.
    """
    connector = _make_unopened_databricks_connector()
    metadata = SimpleNamespace(cols=[
        SimpleNamespace(name='ENTITY1', dtype=Dtype.int64),
        SimpleNamespace(name='TARGET', dtype=Dtype.float64),
        SimpleNamespace(name='W', dtype=Dtype.float64),
        SimpleNamespace(name='audit_blob', dtype=Dtype.unsupported),
    ])

    with (
            patch.object(connector, '_source_reader', return_value='reader'),
            patch.object(connector, 'get_table_metadata',
                         return_value=[metadata]),
            patch.object(connector, 'get_query_metadata',
                         return_value=metadata),
            patch('kumo.connector.source_bridge.stage_source') as stage,
    ):
        connector.export_table_read_only('weights', '/dest', None, sql)

    assert stage.call_args.kwargs['columns'] == {
        'ENTITY1': Dtype.int64,
        'TARGET': Dtype.float64,
        'W': Dtype.float64,
    }


@pytest.mark.parametrize('staged, expect_volume', [(True, True),
                                                   (False, False)])
def test_databricks_source_reader_only_provisions_a_volume_when_staging(
        staged, expect_volume):
    """``_staging_volume`` calls ``ensure_managed_volume``, which is a
    ``CREATE VOLUME`` in the customer's Unity Catalog. A read-only export must
    not reach it at all -- declining the offer after provisioning would have
    already made the write."""
    connector = _make_unopened_databricks_connector()
    connector._context = MagicMock()

    with (
            patch('kumo_databricks.DatabricksSourceReader') as cls,
            patch.object(connector, '_staging_volume',
                         return_value=('/Volumes/v', 'client')) as volume,
    ):
        connector._source_reader(staged=staged)

    assert volume.called is expect_volume
    assert (cls.call_args.kwargs['staging_volume_path']
            is not None) is expect_volume
    assert (cls.call_args.kwargs['workspace_client']
            is not None) is expect_volume


def test_databricks_custom_table_export_rejects_an_all_unreadable_source():
    """Dropping every column would hand the reader an empty projection, which
    it reports as a programming error rather than as the user's schema."""
    connector = _make_unopened_databricks_connector()
    metadata = SimpleNamespace(cols=[
        SimpleNamespace(name='blob', dtype=Dtype.unsupported),
    ])

    with (
            patch.object(connector, '_source_reader', return_value='reader'),
            patch.object(connector, 'get_table_metadata',
                         return_value=[metadata]),
            patch('kumo.connector.source_bridge.stage_source') as stage,
    ):
        with pytest.raises(ConnectorError, match='type Kumo can read'):
            connector.export_table_read_only('weights', '/dest')

    stage.assert_not_called()


@pytest.mark.parametrize('name', [
    'weights',
    'schema.weights',
    'kumo_test.schema.weights',
])
def test_databricks_custom_table_export_accepts_a_qualified_name(name):
    """The reader qualifies with its own catalog and schema, so it is handed
    the bare table. Passing the raw coordinate through would build
    ``kumo_test.schema.'kumo_test.schema.weights'`` and report a table the
    user does have as missing."""
    connector = _make_unopened_databricks_connector()
    metadata = SimpleNamespace(
        cols=[SimpleNamespace(name='ID', dtype=Dtype.int64)])

    with (
            patch.object(connector, '_source_reader', return_value='reader'),
            patch.object(connector, 'get_table_metadata',
                         return_value=[metadata]),
            patch('kumo.connector.source_bridge.stage_source') as stage,
    ):
        connector.export_table_read_only(name, '/dest')

    assert stage.call_args.args == ('reader', 'weights', '/dest')


def test_databricks_graph_table_export_accepts_a_qualified_name():
    """`export_table` reaches the same reader by the same coordinate."""
    connector = _make_unopened_databricks_connector()
    table_fp = SimpleNamespace(cols={'ID': Dtype.int64})

    with (
            patch.object(connector, '_source_reader', return_value='reader'),
            patch('kumo.connector.source_bridge.read_source') as read,
    ):
        connector.export_table('kumo_test.schema.weights', table_fp, 'v1')

    assert read.call_args.args[:2] == ('reader', 'weights')


def test_databricks_custom_table_export_rejects_a_foreign_coordinate():
    """A coordinate outside the connector's namespace is refused by name
    rather than by a missing-table error from the warehouse."""
    connector = _make_unopened_databricks_connector()

    with patch('kumo.connector.source_bridge.stage_source') as stage:
        with pytest.raises(ConnectorError,
                           match='does not match the connector catalog'):
            connector.export_table_read_only('other.schema.weights', '/dest')

    stage.assert_not_called()


def test_databricks_output_write_sanitizes_provider_error(tmp_path, caplog):
    connector = _make_output_databricks_connector()
    secret = 'provider-payload-with-client-secret'
    artifact_path = local_artifact_dataset(tmp_path)

    with patch.object(connector, '_get_df_result',
                      side_effect=RuntimeError(secret)):
        with pytest.raises(
                ConnectorError,
                match='configured Databricks output table') as exc_info:
            connector.write_artifacts_to_table(
                'authorized_schema__kumo__predictions',
                str(artifact_path),
            )

    assert secret not in exc_info.value.get_user_message()
    assert secret not in caplog.text


def test_databricks_output_write_does_not_relabel_an_unrelated_valueerror(
        tmp_path, caplog):
    """``execute`` and ``run`` do real provider work under
    ``write_artifacts_to_table``, and either can raise a bare ``ValueError``
    of their own that has nothing to do with the writer's own validation. A
    ``ValueError`` that reaches the warehouse layer, carrying provider detail,
    must be sanitized the same way any other unrelated failure already is --
    not passed through as the writer's own deliberate refusal message, which
    is the one case this boundary echoes verbatim."""
    connector = _make_output_databricks_connector()
    secret = 'sql-text-with-endpoint-and-local-path-detail'
    artifact_path = local_artifact_dataset(tmp_path)

    with patch.object(connector, '_get_df_result',
                      side_effect=ValueError(secret)):
        with pytest.raises(
                ConnectorError,
                match='configured Databricks output table') as exc_info:
            connector.write_artifacts_to_table(
                'authorized_schema__kumo__predictions',
                str(artifact_path),
            )

    assert secret not in exc_info.value.get_user_message()
    assert secret not in caplog.text


def test_databricks_metadata_normalizes_in_scope_qualified_table():
    connector = _make_unopened_databricks_connector()
    metadata = SimpleNamespace(table_name='orders', source_name='orders')

    with patch.object(
            connector,
            '_fetch_table_metadata',
            return_value={'orders': metadata},
    ) as fetch:
        result = connector._get_table_metadata(
            ['kumo_test.schema.orders'],
            skip_inferral=True,
        )

    fetch.assert_called_once_with(['orders'], True, None)
    assert result[0].table_name == 'kumo_test.schema.orders'
    assert result[0].source_name == 'kumo_test.schema.orders'


def test_databricks_metadata_ignore_errors_preserves_available_tables():
    connector = _make_unopened_databricks_connector()
    metadata = SimpleNamespace(table_name='orders', source_name='orders')

    with patch.object(
            connector,
            '_fetch_table_metadata',
            return_value={'orders': metadata},
    ):
        result = connector._get_table_metadata(
            ['missing', 'kumo_test.schema.orders'],
            skip_inferral=True,
            ignore_errors=True,
        )

    assert [table.table_name for table in result] == ['kumo_test.schema.orders']
    assert result[0].source_name == 'kumo_test.schema.orders'


def test_databricks_metadata_missing_table_is_typed():
    connector = _make_unopened_databricks_connector()

    with (
            patch.object(connector, '_fetch_table_metadata', return_value={}),
            pytest.raises(SourceTableNotFoundError,
                          match="Source table 'missing'"),
    ):
        connector._get_table_metadata(['missing'], skip_inferral=True)


@pytest.mark.parametrize(
    ('source_name', 'message'),
    [
        ('other.schema.orders', 'connector catalog'),
        ('kumo_test.other.orders', 'connector schema'),
        ('a.b.c.d', 'unqualified'),
    ],
)
def test_databricks_rejects_out_of_scope_qualified_table(source_name, message):
    connector = _make_unopened_databricks_connector()

    with (
            patch.object(connector, 'open'),
            pytest.raises(ConnectorError, match=message),
    ):
        connector.table_exists(source_name)


@pytest.fixture
def _no_retry_sleep():
    """`io_utils.retry` calls `time.sleep` between attempts. Mock it
    out so the @retry-decorated tests don't actually wait 6+ seconds
    each — the retry counter still increments correctly.
    """
    with patch('io_utils.retry.time.sleep'):
        yield


def test_databricks_check_schema_retries_http_error_then_succeeds(
        _no_retry_sleep, caplog):
    """`_check_schema` is decorated with `@retry(WarehouseRetriableError,
    tries=3)`. Transient `HTTPError`s are converted into
    `WarehouseRetriableError` and retried up to 3 attempts. This locks
    in the retry budget so accidentally lowering it (e.g. tries=1)
    would fail the test.
    """
    connector = _make_unopened_databricks_connector()

    connector._context = Mock()
    fake_cursor = Mock()
    fake_cursor.__enter__ = Mock(return_value=fake_cursor)
    fake_cursor.__exit__ = Mock(return_value=False)
    connector._context.cursor = Mock(return_value=fake_cursor)

    fake_df_result = Mock()
    fake_df_result.fetchall_arrow.return_value.to_pandas.return_value = (
        pd.DataFrame({'ActionType': ['USAGE']}))

    call_count = {'n': 0}
    provider_secret = 'provider-retry-bearer-secret'

    def _fake_execute(query):
        call_count['n'] += 1
        if call_count['n'] <= 2:
            raise HTTPError(f"503 transient {provider_secret}")
        return fake_df_result

    fake_cursor.execute = Mock(side_effect=_fake_execute)

    with patch.object(connector, '_retrieve_current_user',
                      return_value='svc-account'):
        # No exception should escape after retry exhausts transient
        # failures.
        connector._check_schema('my_schema')
    assert call_count['n'] == 3, (
        f"expected 3 calls (2 retries + 1 success); got {call_count['n']}")
    assert provider_secret not in caplog.text


def test_databricks_check_schema_quotes_exact_target_identifiers():
    connector = DatabricksConnector(
        name='databricks',
        host='workspace.example.com',
        cluster_id='cluster-id',
        warehouse_id='',
        catalog='cat`alog',
        schema_name='sche`ma',
    )
    context = MagicMock()
    cursor = context.cursor.return_value.__enter__.return_value
    cursor.execute.return_value.fetchall_arrow.return_value.to_pandas.return_value = (
        pd.DataFrame({'ActionType': ['USE SCHEMA']}))
    connector._context = context

    with patch.object(connector, '_retrieve_current_user',
                      return_value='user`name'):
        connector._check_schema('sche`ma')

    cursor.execute.assert_called_once_with(
        'show grants `user``name` on SCHEMA `cat``alog`.`sche``ma`')


@pytest.mark.parametrize('table_exists', [False, True])
def test_databricks_write_check_allows_inherited_permission(
    caplog,
    table_exists,
):
    connector = _make_unopened_databricks_connector()
    context = MagicMock()
    cursor = context.cursor.return_value.__enter__.return_value
    schema_result = MagicMock()
    schema_result.fetchall_arrow.return_value.to_pandas.return_value = (
        pd.DataFrame({'ActionType': []}))
    table_result = MagicMock()
    table_result.fetchall_arrow.return_value.to_pandas.return_value = (
        pd.DataFrame({'ActionType': ['SELECT']}))
    cursor.execute.side_effect = [
        schema_result,
        (table_result if table_exists else
         DatabricksErrors.ServerOperationError('TABLE_DOES_NOT_EXIST')),
    ]
    connector._context = context

    with patch.object(connector, '_retrieve_current_user',
                      return_value='service-principal'):
        connector._check_table(
            'existing_predictions' if table_exists else 'new_predictions',
            connector.schema_name,
        )

    assert 'may inherit permission' in caplog.text


@pytest.mark.parametrize(
    ('source_name', 'metadata', 'expected'),
    [
        ('orders', {}, False),
        ('orders', {
            'orders': MagicMock()
        }, True),
        ('tpch.orders', {
            'orders': MagicMock()
        }, True),
        ('samples.tpch.orders', {
            'orders': MagicMock()
        }, True),
    ],
)
def test_databricks_table_exists_has_boolean_missing_contract(
        source_name, metadata, expected):
    connector = _sample_schema_connector()
    with (
            patch.object(connector, 'open'),
            patch.object(connector, '_fetch_table_metadata',
                         return_value=metadata) as fetch,
    ):
        assert connector.table_exists(source_name) is expected

    fetch.assert_called_once_with(
        ['orders'],
        skip_inferral=True,
        schema_name='tpch',
    )


def _databricks_cursor(*, rows=None, execute_error=None):
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    if execute_error is not None:
        cursor.execute.side_effect = execute_error
    else:
        result = cursor.execute.return_value
        result.fetchall.return_value = rows or []
        result.fetchone.return_value = (1, )
    return cursor


def _databricks_server_error(code, detail='provider detail'):
    return DatabricksErrors.ServerOperationError(f'[{code}] {detail}')


def test_databricks_missing_table_data_version_is_typed():
    connector = DatabricksConnector(
        name='databricks',
        host='workspace.example.com',
        cluster_id='',
        warehouse_id='warehouse-id',
        catalog='samples',
        schema_name='tpch',
    )
    connector._context = MagicMock()
    error = _databricks_server_error('TABLE_OR_VIEW_NOT_FOUND.WITH_SUGGESTION')

    with (
            patch.object(connector, '_run_sql', side_effect=error),
            pytest.raises(SourceTableNotFoundError,
                          match="Source table 'missing_table'"),
    ):
        connector._get_data_version('missing_table')


@pytest.mark.parametrize(('code', 'level'), [
    ('EXPECT_TABLE_NOT_VIEW', 'INFO'),
    ('UNEXPECTED_HISTORY_FAILURE', 'WARNING'),
])
def test_databricks_history_fallback_log_preserves_severity(
        code, level, monkeypatch, caplog):
    connector, cursor = _metadata_connector()
    cursor.execute.side_effect = _databricks_server_error(
        code, 'private detail')
    monkeypatch.setattr(databricks_connector, 'hour_version', lambda: '1234')
    with caplog.at_level('INFO', logger=databricks_connector.__name__):
        assert connector._get_data_version('orders') == '1234'
    messages = [
        record for record in caplog.records
        if record.name == databricks_connector.__name__
    ]
    assert len(messages) == 1
    assert messages[0].levelname == level
    assert 'private detail' not in caplog.text


@pytest.mark.parametrize('code', [
    'PARSE_SYNTAX_ERROR',
    'TABLE_OR_VIEW_NOT_FOUND.WITH_SUGGESTION',
    'UNBOUND_SQL_PARAMETER',
    'UNRESOLVED_TABLE_VALUED_FUNCTION',
    'DELTA_TIMESTAMP_GREATER_THAN_COMMIT',
    'INVALID_TEMPORAL_VERSION',
])
def test_databricks_query_diagnostics_allowlist(code):
    detail = databricks_connector._databricks_query_error_detail(
        _databricks_server_error(code, 'secret provider detail'))
    assert detail is not None
    assert 'secret provider detail' not in detail


def test_databricks_query_diagnostics_redact_unknown_code():
    assert databricks_connector._databricks_query_error_detail(
        _databricks_server_error('UNKNOWN_CODE', 'secret')) is None


def test_databricks_binary_types_keep_binary_contract():
    connector = _sample_schema_connector()
    assert can_cast_safely(Dtype.binary, Dtype.binary)
    assert connector.databricks_to_dtype('BINARY') is Dtype.binary
    assert connector.databricks_to_dtype('ARRAY<BINARY>') is Dtype.unsupported


@pytest.mark.parametrize(('error', 'message'), [
    (_databricks_server_error('UNBOUND_SQL_PARAMETER',
                              'secret'), 'unbound parameters'),
    (databricks_connector.DecimalOutOfBoundsError('wide_decimal'),
     'wide_decimal.*Cast the column to DOUBLE or STRING'),
])
def test_databricks_sample_query_surfaces_safe_diagnostic(
        monkeypatch, error, message):
    connector = _sample_schema_connector()
    monkeypatch.setattr(connector, '_get_df_result', Mock())
    monkeypatch.setattr(connector, '_run_sql', Mock(side_effect=error))
    with pytest.raises(ConnectorError, match=message):
        connector.sample_query('SELECT :value', 1)


def _sample_schema_connector(*fallback_cursors, catalog='samples'):
    connector = DatabricksConnector(
        name='databricks',
        host='workspace.example.com',
        cluster_id='',
        warehouse_id='warehouse-id',
        catalog=catalog,
        schema_name='tpch',
    )
    grant_cursor = _databricks_cursor(
        execute_error=_databricks_server_error('SAMPLE_TABLE_PERMISSIONS'))
    context = MagicMock()
    context.cursor.side_effect = [grant_cursor, *fallback_cursors]
    connector._context = context
    return connector


def test_databricks_sample_schema_uses_visibility_and_read_fallback():
    schema_cursor = _databricks_cursor(
        rows=[('information_schema', ), ('tpch', )])
    table_cursor = _databricks_cursor(rows=[('tpch', 'region', False)])
    read_cursor = _databricks_cursor()
    connector = _sample_schema_connector(schema_cursor, table_cursor,
                                         read_cursor)

    with patch.object(connector, '_retrieve_current_user',
                      return_value='service-principal'):
        connector.check_schema_exists('tpch')

    schema_cursor.execute.assert_called_once_with('SHOW SCHEMAS IN `samples`')
    table_cursor.execute.assert_called_once_with(
        'SHOW TABLES IN `samples`.`tpch`')
    read_cursor.execute.assert_called_once_with(
        'SELECT 1 FROM `samples`.`tpch`.`region` LIMIT 1')


@pytest.mark.parametrize(
    ('code', 'expected_message', 'expected_diagnostic'),
    [
        ('PERMISSION_DENIED', 'USE CATALOG permission',
         ConnectorValidationDiagnosticCode.TARGET_ACCESS_DENIED),
        ('CATALOG_DOES_NOT_EXIST', 'Catalog .* does not exist',
         ConnectorValidationDiagnosticCode.TARGET_NOT_FOUND),
        ('SCHEMA_DOES_NOT_EXIST', 'Schema tpch does not exist',
         ConnectorValidationDiagnosticCode.TARGET_NOT_FOUND),
    ],
)
def test_databricks_non_sample_grant_failures_remain_failures(
        code, expected_message, expected_diagnostic):
    connector = _make_unopened_databricks_connector()
    connector.schema_name = 'tpch'
    cursor = _databricks_cursor(execute_error=_databricks_server_error(code))
    context = MagicMock()
    context.cursor.return_value = cursor
    connector._context = context

    with patch.object(connector, '_retrieve_current_user',
                      return_value='service-principal'):
        with pytest.raises(ConnectorError, match=expected_message) as caught:
            connector.check_schema_exists('tpch')

    # The detailed public message is preserved for non-masked callers while a
    # safe diagnostic category (or None) rides alongside it for the masking
    # boundaries to surface.
    assert getattr(caught.value, 'diagnostic_code', None) == expected_diagnostic


def test_databricks_error_class_extraction():
    """The structured error class is read from the provider's own token, in
    both the Unity Catalog (``ErrorClass=``) and legacy (leading ``[...]``)
    message forms, and never from free-form detail."""
    extract = databricks_connector._databricks_error_class
    E = DatabricksErrors.ServerOperationError
    # Unity Catalog: code embedded in the leading metadata bracket.
    assert extract(
        E("[RequestId=abc ErrorClass=SCHEMA_DOES_NOT_EXIST."
          "RESOURCE_DOES_NOT_EXIST] Schema 'c.s' does not exist.")
    ) == 'SCHEMA_DOES_NOT_EXIST.RESOURCE_DOES_NOT_EXIST'
    # Legacy / test form: clean leading token.
    assert extract(E('[SECURABLE_DOES_NOT_EXIST] Database(s) does not exist.')
                   ) == 'SECURABLE_DOES_NOT_EXIST'
    # A token in the free-form detail is NOT a structured class.
    assert extract(E("Schema 'main.PERMISSION_DENIED' does not exist.")) is None


def test_databricks_legacy_metastore_falls_back_to_text_for_not_found():
    """hive_metastore omits a structured code, so the text is the only signal
    available there; it still classifies as TARGET_NOT_FOUND."""
    connector = _make_unopened_databricks_connector()
    connector.catalog = 'hive_metastore'
    connector.schema_name = 'tpch'
    # No leading ``[CODE]`` token, so the structured code is absent.
    error = DatabricksErrors.ServerOperationError(
        'Schema tpch does not exist in the metastore')
    cursor = _databricks_cursor(execute_error=error)
    context = MagicMock()
    context.cursor.return_value = cursor
    connector._context = context

    with patch.object(connector, '_retrieve_current_user',
                      return_value='service-principal'):
        with pytest.raises(ConnectorError,
                           match='Schema tpch does not exist') as caught:
            connector.check_schema_exists('tpch')

    assert getattr(caught.value, 'diagnostic_code',
                   None) == (ConnectorValidationDiagnosticCode.TARGET_NOT_FOUND)


def test_databricks_sample_error_on_customer_catalog_fails_with_safe_code():
    connector = _sample_schema_connector(catalog='main')

    with patch.object(connector, '_retrieve_current_user',
                      return_value='service-principal'):
        with pytest.raises(ConnectorValidationDiagnosticError) as caught:
            connector.check_schema_exists('tpch')

    assert caught.value.code == (
        ConnectorValidationDiagnosticCode.DATABRICKS_SAMPLE_TABLE_PERMISSIONS)


def test_databricks_sample_fallback_requires_exact_schema_visibility():
    connector = _sample_schema_connector(
        _databricks_cursor(rows=[('tpch_private', )]))

    with patch.object(connector, '_retrieve_current_user',
                      return_value='service-principal'):
        with pytest.raises(ConnectorValidationDiagnosticError):
            connector.check_schema_exists('tpch')


def test_databricks_sample_fallback_requires_a_successful_table_read():
    connector = _sample_schema_connector(
        _databricks_cursor(rows=[('tpch', )]),
        _databricks_cursor(rows=[('tpch', 'region', False)]),
        _databricks_cursor(
            execute_error=_databricks_server_error('PERMISSION_DENIED')),
    )

    with patch.object(connector, '_retrieve_current_user',
                      return_value='service-principal'):
        with pytest.raises(ConnectorValidationDiagnosticError):
            connector.check_schema_exists('tpch')


def test_databricks_sample_fallback_rejects_unstructured_code_lookalike():
    connector = _make_unopened_databricks_connector()
    connector.catalog = 'samples'
    connector.schema_name = 'tpch'
    cursor = _databricks_cursor(
        execute_error=DatabricksErrors.ServerOperationError(
            'arbitrary SAMPLE_TABLE_PERMISSIONS'))
    context = MagicMock()
    context.cursor.return_value = cursor
    connector._context = context

    with patch.object(connector, '_retrieve_current_user',
                      return_value='service-principal'):
        with pytest.raises(ConnectorError,
                           match="Unable to verify schema 'tpch'"):
            connector.check_schema_exists('tpch')


def test_databricks_check_schema_retries_exhausted_raises(_no_retry_sleep):
    """If `HTTPError` keeps happening, the `@retry` decorator gives up
    after 3 attempts and raises `WarehouseRetriableError`. Callers
    (`check_schema_exists` and `check_table_write_permissions`) wrap
    that into `ConnectorError` — but `_check_schema` itself surfaces
    the retriable error. Both layers matter; this test pins
    `_check_schema`'s budget so an accidental tries=N change is caught.
    """
    connector = _make_unopened_databricks_connector()

    connector._context = Mock()
    fake_cursor = Mock()
    fake_cursor.__enter__ = Mock(return_value=fake_cursor)
    fake_cursor.__exit__ = Mock(return_value=False)
    connector._context.cursor = Mock(return_value=fake_cursor)

    fake_cursor.execute = Mock(side_effect=HTTPError("503 persistent"))

    with patch.object(connector, '_retrieve_current_user',
                      return_value='svc-account'):
        with pytest.raises(WarehouseRetriableError,
                           match="Databricks request failed"):
            connector._check_schema('my_schema')
    assert fake_cursor.execute.call_count == 3, (
        f"expected exactly 3 attempts before giving up; got "
        f"{fake_cursor.execute.call_count}")


def test_databricks_check_schema_exists_wraps_retriable_as_connector_error(
        _no_retry_sleep):
    """`check_schema_exists` (public surface) wraps the exhausted
    `WarehouseRetriableError` into a `ConnectorError` with a stable
    user-facing message. The operator sees a connector error, not a
    bare retriable. Confirms both layers in the @retry chain.
    """
    connector = _make_unopened_databricks_connector()

    connector._context = Mock()
    fake_cursor = Mock()
    fake_cursor.__enter__ = Mock(return_value=fake_cursor)
    fake_cursor.__exit__ = Mock(return_value=False)
    connector._context.cursor = Mock(return_value=fake_cursor)
    fake_cursor.execute = Mock(side_effect=HTTPError("503 persistent"))

    with patch.object(connector, '_retrieve_current_user',
                      return_value='svc-account'):
        with pytest.raises(ConnectorError,
                           match="Unable to verify schema 'my_schema'"):
            connector.check_schema_exists('my_schema')


def test_databricks_array_and_timestamp_type_surface_is_pinned():
    """The exact set `_readable_columns` drops, because the docstring there
    and the PR description both enumerate it.

    `databricks_to_dtype` matches array types by exact string, so the
    allowlist is five spellings and everything else in the family resolves to
    `Dtype.unsupported` -- which this PR promoted from a dtype label into the
    thing that decides whether a column is selected at all. Widening the
    allowlist is the fix and is deliberately not made here; this test exists
    so that widening it is a visible change rather than a silent one.
    """
    connector = _make_unopened_databricks_connector()
    resolves = {
        'ARRAY<FLOAT>': Dtype.floatlist,
        'ARRAY<DOUBLE>': Dtype.floatlist,
        'ARRAY<INT>': Dtype.intlist,
        'ARRAY<BIGINT>': Dtype.intlist,
        'ARRAY<STRING>': Dtype.stringlist,
        'TIMESTAMP': Dtype.time,
        'TIMESTAMP_NTZ': Dtype.time,
    }
    for data_type, expected in resolves.items():
        assert connector.databricks_to_dtype(data_type) == expected, data_type

    for data_type in (
            'ARRAY<SMALLINT>',
            'ARRAY<TINYINT>',
            'ARRAY<BOOLEAN>',
            'ARRAY<DATE>',
            'ARRAY<TIMESTAMP>',
            'ARRAY<ARRAY<INT>>',
            'ARRAY<STRUCT<a:INT>>',
            'TIMESTAMP_LTZ',
            'STRUCT<a:INT>',
            'MAP<STRING,INT>',
            'VARIANT',
            'INTERVAL DAY TO SECOND',
    ):
        assert connector.databricks_to_dtype(
            data_type) == Dtype.unsupported, data_type


def test_an_array_of_decimal_is_mislabelled_not_dropped():
    """`ARRAY<DECIMAL(p,s)>` is the one member of the family that does not
    reach `Dtype.unsupported`, so `_readable_columns` selects it.

    The `'DECIMAL' in data_type` branch is a substring test and runs first,
    but its `re.match` anchors at position 0 and so fails on the `ARRAY<`
    prefix; the branch's own fallback is `Dtype.int64`. A list column is
    therefore declared a scalar integer. Pre-existing and untouched here --
    the drop filter cannot catch it precisely because it is not labelled
    unsupported -- and pinned so the follow-up that widens the allowlist has
    to decide about it.
    """
    connector = _make_unopened_databricks_connector()
    assert connector.databricks_to_dtype('ARRAY<DECIMAL(10,2)>') == Dtype.int64


@pytest.mark.parametrize('data_type,expected', [
    ('INT', Dtype.int64),
    ('int', Dtype.int64),
    (' INT ', Dtype.int64),
    ('DECIMAL(10,2)', Dtype.float),
    (' DECIMAL(10,2)', Dtype.float),
    (' decimal (10, 2) ', Dtype.float),
    ('ARRAY<DECIMAL(10,2)>', Dtype.int64),
])
def test_databricks_hosted_type_name_normalization(data_type, expected):
    connector = DatabricksConnector.__new__(DatabricksConnector)
    from kumo_databricks.types import dtype_for

    assert connector.databricks_to_dtype(data_type) is expected
    assert dtype_for(data_type) is expected


def test_query_schema_preserves_order_and_quoted_names():
    from unittest.mock import MagicMock

    connector = _make_unopened_databricks_connector()
    connector._context = MagicMock()
    cursor = connector._context.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [('', 'bigint'), ('  ', 'string'),
                                    (' mixed ', 'bigint'),
                                    (' mixed ', 'double')]

    metadata = connector.get_query_metadata('SELECT 1, 1.5')

    assert [(col.name, col.dtype)
            for col in metadata.cols] == [(' mixed ', Dtype.int64),
                                          (' mixed ', Dtype.float)]
    cursor.execute.assert_called_once_with(
        'DESCRIBE QUERY SELECT * FROM (SELECT 1, 1.5)')
    connector._context.cursor.return_value.__exit__.assert_called_once()


def test_query_schema_keeps_restricted_diagnostic_boundary():
    from io_utils.errors import RESTRICTED_LOG_MESSAGE_ATTRIBUTE

    connector = _make_unopened_databricks_connector()
    connector._context = MagicMock()
    cursor = connector._context.cursor.return_value.__enter__.return_value
    error = _databricks_server_error('PARSE_SYNTAX_ERROR', 'private query text')
    cursor.execute.side_effect = error

    with pytest.raises(ConnectorError) as caught:
        connector.get_query_metadata('SELECT bad')

    assert 'private query text' not in str(caught.value)
    assert getattr(error, RESTRICTED_LOG_MESSAGE_ATTRIBUTE) is True
    connector._context.cursor.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize('failures', [1, 3])
@pytest.mark.parametrize('phase', ['execute', 'fetch'])
def test_sql_result_retries_http_errors(_no_retry_sleep, failures, phase):
    import pyarrow as pa

    connector = _make_unopened_databricks_connector()
    connector._context = MagicMock()
    cursor = connector._context.cursor.return_value.__enter__.return_value
    cursor.execute.return_value = cursor
    result = pa.table({'id': [1]})
    if phase == 'execute':
        cursor.execute.side_effect = [HTTPError('transient')] * failures + [
            cursor
        ]
        cursor.fetchall_arrow.return_value = result
    else:
        cursor.fetchall_arrow.side_effect = [HTTPError('transient')
                                             ] * failures + [result]

    if failures == 3:
        with pytest.raises(WarehouseRetriableError, match='request failed'):
            connector._get_df_result(connector.context, 'SELECT 1')
    else:
        frame = connector._get_df_result(connector.context, 'SELECT 1')
        assert frame['id'].tolist() == [1]

    assert cursor.execute.call_count == min(failures + 1, 3)
    assert connector._context.cursor.return_value.__exit__.call_count == min(
        failures + 1, 3)


def test_query_schema_retries_http_error_before_describing(_no_retry_sleep):
    connector = _make_unopened_databricks_connector()
    connector._context = MagicMock()
    cursor = connector._context.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = [HTTPError('transient'), cursor]
    cursor.fetchall.return_value = [('id', 'bigint')]

    metadata = connector.get_query_metadata('SELECT 1')

    assert metadata.cols[0].name == 'id'
    assert cursor.execute.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('same_query', [False, True])
@pytest.mark.parametrize('entrypoint', ['_get_df_result', '_run_sql'])
@pytest.mark.parametrize('fail', [False, True])
async def test_sql_inflight_reuses_only_matching_queries(
        same_query, entrypoint, fail):
    import asyncio
    import threading

    import pyarrow as pa

    from kumo.connector.databricks_connector import _dataframe_result

    connector = _make_unopened_databricks_connector()
    context = MagicMock()
    connector._context = context
    started = asyncio.Event()
    release = threading.Event()
    lock = threading.Lock()
    queries = []
    expected = 1 if same_query else 2
    loop = asyncio.get_running_loop()

    def execute(query):
        with lock:
            queries.append(query)
            if len(queries) >= expected:
                loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        if fail:
            raise RuntimeError('query failed')
        result = MagicMock()
        result.fetchall_arrow.return_value = pa.table({'query': [query]})
        return result

    context.cursor.return_value.__enter__.return_value.execute.side_effect = execute
    requested = ['SELECT 1', 'SELECT 1' if same_query else 'SELECT 2']
    method = getattr(connector, entrypoint)
    tasks = [
        asyncio.create_task(
            method.coroutine(
                context,
                *([_dataframe_result, query]
                  if entrypoint == '_run_sql' else [query])))
        for query in requested
    ]
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
    finally:
        release.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert sorted(queries) == sorted(set(requested))
    if fail:
        assert all(isinstance(error, RuntimeError) for error in results)
        assert [str(error) for error in results] == ['query failed'] * 2
    else:
        assert [frame['query'].iloc[0] for frame in results] == requested
    assert context.cursor.return_value.__exit__.call_count == expected


def _metadata_connector(rows=()):
    connector = _sample_schema_connector()
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.execute.return_value = cursor
    cursor.fetchall.return_value = rows
    connector._context.cursor.side_effect = None
    connector._context.cursor.return_value = cursor
    return connector, cursor


def test_databricks_listing_excludes_names_without_visible_columns():
    connector, cursor = _metadata_connector([
        ('tpch', 'z', False),
        ('tpch', 'missing_columns', False),
        ('tpch', 'a', False),
    ])
    cursor.columns.return_value.fetchall_arrow.return_value = pa.table({
        'TABLE_SCHEM': ['tpch', 'tpch', 'other'],
        'TABLE_NAME': ['z', 'a', 'elsewhere'],
        'COLUMN_NAME': ['id', 'id', 'id'],
        'TYPE_NAME': ['bigint', 'bigint', 'bigint'],
    })

    assert connector.list_tables() == ['a', 'z']
    cursor.columns.assert_called_once_with(catalog_name='samples',
                                           schema_name='tpch')
    cursor.execute.assert_not_called()


@pytest.mark.parametrize('names', [
    ['orders', 'customers'],
    ['ORDERS', 'Customers'],
    ['ORDERS', 'orders', 'tpch.Orders', 'ORDERS'],
])
def test_databricks_multiple_tables_use_one_filtered_catalog_read(names):
    connector, cursor = _metadata_connector()
    cursor.columns.return_value.fetchall_arrow.return_value = pa.table({
        'TABLE_SCHEM': ['tpch', 'tpch', 'tpch'],
        'TABLE_NAME': ['orders', 'customers', 'unselected'],
        'COLUMN_NAME': ['id', 'id', 'id'],
        'TYPE_NAME': ['bigint', 'bigint', 'bigint'],
    })

    metadata = connector._get_table_metadata(names, skip_inferral=True)

    assert [table.table_name for table in metadata] == names
    assert [table.source_name for table in metadata] == names
    assert all([(column.name, column.dtype)
                for column in table.cols] == [('id', Dtype.int64)]
               for table in metadata)
    cursor.columns.assert_called_once_with(catalog_name='samples',
                                           schema_name=None)
    cursor.execute.assert_not_called()


def test_databricks_exact_schema_uses_describe_with_complete_types():
    connector, cursor = _metadata_connector([
        (' amount ', 'decimal(38,18)', None),
        ('events', 'array<double>', None),
        ('#tag', 'string', None),
        ('# Partition Information', '', None),
        ('events', 'array<double>', None),
    ])
    columns = connector.get_column_metadata(connector.context, 'tpch',
                                            ['orders'])
    assert columns == {
        ('orders', ' amount '): Dtype.float,
        ('orders', 'events'): Dtype.floatlist,
        ('orders', '#tag'): Dtype.string
    }
    cursor.execute.assert_called_once_with(
        'DESCRIBE EXTENDED `samples`.`tpch`.`orders`')
    cursor.columns.assert_not_called()


def test_databricks_exact_schema_missing_table_remains_absent():
    connector, cursor = _metadata_connector()
    cursor.execute.side_effect = _databricks_server_error(
        'TABLE_OR_VIEW_NOT_FOUND')
    assert connector.get_column_metadata(connector.context, 'tpch',
                                         ['missing']) == {}


@pytest.mark.parametrize(('column', 'primary_key', 'projection'), [
    ('amount', 'id', '`amount`, `id`'),
    ('unit price', 'key`id', '`unit price`, `key``id`'),
])
def test_databricks_sample_retains_projection_and_limit_policy(
        column, primary_key, projection):
    import pyarrow as pa

    connector, cursor = _metadata_connector()
    cursor.fetchall_arrow.return_value = pa.table({
        column: [1.25],
        primary_key: [7]
    })
    result = connector._read_table_sample(
        connector.context, 'orders', 5, 0.5,
        pkey=BaseColumnConfig(name=primary_key, dtype=Dtype.int64),
        cols=[BaseColumnConfig(name=column, dtype=Dtype.float64)])
    assert result.index.tolist() == [7]
    assert result[column].tolist() == [1.25]
    cursor.execute.assert_called_once_with(
        f'SELECT {projection} FROM `tpch`.`orders` LIMIT 5')


def test_databricks_fraction_only_sample_fails_before_query():
    connector, cursor = _metadata_connector()
    with pytest.raises(ConnectorError, match='requires num_rows'):
        connector._read_table_sample(connector.context, 'orders', None, 0.5)
    cursor.execute.assert_not_called()


@pytest.mark.parametrize('fails', [False, True])
def test_databricks_query_sample_resets_shared_session_timeout(fails):
    import pyarrow as pa

    connector, cursor = _metadata_connector()
    cursor.fetchall_arrow.return_value = pa.table({'id': [7]})
    if fails:
        cursor.execute.side_effect = [
            cursor, _databricks_server_error('QUERY_FAILED'), cursor
        ]
        with pytest.raises(ConnectorError, match='sample query failed'):
            connector.sample_query('SELECT 7 AS id', 3)
    else:
        result = connector.sample_query('SELECT 7 AS id', 3)
        assert result['id'].tolist() == [7]
    assert [call.args[0] for call in cursor.execute.call_args_list] == [
        'SET STATEMENT_TIMEOUT = 60',
        'SELECT * FROM (SELECT 7 AS id) LIMIT 3',
        'SET STATEMENT_TIMEOUT = 0',
    ]


def test_databricks_version_retries_transport_before_falling_back(
        _no_retry_sleep):
    connector, cursor = _metadata_connector([(7, '2026-08-27T15:07:20')])
    cursor.execute.side_effect = [HTTPError('unavailable'), cursor]
    assert connector._get_data_version('orders') == '7_1787843240'
    assert cursor.execute.call_count == 2
    assert cursor.__exit__.call_count == 2
