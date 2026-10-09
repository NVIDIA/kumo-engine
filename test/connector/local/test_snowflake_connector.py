# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from test.connector.artifact_staging_contract import (
    assert_indexed_parquet_parts,
    local_artifact_dataset,
)
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumoapi.typing import Dtype
from pyarrow.lib import ArrowInvalid
from snowflake.connector.cursor import SnowflakeCursor
from snowflake.connector.errors import ProgrammingError

import kumo.connector.snowflake_connector as snowflake_connector
import kumo.connector.snowflake_session as snowflake_session
from kumo.config import BaseColumnConfig
from kumo.config.graph import PrimaryKeyConfig
from kumo.connector.connector import (
    Connector,
    ConnectorError,
    SourceTableUnavailableError,
)
from kumo.connector.snowflake_connector import SnowflakeConnector
from kumo.connector.sql_connector import InternalWriteMode, SQLConnector
from kumo.testing.snowflake import disable_snowflake_env_credentials
from statedb.utils.errors import ConnectorValidationDiagnosticCode


def _parquet_bytes(directory) -> int:
    """What SOURCE_SIZE means: the parquet that landed, not the Arrow."""
    return sum(part.stat().st_size
               for part in Path(directory).glob('*.parquet'))


def test_snowflake_connector_config():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'db',
                                   'schema')

    assert asdict(connector.config()) == {
        '_target_': 'kumo.connector.snowflake_connector.SnowflakeConnector',
        'name': 'snowflake',
        'account': 'acc',
        'warehouse': 'warehouse',
        'database': 'db'.upper(),
        'schema_name': 'schema'.upper(),
        'role': None,
    }
    assert connector == Connector.from_config(connector.config())


def test_credential_only_connector_allows_an_empty_target():
    connector = SnowflakeConnector('identity_validation', 'acc', '', '', '')

    assert connector.warehouse == ''
    assert connector.database == ''
    assert connector.schema_name == ''


def test_quoted_target_config_round_trips_without_losing_case():
    connector = SnowflakeConnector('snowflake', 'acc', '"warehouse"',
                                   '"database"', '"schema"', role='"role"')

    assert asdict(connector.config()) == {
        '_target_': 'kumo.connector.snowflake_connector.SnowflakeConnector',
        'name': 'snowflake',
        'account': 'acc',
        'warehouse': '"warehouse"',
        'database': '"database"',
        'schema_name': '"schema"',
        'role': '"role"',
    }
    assert connector == Connector.from_config(connector.config())
    assert connector.resolved_warehouse == 'warehouse'
    assert connector.resolved_database == 'database'
    assert connector.resolved_schema == 'schema'
    assert connector.resolved_role == 'role'
    assert connector._qualified_schema_name() == '"database"."schema"'
    assert connector._qualified_table_name('table') == (
        '"database"."schema"."table"')


def test_resolved_target_names_follow_the_serialized_fields():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'database',
                                   'schema', role='role')

    connector.warehouse = 'next_warehouse'
    connector.database = 'next_database'
    connector.schema_name = '"next_schema"'
    connector.role = '"next_role"'

    assert connector.resolved_warehouse == 'NEXT_WAREHOUSE'
    assert connector.resolved_database == 'NEXT_DATABASE'
    assert connector.resolved_schema == 'next_schema'
    assert connector.resolved_role == 'next_role'
    assert connector._qualified_schema_name() == '"NEXT_DATABASE"."next_schema"'


def test_quoted_target_syntax_reaches_the_connection_driver(monkeypatch):
    connector = SnowflakeConnector('snowflake', 'acc', '"warehouse"',
                                   '"database"', '"schema"', role='"role"')
    connector.get_user_pass_or_key = Mock(return_value={
        'user': 'user',
        'password': 'password'
    })
    connection = Mock()
    connect = Mock(return_value=connection)
    monkeypatch.setattr(snowflake_connector, 'connect', connect)
    monkeypatch.setattr(snowflake_connector.snowflake_session_logging,
                        'ensure_logged_connector_connection',
                        lambda opened: opened)

    assert connector._open() is connection

    settings = connect.call_args.kwargs
    assert settings['warehouse'] == '"warehouse"'
    assert settings['database'] == '"database"'
    assert settings['schema'] == '"schema"'
    assert settings['role'] == '"role"'


def test_information_schema_names_are_bound_not_interpolated():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'database',
                                   '"O\'Reilly"')
    connector._context = Mock()
    result = pd.DataFrame({'TABLE_NAME': ["t'a"], 'BYTES': [12]})

    with patch.object(connector, 'execute_meta_query',
                      return_value=result) as execute:
        assert connector._query_info_table("t'a", 'BYTES') == 12

    _, query = execute.call_args.args
    assert "O'Reilly" not in query
    assert "t'a" not in query
    assert execute.call_args.kwargs['params'] == ("O'Reilly", "t'a")


def test_all_exact_information_schema_names_are_bound():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'database',
                                   '"O\'Reilly"')
    connector._context = Mock()
    cursor = connector.context.cursor.return_value
    cursor.connection._paramstyle = 'pyformat'
    table = "x' OR 1=1 --"
    altered = pd.Timestamp('2026-01-01')
    cursor.fetchall.side_effect = [[('BASE TABLE', )], [(table, altered)],
                                   [('VIEW', )]]

    assert connector._get_data_version(table) == str(int(altered.timestamp()))
    assert connector.is_table_view(table)
    assert cursor.execute.call_count == 3
    for call in cursor.execute.call_args_list:
        query, params = call.args
        assert "O'Reilly" not in query and table not in query
        assert params == ("O'Reilly", table)

    catalog_connector = SnowflakeConnector('snowflake', 'acc', 'warehouse',
                                           '"data.base"', '"O\'Reilly"')
    context = MagicMock()
    cursor = context.cursor.return_value.__enter__.return_value
    cursor.connection._paramstyle = 'pyformat'
    cursor.fetchall.return_value = []

    catalog_connector.get_column_metadata(context)

    query, params = cursor.execute.call_args.args
    assert '"data.base"."INFORMATION_SCHEMA"."COLUMNS"' in query
    assert '"data.base"."INFORMATION_SCHEMA"."TABLES"' in query
    assert '"O\'Reilly"' not in query
    assert params == ["O'Reilly"]


def test_execute_meta_query_forwards_bound_values_to_the_driver():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'database',
                                   'schema')
    cursor = Mock()
    cursor.description = [{'name': 'TABLE_NAME'}]
    cursor.fetchall.return_value = [('orders', )]
    context = Mock()
    context.cursor.return_value = cursor
    query = ('select TABLE_NAME from INFORMATION_SCHEMA.TABLES '
             'where TABLE_SCHEMA = %s')

    result = connector.execute_meta_query(context, query, params=("O'Reilly", ))

    cursor.execute.assert_called_once_with(query, ("O'Reilly", ))
    assert result['TABLE_NAME'].tolist() == ['orders']


def test_select_query():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'db',
                                   'schema')
    query = connector.get_query(table_name='test',
                                pkey=PrimaryKeyConfig('pkey'),
                                cols=[BaseColumnConfig('col1')])
    assert query == ('SELECT "col1", "pkey" from '
                     '"DB"."SCHEMA"."test"')
    query = connector.get_query(table_name='test',
                                pkey=PrimaryKeyConfig('pkey'))
    assert query == 'SELECT * from "DB"."SCHEMA"."test"'
    query = connector.get_query(table_name='test')
    assert query == 'SELECT * from "DB"."SCHEMA"."test"'
    query = connector.get_query(table_name='test',
                                pkey=PrimaryKeyConfig('pkey'), cols=[])
    assert query == 'SELECT "pkey" from "DB"."SCHEMA"."test"'
    with pytest.raises(ValueError):
        connector.get_query(table_name='test', cols=[])


def test_select_and_drop_quote_exact_names_containing_quotes():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'db',
                                   'schema')
    query = connector.get_query(table_name='ta"ble',
                                pkey=BaseColumnConfig('p"key'),
                                cols=[BaseColumnConfig('co"l')])
    assert query == ('SELECT "co""l", "p""key" from '
                     '"DB"."SCHEMA"."ta""ble"')

    connector._context = Mock()
    with patch.object(connector, 'execute_meta_query') as execute:
        connector.drop_table_if_exists('ta"ble')
    execute.assert_called_once_with(
        connector.context, 'DROP TABLE IF EXISTS "DB"."SCHEMA"."ta""ble"')


def test_platform_snowpark_builder_gets_syntax_and_cache_gets_identity(
        monkeypatch):
    connector = SnowflakeConnector('snowflake', 'acc', '"warehouse"',
                                   '"database"', '"schema"', role='"role"')
    connector.get_user_pass_or_key = Mock(return_value={
        'user': 'user',
        'password': 'password'
    })
    opened = object()
    captured = {}

    class _Builder:
        def configs(self, settings):
            captured.update(settings)
            return self

        def create(self):
            return opened

    class _Session:
        builder = _Builder()

    monkeypatch.setattr(snowflake_session, 'SnowparkSession', _Session)
    monkeypatch.setattr(snowflake_session, 'ensure_logged_snowpark_session',
                        lambda session: session)

    assert snowflake_session.build_snowpark_session_from_connector(
        connector) is opened
    assert captured['warehouse'] == '"warehouse"'
    assert captured['database'] == '"database"'
    assert captured['schema'] == '"schema"'
    assert captured['role'] == '"role"'

    key = snowflake_session._credentials_key(connector)
    assert key.warehouse == 'warehouse'
    assert key.database == 'database'
    assert key.schema_name == 'schema'
    assert key.role == 'role'


def test_get_df_result_requests_stable_timestamp_precision():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'db',
                                   'schema')
    raw_batches = [
        pa.table({
            'DECEASED_DATE':
            pa.array([1_577_836_800_123_456_789], type=pa.timestamp('ns')),
            'CREATED_AT':
            pa.array([datetime(2020, 1, 1)], type=pa.timestamp('ns')),
            'LABEL':
            pa.array(['2020-01-01'], type=pa.string()),
        }),
        pa.table({
            'DECEASED_DATE':
            pa.array([datetime(9999, 1, 1)], type=pa.timestamp('us')),
            'CREATED_AT':
            pa.array([None], type=pa.timestamp('us')),
            'LABEL':
            pa.array([None], type=pa.string()),
        }),
    ]
    stable_schema = pa.schema([
        ('DECEASED_DATE', pa.timestamp('us')),
        ('CREATED_AT', pa.timestamp('us')),
        ('LABEL', pa.string()),
    ])

    # This is the real PyArrow operation that failed in the incident.
    with pytest.raises(ArrowInvalid, match='Schema at index 1 was different'):
        pa.concat_tables(raw_batches)

    class PrecisionAwareCursor:
        rowcount = 2
        description = [{'name': 'DECEASED_DATE'}]

        def fetch_arrow_all(self, force_microsecond_precision=False):
            batches = raw_batches
            if force_microsecond_precision:
                # This models the connector's documented truncation of
                # timestamp scale 7-9 when forcing microsecond precision.
                batches = [
                    batch.cast(stable_schema, safe=False) for batch in batches
                ]
            return pa.concat_tables(batches)

    cursor = PrecisionAwareCursor()
    context = Mock()
    context.cursor.return_value.execute.return_value = cursor

    result = connector._get_df_result(
        context, 'SELECT * FROM RECOMMENDATIONS_USER_INPUT_VIEW')

    assert result.columns.tolist() == ['DECEASED_DATE', 'CREATED_AT', 'LABEL']
    assert result['DECEASED_DATE'].tolist() == [
        datetime(2020, 1, 1, 0, 0, 0, 123456),
        datetime(9999, 1, 1),
    ]
    assert result['DECEASED_DATE'].dtype == object
    assert result['CREATED_AT'].dtype == 'datetime64[ns]'
    assert result['CREATED_AT'].iloc[0] == pd.Timestamp('2020-01-01')
    assert pd.isna(result['CREATED_AT'].iloc[1])
    assert result['LABEL'].dtype == object
    assert result['LABEL'].tolist() == ['2020-01-01', None]


def test_restore_in_range_timestamp_precision_preserves_timezones():
    table = pa.table({
        'IN_RANGE':
        pa.array([datetime(2020, 1, 1, tzinfo=timezone.utc), None],
                 type=pa.timestamp('us', tz='UTC')),
        'OUT_OF_RANGE':
        pa.array([datetime(9999, 1, 1, tzinfo=timezone.utc), None],
                 type=pa.timestamp('us', tz='UTC')),
        'ALL_NULL':
        pa.array([None, None], type=pa.timestamp('us')),
        'LABEL':
        pa.array(['2020-01-01', None], type=pa.string()),
    })

    restored = snowflake_connector._restore_in_range_timestamp_precision(table)
    result = restored.to_pandas(timestamp_as_object=True)

    assert restored.schema.field('IN_RANGE').type == pa.timestamp(
        'ns', tz='UTC')
    assert restored.schema.field('OUT_OF_RANGE').type == pa.timestamp(
        'us', tz='UTC')
    assert restored.schema.field('ALL_NULL').type == pa.timestamp('ns')
    assert result['IN_RANGE'].dtype == 'datetime64[ns, UTC]'
    assert result['OUT_OF_RANGE'].dtype == object
    assert result['ALL_NULL'].dtype == 'datetime64[ns]'
    assert result['OUT_OF_RANGE'].iloc[0] == datetime(9999, 1, 1,
                                                      tzinfo=timezone.utc)
    assert result['LABEL'].tolist() == ['2020-01-01', None]


def test_get_df_result_supports_cursor_wrapper_without_precision_keyword():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'db',
                                   'schema')
    arrow_table = pa.table({'ID': pa.array([1], type=pa.int64())})

    class LegacyCursor:
        rowcount = 1
        description = [{'name': 'ID'}]

        def fetch_arrow_all(self):
            return arrow_table

    context = Mock()
    context.cursor.return_value.execute.return_value = LegacyCursor()

    result = connector._get_df_result(context, 'SELECT ID FROM TABLE')

    pd.testing.assert_frame_equal(result, arrow_table.to_pandas())


def test_microsecond_precision_kwargs_match_snowflake_cursor_api():
    assert snowflake_connector._microsecond_precision_kwargs(
        SnowflakeCursor.fetch_arrow_all) == {
            'force_microsecond_precision': True
        }


def test_snowflake_malformed_rows_falls_back_to_count_query():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'db',
                                   'schema')
    connector._context = Mock()
    show_tables_df = pd.DataFrame([{
        'name': 'P_USER',
        'kind': 'TABLE',
        'created_on': '2024-01-01 00:00:00.000',
        'rows': 'not-an-int',
        'bytes': 100,
    }])

    with patch.object(connector, 'execute_meta_query',
                      return_value=show_tables_df), patch.object(
                          SQLConnector, '_get_table_num_rows',
                          return_value=1234) as mock_parent_count:
        assert connector._get_table_num_rows('P_USER') == 1234
        mock_parent_count.assert_called_once_with(table_name='P_USER')


def test_snowflake_malformed_bytes_falls_back_to_info_table():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'db',
                                   'schema')
    connector._context = Mock()
    show_tables_df = pd.DataFrame([{
        'name': 'P_USER',
        'kind': 'TABLE',
        'created_on': '2024-01-01 00:00:00.000',
        'rows': 10,
        'bytes': 'invalid-bytes',
    }])

    with patch.object(connector, 'execute_meta_query',
                      return_value=show_tables_df), patch.object(
                          connector, '_query_info_table',
                          return_value=777) as mock_info_table:
        assert connector._get_table_source_size('P_USER') == 777
        mock_info_table.assert_called_once_with('P_USER', 'BYTES')


def test_snowflake_query():
    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='FINANCIAL')
    assert connector.get_query('test_table') == (
        'SELECT * from "KUMO"."FINANCIAL"."test_table"')
    assert connector.get_query(
        'test_table',
        limit=10) == ('SELECT * from "KUMO"."FINANCIAL"."test_table" LIMIT 10')
    connector.close()


def test_throw_error_if_no_credentials():
    connector = SnowflakeConnector('snowflake', account='example-account',
                                   warehouse='WH_XS', database='KUMO',
                                   schema_name='FINANCIAL')
    with disable_snowflake_env_credentials():
        with pytest.raises(ConnectorError,
                           match=r"No credentials were provided"):
            connector.open()


class TestSnowflakeAccessCheck:

    _credentials = {'user': 'test-user', 'password': 'test-password'}

    @patch('kumo.connector.snowflake_connector.connect')
    def test_no_tables_found(self, mock_connect):
        """Test when database has no tables"""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # fetchone returns None (no tables)
        mock_cursor.fetchone.return_value = None

        connector = SnowflakeConnector('snowflake', 'test_account', 'test_wh',
                                       'test_db', 'test_sch')
        # shouldn't raise an error since not having
        # any tables might be a valid condition - say
        # when the DB has only views or the user has
        # access to only views...
        connector.open(self._credentials)

    @patch('kumo.connector.snowflake_connector.connect')
    def test_permission_validation_quotes_exact_target_identifiers(
            self, mock_connect):
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = None

        connector = SnowflakeConnector(
            'snowflake',
            'test_account',
            '"ware""house"',
            '"data""base"',
            '"sche""ma"',
        )
        connector.open(self._credentials)

        calls = mock_cursor.execute.call_args_list
        assert calls[0].args == ('USE DATABASE "data""base"', )
        assert calls[1].args == ('USE SCHEMA "sche""ma"', )
        assert calls[2].args == ('USE WAREHOUSE "ware""house"', )
        assert calls[3].args[1] == ('sche"ma', )
        assert "WHERE TABLE_SCHEMA = %s" in calls[3].args[0]

    @patch('kumo.connector.snowflake_connector.connect')
    def test_permission_denied_on_select(self, mock_connect):
        """Test when user lacks SELECT permission on table"""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # First fetchone returns table name
        mock_cursor.fetchone.return_value = ('CUSTOMERS', )

        # An access-control violation carries SQLSTATE 42501 ("Insufficient
        # privileges to operate on ...") and must classify as access-denied.
        mock_cursor.execute.side_effect = [
            None,  # USE DATABASE
            None,  # USE SCHEMA
            None,  # USE WAREHOUSE
            ProgrammingError(
                "SQL access control error: Insufficient privileges", errno=3001,
                sqlstate='42501'),  # INFORMATION_SCHEMA query
        ]
        connector = SnowflakeConnector('snowflake', 'test_account', 'test_wh',
                                       'test_db', 'test_sch')
        with pytest.raises(ConnectorError, match="Access checks failed") as exc:
            connector.open(self._credentials)
        assert exc.value.diagnostic_code == (
            ConnectorValidationDiagnosticCode.TARGET_ACCESS_DENIED)

    @patch('kumo.connector.snowflake_connector.connect')
    def test_permission_denied_on_database(self, mock_connect):
        """Test when user lacks access to database"""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # USE DATABASE raises ProgrammingError
        mock_cursor.execute.side_effect = ProgrammingError(
            "Database 'TEST_DB' does not exist or not authorized")

        connector = SnowflakeConnector('snowflake', 'test_account', 'test_wh',
                                       'test_db', 'test_sch')
        with pytest.raises(ConnectorError, match="Access checks failed"):
            connector.open(self._credentials)

    @patch('kumo.connector.snowflake_connector.connect')
    def test_schema_not_found(self, mock_connect):
        """Test when schema doesn't exist"""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # USE SCHEMA raises error
        mock_cursor.execute.side_effect = [
            None,  # USE DATABASE
            ProgrammingError(
                "Schema 'NONEXISTENT' does not exist")  # USE SCHEMA
        ]

        connector = SnowflakeConnector('snowflake', 'test_account', 'test_wh',
                                       'test_db', 'test_sch')

        with pytest.raises(ConnectorError, match="Access checks failed"):
            connector.open(self._credentials)

    @pytest.mark.parametrize('errno', [2003, 2043])
    @patch('kumo.connector.snowflake_connector.connect')
    def test_object_not_found_or_unauthorized_is_classified(
            self, mock_connect, errno):
        """Snowflake object-resolution errnos (002003 "does not exist or not
        authorized" and 002043 "does not exist, or operation cannot be
        performed") are genuine target-config problems -> TARGET_INACCESSIBLE."""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # USE DATABASE/SCHEMA/WAREHOUSE on a wrong/ungranted object.
        mock_cursor.execute.side_effect = ProgrammingError(
            msg=f"{errno:06d}: Object does not exist or not authorized.",
            errno=errno)

        connector = SnowflakeConnector('snowflake', 'test_account', 'test_wh',
                                       'test_db', 'test_sch')

        with pytest.raises(ConnectorError, match="Access checks failed") as err:
            connector.open(self._credentials)
        assert err.value.diagnostic_code == (
            ConnectorValidationDiagnosticCode.TARGET_INACCESSIBLE)

    @patch('kumo.connector.snowflake_connector.connect')
    def test_operational_failure_is_unclassified(self, mock_connect):
        """A transient/operational Snowflake error keeps the guidance message
        but stays unclassified so it is not mislabeled non-retriable."""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # errno 604 is a statement cancellation, not a target-config failure.
        mock_cursor.execute.side_effect = ProgrammingError(
            msg="000604: SQL execution canceled", errno=604)

        connector = SnowflakeConnector('snowflake', 'test_account', 'test_wh',
                                       'test_db', 'test_sch')

        with pytest.raises(ConnectorError, match="Access checks failed") as err:
            connector.open(self._credentials)
        assert err.value.diagnostic_code is None

    @patch('kumo.connector.snowflake_connector.connect')
    def test_empty_table(self, mock_connect):
        """Test when table exists but is empty"""
        mock_cursor = Mock()
        mock_conn = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Table exists, but no data
        mock_cursor.fetchone.side_effect = [
            ('CUSTOMERS', ),  # Table name
            None  # No data in table
        ]

        connector = SnowflakeConnector('snowflake', 'test_account', 'test_wh',
                                       'test_db', 'test_sch')

        connector.open(self._credentials)


def _describe_meta(**cols: int) -> list:
    """`cursor.describe` metadata: column name -> FIELD_ID type code."""
    from types import SimpleNamespace
    return [
        SimpleNamespace(name=name, type_code=code)
        for name, code in cols.items()
    ]


def _make_snowflake_connector_with_mock_context() -> SnowflakeConnector:
    """Build a SnowflakeConnector with `_context` pre-populated by a Mock.

    Setting `_context` makes `is_connected` True so the connector won't
    try to authenticate against Snowflake when `context` is accessed.
    Callers can then `patch.object(connector, 'execute_meta_query', ...)`
    to control the SQL surface.
    """
    connector = SnowflakeConnector(
        'snowflake',
        account='acct',
        warehouse='WH_XS',
        database='KUMO',
        schema_name='TEST_SCHEMA',
    )
    connector._context = Mock()
    return connector


def test_is_table_view_preserves_ambiguous_visibility_signal():
    connector = _make_snowflake_connector_with_mock_context()

    cursor = connector.context.cursor.return_value
    cursor.connection._paramstyle = 'pyformat'
    cursor.fetchall.return_value = []
    with pytest.raises(SourceTableUnavailableError,
                       match='could not be found or accessed'):
        connector.is_table_view('HIDDEN_OR_MISSING')
    cursor.close.assert_called_once()


def test_snowflake_metadata_ignore_errors_preserves_available_tables():
    connector = _make_snowflake_connector_with_mock_context()
    metadata = Mock(table_name='ORDERS', source_name='ORDERS')

    with patch.object(
            connector,
            '_fetch_table_metadata',
            return_value={'ORDERS': metadata},
    ):
        result = connector._get_table_metadata(
            ['MISSING', 'ORDERS'],
            skip_inferral=True,
            ignore_errors=True,
        )

    assert result == [metadata]


def test_snowflake_metadata_missing_table_is_typed():
    connector = _make_snowflake_connector_with_mock_context()

    with (
            patch.object(connector, '_fetch_table_metadata', return_value={}),
            pytest.raises(SourceTableUnavailableError,
                          match="Source table 'MISSING'"),
    ):
        connector._get_table_metadata(['MISSING'], skip_inferral=True)


def _make_programming_error(msg: str) -> ProgrammingError:
    """ProgrammingError requires positional args in its accepted signature."""
    return ProgrammingError(msg=msg)


@pytest.mark.parametrize(('row', 'expected'), [((1, ), True), (None, False)])
def test_table_exists_uses_parameterized_exact_lookup(row, expected):
    connector = _make_snowflake_connector_with_mock_context()
    cursor = connector.context.cursor.return_value
    cursor.fetchone.return_value = row
    cursor.connection._paramstyle = 'pyformat'

    assert connector.table_exists('CUSTOMERS') is expected

    cursor.execute.assert_called_once_with(
        'SELECT 1 FROM "KUMO"."INFORMATION_SCHEMA"."TABLES" '
        'WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s LIMIT 1',
        ['TEST_SCHEMA', 'CUSTOMERS'],
    )
    cursor.close.assert_called_once_with()


def test_overwrite_table_uses_hosted_destination_column_order(monkeypatch):
    connector = _make_snowflake_connector_with_mock_context()
    connector._context = MagicMock()
    inventories = {
        'STAGING': ['TIMESTAMP', 'SCORE', 'ENTITY'],
        'PREDICTIONS': ['SCORE', 'ENTITY', 'TIMESTAMP'],
    }
    catalog_calls = []

    def _columns(cursor, *, database, schema, table):
        catalog_calls.append((cursor, database, schema, table))
        return {(table, column): None for column in inventories[table]}

    monkeypatch.setattr('kumo_snowflake.catalog.columns', _columns)
    monkeypatch.setattr(connector, 'table_exists', lambda _name: True)
    execute = Mock()
    monkeypatch.setattr(connector, 'execute_meta_query', execute)

    connector.overwrite_table('STAGING', 'PREDICTIONS')

    cursor = connector.context.cursor.return_value.__enter__.return_value
    assert catalog_calls == [
        (cursor, 'KUMO', 'TEST_SCHEMA', 'STAGING'),
        (cursor, 'KUMO', 'TEST_SCHEMA', 'PREDICTIONS'),
    ]
    execute.assert_called_once_with(
        connector.context,
        'INSERT OVERWRITE INTO "PREDICTIONS" '
        '("SCORE", "ENTITY", "TIMESTAMP") '
        'SELECT "SCORE", "ENTITY", "TIMESTAMP" FROM "STAGING"',
    )
    assert connector.context.cursor.return_value.__exit__.call_count == 2


def test_check_table_write_permissions_schema_mismatch():
    """The connector binds itself to a specific schema at construction.
    Caller-provided `table_schema` must match that schema (case-insensitive)
    so the write goes to a known destination — otherwise the caller is
    using the wrong connector for the table and the operation must fail
    before any privilege check runs.
    """
    connector = _make_snowflake_connector_with_mock_context()
    with pytest.raises(ConnectorError, match="does not match connector schema"):
        connector.check_table_write_permissions(table_name='t',
                                                table_schema='OTHER_SCHEMA',
                                                mode=InternalWriteMode.APPEND)


def test_check_table_write_permissions_invalid_mode():
    """Only APPEND and OVERWRITE are valid write modes; any other value
    means the caller passed in a mis-typed enum (or a stale string).
    Reject before any privilege check so the failure is obvious.
    """
    connector = _make_snowflake_connector_with_mock_context()
    # CREATE/DROP succeeds; table exists; mode is invalid.
    with patch.object(connector, 'execute_meta_query',
                      return_value=pd.DataFrame()), patch.object(
                          connector, 'table_exists', return_value=True):
        with pytest.raises(ConnectorError, match=r"Invalid Snowflake"):
            connector.check_table_write_permissions(table_name='t',
                                                    table_schema='TEST_SCHEMA',
                                                    mode='SOMETHING_BOGUS')


def test_check_table_write_permissions_invalid_mode_when_table_missing():
    """Invalid modes are rejected before any provider operation."""
    connector = _make_snowflake_connector_with_mock_context()
    with patch.object(connector,
                      'execute_meta_query') as mock_exec, patch.object(
                          connector, 'table_exists') as mock_exists:
        with pytest.raises(ConnectorError, match='Invalid Snowflake'):
            connector.check_table_write_permissions(
                table_name='nonexistent_t',
                table_schema='TEST_SCHEMA',
                mode='SOMETHING_BOGUS',
            )
    mock_exec.assert_not_called()
    mock_exists.assert_not_called()


def test_check_table_write_permissions_create_table_denied():
    """If the role can't CREATE TABLE on the connector's schema then
    *every* downstream write — staging table, INSERT OVERWRITE, etc. —
    will fail later. The check must surface that early as a ConnectorError
    naming the CREATE TABLE privilege, so the operator knows exactly
    what to grant.
    """
    connector = _make_snowflake_connector_with_mock_context()
    err = _make_programming_error(
        "SQL access control error: Insufficient privileges to operate on "
        "schema 'TEST_SCHEMA'")

    def _fake_exec(_ctx, query, creds_clause=None):
        if query.startswith('CREATE TABLE'):
            raise err
        return pd.DataFrame()

    with patch.object(connector, 'execute_meta_query', side_effect=_fake_exec):
        with pytest.raises(ConnectorError, match="CREATE TABLE privileges"):
            connector.check_table_write_permissions(
                table_name='t', table_schema='TEST_SCHEMA',
                mode=InternalWriteMode.APPEND)


def test_check_table_write_permissions_nonexistent_table_skips_insert_check():
    """When the destination table doesn't exist yet, the connector will
    create it on first write — so INSERT/DELETE checks are unnecessary
    and would only generate noise. The early return is the contract the
    integration test was trying to lock in.
    """
    connector = _make_snowflake_connector_with_mock_context()

    with patch.object(connector, 'execute_meta_query',
                      return_value=pd.DataFrame()) as mock_exec, patch.object(
                          connector, 'table_exists', return_value=False):
        connector.check_table_write_permissions(table_name='t',
                                                table_schema='TEST_SCHEMA',
                                                mode=InternalWriteMode.APPEND)
        # Verify INSERT was never tried — only CREATE/DROP probe queries.
        joined = ' | '.join(call.args[1] for call in mock_exec.call_args_list)
        assert 'INSERT' not in joined.upper()


def test_check_table_write_permissions_insert_denied():
    """When the destination table exists, APPEND mode needs INSERT.
    Failing INSERT privileges must produce a ConnectorError that names
    the table and the missing INSERT permission — otherwise the operator
    has to guess which of CREATE/INSERT/DELETE went wrong.
    """
    connector = _make_snowflake_connector_with_mock_context()
    insert_err = _make_programming_error(
        "SQL access control error: Insufficient privileges to INSERT")

    def _fake_exec(_ctx, query, creds_clause=None):
        if query.startswith('INSERT INTO'):
            raise insert_err
        return pd.DataFrame()

    with patch.object(connector, 'execute_meta_query',
                      side_effect=_fake_exec), patch.object(
                          connector, 'table_exists', return_value=True):
        with pytest.raises(ConnectorError, match="Cannot INSERT"):
            connector.check_table_write_permissions(
                table_name='existing_table', table_schema='TEST_SCHEMA',
                mode=InternalWriteMode.APPEND)


def test_check_table_write_permissions_overwrite_denied():
    """OVERWRITE needs INSERT + DELETE. We can't tell which one failed
    from the surface here (Snowflake collapses them into a single error
    on INSERT OVERWRITE), so the message must hint at both grants.
    """
    connector = _make_snowflake_connector_with_mock_context()
    overwrite_err = _make_programming_error(
        "SQL access control error: Insufficient privileges to DELETE")

    def _fake_exec(_ctx, query, creds_clause=None):
        if query.startswith('INSERT OVERWRITE'):
            raise overwrite_err
        return pd.DataFrame()

    with patch.object(connector, 'execute_meta_query',
                      side_effect=_fake_exec), patch.object(
                          connector, 'table_exists', return_value=True):
        with pytest.raises(ConnectorError, match="overwrite existing table"):
            connector.check_table_write_permissions(
                table_name='existing_table', table_schema='TEST_SCHEMA',
                mode=InternalWriteMode.OVERWRITE)


def test_check_table_write_permissions_overwrite_success():
    """Happy path: CREATE/DROP/INSERT OVERWRITE all succeed → no error
    raised. Locks in that the method doesn't add spurious behavior on
    success (e.g. accidentally raising on empty DataFrame results).
    """
    connector = _make_snowflake_connector_with_mock_context()
    with patch.object(connector, 'execute_meta_query',
                      return_value=pd.DataFrame()) as mock_exec, patch.object(
                          connector, 'table_exists', return_value=True):
        connector.check_table_write_permissions(
            table_name='t', table_schema='TEST_SCHEMA',
            mode=InternalWriteMode.OVERWRITE)
    sql = ' | '.join(call.args[1] for call in mock_exec.call_args_list)
    assert '"KUMO"."TEST_SCHEMA".' in sql
    assert 'INSERT OVERWRITE INTO "KUMO"."TEST_SCHEMA"."t"' in sql


def test_check_table_write_probe_cleanup_failure_is_sanitized(caplog):
    connector = _make_snowflake_connector_with_mock_context()
    secret = 'provider-payload-with-session-token'
    cleanup_error = _make_programming_error(secret)

    def _fake_exec(_ctx, query, creds_clause=None):
        if query.startswith('DROP TABLE'):
            raise cleanup_error
        return pd.DataFrame()

    with patch.object(connector, 'execute_meta_query', side_effect=_fake_exec):
        with pytest.raises(ConnectorError,
                           match='probe cleanup failed') as exc_info:
            connector.check_table_write_permissions(
                table_name='t',
                table_schema='TEST_SCHEMA',
                mode=InternalWriteMode.APPEND,
            )

    assert secret not in exc_info.value.get_user_message()
    assert secret not in caplog.text


def test_check_table_write_denial_does_not_surface_provider_payload(caplog):
    connector = _make_snowflake_connector_with_mock_context()
    secret = 'provider-payload-with-private-key'
    create_error = _make_programming_error(secret)

    with patch.object(
            connector,
            'execute_meta_query',
            side_effect=create_error,
    ):
        with pytest.raises(ConnectorError,
                           match='CREATE TABLE privileges') as exc_info:
            connector.check_table_write_permissions(
                table_name='t',
                table_schema='TEST_SCHEMA',
                mode=InternalWriteMode.APPEND,
            )

    assert secret not in exc_info.value.get_user_message()
    assert secret not in caplog.text


def test_snowflake_output_write_does_not_surface_provider_payload(caplog):
    connector = _make_snowflake_connector_with_mock_context()
    secret = 'provider-payload-with-password'

    with patch.object(
            connector,
            'execute_meta_query',
            side_effect=RuntimeError(secret),
    ):
        with pytest.raises(
                ConnectorError,
                match='configured Snowflake output table') as exc_info:
            connector.write_artifacts_to_table(
                'predictions', 's3://internal-artifacts/predictions/')

    assert secret not in exc_info.value.get_user_message()
    assert secret not in caplog.text


@pytest.mark.parametrize('query,columns', [
    (False, None),
    (False, {
        'ID': Dtype.int64,
        'TS': Dtype.time
    }),
    (True, {
        'ID': Dtype.int64,
        'TS': Dtype.time
    }),
])
def test_snowflake_custom_exports_use_staged_reader(tmp_path, monkeypatch,
                                                    query, columns):
    from types import SimpleNamespace

    from kumo_snowflake import source
    from kumo_snowflake.staging import UnloadResult

    connector = _make_snowflake_connector_with_mock_context()
    connector._context.cursor.return_value.describe.return_value = (
        _describe_meta(ID=0, TS=7))
    connector.get_query_metadata = Mock(return_value=Mock(cols=[
        SimpleNamespace(name='ID', dtype=Dtype.int64),
        SimpleNamespace(name='TS', dtype=Dtype.time),
    ]))
    unload = Mock(return_value=UnloadResult(rows_unloaded=2, input_bytes=32))
    clear = Mock()
    monkeypatch.setattr(source, 'create_stage', Mock())
    monkeypatch.setattr(source, 'clear_prefix', clear)
    monkeypatch.setattr(source, 'unload', unload)
    monkeypatch.setattr(source, 'list_prefix', Mock(return_value=[]))
    monkeypatch.setattr(source, 'presigned_urls', Mock(return_value=[]))

    def download(files, urls, dest, **kwargs):
        path = os.path.join(dest, 'part.parquet')
        pq.write_table(pa.table({'ID': [1, 2]}), path)
        return (path, )

    monkeypatch.setattr(source, 'download', download)
    destination = str(tmp_path / 'persisted')
    os.makedirs(destination)
    pq.write_table(pa.table({'ID': [99]}),
                   os.path.join(destination, 'old.parquet'))
    sql = 'SELECT id AS "ID", ts AS "TS" FROM events'
    if query:
        fingerprint = Mock(ingest_path_fpid='sql-result', cols=columns)
        with patch('kumo.connector.source_bridge.get_materialize_path',
                   return_value=destination):
            metrics = connector.export_query(sql, fingerprint, 'version')
        expected_source = f'({sql})'
    else:
        metrics = connector.export_table_read_only('LOAN', destination, columns)
        expected_source = '(SELECT * FROM "KUMO"."TEST_SCHEMA"."LOAN")'
    assert unload.call_args.kwargs['source'] == expected_source
    assert unload.call_args.kwargs['projection'] == (
        '"ID", CAST(CONVERT_TIMEZONE(\'UTC\', "TS") AS TIMESTAMP_NTZ) AS "TS"')
    assert clear.call_count == 2
    assert metrics.num_rows == 2
    assert metrics.source_size == _parquet_bytes(destination)
    assert pq.read_table(destination).to_pydict() == {'ID': [1, 2]}
    connector._context.cursor.return_value.fetch_arrow_batches.assert_not_called(
    )


def test_snowflake_validate_permissions_timeout_fails_closed():
    timeout_const = (
        snowflake_connector.SNOWFLAKE_CONNECTOR_PERMISSIONS_CHECK_TIMEOUT)

    connector = _make_snowflake_connector_with_mock_context()
    # Replace the ThreadPoolExecutor used inside validate_permissions with
    # a fake whose `future.result(timeout=...)` raises FuturesTimeoutError
    # synchronously, so the test doesn't actually wait 30 seconds.
    fake_future = Mock()
    fake_future.result.side_effect = FuturesTimeoutError("timed out")
    fake_executor = Mock()
    fake_executor.__enter__ = Mock(return_value=fake_executor)
    fake_executor.__exit__ = Mock(return_value=False)
    fake_executor.submit = Mock(return_value=fake_future)

    with patch('kumo.connector.snowflake_connector.ThreadPoolExecutor',
               return_value=fake_executor):
        with pytest.raises(ConnectorError, match='validation timed out'):
            connector.validate_permissions()

    fake_executor.submit.assert_called_once()
    fake_future.result.assert_called_once_with(timeout=timeout_const)
    fake_future.cancel.assert_called_once_with()
    fake_executor.shutdown.assert_called_once_with(wait=False,
                                                   cancel_futures=True)


def test_snowflake_download_copies_kumo_materialization(tmp_path):
    connector = _make_snowflake_connector_with_mock_context()
    source = tmp_path / 'persisted'
    source.mkdir()
    pq.write_table(pa.table({'ID': [1, 2]}), source / 'part.parquet')
    local_dest = tmp_path / 'local'
    table_fp = Mock(ingest_path_fpid='fpid_abc')

    with patch('kumo.data.utils.get_materialize_path',
               return_value=str(source)):
        connector.download('LOAN', table_fp, '1', str(local_dest))

    assert pq.read_table(local_dest / 'part.parquet').to_pydict() == {
        'ID': [1, 2]
    }


def _make_mock_connector():
    connector = SnowflakeConnector('snowflake', 'acc', 'warehouse', 'db',
                                   'schema')
    connector._context = Mock()
    return connector


def _install_common_mocks(monkeypatch, connector, sql_log):
    """Log metadata SQL and report that the destination table exists."""
    def _fake_meta_query(ctx, query, creds_clause=None):
        full = f"{query}\n{creds_clause}" if creds_clause else query
        sql_log.append(full)
        return pd.DataFrame()

    monkeypatch.setattr(connector, 'execute_meta_query', _fake_meta_query)
    monkeypatch.setattr(connector, 'table_exists', lambda name: True)


@pytest.mark.parametrize(
    ('artifact_path', 'credentials', 'from_clause', 'purge'),
    [
        (
            's3://bucket/preds/',
            {
                'AccessKeyId': 'AKIA',
                'SecretAccessKey': 'secret',
                'SessionToken': 'token',
            },
            "FROM 's3://bucket/preds/'",
            'FALSE',
        ),
        ('@MY_STAGE/predictions/', None, 'FROM @MY_STAGE/predictions/',
         'FALSE'),
    ],
    ids=['credentialed_s3', 'explicit_stage'],
)
def test_write_artifact_source_auth_modes(monkeypatch, artifact_path,
                                          credentials, from_clause, purge):
    sql_log = []
    connector = _make_mock_connector()
    _install_common_mocks(monkeypatch, connector, sql_log)
    credential_lookup = Mock(return_value=credentials)
    monkeypatch.setattr(snowflake_connector, 'get_temporary_aws_credential',
                        credential_lookup)

    connector.write_artifacts_to_table('PREDS', artifact_path, replace=False)

    assert not any(s.startswith('SHOW SCHEMAS') for s in sql_log), sql_log
    assert not any(s.startswith('PUT file://') for s in sql_log), sql_log
    assert not any('__kumo_artifact_stage_' in s for s in sql_log), sql_log
    copy_sqls = [s for s in sql_log if s.startswith('COPY INTO')]
    assert len(copy_sqls) == 1, sql_log
    assert from_clause in copy_sqls[0]
    assert f'PURGE = {purge}' in copy_sqls[0]
    assert ('CREDENTIALS = (' in copy_sqls[0]) is (credentials is not None)
    if credentials is None:
        credential_lookup.assert_not_called()
    else:
        credential_lookup.assert_called_once_with(artifact_path)


@pytest.mark.parametrize(
    ('embeddings', 'shards', 'nested', 'table_name'),
    [
        (False, 2, True, 'PREDS'),
        (True, 1, False, 'EMBEDDINGS'),
    ],
    ids=['nested_predictions', 'embedding'],
)
def test_write_local_artifacts_uses_isolated_user_stage(monkeypatch, tmp_path,
                                                        embeddings, shards,
                                                        nested, table_name):
    artifact_path = local_artifact_dataset(tmp_path, shards=shards,
                                           embeddings=embeddings, nested=nested)
    sql_log = []
    connector = _make_mock_connector()
    _install_common_mocks(monkeypatch, connector, sql_log)
    monkeypatch.setattr(
        snowflake_connector,
        'get_temporary_aws_credential',
        lambda _path: pytest.fail('stage loads must not request AWS creds'),
    )

    connector.write_artifacts_to_table(table_name, str(artifact_path))

    cursor = connector.context.cursor.return_value
    assert cursor._upload.call_count == shards
    locations = [
        call_.kwargs['stage_location'].rstrip('/')
        for call_ in cursor._upload.call_args_list
    ]
    stage_root = assert_indexed_parquet_parts(locations, suffix='') + '/'
    assert stage_root.startswith('@~/__kumo_artifact_')
    cursor.close.assert_called_once_with()
    copy_sql = next(sql for sql in sql_log if sql.startswith('COPY INTO'))
    assert f'FROM {stage_root}' in copy_sql
    assert 'PURGE = TRUE' in copy_sql
    assert f'REMOVE {stage_root}' in sql_log


def test_write_artifacts_to_table_partial_local_upload_is_removed(
        monkeypatch, tmp_path, caplog):
    artifact_path = local_artifact_dataset(tmp_path)
    sql_log = []
    connector = _make_mock_connector()
    _install_common_mocks(monkeypatch, connector, sql_log)
    secret = 'provider-response-with-private-path'
    connector.context.cursor.return_value._upload.side_effect = RuntimeError(
        secret)

    with pytest.raises(ConnectorError,
                       match='configured Snowflake output table'):
        connector.write_artifacts_to_table('PREDS', str(artifact_path))

    remove_sql = [sql for sql in sql_log if sql.startswith('REMOVE @~/')]
    assert len(remove_sql) == 1
    connector.context.cursor.return_value.close.assert_called_once_with()
    assert secret not in caplog.text


def test_query_schema_preserves_order_names_and_closes_cursor():
    from types import SimpleNamespace

    connector = _make_snowflake_connector_with_mock_context()
    cursor = connector._context.cursor.return_value
    cursor.describe.return_value = [
        SimpleNamespace(name=' mixed ', type_code=0, scale=0),
        SimpleNamespace(name=' mixed ', type_code=0, scale=2),
    ]

    metadata = connector.get_query_metadata('SELECT 1, 1.5')

    assert [(col.name, col.dtype)
            for col in metadata.cols] == [(' mixed ', Dtype.int64),
                                          (' mixed ', Dtype.float64)]
    cursor.describe.assert_called_once_with('SELECT 1, 1.5')
    cursor.close.assert_called_once()


def test_query_schema_closes_cursor_on_describe_failure():
    connector = _make_snowflake_connector_with_mock_context()
    cursor = connector._context.cursor.return_value
    cursor.describe.side_effect = RuntimeError('compile failed')

    with pytest.raises(ConnectorError, match='failed to describe'):
        connector.get_query_metadata('SELECT bad')

    cursor.close.assert_called_once()


@pytest.fixture(scope='module')
def generated_auth_key():
    from cryptography.hazmat.primitives.asymmetric import rsa

    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.mark.parametrize('encrypted', [False, True])
@pytest.mark.parametrize('escaped', [False, True])
@pytest.mark.parametrize('traditional', [False, True])
def test_hosted_and_local_key_credentials_produce_same_der(
        generated_auth_key, encrypted, escaped, traditional, caplog):
    from cryptography.hazmat.primitives import serialization
    from kumo_core.sources.principals import Principal
    from kumo_snowflake.session import _authentication
    from kumoapi.data_source import (
        SnowflakeAuthenticationKind,
        SnowflakeIdentityConfig,
        SnowflakeKeyPairCredentials,
    )
    from pydantic import SecretStr

    passphrase = 'test-only-passphrase' if encrypted else None
    encryption = (serialization.BestAvailableEncryption(passphrase.encode())
                  if encrypted else serialization.NoEncryption())
    pem = generated_auth_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL if traditional else
        serialization.PrivateFormat.PKCS8, encryption).decode()
    if escaped:
        pem = pem.replace('\n', '\\n')
    material = SnowflakeKeyPairCredentials(
        private_key=SecretStr(pem),
        private_key_passphrase=SecretStr(passphrase) if passphrase else None)
    principal = Principal(
        name='test-key', config=SnowflakeIdentityConfig(
            account='test-account', username='test-user',
            authentication=SnowflakeAuthenticationKind.KEY_PAIR),
        credentials=material)
    connector = _make_snowflake_connector_with_mock_context()
    hosted = connector.get_user_pass_or_key(
        'Snowflake', 'UNUSED_USER_ENV', 'UNUSED_PASS_ENV', credentials={
            'user': SecretStr('test-user'),
            'private_key': material.private_key,
            'private_key_passphrase': material.private_key_passphrase,
        }, key_encoding=serialization.Encoding.DER)
    local = _authentication(principal)
    expected = generated_auth_key.private_bytes(
        serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption())
    assert hosted == {'user': 'test-user', 'private_key': expected}
    assert local == {
        'account': 'test-account',
        'user': 'test-user',
        'private_key': expected
    }
    assert pem not in caplog.text
    assert 'test-only-passphrase' not in caplog.text


@pytest.mark.parametrize('encoding', [None, 'PEM'])
def test_hosted_key_encoding_contract_is_preserved(generated_auth_key,
                                                   encoding):
    from cryptography.hazmat.primitives import serialization
    from pydantic import SecretStr

    pem = generated_auth_key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption())
    connector = _make_snowflake_connector_with_mock_context()
    result = connector.get_user_pass_or_key(
        'Snowflake', 'UNUSED_USER_ENV', 'UNUSED_PASS_ENV', credentials={
            'user': SecretStr('test-user'),
            'private_key': SecretStr(pem.decode()),
        }, key_encoding=serialization.Encoding.PEM if encoding else None)
    actual = result['private_key']
    if encoding:
        assert actual == pem
    else:
        assert actual.private_numbers() == generated_auth_key.private_numbers()


@pytest.mark.parametrize('failure',
                         ['invalid', 'missing_passphrase', 'wrong_passphrase'])
def test_key_material_failures_preserve_redaction(generated_auth_key, failure,
                                                  caplog):
    from cryptography.hazmat.primitives import serialization
    from kumo_snowflake.session import _private_key_der
    from pydantic import SecretStr

    pem = generated_auth_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(
            b'test-only-passphrase')).decode()
    if failure == 'invalid':
        pem = 'invalid-private-material'
    passphrase = 'incorrect-passphrase' if failure == 'wrong_passphrase' else None
    connector = _make_snowflake_connector_with_mock_context()
    with pytest.raises(ConnectorError) as caught:
        connector.get_user_pass_or_key(
            'Snowflake', 'UNUSED_USER_ENV', 'UNUSED_PASS_ENV', credentials={
                'user':
                SecretStr('test-user'),
                'private_key':
                SecretStr(pem),
                'private_key_passphrase':
                SecretStr(passphrase) if passphrase else None,
            }, key_encoding=serialization.Encoding.DER)
    with pytest.raises((TypeError, ValueError)):
        _private_key_der(pem, passphrase)
    assert 'private key provided is invalid' in caught.value.get_user_message()
    assert pem not in caught.value.get_user_message()
    assert pem not in caplog.text
    assert 'incorrect-passphrase' not in caplog.text


def test_key_credentials_require_username(generated_auth_key):
    from cryptography.hazmat.primitives import serialization
    from pydantic import SecretStr

    pem = generated_auth_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()).decode()
    connector = _make_snowflake_connector_with_mock_context()
    with pytest.raises(ConnectorError, match='need a username'):
        connector.get_user_pass_or_key(
            'Snowflake', 'UNUSED_USER_ENV', 'UNUSED_PASS_ENV', credentials={
                'user': SecretStr(''),
                'private_key': SecretStr(pem),
            }, key_encoding=serialization.Encoding.DER)


def test_catalog_metadata_preserves_composite_primary_key_column_order(
        monkeypatch):
    from unittest.mock import MagicMock

    connector = SnowflakeConnector('catalog', account='account', warehouse='WH',
                                   database='DB', schema_name='PUBLIC')
    connector._context = MagicMock()
    cursor = connector._context.cursor.return_value.__enter__.return_value
    cursor.connection._paramstyle = 'pyformat'
    cursor.fetchall.return_value = [
        ('TABLE', 'Z_FIRST', 'NUMBER', 0),
        ('TABLE', 'A_SECOND', 'NUMBER', 0),
        ('TABLE', 'AMOUNT', 'NUMBER', 18),
    ]
    monkeypatch.setattr(
        connector, 'get_primary_keys', lambda context: {
            ('TABLE', name): PrimaryKeyConfig(name=name)
            for name in ('A_SECOND', 'Z_FIRST')
        })
    monkeypatch.setattr(connector, 'get_foreign_keys', lambda context: {})

    metadata = connector._fetch_table_metadata()['TABLE']

    assert metadata.pkey.name == 'Z_FIRST'
    assert [(col.name, col.dtype)
            for col in metadata.cols] == [('A_SECOND', Dtype.int64),
                                          ('AMOUNT', Dtype.float64)]
    query, params = cursor.execute.call_args.args
    assert query.endswith('ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION')
    assert params == ['PUBLIC']
    connector._context.cursor.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize('operation,phase', [
    ('columns', 'execute'),
    ('columns', 'fetchall'),
    ('list', 'execute'),
    ('view', 'execute'),
])
def test_catalog_metadata_failure_keeps_diagnostic_boundary_and_closes_cursor(
        operation, phase):
    from unittest.mock import MagicMock

    from io_utils.errors import (
        RESTRICTED_LOG_MESSAGE_ATTRIBUTE,
        SAFE_LOG_MESSAGE_ATTRIBUTE,
    )

    connector = _make_snowflake_connector_with_mock_context()
    context = MagicMock()
    connector._context = context
    cursor = context.cursor.return_value
    cursor.__enter__.return_value = cursor
    cursor.connection._paramstyle = 'pyformat'
    failure = _make_programming_error('catalog metadata unavailable')
    getattr(cursor, phase).side_effect = failure

    with pytest.raises(ProgrammingError) as caught:
        if operation == 'columns':
            connector.get_column_metadata(context)
        elif operation == 'list':
            connector.list_tables()
        else:
            connector.is_table_view('TABLE')

    assert caught.value is failure
    assert getattr(failure, RESTRICTED_LOG_MESSAGE_ATTRIBUTE)
    assert getattr(
        failure,
        SAFE_LOG_MESSAGE_ATTRIBUTE) == 'Snowflake metadata query failed'
    if operation == 'view':
        cursor.close.assert_called_once()
    else:
        cursor.__exit__.assert_called_once()


def test_sample_query_bounds_execution_and_returns_source_values():
    connector = _make_snowflake_connector_with_mock_context()
    cursor = connector.context.cursor.return_value
    cursor.fetch_arrow_all.return_value = pa.table({
        'ID': [1, 2],
        'VALUE': [1.123456789012345, None],
    })

    result = connector.sample_query('SELECT ID, VALUE FROM events', 2)

    assert result.ID.tolist() == [1, 2]
    assert result.VALUE.iloc[0] == 1.123456789012345
    assert pd.isna(result.VALUE.iloc[1])
    query, = cursor.execute.call_args.args
    assert query == 'SELECT * FROM (SELECT ID, VALUE FROM events) LIMIT 2'
    assert int(cursor.execute.call_args.kwargs['_statement_params']
               ['STATEMENT_TIMEOUT_IN_SECONDS']) > 0
    cursor.close.assert_called_once()
