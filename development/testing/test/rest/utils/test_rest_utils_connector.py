# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from kumo_core.types import ConnectorType
from kumo_core.types.schema import BaseColumnConfig
from kumoapi.typing import Dtype, Stype

from kumo.config.graph import TableMetadataConfig
from kumo.connector import (
    ParquetConnector,
    SnowflakeConnector,
    snowflake_connector,
)
from kumo.connector.connector import NUM_SAMPLE_ROWS
from kumo.testing import onlyFullTest, withSnowflakeCredentials
from rest.utils.connector import (
    get_query_columns_response,
    get_query_sample_rows_response,
    get_sample_rows_response,
    get_table_cols,
    stringify_sample_dataframe,
)
from statedb.utils.errors import (
    ConnectorValidationError,
    CustomSqlQueryError,
    sql_query_error_message,
)


def _fake_sql_connector(scan_estimate_bytes=None):
    """A SQL-capable fake warehouse whose describe/sample surface is recorded
    and whose physical-read surface asserts if ever hit (so a test can prove
    the query path never resolves a physical table)."""
    from test.graph.test_custom_sql_infer import _FakeSQLConnector

    query_cols = [
        BaseColumnConfig(name='id', dtype=Dtype.string),
        BaseColumnConfig(name='val', dtype=Dtype.float32),
    ]
    sample = pd.DataFrame({
        'id': [f'id{i}' for i in range(50)],
        'val': [float(i) for i in range(50)],
    })
    # Only a warehouse that gates is asked for an estimate.
    return _FakeSQLConnector('wh', query_cols, sample,
                             scan_estimate_bytes=scan_estimate_bytes,
                             gates_query_scan=scan_estimate_bytes is not None)


def test_get_query_columns_response_discovers_without_scan():
    """Query column discovery describes the query for its output columns
    (name + dtype + heuristic stype) and never scans data or resolves a
    physical table."""
    connector = _fake_sql_connector()

    resp = get_query_columns_response(connector,
                                      'SELECT id, val FROM transactions',
                                      ConnectorType.SNOWFLAKE)

    by_name = {c.name: c for c in resp.source_cols}
    assert list(by_name) == ['id', 'val']
    assert by_name['val'].dtype.value == 'float32'
    assert by_name['val'].stype is not None  # heuristic stype resolved
    assert connector.get_query_metadata_calls  # query was described
    assert connector.sample_query_calls == []  # no data scan / Option-A sample
    assert connector.physical_metadata_calls == []  # no physical lookup


def test_get_query_columns_response_rejects_obvious_write():
    """A data-modifying / multi-statement query is rejected before any
    warehouse describe by the advisory read-only pre-filter, and surfaces a
    clean public message (not a sanitized generic connector error)."""
    connector = _fake_sql_connector()

    with pytest.raises(ConnectorValidationError, match='single SQL statement'):
        get_query_columns_response(connector, 'SELECT 1; DROP TABLE t',
                                   ConnectorType.SNOWFLAKE)

    assert connector.get_query_metadata_calls == []


def test_get_query_columns_response_rejects_cross_namespace_read():
    connector = _fake_sql_connector()

    with pytest.raises(ConnectorValidationError, match='outside'):
        get_query_columns_response(connector,
                                   'SELECT * FROM OTHER.PRIVATE.transactions',
                                   ConnectorType.SNOWFLAKE)

    assert connector.get_query_metadata_calls == []


def test_get_query_sample_rows_response_bounded():
    """The "Load sample" path returns the query columns and at most ``num_rows``
    stringified rows from the bounded ``sample_query`` — never a physical
    read."""
    connector = _fake_sql_connector()

    resp = get_query_sample_rows_response(connector,
                                          'SELECT id, val FROM transactions',
                                          ConnectorType.SNOWFLAKE, num_rows=5)

    assert [c.name for c in resp.columns] == ['id', 'val']
    assert len(resp.rows) == 5
    assert resp.rows[0] == ['id0', '0.0']
    assert connector.sample_query_calls  # bounded sample ran
    assert connector.physical_metadata_calls == []


def test_get_query_sample_rows_response_serializes_for_router():
    """With ``serialize_to_json`` (the request-router default) the preview is
    returned as ``sample_df_serialized`` in the same shape as the physical
    sample path, so the response is UI-shape-compatible."""
    connector = _fake_sql_connector()

    resp = get_query_sample_rows_response(connector,
                                          'SELECT id, val FROM transactions',
                                          ConnectorType.SNOWFLAKE, num_rows=5,
                                          serialize_to_json=True)

    assert resp.sample_df_serialized is not None
    assert resp.rows == []


def test_stringify_sample_dataframe_handles_non_utf8_binary():
    rows = stringify_sample_dataframe(
        pd.DataFrame({'binary': [b'\xff\x00',
                                 bytearray(b'\x01'), None]}))

    assert rows.values.tolist() == [['0xff00'], ['0x01'], [None]]


def test_get_query_response_strips_trailing_terminator_for_subquery():
    """A trailing `;`/comment is stripped before the query reaches
    describe/sample, so a valid `SELECT …;` is not turned into invalid subquery
    SQL when embedded as `SELECT * FROM (<sql>) LIMIT n`."""
    connector = _fake_sql_connector()

    get_query_columns_response(connector, 'SELECT id, val FROM transactions;',
                               ConnectorType.SNOWFLAKE)
    get_query_sample_rows_response(connector,
                                   'SELECT id, val FROM transactions;',
                                   ConnectorType.SNOWFLAKE, num_rows=3)

    assert connector.get_query_metadata_calls == [
        'SELECT id, val FROM transactions'
    ] * 2
    assert connector.sample_query_calls == ['SELECT id, val FROM transactions']


def test_get_query_sample_rows_response_cost_gated_returns_columns_only():
    """A cost-gated query (e.g. BigQuery, where LIMIT does not bound the scan)
    returns the columns with no rows and never runs `sample_query`, so the
    "Load sample" button can't trigger a full scan of an arbitrary query."""
    from kumo.graph.table import SQL_SAMPLE_MAX_SCAN_BYTES

    connector = _fake_sql_connector(
        scan_estimate_bytes=SQL_SAMPLE_MAX_SCAN_BYTES + 1)

    resp = get_query_sample_rows_response(connector,
                                          'SELECT id, val FROM transactions',
                                          ConnectorType.SNOWFLAKE, num_rows=5)

    assert [c.name for c in resp.columns] == ['id', 'val']
    assert resp.rows == []
    assert connector.sample_query_calls == []  # gate blocked the scan


def test_get_query_sample_rows_response_rejects_obvious_write():
    """A data-modifying query is rejected before sampling, surfacing a clean
    public message."""
    connector = _fake_sql_connector()

    with pytest.raises(ConnectorValidationError, match='read-only'):
        get_query_sample_rows_response(connector, 'DELETE FROM t',
                                       ConnectorType.SNOWFLAKE, num_rows=5)

    assert connector.sample_query_calls == []


def test_get_query_sample_rows_response_clamps_oversized_num_rows():
    """An oversized preview request is clamped to the connector-wide sample cap
    so the pushed-down ``LIMIT`` can never be unbounded."""
    connector = _fake_sql_connector()

    with patch.object(connector, 'sample_query',
                      wraps=connector.sample_query) as spy:
        get_query_sample_rows_response(connector,
                                       'SELECT id, val FROM transactions',
                                       ConnectorType.SNOWFLAKE,
                                       num_rows=NUM_SAMPLE_ROWS + 10_000)

    assert spy.call_args.args[1] == NUM_SAMPLE_ROWS


class _WarehouseError(Exception):
    """Stand-in for a warehouse driver error (e.g. Snowflake compilation
    error): a non-``ValueError`` exception raised by describe/sample."""


def test_get_query_columns_response_surfaces_warehouse_describe_error():
    """A warehouse that rejects the query at describe time (e.g. an unknown
    function) surfaces the warehouse's own reason as a ``CustomSqlQueryError``,
    not the misleading generic "Connector source access failed"."""
    connector = _fake_sql_connector()
    reason = "SQL compilation error: Unknown function AI_GEN"
    connector.get_query_metadata = MagicMock(
        side_effect=_WarehouseError(reason))

    with pytest.raises(CustomSqlQueryError) as caught:
        get_query_columns_response(connector,
                                   "SELECT ai_gen('hi') AS ai FROM t",
                                   ConnectorType.SNOWFLAKE)

    message = str(caught.value)
    assert reason in message  # the warehouse's actionable reason is preserved
    assert message.startswith("The warehouse could not run this query.")
    assert "source access" not in message.lower()
    # A CustomSqlQueryError is on the activity-boundary allowlist.
    assert isinstance(caught.value, ConnectorValidationError)
    # The raw error stays chained for the server log, not flattened away.
    assert isinstance(caught.value.__cause__, _WarehouseError)


def test_get_query_sample_rows_response_surfaces_warehouse_sample_error():
    """A warehouse that rejects the bounded sample (after a successful describe)
    surfaces the warehouse's reason as a ``CustomSqlQueryError``."""
    connector = _fake_sql_connector()
    reason = "Statement reached its statement timeout"
    connector.sample_query = MagicMock(side_effect=_WarehouseError(reason))

    with pytest.raises(CustomSqlQueryError) as caught:
        get_query_sample_rows_response(connector,
                                       "SELECT id, val FROM transactions",
                                       ConnectorType.SNOWFLAKE, num_rows=5)

    message = str(caught.value)
    assert reason in message
    assert message.startswith("The warehouse could not run this query.")
    assert "source access" not in message.lower()


def test_get_query_sample_rows_response_surfaces_scan_estimate_error():
    """The BigQuery-style scan-estimate probe is a warehouse call too, so a
    failure there is surfaced as a ``CustomSqlQueryError`` rather than flattened
    to a generic access error."""
    connector = _fake_sql_connector()
    reason = "Query would process an unbounded amount of data"
    connector.get_query_scan_estimate_bytes = MagicMock(
        side_effect=_WarehouseError(reason))

    with pytest.raises(CustomSqlQueryError) as caught:
        get_query_sample_rows_response(connector,
                                       "SELECT id, val FROM transactions",
                                       ConnectorType.BIGQUERY, num_rows=5)

    assert reason in str(caught.value)
    assert connector.sample_query_calls == []  # never reached the sample scan


def test_get_query_columns_response_open_failure_stays_unwrapped():
    """A connection/auth failure happens in ``open`` — before any query runs —
    so it propagates unwrapped (the activity boundary keeps classifying it as
    "source access"), never mislabeled as a query failure."""
    connector = _fake_sql_connector()
    connector.open = MagicMock(side_effect=_WarehouseError("auth token secret"))
    connector.get_query_metadata = MagicMock()

    with pytest.raises(_WarehouseError):
        get_query_columns_response(connector,
                                   "SELECT id, val FROM transactions",
                                   ConnectorType.SNOWFLAKE)

    # The query was never described: the failure is pre-query, not a query
    # rejection, so it must not be turned into a CustomSqlQueryError here.
    connector.get_query_metadata.assert_not_called()


def test_query_advisory_read_only_message_has_no_warehouse_prefix():
    """The advisory read-only rejection keeps its own clean message (it runs
    before ``open``), and is never dressed up as a warehouse query failure."""
    connector = _fake_sql_connector()

    with pytest.raises(ConnectorValidationError) as caught:
        get_query_columns_response(connector, "DELETE FROM t",
                                   ConnectorType.SNOWFLAKE)

    message = str(caught.value)
    assert "read-only" in message
    assert not isinstance(caught.value, CustomSqlQueryError)
    assert "warehouse could not run" not in message.lower()


def test_sql_query_error_message_sanitizes_reason():
    """The shared sanitizer collapses whitespace and bounds length so a
    multi-line/oversized provider dump can never reach the client verbatim."""
    short = sql_query_error_message(_WarehouseError("line one\n  line two"))
    assert short == "The warehouse could not run this query. line one line two"

    long = sql_query_error_message(_WarehouseError("x" * 1000))
    # Prefix + space + <=400 chars of reason + single-char ellipsis.
    assert long.endswith("…")
    assert len(long) <= len("The warehouse could not run this query. ") + 401


def test_get_table_cols_preserves_inferred_multicategorical_delimiter():
    metadata = TableMetadataConfig(
        table_name='items',
        source_name='items',
        data_source='warehouse',
        cols=[
            BaseColumnConfig(
                name='tags',
                stype=Stype.multicategorical,
                dtype=Dtype.string,
                multicategorical_delimiter='|',
            ),
        ],
    )

    response = get_table_cols(metadata)

    assert response.source_cols[0].multicategorical_delimiter == '|'


def test_get_sample_rows_response():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', '..', 'parquet_data', 'FINANCIAL')
    connector = ParquetConnector('parquet', root_dir=root_dir)

    sample, _ = get_sample_rows_response(connector, 'TRANS', 12,
                                         skip_cache=True)
    assert len(sample.columns) == 10
    assert sample.columns[0].name == 'TRANS_ID'
    assert sample.columns[2].name == 'DATE'
    assert sample.columns[4].name == 'OPERATION'
    assert sample.columns[6].name == 'BALANCE'
    assert sample.columns[8].name == 'BANK'
    assert len(sample.rows) == 12
    assert sample.response_token == ''


@onlyFullTest
@pytest.mark.snowflake
@withSnowflakeCredentials
def test_get_sample_rows_response_snowflake():
    credentials = {
        'user': os.getenv(snowflake_connector._ENV_SNOWFLAKE_USER),
        'private_key': os.getenv(snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY)
    }

    connector = SnowflakeConnector('snowflake', account='xva19026',
                                   warehouse='WH_L', database='KUMO',
                                   schema_name='SPARK_SCALE_TEST')
    connector.open(credentials=credentials)

    sample, _ = get_sample_rows_response(connector, 'TEST300G1024', 10000,
                                         skip_cache=True)
    assert len(sample.columns) == 1024
    assert len(sample.rows) == 10000
    assert sample.response_token == ''


def test_get_sample_rows_async():
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', '..', 'parquet_data', 'FINANCIAL')
    connector = ParquetConnector('parquet', root_dir=root_dir)

    sample_rows, fut = get_sample_rows_response(connector, 'TRANS', 12,
                                                skip_cache=True, is_async=True)
    response_token = sample_rows.response_token
    assert response_token != ''
    assert len(response_token) == 32

    with pytest.raises(ValueError):
        _, _ = get_sample_rows_response(connector, 'TRANS', 12, skip_cache=True,
                                        response_token=response_token + '1')

    sample, _ = get_sample_rows_response(connector, 'TRANS', 12,
                                         skip_cache=True,
                                         response_token=response_token,
                                         is_async=True)
    if sample.response_token != '':
        assert sample.response_token == response_token

        # Force table sample fetching to finish.
        if fut is not None:
            fut.result()
        time.sleep(0.1)
        sample, _ = get_sample_rows_response(connector, 'TRANS', 12,
                                             skip_cache=True,
                                             response_token=response_token,
                                             is_async=True)

    assert len(sample.columns) == 10
    assert sample.columns[0].name == 'TRANS_ID'
    assert sample.columns[2].name == 'DATE'
    assert sample.columns[4].name == 'OPERATION'
    assert sample.columns[6].name == 'BALANCE'
    assert sample.columns[8].name == 'BANK'
    assert len(sample.rows) == 12


def test_get_sample_rows_failure(caplog):
    root_dir = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(root_dir, '..', '..', 'parquet_data', 'FINANCIAL')
    connector = ParquetConnector('parquet', root_dir=root_dir)

    provider_secret = 'known-provider-error-secret'
    sample_rows, fut = get_sample_rows_response(
        connector, 'TRANS', 12, skip_cache=True, is_async=True,
        mock_error=FileNotFoundError(provider_secret))
    response_token = sample_rows.response_token
    assert response_token != ''
    assert len(response_token) == 32

    with pytest.raises(ValueError,
                       match='Unable to prepare sample rows') as error:
        if fut is not None:
            fut.result()
        _, _ = get_sample_rows_response(connector, 'TRANS', 12, skip_cache=True,
                                        response_token=response_token,
                                        is_async=True)
    assert provider_secret not in str(error.value)
    assert provider_secret not in caplog.text

    root_dir = osp.join(root_dir, '..', '..', 'parquet_data', 'SYNTHETIC')
    connector = ParquetConnector('parquet_connector_2', root_dir=root_dir,
                                 file_type='parquet')

    sample_rows, fut = get_sample_rows_response(
        connector, source_table_name='unsupported_nested_schema', num_rows=2,
        is_async=True)
    response_token = sample_rows.response_token
    assert response_token != ''
    assert len(response_token) == 32

    with pytest.raises(ValueError, match='Unable to prepare sample rows'):
        if fut is not None:
            fut.result()
        _, _ = get_sample_rows_response(
            connector, source_table_name='unsupported_nested_schema',
            num_rows=2, response_token=response_token, is_async=True)
