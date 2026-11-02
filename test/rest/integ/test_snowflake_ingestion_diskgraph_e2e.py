# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snowflake diskgraph ingestion end-to-end tests.

These tests exercise the diskgraph table and graph ingestion paths —
``ingest_table_data_activity_v2`` →
``SnowflakeConnector.export_table`` (temporary-stage Parquet unload into
Kumo-owned storage) → ``connector.download`` → diskgraph ingest stats —
against the real Snowflake test account
(``xva19026``, database ``KUMO``, schema ``FINANCIAL_PYTEST``).
Training is intentionally NOT exercised: the contract under test
is the connector + ingestion workflow alone. Most tests run
``POST /v0/snapshots/graph`` and stop after graph refresh; the wide-integer
test uses ``POST /v1/tablesnapshots`` to exercise standalone column stats.

The graph-refresh coverage mirrors ``test_databricks_diskgraph_e2e.py``
(PR #28617). The activity-level short-circuit ported with that PR already
serves Snowflake.

Source data is the financial-pytest subset in
``KUMO.FINANCIAL_PYTEST``:
  - ``ACCOUNT`` (~4,500 rows)
  - ``LOAN`` (~682 rows; smallest, used for single-table test)
  - ``TRANS`` (largest)
  - ``EMPTY_TABLE`` (zero-row table for empty-ingest assertions)

Tests:
  - ``test_export_single_table_graph_to_staging``: 1-table graph
    (LOAN) refresh; parquet appears at the Kumo-owned ingest path.
  - ``test_export_multi_table_graph_to_staging``: 3-table graph
    (ACCOUNT + LOAN + TRANS) refresh; every table lands at its
    own ``ingest_path_fpid``.
  - ``test_export_short_circuit_unchanged_version``: refresh twice
    on the same connector + data version; materialized parquet metadata
    is byte-identical
    across runs (the diskgraph short-circuit at
    ``ingest_table_data_activity_v2.py`` skips re-export when the
    local NVMe data is still present).
  - ``test_export_mid_batch_failure``: graph references a
    non-existent source table; the API rejects the graph creation
    with HTTP 422 + ``"does not exist"`` before ingestion runs.
  - ``test_export_bad_credentials_raises``: profile registration with
    invalid password credentials fails with HTTP 422 before a connector can
    be created.
  - ``test_export_empty_table``: source table
    ``KUMO.FINANCIAL_PYTEST.EMPTY_TABLE`` has the fixture's
    declared columns but zero rows; graph creation passes,
    refresh proceeds, and the ingestion activity surfaces a
    ``"Table ... has 0 rows"`` validation error against the
    per-table snapshot (no exception raised, by design — see
    ``ingest_table_data_activity_v2.py:103-111``).
  - ``test_snowflake_wide_integer_rejected_during_table_refresh``: a real
    Snowflake ``NUMBER(38,0)`` result retains its staged decimal representation;
    an in-range value produces exact stats and an out-of-range value fails
    the refresh instead of being silently published as null.

Prerequisites:
  - Temporal dev server: ``make up-local-temporal``
  - ``diskgraph`` Python package installed (CI uses
    ``install-diskgraph: 'true'`` in ``testing_private.yml``)
  - ``DEPLOY_ENV_CONFIG_FILE`` pointing at a diskgraph deploy config
  - Snowflake credentials available via ``withSnowflakeCredentials``
    (AWS Secrets Manager ``xva19026.snowflakecomputing.com`` or
    ``SNOWFLAKE_USER`` / ``SNOWFLAKE_PRIVATE_KEY`` env vars)
  - The Kumo service account needs only read access to the source tables;
    ingestion creates a session-owned temporary stage, then drops it.

Usage:
    python -m pytest \\
        test/rest/integ/test_snowflake_ingestion_diskgraph_e2e.py \\
        -xvs --timeout=1800 --runsnowflake
"""
import dataclasses
import io
import logging
import os
import time
from decimal import Decimal
from test.rest.integ.test_diskgraph_e2e import (
    MAX_POLL_SECONDS,
    POLL_INTERVAL_SECONDS,
)

import kumoapi.data_source as data_source_api
import kumoapi.table as table_api
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from kumoapi.data_snapshot import (
    GraphSnapshotStage,
    TableSnapshotResource,
    TableSnapshotStage,
)
from kumoapi.typing import Dtype, Stype
from pydantic import SecretStr

from io_utils import fsutil
from io_utils.json import to_json
from kumo.connector import snowflake_connector
from kumo.data import MaterializationStage
from kumo.data.utils import get_materialize_path
from kumo.testing import withSnowflakeCredentials
from kumo.testing.rest import (
    create_identity_backed_connector,
    financial_snowflake_config,
    fixture_admin_user_authorization_context,
    get_and_expect_success,
    post_and_expect_success,
    randomize_fixture_ids_v2,
    remap_fixture_connector_ids_v2,
    snowflake_test_identity,
)
from rest.config.api_command import FixtureDefinition, FixturePQueryRecord
from rest.config.credentials import KeyPair
from rest.config.graph_v2 import (
    GraphDefinition,
    GraphSnapshotRequest,
    GraphSnapshotResponse,
)
from rest.config.status import IDResponse
from rest.config.table_v2 import (
    CustomSQLSourceTable,
    SnowflakeSourceTable,
    TableDefinition,
)
from statedb.config.common import JobStatus, ProjectID

log = logging.getLogger(__name__)


_CONNECTOR_NAME = "financial_snowflake_diskgraph"

# Source tables hand-provisioned in xva19026.KUMO.FINANCIAL_PYTEST.
_DATABASE = 'KUMO'
_SCHEMA = 'FINANCIAL_PYTEST'
_TABLE_ACCOUNT = 'ACCOUNT'
_TABLE_LOAN = 'LOAN'
_TABLE_TRANS = 'TRANS'
_TABLE_EMPTY = 'EMPTY_TABLE'
# Intentionally non-existent name used to drive the mid-batch
# failure surface.
_TABLE_DOES_NOT_EXIST = 'DOES_NOT_EXIST'
_I64_MAX = 9_223_372_036_854_775_807
_WIDE_INTEGER = 123_456_789_012_345_678_901_234_567_890

# Pool fixtures (storage_backend, _diskgraph_deploy_config,
# _pool_subprocesses) live in test/rest/integ/conftest.py; the
# diskgraph workflow needs the pool subprocesses to run.
# Snowflake streams into Kumo persistence before copying to the local
# dataplane, so we pin storage_backend to s3 to match the existing
# Snowflake training e2e — the local
# parametrization hits a kumo-side stats persistence issue
# unrelated to the Snowflake connector path.
pytestmark = [
    pytest.mark.ci_gate('snowflake-e2e'),
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", [
        pytest.param("s3", marks=pytest.mark.private_resource('aws')),
    ], indirect=True),
]


@pytest.fixture
def test_cleanup_options():
    """Preserve state across the test — don't reset DB between operations."""
    from rest.testing.config import TestFunctionCleanupOptions
    return TestFunctionCleanupOptions(
        should_reset_rest_db=False,
        should_reset_ml_db=False,
        should_reset_data_backend_cache=True,
        should_delete_data_persist_dir=True,
    )


# ----------------------------------------------------------------------------
# Table definition factories
# ----------------------------------------------------------------------------
#
# Column types mirror the existing training-e2e fixture
# (``test/rest/integ/test_cases/test_snowflake_e2e_fixtures.json``).
# Snowflake schema is static, so we hard-code the columns rather
# than discovering them at runtime.


def _account_table_def(connector_name: str,
                       source_name: str = _TABLE_ACCOUNT) -> TableDefinition:
    return TableDefinition(
        cols=[
            table_api.Column(name="FREQUENCY", stype=Stype.categorical,
                             dtype=Dtype.string, timestamp_format=None),
            table_api.Column(name="DATE", stype=Stype.timestamp,
                             dtype=Dtype.date, timestamp_format=None),
            table_api.Column(name="DISTRICT_ID", stype=Stype.ID,
                             dtype=Dtype.int, timestamp_format=None),
            table_api.Column(name="ACCOUNT_ID", stype=Stype.ID, dtype=Dtype.int,
                             timestamp_format=None),
        ],
        source_table=SnowflakeSourceTable(
            snowflake_connector_id=connector_name,
            database=_DATABASE,
            schema_name=_SCHEMA,
            table=source_name,
        ),
        pkey="ACCOUNT_ID",
        time_col=None,
        end_time_col=None,
    )


def _loan_table_def(connector_name: str,
                    source_name: str = _TABLE_LOAN) -> TableDefinition:
    return TableDefinition(
        cols=[
            table_api.Column(name="AMOUNT", stype=Stype.numerical,
                             dtype=Dtype.int, timestamp_format=None),
            table_api.Column(name="DURATION", stype=Stype.categorical,
                             dtype=Dtype.int, timestamp_format=None),
            table_api.Column(name="PAYMENTS", stype=Stype.numerical,
                             dtype=Dtype.float, timestamp_format=None),
            table_api.Column(name="STATUS", stype=Stype.categorical,
                             dtype=Dtype.string, timestamp_format=None),
            table_api.Column(name="ACCOUNT_ID", stype=Stype.ID, dtype=Dtype.int,
                             timestamp_format=None),
            table_api.Column(name="LOAN_ID", stype=Stype.ID, dtype=Dtype.int,
                             timestamp_format=None),
            table_api.Column(name="DATE", stype=Stype.timestamp,
                             dtype=Dtype.date, timestamp_format=None),
        ],
        source_table=SnowflakeSourceTable(
            snowflake_connector_id=connector_name,
            database=_DATABASE,
            schema_name=_SCHEMA,
            table=source_name,
        ),
        pkey="LOAN_ID",
        time_col="DATE",
        end_time_col=None,
    )


def _trans_table_def(connector_name: str,
                     source_name: str = _TABLE_TRANS) -> TableDefinition:
    return TableDefinition(
        cols=[
            table_api.Column(name="TYPE", stype=Stype.categorical,
                             dtype=Dtype.string, timestamp_format=None),
            table_api.Column(name="OPERATION", stype=Stype.categorical,
                             dtype=Dtype.string, timestamp_format=None),
            table_api.Column(name="AMOUNT", stype=Stype.numerical,
                             dtype=Dtype.float, timestamp_format=None),
            table_api.Column(name="BALANCE", stype=Stype.numerical,
                             dtype=Dtype.float, timestamp_format=None),
            table_api.Column(name="K_SYMBOL", stype=Stype.categorical,
                             dtype=Dtype.string, timestamp_format=None),
            table_api.Column(name="ACCOUNT_ID", stype=Stype.ID, dtype=Dtype.int,
                             timestamp_format=None),
            table_api.Column(name="TRANS_ID", stype=Stype.ID, dtype=Dtype.int,
                             timestamp_format=None),
            table_api.Column(name="DATE", stype=Stype.timestamp,
                             dtype=Dtype.date, timestamp_format=None),
        ],
        source_table=SnowflakeSourceTable(
            snowflake_connector_id=connector_name,
            database=_DATABASE,
            schema_name=_SCHEMA,
            table=source_name,
        ),
        pkey="TRANS_ID",
        time_col="DATE",
        end_time_col=None,
    )


def _wide_integer_table_def(connector_id: str, value: int) -> TableDefinition:
    return TableDefinition(
        cols=[
            table_api.Column(name='LOAN_ID', stype=Stype.ID, dtype=Dtype.int),
            table_api.Column(name='WIDE_INT', stype=Stype.numerical,
                             dtype=Dtype.int),
        ],
        source_table=CustomSQLSourceTable(
            connector_id=connector_id,
            table='snowflake_wide_integer_bounds',
            sql=('SELECT LOAN_ID, '
                 f"CAST('{value}' AS NUMBER(38,0)) AS WIDE_INT "
                 f'FROM {_DATABASE}.{_SCHEMA}.{_TABLE_LOAN} LIMIT 1'),
        ),
        pkey='LOAN_ID',
    )


# ----------------------------------------------------------------------------
# FixtureDefinition builders
# ----------------------------------------------------------------------------
#
# All builders embed a dummy ``FixturePQueryRecord`` to satisfy
# ``FixtureDefinition``'s schema and ``randomize_fixture_ids_v2``;
# the query string is never executed because the tests stop at
# graph refresh.

_DUMMY_QUERY_STR = "PREDICT LOAN.AMOUNT FOR EACH LOAN.LOAN_ID"


def _build_single_table_snowflake_fixture(
    connector_name: str = _CONNECTOR_NAME, ) -> FixtureDefinition:
    """Single-table fixture (LOAN only — smallest financial table)."""
    loan = _loan_table_def(connector_name)

    graph = GraphDefinition(tables={"LOAN": loan}, col_groups=[])

    return FixtureDefinition(
        tables=[loan],
        graphs=[graph],
        queries=[
            FixturePQueryRecord(
                query_str=_DUMMY_QUERY_STR,
                graph_fpid=graph.fpid,
                pquery_name="single_table_dummy",
            ),
        ],
    )


def _build_snowflake_fixture(
    connector_name: str = _CONNECTOR_NAME,
    account_source: str = _TABLE_ACCOUNT,
    loan_source: str = _TABLE_LOAN,
    trans_source: str = _TABLE_TRANS,
) -> FixtureDefinition:
    """3-table fixture (ACCOUNT + LOAN + TRANS).

    Source-table names are parameterized so the failure-mode tests
    can swap in non-existent tables for the mid-batch-failure path.
    """
    account = _account_table_def(connector_name, account_source)
    loan = _loan_table_def(connector_name, loan_source)
    trans = _trans_table_def(connector_name, trans_source)

    from kumoapi.graph import ColumnKey, ColumnKeyGroup
    graph = GraphDefinition(
        tables={
            "ACCOUNT": account,
            "LOAN": loan,
            "TRANS": trans,
        },
        col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey(table_name="ACCOUNT", col_name="ACCOUNT_ID"),
                ColumnKey(table_name="LOAN", col_name="ACCOUNT_ID"),
                ColumnKey(table_name="TRANS", col_name="ACCOUNT_ID"),
            )),
        ],
    )

    return FixtureDefinition(
        tables=[account, loan, trans],
        graphs=[graph],
        queries=[
            FixturePQueryRecord(
                query_str=_DUMMY_QUERY_STR,
                graph_fpid=graph.fpid,
                pquery_name="multi_table_dummy",
            ),
        ],
    )


def _build_empty_table_fixture(
        connector_name: str = _CONNECTOR_NAME) -> FixtureDefinition:
    """Single-table fixture pointing at the zero-row test table."""
    loan = _loan_table_def(connector_name, _TABLE_EMPTY)

    graph = GraphDefinition(tables={"LOAN": loan}, col_groups=[])

    return FixtureDefinition(
        tables=[loan],
        graphs=[graph],
        queries=[
            FixturePQueryRecord(
                query_str=_DUMMY_QUERY_STR,
                graph_fpid=graph.fpid,
                pquery_name="empty_table_dummy",
            ),
        ],
    )


# ----------------------------------------------------------------------------
# Materialization helpers
# ----------------------------------------------------------------------------


def _parse_data_version(table_snapshot_id: str) -> str:
    if '@' not in table_snapshot_id:
        raise ValueError(
            f'unexpected TableSnapshotID format: {table_snapshot_id!r}')
    return table_snapshot_id.rsplit('@', 1)[1]


def _materialize_path(response: GraphSnapshotResponse, source_name: str) -> str:
    for table_name, api_table_def in response.definition.tables.items():
        table_def = TableDefinition(**dataclasses.asdict(api_table_def))
        if table_def.get_source_table_name() != source_name:
            continue
        snapshot_id = str(response.snapshot_resource.table_ids[table_name])
        return get_materialize_path(
            table_def.fingerprint(),
            _parse_data_version(snapshot_id),
            MaterializationStage.INGEST,
        )
    raise AssertionError(f'No snapshot table uses source {source_name!r}')


def _table_materialize_path(resource: TableSnapshotResource) -> str:
    assert resource.table_definition is not None
    table_def = TableDefinition(**dataclasses.asdict(resource.table_definition))
    return get_materialize_path(
        table_def.fingerprint(),
        _parse_data_version(str(resource.id)),
        MaterializationStage.INGEST,
    )


def _list_staged_files(path: str) -> dict[str, tuple[int, float]]:
    if not fsutil.exists(path):
        return {}
    return {
        entry: (int(fsutil.filesize(entry)
                    or 0), fsutil.get_last_modified_time(entry))
        for entry in fsutil.listdir(path) if entry.endswith('.parquet')
    }


def _cleanup_materializations(paths: list[str]) -> None:
    for path in paths:
        try:
            fsutil.rm(path, ignore_errors=True)
        except Exception as error:  # noqa: BLE001 — diagnostics only
            log.warning('Cleanup failed for %s (error_type=%s)', path,
                        type(error).__name__)


def _assert_staged_parquet(staged: dict[str, tuple[int, float]],
                           source_name: str, path: str) -> None:
    assert staged, (
        f"expected Kumo-owned parquet for {source_name!r} under {path}, "
        "but found none")
    for parquet_path, (size, _last_modified) in staged.items():
        assert size > 0, f"empty staged parquet at {parquet_path}"


def _assert_wide_integer_parquet(path: str, expected: int,
                                 expect_decimal: bool) -> None:
    staged = _list_staged_files(path)
    _assert_staged_parquet(staged, 'NUMBER(38,0) Custom SQL', path)
    values = []
    for parquet_path in staged:
        parquet_file = pq.ParquetFile(
            io.BytesIO(fsutil.read_binary(parquet_path)))
        field = parquet_file.schema_arrow.field('WIDE_INT')
        if expect_decimal:
            assert pa.types.is_decimal128(field.type), field.type
            assert field.type.precision == 38
            assert field.type.scale == 0
        else:
            assert pa.types.is_int64(field.type), field.type
        values.extend(
            parquet_file.read(
                columns=['WIDE_INT']).column('WIDE_INT').to_pylist())
    expected_value = Decimal(expected) if expect_decimal else expected
    assert values == [expected_value]


# ----------------------------------------------------------------------------
# Refresh + wait helpers
# ----------------------------------------------------------------------------


def _connector_name(fixture: FixtureDefinition) -> str:
    table = next(iter(fixture.graphs[0].tables.values()))
    return str(table.get_connector_id())


def _register_connector(client: TestClient, fixture: FixtureDefinition):
    key_pair = KeyPair(
        user=SecretStr(os.environ[snowflake_connector._ENV_SNOWFLAKE_USER]),
        private_key=SecretStr(
            os.environ[snowflake_connector._ENV_SNOWFLAKE_PRIVATE_KEY]),
    )
    identity_config, credentials = snowflake_test_identity(key_pair)
    return create_identity_backed_connector(
        client,
        identity_config=identity_config,
        credentials=credentials,
        connector_config=financial_snowflake_config(_connector_name(fixture)),
    )


async def _setup_and_refresh(
    client: TestClient,
    fixture: FixtureDefinition,
):
    """Register connectors + graph and trigger a graph refresh.

    Returns the resulting ``graph_snapshot_id``. Same flow as the
    Databricks ingestion-only file (``test_databricks_diskgraph_e2e.py``):
      1. POST ``/v0/projects/default/connectors/`` for each connector in the fixture.
      2. POST ``/v1/graphs/?skip_validation=true`` to register the
         graph definition.
      3. Call ``data_snapshot_api_driver.create_graph_snapshot``
         directly via the in-process driver to kick off
         ``GraphRefreshWorkflow``.
    """
    temporary_connector_id = _connector_name(fixture)
    setup = _register_connector(client, fixture)
    fixture = remap_fixture_connector_ids_v2(
        fixture, {temporary_connector_id: setup.connector_id})

    assert len(fixture.graphs) == 1, (
        "ingestion tests use exactly one graph per fixture")
    graph_def = fixture.graphs[0]
    # ``?skip_validation=true`` so the API doesn't reject graphs
    # that reference deliberately-bad sources (e.g. the mid-batch
    # failure test points one table at a non-existent name).
    post_and_expect_success(
        client,
        '/v1/graphs/?skip_validation=true',
        graph_def,
        IDResponse,
        project_id=setup.project_id,
    )

    driver = client.app_builder._driver.data_snapshot_api_driver
    with fixture_admin_user_authorization_context():
        snapshot_id = await driver.create_graph_snapshot(
            graph_fpid=graph_def.fpid,
            refresh_source=False,
            project_id=ProjectID(setup.project_id),
        )
    return (snapshot_id, ProjectID(setup.project_id), setup.connector_id,
            graph_def.fpid)


def _wait_for_graph_snapshot(
    client: TestClient,
    graph_snapshot_id: str,
    project_id: ProjectID,
) -> GraphSnapshotResponse:
    """Poll ``POST /v0/snapshots/graph`` until INGEST stage is terminal."""
    deadline = time.time() + MAX_POLL_SECONDS
    while time.time() < deadline:
        resp = post_and_expect_success(
            client,
            '/v0/snapshots/graph',
            GraphSnapshotRequest(graph_snapshot_id=graph_snapshot_id),
            GraphSnapshotResponse,
            project_id=str(project_id),
        )
        ingest_state = resp.snapshot_resource.stages.get(
            GraphSnapshotStage.INGEST)
        if ingest_state and ingest_state.status.is_terminal:
            return resp
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Graph snapshot {graph_snapshot_id} did not complete INGEST "
        f"stage within {MAX_POLL_SECONDS}s")


def _wait_for_table_snapshot(
    client: TestClient,
    table_snapshot_id: str,
    project_id: ProjectID,
    terminal_stage: TableSnapshotStage,
) -> TableSnapshotResource:
    deadline = time.time() + MAX_POLL_SECONDS
    last_resource = None
    while time.time() < deadline:
        last_resource = get_and_expect_success(
            client,
            f'/v1/tablesnapshots/{table_snapshot_id}',
            {'project_id': str(project_id)},
            TableSnapshotResource,
        )
        state = last_resource.stages.get(terminal_stage)
        if state is not None and state.status.is_terminal:
            return last_resource
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Table snapshot {table_snapshot_id} did not complete "
                       f"{terminal_stage} within {MAX_POLL_SECONDS}s; "
                       f"last_resource={last_resource}")


def _snapshot_stat_value(stats: dict, key: str):
    value = stats[key]
    if isinstance(value, dict) and 'value' in value:
        return value['value']
    return value.value if hasattr(value, 'value') else value


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


@pytest.mark.snowflake
@pytest.mark.asyncio
@pytest.mark.timeout(900)
@withSnowflakeCredentials
async def test_snowflake_wide_integer_rejected_during_table_refresh(
    client: TestClient,
    public_api_client: TestClient,
    temporal_client,
    storage_backend,
):
    """Reject an out-of-range NUMBER(38,0) instead of publishing null stats."""
    fixture = randomize_fixture_ids_v2(_build_single_table_snowflake_fixture())
    setup = _register_connector(client, fixture)
    materialize_paths: list[str] = []
    in_range_table = _wide_integer_table_def(setup.connector_id, _I64_MAX)
    out_of_range_table = _wide_integer_table_def(setup.connector_id,
                                                 _WIDE_INTEGER)

    try:
        for name_alias, table_def in (
            ('wide_integer_in_range', in_range_table),
            ('wide_integer_out_of_range', out_of_range_table),
        ):
            post_and_expect_success(
                public_api_client,
                '/v1/tables',
                table_def,
                IDResponse,
                name_alias=name_alias,
                project_id=setup.project_id,
            )

        in_range_id = post_and_expect_success(
            public_api_client,
            '/v1/tablesnapshots',
            in_range_table,
            IDResponse,
            project_id=setup.project_id,
            refresh_source=True,
        ).id
        in_range = _wait_for_table_snapshot(
            public_api_client,
            in_range_id,
            ProjectID(setup.project_id),
            TableSnapshotStage.FULL_COL_STATS,
        )
        assert in_range.stages[
            TableSnapshotStage.INGEST].status == JobStatus.DONE
        assert (in_range.stages[TableSnapshotStage.FULL_COL_STATS].status ==
                JobStatus.DONE)
        in_range_path = _table_materialize_path(in_range)
        materialize_paths.append(in_range_path)
        _assert_wide_integer_parquet(in_range_path, _I64_MAX,
                                     expect_decimal=True)
        assert in_range.column_stats is not None
        wide_stats = next(column.stats for column in in_range.column_stats
                          if column.column_name == 'WIDE_INT')
        assert _snapshot_stat_value(wide_stats, 'MIN') == _I64_MAX
        assert _snapshot_stat_value(wide_stats, 'MAX') == _I64_MAX
        assert _snapshot_stat_value(wide_stats, 'NA_FRACTION') == 0

        out_of_range_id = post_and_expect_success(
            public_api_client,
            '/v1/tablesnapshots',
            out_of_range_table,
            IDResponse,
            project_id=setup.project_id,
            refresh_source=True,
        ).id
        out_of_range = _wait_for_table_snapshot(
            public_api_client,
            out_of_range_id,
            ProjectID(setup.project_id),
            TableSnapshotStage.MIN_COL_STATS,
        )
        assert (out_of_range.stages[TableSnapshotStage.INGEST].status ==
                JobStatus.DONE)
        min_stats = out_of_range.stages[TableSnapshotStage.MIN_COL_STATS]
        assert min_stats.status == JobStatus.FAILED
        assert min_stats.error is not None
        assert "column 'WIDE_INT'" in min_stats.error
        assert 'signed 64-bit integer range' in min_stats.error
        assert 'DOUBLE or STRING in Custom SQL' in min_stats.error
        assert (out_of_range.stages[TableSnapshotStage.FULL_COL_STATS].status ==
                JobStatus.CANCELLED)
        assert out_of_range.column_stats is None
        out_of_range_path = _table_materialize_path(out_of_range)
        materialize_paths.append(out_of_range_path)
        _assert_wide_integer_parquet(out_of_range_path, _WIDE_INTEGER,
                                     expect_decimal=True)
    finally:
        _cleanup_materializations(materialize_paths)


@pytest.mark.snowflake
@pytest.mark.asyncio
@pytest.mark.timeout(900)
@withSnowflakeCredentials
async def test_export_single_table_graph_to_staging(
    client: TestClient,
    temporal_client,
    storage_backend,
):
    """Single-table graph: ingestion writes Kumo-owned parquet.

    Drives ``ingest_table_data_activity_v2`` →
    ``SnowflakeConnector.export_table`` for a graph that has one
    table (LOAN, smallest financial table for fast CI) and asserts
    its materialization contains parquet.
    """
    fixture = _build_single_table_snowflake_fixture()
    fixture = randomize_fixture_ids_v2(fixture)
    materialize_paths: list[str] = []

    try:
        graph_snapshot_id, project_id, _connector_id, _ = (await
                                                           _setup_and_refresh(
                                                               client, fixture))
        resp = _wait_for_graph_snapshot(client, graph_snapshot_id, project_id)
        ingest = resp.snapshot_resource.stages[GraphSnapshotStage.INGEST]
        assert ingest.status == JobStatus.DONE, (
            f"INGEST stage failed: status={ingest.status}, "
            f"error={ingest.error}")

        path = _materialize_path(resp, _TABLE_LOAN)
        materialize_paths.append(path)
        staged = _list_staged_files(path)
        _assert_staged_parquet(staged, _TABLE_LOAN, path)
        log.info(
            "Single-table staging assertion passed for %s "
            "(%d staged files)", _TABLE_LOAN, len(staged))
    finally:
        _cleanup_materializations(materialize_paths)


@pytest.mark.snowflake
@pytest.mark.asyncio
@pytest.mark.timeout(900)
@withSnowflakeCredentials
async def test_export_multi_table_graph_to_staging(
    client: TestClient,
    temporal_client,
    storage_backend,
):
    """3-table graph: each table lands at its own Kumo-owned ingest path.

    Production graph refresh fans out per-table
    ``TableRefreshWorkflow`` instances in parallel; each runs its
    own ``ingest_table_data_activity_v2`` → ``export_table``.
    Test that ACCOUNT, LOAN, and TRANS all land at distinct paths.
    """
    fixture = _build_snowflake_fixture()
    fixture = randomize_fixture_ids_v2(fixture)
    table_to_path: dict[str, str] = {}

    try:
        graph_snapshot_id, project_id, _connector_id, _ = (await
                                                           _setup_and_refresh(
                                                               client, fixture))
        resp = _wait_for_graph_snapshot(client, graph_snapshot_id, project_id)
        ingest = resp.snapshot_resource.stages[GraphSnapshotStage.INGEST]
        assert ingest.status == JobStatus.DONE, (
            f"INGEST stage failed: status={ingest.status}, "
            f"error={ingest.error}")

        table_to_path = {
            source: _materialize_path(resp, source)
            for source in (_TABLE_ACCOUNT, _TABLE_LOAN, _TABLE_TRANS)
        }
        assert len(set(table_to_path.values())) == len(table_to_path)
        for source, path in table_to_path.items():
            staged = _list_staged_files(path)
            _assert_staged_parquet(staged, source, path)
        log.info("Multi-table staging assertion passed for %d tables",
                 len(table_to_path))
    finally:
        _cleanup_materializations(list(table_to_path.values()))


@pytest.mark.snowflake
@pytest.mark.asyncio
@pytest.mark.timeout(1500)
@withSnowflakeCredentials
async def test_export_short_circuit_unchanged_version(
    client: TestClient,
    temporal_client,
    storage_backend,
):
    """Re-running graph refresh with unchanged data does not re-export.

    Run 1 primes Kumo-owned persistence and the materialization-stage cache
    for every table. Run 2 hits
    the same connector + data version; for each table the activity
    sees the same ``(fingerprint, data_version)`` key, finds the
    local NVMe ingest data still on disk, and short-circuits
    before calling ``export_table``.

    Signal: ``(path, size, last_modified)`` tuples in Kumo persistence. If
    the activity re-ran ``export_table``, the files would have fresh timestamps.
    Comparing the set across runs gives a clean equality check.
    """
    fixture = _build_snowflake_fixture()
    fixture = randomize_fixture_ids_v2(fixture)
    table_to_path: dict[str, str] = {}

    try:
        first_id, project_id, _connector_id, graph_fpid = (await
                                                           _setup_and_refresh(
                                                               client, fixture))
        first_resp = _wait_for_graph_snapshot(client, first_id, project_id)
        first_ingest = first_resp.snapshot_resource.stages[
            GraphSnapshotStage.INGEST]
        assert first_ingest.status == JobStatus.DONE, (
            f"First refresh failed: {first_ingest.error}")

        table_to_path = {
            source: _materialize_path(first_resp, source)
            for source in (_TABLE_ACCOUNT, _TABLE_LOAN, _TABLE_TRANS)
        }
        before: dict[str, dict[str, tuple[int, float]]] = {}
        for source, path in table_to_path.items():
            staged = _list_staged_files(path)
            _assert_staged_parquet(staged, source, path)
            before[source] = staged
        log.info("Captured materialization listings for %d tables after run",
                 len(before))

        # Trigger the same graph refresh a second time. Same
        # connector + same fixture → same fingerprint + same
        # data_version, so the activity should short-circuit before
        # re-exporting.
        driver = client.app_builder._driver.data_snapshot_api_driver
        with fixture_admin_user_authorization_context():
            second_id = await driver.create_graph_snapshot(
                graph_fpid=graph_fpid,
                refresh_source=False,
                project_id=project_id,
            )
        second_resp = _wait_for_graph_snapshot(client, second_id, project_id)
        second_ingest = second_resp.snapshot_resource.stages[
            GraphSnapshotStage.INGEST]
        assert second_ingest.status == JobStatus.DONE, (
            f"Second refresh failed: {second_ingest.error}")

        for source, path in table_to_path.items():
            after = _list_staged_files(path)
            assert after == before[source], (
                f"Second ingestion re-wrote Kumo-owned files for "
                f"{source!r} — diskgraph short-circuit failed. "
                f"before={before[source]}, after={after}")
        log.info("Short-circuit assertion passed for %d tables.",
                 len(table_to_path))
    finally:
        _cleanup_materializations(list(table_to_path.values()))


@pytest.mark.snowflake
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@withSnowflakeCredentials
async def test_export_mid_batch_failure(
    client: TestClient,
    temporal_client,
    storage_backend,
):
    """One bad table → graph creation rejects with 422 before ingestion.

    Builds a 3-table fixture where ``TRANS`` references a
    non-existent source table. Kumo validates source-table
    existence at graph creation time (``POST /v1/graphs/``) — even
    with ``skip_validation=true``, the connector's table catalog
    is consulted and the bad reference is rejected with HTTP 422
    plus a ``"does not exist"`` message. That's the user-visible
    failure surface for a "halfway through batch" scenario; the
    ingestion activity never runs because the graph never gets
    registered.
    """
    fixture = _build_snowflake_fixture(trans_source=_TABLE_DOES_NOT_EXIST)
    fixture = randomize_fixture_ids_v2(fixture)

    # Register the connector first so the API has somewhere to
    # look up the bad table reference.
    setup = _register_connector(client, fixture)

    graph_def = fixture.graphs[0]
    resp = client.post('/v1/graphs/?skip_validation=true',
                       data=to_json(graph_def, insecure=True),
                       params={'project_id': setup.project_id},
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 422, (
        f"expected 422 for graph with non-existent source "
        f"{_TABLE_DOES_NOT_EXIST!r}, got {resp.status_code}: "
        f"{resp.text}")
    detail = resp.json().get('detail', '')
    assert 'does not exist' in detail, (
        f"expected 'does not exist' in error detail, got: {detail!r}")
    log.info("Mid-batch (bad-table) failure surfaced at API: %s", detail)


@pytest.mark.snowflake
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@withSnowflakeCredentials
async def test_export_bad_credentials_raises(
    client: TestClient,
    temporal_client,
    storage_backend,
):
    """Bad username/password credentials fail identity validation."""
    response = client.post(
        '/v0/identities',
        data=to_json(
            data_source_api.CreateIdentityArgs(
                display_name='invalid-snowflake-identity',
                config=data_source_api.SnowflakeIdentityConfig(
                    account='xva19026',
                    username='bad_user',
                    authentication=(
                        data_source_api.SnowflakeAuthenticationKind.PASSWORD),
                ),
                credentials=data_source_api.SnowflakePasswordCredentials(
                    password=SecretStr('bad_password')),
            ),
            insecure=True,
        ),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422, response.text


@pytest.mark.snowflake
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@withSnowflakeCredentials
async def test_export_empty_table(
    client: TestClient,
    temporal_client,
    storage_backend,
):
    """Empty source table → ingestion surfaces a ``"0 rows"`` validation error.

    The empty fixture (``KUMO.FINANCIAL_PYTEST.EMPTY_TABLE``)
    carries the LOAN schema the fixture declares, so graph creation
    passes schema validation. Refresh proceeds normally: the
    Snowflake stream writes a zero-row parquet, then
    ``ingest_table_data_activity_v2.py:103-111`` notices
    ``NUM_ROWS == 0`` and appends a ``ValidationError`` ("Table ...
    has 0 rows") to the per-table ``validation_response`` without
    raising. The graph snapshot's INGEST stage records that
    validation error against the empty table.

    Test asserts the ``"0 rows"`` string surfaces somewhere
    user-visible in the snapshot — graph INGEST stage error /
    warnings, or the per-table snapshot's stage payload.
    """
    fixture = _build_empty_table_fixture()
    fixture = randomize_fixture_ids_v2(fixture)

    graph_snapshot_id, project_id, _, _ = await _setup_and_refresh(
        client, fixture)
    resp = _wait_for_graph_snapshot(client, graph_snapshot_id, project_id)

    ingest = resp.snapshot_resource.stages[GraphSnapshotStage.INGEST]
    log.info(
        "Empty-table refresh ended with INGEST status=%s, "
        "error=%s, warnings=%s", ingest.status, ingest.error, ingest.warnings)

    candidates: list[str] = []
    if ingest.error:
        candidates.append(ingest.error)
    candidates.extend(ingest.warnings or [])

    table_snap_id = resp.snapshot_resource.table_ids["LOAN"]
    try:
        table_resp = client.get(f'/v0/snapshots/table/{table_snap_id}')
        if table_resp.status_code == 200:
            payload = table_resp.json()
            snap = payload.get('snapshot') or {}
            for stage in (snap.get('stages') or {}).values():
                if isinstance(stage, dict):
                    if stage.get('error'):
                        candidates.append(stage['error'])
                    candidates.extend(stage.get('warnings') or [])
    except Exception as e:  # noqa: BLE001 — diagnostics only
        log.warning("Could not fetch table snapshot %s: %s", table_snap_id, e)

    assert any('0 rows' in (msg or '') for msg in candidates), (
        f"expected '0 rows' to surface in snapshot for empty table; "
        f"candidates={candidates}, ingest_status={ingest.status}")
    log.info("Empty-table '0 rows' validation surfaced in snapshot.")
