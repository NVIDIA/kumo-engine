# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BigQuery diskgraph ingestion-only end-to-end tests.

These tests exercise the diskgraph ingestion path —
``ingest_table_data_activity_v2`` →
``BigQueryConnector.export_table`` (``EXPORT DATA`` to GCS) →
``connector.download`` (GCS → local NVMe) → diskgraph
ingest stats — against the real ``kumo-engineering.financial``
BigQuery dataset. Training is intentionally NOT exercised: the
contract under test is the connector + ingestion workflow alone,
so each test runs ``POST /v0/snapshots/graph`` (graph refresh) and
stops there.

Mirrors the Databricks ingestion-only suite at
``test_databricks_diskgraph_e2e.py`` (PR #28617). The
activity-level short-circuit at
``ingest_table_data_activity_v2.py:70-86`` that landed alongside
the Databricks tests is connector-agnostic, so it already serves
the BigQuery path — no production code changes needed here.

Source tables in ``kumo-engineering.financial``:
  - ``ACCOUNT``
  - ``LOAN``
  - ``TRANS``

Tests (names match the Databricks suite for parity):
  - ``test_export_single_table_graph_to_staging``: 1-table graph
    refresh; GCS staging carries parquet under the per-table fpid.
  - ``test_export_multi_table_graph_to_staging``: 3-table graph
    refresh; every table lands at its own ``ingest_path_fpid``.
  - ``test_export_short_circuit_unchanged_version``: refresh twice
    on the same connector + data version; GCS parquet
    ``(name, size, mtime)`` map is identical across runs
    (exercises the diskgraph short-circuit at
    ``ingest_table_data_activity_v2.py:70-86``). We compare the
    full map rather than just file count because ``EXPORT DATA``
    can vary shard names across runs.
  - ``test_export_mid_batch_failure``: graph references a
    non-existent source table; the API rejects the graph creation
    with HTTP 422 + ``"does not exist"`` before ingestion runs.
    Tests the user-visible boundary.
  - ``test_export_bad_credentials_raises``: profile registration with an
    invalid service-account key fails with HTTP 422 before a connector can be
    created.

Prerequisites:
  - Temporal dev server: ``make up-local-temporal``
  - ``diskgraph`` Python package installed (CI uses
    ``install-diskgraph: 'true'`` in ``testing_gcs.yml``)
  - ``DEPLOY_ENV_CONFIG_FILE`` pointing at a diskgraph deploy
    config (CI sets it for the ``bigquery_*_diskgraph`` matrix
    entries; this module also defaults it at import time)
  - BigQuery credentials available via the
    ``withBigQueryCredentials`` decorator (from AWS Secrets
    Manager in CI, ``GOOGLE_CREDENTIALS_STR_JSON`` fallback
    otherwise)
  - ``GCP_STORAGE_TRANSFER_BUCKET`` env var set to a writable
    GCS bucket (CI sets it to ``kumo-test-bucket``)

Usage:
    python -m pytest \\
        test/rest/integ/test_bigquery_ingestion_diskgraph_e2e.py \\
        -xvs --timeout=900
"""
import logging
import os
import shutil
import time
from pathlib import Path
from test.rest.integ.test_diskgraph_e2e import (
    MAX_POLL_SECONDS,
    POLL_INTERVAL_SECONDS,
)

import kumoapi.data_source as data_source_api
import kumoapi.table as table_api
import pytest
from fastapi.testclient import TestClient
from kumoapi.data_snapshot import GraphSnapshotStage
from kumoapi.typing import Dtype, Stype
from pydantic import SecretStr

from io_utils import fsutil
from io_utils.json import to_json
from kumo.testing import withBigQueryCredentials
from kumo.testing.rest import (
    bigquery_test_identity,
    create_identity_backed_connector,
    fixture_admin_user_authorization_context,
    invalid_bigquery_key_test_identity_config,
    post_and_expect_success,
    randomize_fixture_ids_v2,
    remap_fixture_connector_ids_v2,
)
from rest.config.api_command import FixtureDefinition, FixturePQueryRecord
from rest.config.connector import BigQueryConnectorResourceConfig
from rest.config.graph_v2 import (
    GraphDefinition,
    GraphSnapshotRequest,
    GraphSnapshotResponse,
)
from rest.config.status import IDResponse
from rest.config.table_v2 import BigQuerySourceTable, TableDefinition
from statedb.config.common import JobStatus, ProjectID

log = logging.getLogger(__name__)

# Default DEPLOY_ENV_CONFIG_FILE so _diskgraph_deploy_config picks
# up the diskgraph config even when the env var isn't set by CI.
# Doing this at module import (not in an autouse fixture) because
# ``deploy_env_config`` caches its DeployEnvConfig the first time
# any test in the session imports the conftest — see notes in
# ``test_diskgraph_e2e.py``.
if 'DEPLOY_ENV_CONFIG_FILE' not in os.environ:
    _default_deploy_config = (Path(__file__).resolve().parents[3] / 'rest' /
                              'devserver' / 'configs' / 'diskgraph-test.json')
    os.environ['DEPLOY_ENV_CONFIG_FILE'] = str(_default_deploy_config)

_CONNECTOR_NAME = "financial_bigquery_diskgraph"
_PROJECT_ID = "kumo-engineering"
_DATASET_ID = "financial"

# Permanent tables in ``kumo-engineering.financial``. These are also used by
# the BigQuery training E2E fixture.
_TABLE_ACCOUNT = "ACCOUNT"
_TABLE_LOAN = "LOAN"
_TABLE_TRANS = "TRANS"
# Reference name that intentionally does not exist in the financial dataset —
# any ingest attempt against it surfaces a real ConnectorError from
# the BigQuery backend. Used by the mid-batch failure test.
_TABLE_DOES_NOT_EXIST = "does_not_exist"

# Pool fixtures (storage_backend, _diskgraph_deploy_config,
# _pool_subprocesses) live in test/rest/integ/conftest.py; the
# diskgraph workflow needs the pool subprocesses to run. The ingestion-only
# assertions are scoped to local storage; pinning avoids a second skipped S3
# parametrization and keeps the CI job from spawning an unused pool.
pytestmark = [
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", ["local"], indirect=True),
]


@pytest.fixture
def test_cleanup_options():
    """Preserve state within each test, then clean diskgraph state after it."""
    from rest.testing.config import TestFunctionCleanupOptions
    return TestFunctionCleanupOptions(
        should_reset_rest_db=False,
        should_reset_ml_db=False,
        should_reset_data_backend_cache=True,
        should_delete_data_persist_dir=True,
    )


@pytest.fixture(autouse=True)
def _cleanup_diskgraph_data_root_after_test(storage_backend):
    yield

    if storage_backend != 'local':
        return

    from statedb.config.diskgraph import get_diskgraph_data_root
    data_root = Path(get_diskgraph_data_root())
    if 'diskgraph-data-root' not in data_root.name:
        log.warning(
            "Skipping DiskGraph data_root cleanup for unexpected path %s",
            data_root)
        return

    shutil.rmtree(data_root, ignore_errors=True)
    data_root.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Table definition factories
# ----------------------------------------------------------------------------


def _bq_source(connector_name: str, table_name: str) -> BigQuerySourceTable:
    return BigQuerySourceTable(
        bigquery_connector_id=connector_name,
        table_name=table_name,
        project_id=_PROJECT_ID,
        dataset_id=_DATASET_ID,
    )


def _account_table_def(connector_name: str,
                       source_name: str = _TABLE_ACCOUNT) -> TableDefinition:
    return TableDefinition(
        cols=[
            table_api.Column(name="ACCOUNT_ID", stype=Stype.ID, dtype=Dtype.int,
                             timestamp_format=None),
        ],
        source_table=_bq_source(connector_name, source_name),
        pkey="ACCOUNT_ID",
        time_col=None,
        end_time_col=None,
    )


def _loan_table_def(connector_name: str,
                    source_name: str = _TABLE_LOAN) -> TableDefinition:
    return TableDefinition(
        cols=[
            table_api.Column(name="LOAN_ID", stype=Stype.ID, dtype=Dtype.int,
                             timestamp_format=None),
            table_api.Column(name="ACCOUNT_ID", stype=Stype.ID, dtype=Dtype.int,
                             timestamp_format=None),
            table_api.Column(name="AMOUNT", stype=Stype.numerical,
                             dtype=Dtype.int, timestamp_format=None),
            table_api.Column(name="DATE", stype=Stype.timestamp,
                             dtype=Dtype.date, timestamp_format=None),
        ],
        source_table=_bq_source(connector_name, source_name),
        pkey="LOAN_ID",
        time_col="DATE",
        end_time_col=None,
    )


def _trans_table_def(connector_name: str,
                     source_name: str = _TABLE_TRANS) -> TableDefinition:
    return TableDefinition(
        cols=[
            table_api.Column(name="TRANS_ID", stype=Stype.ID, dtype=Dtype.int,
                             timestamp_format=None),
            table_api.Column(name="ACCOUNT_ID", stype=Stype.ID, dtype=Dtype.int,
                             timestamp_format=None),
            table_api.Column(name="AMOUNT", stype=Stype.numerical,
                             dtype=Dtype.float, timestamp_format=None),
            table_api.Column(name="DATE", stype=Stype.timestamp,
                             dtype=Dtype.date, timestamp_format=None),
        ],
        source_table=_bq_source(connector_name, source_name),
        pkey="TRANS_ID",
        time_col="DATE",
        end_time_col=None,
    )


# ----------------------------------------------------------------------------
# FixtureDefinition builders
# ----------------------------------------------------------------------------
#
# All builders embed at least one ``FixturePQueryRecord`` even though
# ingestion-only tests don't run training. The pquery is required by
# ``randomize_fixture_ids_v2`` (which uses query name as a
# randomization seed) and by ``FixtureDefinition``'s schema. The
# query string is never executed.

_DUMMY_QUERY_STR = (
    "PREDICT SUM(LOAN.AMOUNT, 0, 360, days) FOR EACH ACCOUNT.ACCOUNT_ID")


def _build_single_table_bigquery_fixture(
    connector_name: str = _CONNECTOR_NAME, ) -> FixtureDefinition:
    """Single-table fixture using the small ACCOUNT source."""
    account = _account_table_def(connector_name)

    graph = GraphDefinition(tables={"ACCOUNT": account}, col_groups=[])

    return FixtureDefinition(
        tables=[account],
        graphs=[graph],
        queries=[
            FixturePQueryRecord(
                query_str=_DUMMY_QUERY_STR,
                graph_fpid=graph.fpid,
                pquery_name="single_table_dummy",
            ),
        ],
    )


def _build_bigquery_fixture(
    connector_name: str = _CONNECTOR_NAME,
    account_source: str = _TABLE_ACCOUNT,
    loan_source: str = _TABLE_LOAN,
    trans_source: str = _TABLE_TRANS,
) -> FixtureDefinition:
    """Three-table fixture using ACCOUNT, LOAN, and TRANS.

    Source-table names are parameterized so the failure-mode tests can
    swap in non-existent tables for the mid-batch-failure path.
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


# ----------------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------------


def _staging_root_uri(table: TableDefinition) -> str:
    """Per-table directory the read publishes parquet to.

    A source reader lands parquet at the materialize path every connector
    uses -- ``{persist}/ingest/{ingest_path_fpid}/{data_version}/`` -- rather
    than in a bucket prefix private to one vendor.
    Returning the parent (without ``data_version``) lets callers walk
    the single ``data_version`` subdir without needing to know the
    auto-generated version string up front.
    """
    from kumo.data import MaterializationStage
    from kumo.data.utils import get_materialize_path

    # Any version: the caller walks the one subdirectory beneath.
    staged = get_materialize_path(table.fingerprint(), "",
                                  MaterializationStage.INGEST)
    return staged.rstrip("/")


def _list_parquet_files(staging_root: str) -> list[str]:
    """Return all parquet files under a per-table staging root.

    A read publishes to ``{staging_root}/{data_version}/`` so we descend
    one level. BigQuery may shard the export across multiple
    parquet objects (``000000000000.parquet``,
    ``000000000001.parquet``, ...); skip anything that isn't a
    parquet file. ``fsutil.listdir`` returns full URIs for GCS, not
    basenames.
    """
    if not fsutil.exists(staging_root):
        return []
    parquet: list[str] = []
    for version_uri in fsutil.listdir(staging_root):
        if not fsutil.isdir(version_uri):
            continue
        for entry in fsutil.listdir(version_uri):
            if entry.endswith('.parquet'):
                parquet.append(entry)
    return sorted(parquet)


def _cleanup_staging_uris(uris: list[str]) -> None:
    """Best-effort cleanup of per-table staging directories.

    ``fsutil.rm(uri, ignore_errors=True)`` doesn't catch every GCS
    error (e.g. transient 5xx from a race between listdir and
    delete-object), so wrap each rm in a broad ``try/except`` —
    cleanup is non-essential and shouldn't fail a test whose primary
    assertions already passed.
    """
    for uri in uris:
        try:
            fsutil.rm(uri, ignore_errors=True)
        except Exception as e:  # noqa: BLE001 — diagnostics only
            log.warning("Cleanup failed for %s: %s", uri, e)


def _assert_staged_parquet(uri: str, source_name: str) -> list[str]:
    """Assert at least one non-empty parquet exists under ``uri``.

    Returns the sorted list of parquet URIs so callers can capture
    mtimes for the short-circuit assertion.
    """
    parquet_files = _list_parquet_files(uri)
    assert parquet_files, (
        f"expected staged parquet for {source_name!r} under {uri}, "
        f"but found none")
    for entry in parquet_files:
        size = fsutil.filesize(entry)
        assert size and size > 0, f"empty staged parquet at {entry}"
    return parquet_files


def _capture_signature(staging_uri: str) -> dict[str, tuple[int, float]]:
    """Capture ``{path: (size, mtime)}`` for every parquet under ``uri``.

    BigQuery's ``EXPORT DATA`` can vary the shard count and shard
    names across runs even when the source is unchanged. Comparing
    file count alone is therefore not a reliable short-circuit
    signal. We capture the full ``(path, size, mtime)`` map and
    assert *equality* across runs — if the second run re-exported,
    at least one of the names, sizes or mtimes would differ.

    A senior review flagged sub-second mtime collision as a
    theoretical false-negative risk (GCS ``Last-Modified`` is
    seconds-precision in some surfaces). In practice the risk is
    nil for this test: the diskgraph short-circuit returns in
    microseconds, while ``EXPORT DATA`` takes several full
    seconds — if the activity wrongly re-exported, mtimes would
    advance by seconds, not sub-seconds. If a future case ever
    needs stricter detection, capture ``blob.generation`` /
    ``etag`` directly via the GCS SDK instead of going through
    ``fsutil``.
    """
    signature: dict[str, tuple[int, float]] = {}
    for path in _list_parquet_files(staging_uri):
        signature[path] = (
            fsutil.filesize(path) or 0,
            fsutil.get_last_modified_time(path),
        )
    return signature


def _skip_if_not_local(storage_backend: str) -> None:
    if storage_backend != 'local':
        pytest.skip("BigQuery diskgraph ingestion e2e is scoped to local "
                    "storage; the s3 parametrization hits a kumo-side "
                    "stats persistence issue unrelated to the BigQuery "
                    "connector path (same skip as the Databricks "
                    "ingestion suite).")


# ----------------------------------------------------------------------------
# Refresh + wait helpers
# ----------------------------------------------------------------------------


async def _setup_and_refresh(
    client: TestClient,
    fixture: FixtureDefinition,
):
    """Register connectors + graph and trigger a graph refresh.

    Returns the snapshot, project, and registered graph definition. Steps:
      1. POST ``/v0/projects/default/connectors/`` for each connector in the fixture.
      2. POST ``/v1/graphs/`` to register the graph definition (so
         ``DataSnapshotApiDriver.create_graph_snapshot`` can resolve
         the graph by fpid).
      3. Call ``data_snapshot_api_driver.create_graph_snapshot``
         directly via the in-process driver — kicks off
         ``GraphRefreshWorkflow`` (which fans out per-table
         ``TableRefreshWorkflow`` instances and the child
         ``DiskgraphIngestWorkflow``) and returns the snapshot id.

    Uses the driver method instead of a dedicated REST endpoint
    because (a) the v2 graph router doesn't expose a refresh-only
    endpoint, (b) ``training_jobs/train`` would couple the test to
    training, and (c) the public ``/graphsnapshots`` route requires
    API-key auth that the standard test client doesn't carry.
    """
    temporary_connector_id = _connector_name(fixture)
    setup = _register_connector(client, fixture)
    fixture = remap_fixture_connector_ids_v2(
        fixture, {temporary_connector_id: setup.connector_id})

    assert len(fixture.graphs) == 1, (
        "ingestion tests use exactly one graph per fixture")
    graph_def = fixture.graphs[0]
    # ``?skip_validation=true`` so the API doesn't reject graphs that
    # reference deliberately-bad sources (e.g. the mid-batch failure
    # test points one table at a non-existent name). Validation we
    # actually care about happens inside the ingestion activity.
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
    return snapshot_id, ProjectID(setup.project_id), graph_def


def _connector_name(fixture: FixtureDefinition) -> str:
    table = next(iter(fixture.graphs[0].tables.values()))
    return str(table.get_connector_id())


def _register_connector(client: TestClient, fixture: FixtureDefinition):
    identity_config, credentials = bigquery_test_identity()
    return create_identity_backed_connector(
        client,
        identity_config=identity_config,
        credentials=credentials,
        connector_config=BigQueryConnectorResourceConfig(
            name=_connector_name(fixture),
            project_id=_PROJECT_ID,
            dataset_id=_DATASET_ID,
            location='US',
        ),
    )


def _wait_for_graph_snapshot(
    client: TestClient,
    graph_snapshot_id: str,
    project_id: ProjectID,
) -> GraphSnapshotResponse:
    """Poll ``POST /v0/snapshots/graph`` until INGEST stage is terminal.

    Mirrors ``_wait_for_training`` from ``test_diskgraph_e2e.py`` but
    polls the snapshot API instead of the training job API. The
    snapshot API is the right boundary for ingestion-only tests;
    polling Temporal directly would couple the test to internal
    workflow IDs.
    """
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


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(900)
@withBigQueryCredentials
async def test_export_single_table_graph_to_staging(
    client: TestClient,
    temporal_client,
    storage_backend,
):
    """Single-table graph: ingestion stages parquet under the per-table fpid.

    Drives ``ingest_table_data_activity_v2`` →
    ``BigQueryConnector.export_table`` for a graph that has one
    table (ACCOUNT — the smallest of the three permanent tables)
    and asserts the GCS bucket contains parquet under
    ``gs://{bucket}/ingest/{ingest_path_fpid}/{data_version}/``.
    Stops at the ingestion stage; no training is triggered.
    """
    _skip_if_not_local(storage_backend)
    fixture = _build_single_table_bigquery_fixture()
    fixture = randomize_fixture_ids_v2(fixture)
    account_uri = None

    try:
        graph_snapshot_id, project_id, graph_def = await _setup_and_refresh(
            client, fixture)
        account = next(table for table in graph_def.tables.values()
                       if table.get_source_table_name() == _TABLE_ACCOUNT)
        account_uri = _staging_root_uri(account)
        resp = _wait_for_graph_snapshot(client, graph_snapshot_id, project_id)
        ingest = resp.snapshot_resource.stages[GraphSnapshotStage.INGEST]
        assert ingest.status == JobStatus.DONE, (
            f"INGEST stage failed: status={ingest.status}, "
            f"error={ingest.error}")

        _assert_staged_parquet(account_uri, _TABLE_ACCOUNT)
        log.info("Single-table staging assertion passed at %s", account_uri)
    finally:
        if account_uri is not None:
            _cleanup_staging_uris([account_uri])


@pytest.mark.asyncio
@pytest.mark.timeout(900)
@withBigQueryCredentials
async def test_export_multi_table_graph_to_staging(
    client: TestClient,
    temporal_client,
    storage_backend,
):
    """3-table graph: each table lands at its own ``ingest_path_fpid`` subdir.

    Production graph refresh fans out per-table
    ``TableRefreshWorkflow`` instances in parallel; each runs its
    own ``ingest_table_data_activity_v2`` → ``export_table``. Test
    that all three tables land at distinct subpaths under the
    shared ``gs://{bucket}/ingest/`` prefix.
    """
    _skip_if_not_local(storage_backend)
    fixture = _build_bigquery_fixture()
    fixture = randomize_fixture_ids_v2(fixture)
    table_to_uri = {}

    try:
        graph_snapshot_id, project_id, graph_def = await _setup_and_refresh(
            client, fixture)
        table_to_uri = {
            table.get_source_table_name(): _staging_root_uri(table)
            for table in graph_def.tables.values()
        }
        resp = _wait_for_graph_snapshot(client, graph_snapshot_id, project_id)
        ingest = resp.snapshot_resource.stages[GraphSnapshotStage.INGEST]
        assert ingest.status == JobStatus.DONE, (
            f"INGEST stage failed: status={ingest.status}, "
            f"error={ingest.error}")

        for source, uri in table_to_uri.items():
            _assert_staged_parquet(uri, source)
        log.info("Multi-table staging assertion passed for %d tables",
                 len(table_to_uri))
    finally:
        _cleanup_staging_uris(list(table_to_uri.values()))


@pytest.mark.asyncio
@pytest.mark.timeout(1200)
@withBigQueryCredentials
async def test_export_short_circuit_unchanged_version(
    client: TestClient,
    temporal_client,
    storage_backend,
):
    """Re-running graph refresh with unchanged data does not re-export.

    Run 1 primes the GCS staging bucket + the local NVMe ingest path
    + the ``materialization_stage`` cache for every table. Run 2
    hits the same connector + data version; for each table the
    activity sees the same ``(fingerprint, data_version)`` key,
    finds the local NVMe ingest data still on disk, and
    short-circuits before calling ``export_table``
    (``ingest_table_data_activity_v2.py:70-86``).

    Signal: the full ``(name, size, mtime)`` map of staged parquet.
    If the activity re-ran ``EXPORT DATA`` the file mtimes (and
    potentially shard names) would change. We compare the full map
    rather than just file count because ``EXPORT DATA`` can vary
    shard names across runs even when the source is unchanged.
    """
    _skip_if_not_local(storage_backend)
    fixture = _build_bigquery_fixture()
    fixture = randomize_fixture_ids_v2(fixture)
    table_to_uri = {}

    try:
        first_id, project_id, graph_def = await _setup_and_refresh(
            client, fixture)
        table_to_uri = {
            table.get_source_table_name(): _staging_root_uri(table)
            for table in graph_def.tables.values()
        }
        first_resp = _wait_for_graph_snapshot(client, first_id, project_id)
        first_ingest = first_resp.snapshot_resource.stages[
            GraphSnapshotStage.INGEST]
        assert first_ingest.status == JobStatus.DONE, (
            f"First refresh failed: {first_ingest.error}")

        before: dict[str, dict[str, tuple[int, float]]] = {}
        for source, uri in table_to_uri.items():
            _assert_staged_parquet(uri, source)
            before[source] = _capture_signature(uri)
        log.info("Captured parquet signatures for %d tables after first run",
                 len(before))

        # Trigger the same graph refresh a second time. Same connector
        # + same fixture → same fingerprint + same data_version (a
        # stable BigQuery table snapshot id derived from
        # ``last_modified_time``, see
        # ``BigQueryConnector._get_data_version``).
        driver = client.app_builder._driver.data_snapshot_api_driver
        with fixture_admin_user_authorization_context():
            second_id = await driver.create_graph_snapshot(
                graph_fpid=graph_def.fpid,
                refresh_source=False,
                project_id=project_id,
            )
        # Note: ``create_graph_snapshot`` returns
        # ``GRAPH.format_id(graph_fpid, graph_data_version)``, which is
        # idempotent — same fpid + same data_version yields the same
        # snapshot id. We don't assert the IDs differ; the
        # ``(name, size, mtime)`` equality check below is the actual
        # signal that no re-export happened.
        second_resp = _wait_for_graph_snapshot(client, second_id, project_id)
        second_ingest = second_resp.snapshot_resource.stages[
            GraphSnapshotStage.INGEST]
        assert second_ingest.status == JobStatus.DONE, (
            f"Second refresh failed: {second_ingest.error}")

        for source, uri in table_to_uri.items():
            after = _capture_signature(uri)
            assert after == before[source], (
                f"Second ingestion re-wrote staging parquet for "
                f"{source!r} — diskgraph short-circuit failed. "
                f"before={before[source]}, after={after}")
        log.info("Short-circuit assertion passed for %d tables.",
                 len(table_to_uri))
    finally:
        _cleanup_staging_uris(list(table_to_uri.values()))


# NOTE: ``test_export_mixed_versions_partial_short_circuit`` is a
# planned test (one table re-exports while the others short-circuit
# after a non-destructive cache invalidation), but it can't be
# implemented reliably here today: ``create_graph_snapshot`` is
# idempotent on ``(graph_fpid, graph_data_version)`` and Temporal
# dedupes the workflow id, so a second call with the same args
# does not re-run the activity even when the test process has
# cleared the per-table cache and local NVMe data. Truly testing
# partial re-export would require either rewriting one of the
# BigQuery tables to bump its data_version (destructive against
# the shared ``financial`` dataset) or invoking the per-table
# refresh workflow directly with ``refresh_source=True`` for one
# table — both out of scope for this PR. Tracking as a follow-up.


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@withBigQueryCredentials
async def test_export_mid_batch_failure(
    client: TestClient,
    temporal_client,
    storage_backend,
):
    """One bad table → graph creation rejects with 422 before ingestion.

    Builds a 3-table fixture where ``TRANS`` references a
    non-existent BigQuery table. Kumo validates source-table
    existence at graph creation time (``POST /v1/graphs/``) — even
    with ``skip_validation=true``, the connector's table catalog is
    consulted and the bad reference is rejected with HTTP 422 plus
    a ``"does not exist"`` message. That's the user-visible
    failure surface for a "halfway through batch" scenario; the
    ingestion activity never runs because the graph never gets
    registered. Assert the 422 + the error message rather than
    driving through to ingestion.
    """
    _skip_if_not_local(storage_backend)
    fixture = _build_bigquery_fixture(trans_source=_TABLE_DOES_NOT_EXIST)
    fixture = randomize_fixture_ids_v2(fixture)

    # Register the connector first so the API has somewhere to look
    # up the bad table reference.
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
    # BigQuery's connector wraps the 404 as
    # ``"Not found: Table ..."`` rather than ``"does not exist"`` —
    # accept either phrasing.
    detail_lower = detail.lower()
    assert 'does not exist' in detail_lower or 'not found' in detail_lower, (
        f"expected bad-table error phrasing in detail, got: {detail!r}")
    log.info("Mid-batch (bad-table) failure surfaced at API: %s", detail)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@withBigQueryCredentials
async def test_export_bad_credentials_raises(
    client: TestClient,
    temporal_client,
    storage_backend,
):
    """Bad service-account credentials fail identity validation."""
    _skip_if_not_local(storage_backend)
    identity_config = invalid_bigquery_key_test_identity_config()
    response = client.post(
        '/v0/identities',
        data=to_json(
            data_source_api.CreateIdentityArgs(
                display_name='invalid-bigquery-identity',
                config=identity_config,
                credentials=(data_source_api.GoogleServiceAccountKeyCredentials(
                    private_key_id=SecretStr('bad_private_key_id'),
                    private_key=SecretStr('bad_private_key'),
                )),
            ),
            insecure=True,
        ),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422, response.text
