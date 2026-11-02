# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diskgraph end-to-end test via public /v1 REST API (SDK path).

Mirrors the exact SDK call sequence (kumo-sdk) through the public REST router
(rest/public/app.py). All pipeline stages go through /v1 endpoints — no /v0
fallbacks.

The SDK flow is:
  1. pquery.generate_training_table(graph_snapshot_id=None)
     → server handles graph refresh internally
  2. trainer.fit(graph, train_table)
     → graph.snapshot() creates explicit snapshot, passed to training
  3. pquery.generate_prediction_table(graph_snapshot_id=None)
     → server handles graph refresh internally
  4. trainer.predict(graph, prediction_table)
     → graph.snapshot() creates explicit snapshot, passed to prediction

Tests:
  1. test_diskgraph_decoupled_training_and_prediction
     Full SDK flow with train-table reuse across two training jobs.

  2. test_diskgraph_snapshot_reuse
     Single snapshot used for two independent train-table-gen jobs.

  3. test_diskgraph_graph_refresh_idempotency
     Re-creating a graph snapshot produces identical stats.

Prerequisites:
    - Temporal dev server running: make up-local-temporal
    - diskgraph Python package installed

Usage:
    python -m pytest test/rest/integ/test_diskgraph_public_api_e2e.py \\
      -xvs --timeout=3600
"""
import copy
import dataclasses
import io
import logging
import math
import os
import time
from pathlib import Path
from test.rest.integ.fixture_connector_args import (
    load_selected_fixture,
    provision_fixture_connectors,
)
from test.rest.integ.public_response_reader import (
    flatten_public_parquet_refs,
    read_public_response_bytes,
)
from test.rest.integ.training_budget import clamp_e2e_training_budget
from typing import Any, Dict, List, Optional, Tuple

import kumoapi.graph as graph_api
import kumoapi.jobs as api
import kumoapi.model_plan as plan_api
import kumoapi.pquery as pquery_api
import pytest
from fastapi.testclient import TestClient
from kumoapi.common import JobStatus
from kumoapi.data_snapshot import (
    GraphSnapshotID,
    GraphSnapshotResource,
    GraphSnapshotStage,
    TableSnapshotID,
    TableSnapshotResource,
    TableSnapshotStage,
)
from kumoapi.model_plan import RunMode

from io_utils.json import load_json
from kumo.testing.rest import get_and_expect_success, post_and_expect_success
from rest.config.api_command import FixtureDefinition
from rest.config.graph_v2 import GraphDefinition
from rest.config.status import EmptyResponse, IDResponse
from rest.public.routers.pquery import _public_pquery_from_internal
from rest.routers.fixture_utils import fixture_to_pquery_resources_v2
from temporalio_commons import TemporalClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — same dataset / expectations as test_diskgraph_e2e.py
# ---------------------------------------------------------------------------


FIXTURE_PATH = Path(__file__).parent / "test_cases" / "test_e2e_fixtures.json"

# Pool fixtures (storage_backend, _diskgraph_deploy_config, _pool_subprocesses)
# live in ``test/rest/integ/conftest.py`` and are activated for every test in
# this module via ``pytestmark`` below. The pool subprocesses are required
# even though this file exercises the public API surface — workflow dispatch
# routes through ``SnapshotActivities`` on the pool task queues, so the
# subprocesses must be running for ingest/train/predict to make progress.
pytestmark = [
    pytest.mark.ci_group('aws-public-api'),
    pytest.mark.usefixtures("_pool_subprocesses"),
]

PQL_TEST_CASE = "saas_temporal_regression_base"

POLL_INTERVAL_SECONDS = 5
MAX_POLL_SECONDS = 1200

# PKDD'99 Financial dataset exact values (deterministic for this fixture).
EXPECTED_TRAIN_TABLE_ROWS = 18_000
EXPECTED_NUM_ENTITIES = 4_500  # ACCOUNT table size
EXPECTED_BP_PREDICTION_ROWS = 4_500  # one prediction per entity
EXPECTED_TABLE_NAMES = {'ACCOUNT', 'LOAN', 'TRANS'}

# RMSE upper bound for the regression task catches divergence.
EXPECTED_TEST_RMSE_MAX = 80_000

# Relative tolerance for floating-point stat comparisons.
_FLOAT_REL_TOL = 1e-6

# Expected column stats per table.
# Dict[table_name, Dict[col_name, (total, num_unique, stype)]]
# We validate COUNT and NUM_UNIQUE via the V1 ColumnStats.stats dict keys.
#
# stype drives NUM_UNIQUE assertion mode:
#   timestamp / categorical / ID → HLL-backed → ±5% tolerance.
#   numerical / sequence         → exact COUNT → strict equality.
#
# Timestamp num_unique values are HyperLogLog estimates of *distinct*
# timestamps (not row counts). Pre kumo-diskgraph commit 1eb4bf9
# the Timestamp col_stats variant had no HLL and emitted total row count
# for NUM_UNIQUE — the previous values (4500, 682, 5000) were calibrated
# against that bug. With the HLL fix in the wheel, NUM_UNIQUE now
# reflects actual cardinality.
#
# Verification (`pandas.read_csv(...; sep=';').DATE.nunique()`):
#   ACCOUNT.DATE:  4500 rows, 1535 distinct dates (1993-01-01..1997-12-29, 1823d)
#   LOAN.DATE:      682 rows,  559 distinct dates (1993-07-05..1998-12-08, 1982d)
#   TRANS.DATE:    5000 rows,  181 distinct dates (1993-01-01..1993-06-30,  180d)
EXPECTED_COL_COUNTS: Dict[str, Dict[str, Tuple[int, int, str]]] = {
    'ACCOUNT': {
        'FREQUENCY': (4500, 3, 'categorical'),
        'DATE': (4500, 1535, 'timestamp'),
        'DISTRICT_ID': (4500, 77, 'ID'),
        'ACCOUNT_ID': (4500, 4500, 'ID'),
    },
    'LOAN': {
        'AMOUNT': (682, 682, 'numerical'),
        'DURATION': (682, 5, 'numerical'),
        'PAYMENTS': (682, 682, 'numerical'),
        'STATUS': (682, 4, 'categorical'),
        'ACCOUNT_ID': (682, 682, 'ID'),
        'LOAN_ID': (682, 678, 'ID'),
        'DATE': (682, 559, 'timestamp'),
    },
    'TRANS': {
        'TYPE': (5000, 3, 'categorical'),
        'OPERATION': (5000, 4, 'categorical'),
        'AMOUNT': (5000, 5000, 'numerical'),
        'BALANCE': (5000, 5000, 'numerical'),
        'K_SYMBOL': (5000, 7, 'categorical'),
        'ACCOUNT_ID': (5000, 556, 'ID'),
        'TRANS_ID': (5000, 5000, 'ID'),
        'DATE': (5000, 181, 'timestamp'),
    },
}

# Expected edge stats (V1 format: src=FK table, dst=PK table).
# Tuples: (src_table, fkey, dst_table, total_num_edges,
#           abs_src_in_dst, abs_dst_in_src,
#           pct_src_in_dst, pct_dst_in_src)
_EdgeSpec = Tuple[str, str, str, int, float, float, float, float]

EXPECTED_EDGE_STATS: List[_EdgeSpec] = [
    # LOAN.ACCOUNT_ID -> ACCOUNT: 682 edges
    # src_in_dst: 682/682 = 100% (all LOANs match an ACCOUNT)
    # dst_in_src: 682/4500 = 15.15% (fraction of ACCOUNTs with a LOAN)
    ('LOAN', 'ACCOUNT_ID', 'ACCOUNT', 682, 682, 682, 682 / 682 * 100,
     682 / 4500 * 100),
    # TRANS.ACCOUNT_ID -> ACCOUNT: 5000 edges
    # src_in_dst: 5000/5000 = 100% (all TRANS match an ACCOUNT)
    # dst_in_src: 554/4500 = 12.3% (fraction of ACCOUNTs with a TRANS)
    ('TRANS', 'ACCOUNT_ID', 'ACCOUNT', 5000, 5000, 554, 5000 / 5000 * 100,
     554 / 4500 * 100),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx_eq(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if math.isnan(expected):
            return math.isnan(float(actual))
        return math.isclose(float(actual), expected, rel_tol=_FLOAT_REL_TOL)
    return actual == expected


# -- Setup helpers ----------------------------------------------------------


async def _create_connectors_v1(
    client: TestClient,
    fixture: FixtureDefinition,
) -> str:
    """Provision project-owned integration connectors."""
    project_id, _ = await provision_fixture_connectors(
        client, fixture, api_prefix='/v1', selected_sources_only=True)
    return str(project_id)


async def _setup_pquery_and_graph(
    client: TestClient,
    fixture: FixtureDefinition,
) -> Tuple[str, str, pquery_api.PQueryResource]:
    """Create PQuery + Graph via V1, return (pquery_id, graph_id, args)."""
    resources = await fixture_to_pquery_resources_v2.coroutine(fixture)
    assert len(resources) == 1, (
        f"Expected 1 PQuery resource, got {len(resources)}")
    args = resources[0]
    pquery_id = args.name
    assert pquery_id is not None

    # PredictiveQuery.save() → POST /v1/predictive_queries/
    post_and_expect_success(client, '/v1/predictive_queries/?dialect=PQLANG_V2',
                            _public_pquery_from_internal(args), EmptyResponse)

    # Graph.save() → POST /v1/graphs
    graph_id = post_and_expect_success(
        client, f'/v1/graphs?name_alias={pquery_id}_graph', args.graph,
        IDResponse).id
    expected_fpid = GraphDefinition(**dataclasses.asdict(args.graph)).fpid
    assert graph_id == f'graph-{expected_fpid}'

    return pquery_id, graph_id, args


# -- Snapshot helpers -------------------------------------------------------


def _wait_for_graph_snapshot(
    client: TestClient,
    snapshot_id: GraphSnapshotID,
    wait_for_col_stats: bool = True,
    wait_for_edge_stats: bool = True,
) -> GraphSnapshotResource:
    """Poll graph + table snapshot stages until terminal.

    Mirrors the SDK's Graph.snapshot() blocking loop which polls both
    graph-level and per-table stages.  When wait_for_col_stats=True, also
    waits for FULL_COL_STATS stage on each table (diskgraph colstats
    pipeline runs after per-table ingest).  When wait_for_edge_stats=True,
    also waits for the edge-health endpoint to report ready.
    """
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        resource = get_and_expect_success(client,
                                          f'/v1/graphsnapshots/{snapshot_id}',
                                          {}, GraphSnapshotResource)

        graph_stage = resource.stages.get(GraphSnapshotStage.INGEST)
        if graph_stage is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        all_done = True
        for _, table_id in resource.table_ids.items():
            table_resource = get_and_expect_success(
                client, f'/v1/tablesnapshots/{table_id}', {},
                TableSnapshotResource)
            ingest_stage = table_resource.stages.get(TableSnapshotStage.INGEST)
            if ingest_stage is None or not ingest_stage.status.is_terminal:
                all_done = False
                break

            if wait_for_col_stats:
                full_stats_stage = table_resource.stages.get(
                    TableSnapshotStage.FULL_COL_STATS)
                if (full_stats_stage is None
                        or not full_stats_stage.status.is_terminal):
                    all_done = False
                    break

        if all_done and graph_stage.status.is_terminal:
            if graph_stage.status == JobStatus.FAILED:
                raise RuntimeError(f"Graph snapshot {snapshot_id} failed: "
                                   f"error={graph_stage.error}")
            if wait_for_edge_stats:
                edge_health = get_and_expect_success(
                    client, f'/v1/graphsnapshots/edge_health/{snapshot_id}', {},
                    graph_api.EdgeHealthResponse)
                if not edge_health.is_ready:
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue
            return resource

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Graph snapshot {snapshot_id} did not complete in {MAX_POLL_SECONDS}s")


# -- Stats validation helpers (V1 types) ------------------------------------


def _snapshot_stat_value(stats: Dict[str, Any], key: str) -> Any:
    payload = stats[key]
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    if hasattr(payload, "value"):
        return payload.value
    return payload


def _check_column_stats_v1(
    client: TestClient,
    table_snapshot_ids: Dict[str, TableSnapshotID],
) -> None:
    """Validate column stats via GET /v1/tablesnapshots/{id}.

    Checks COUNT and NUM_UNIQUE for every column in every table.
    """
    actual_tables = set(table_snapshot_ids.keys())
    expected_tables = set(EXPECTED_COL_COUNTS.keys())
    assert actual_tables == expected_tables, (
        f"Table mismatch: got {actual_tables}, expected {expected_tables}")

    for table_name, table_snapshot_id in table_snapshot_ids.items():
        resource = get_and_expect_success(
            client, f'/v1/tablesnapshots/{table_snapshot_id}', {},
            TableSnapshotResource)

        assert resource.column_stats is not None, (
            f"No column_stats for table {table_name}")
        assert len(resource.column_stats) > 0, (
            f"Empty column_stats for table {table_name}")

        expected_cols = EXPECTED_COL_COUNTS[table_name]
        actual_by_name = {cs.column_name: cs for cs in resource.column_stats}

        assert set(actual_by_name.keys()) == set(
            expected_cols.keys()), (f"Column mismatch for {table_name}: "
                                    f"got {sorted(actual_by_name.keys())}, "
                                    f"expected {sorted(expected_cols.keys())}")

        for col_name, (exp_total, exp_unique,
                       exp_stype) in (expected_cols.items()):
            cs = actual_by_name[col_name]
            stats = cs.stats
            prefix = f"{table_name}.{col_name}"

            # V1 stats keys are stringified ColStatType enum values.
            if 'COUNT' in stats:
                count = _snapshot_stat_value(stats, 'COUNT')
                assert count == exp_total, (
                    f"{prefix}: COUNT {count} != {exp_total}")
            if 'NUM_UNIQUE' in stats:
                num_unique = _snapshot_stat_value(stats, 'NUM_UNIQUE')
                # HLL-backed for timestamp / categorical / ID — apply ±5%.
                # Numerical / sequence use exact COUNT — must match exactly.
                if exp_stype in ('timestamp', 'categorical', 'ID'):
                    tol = max(1, int(round(exp_unique * 0.05)))
                    assert abs(num_unique - exp_unique) <= tol, (
                        f"{prefix}: NUM_UNIQUE {num_unique} not "
                        f"within ±5% ({tol}) of {exp_unique}")
                else:
                    assert num_unique == exp_unique, (
                        f"{prefix}: NUM_UNIQUE {num_unique} != "
                        f"{exp_unique} (exact required for {exp_stype})")

        log.info("V1 column stats validated for %s: %d columns OK", table_name,
                 len(resource.column_stats))


def _check_edge_stats_v1(
    client: TestClient,
    snapshot_id: GraphSnapshotID,
) -> None:
    """Validate edge stats via GET /v1/graphsnapshots/edge_health/{id}."""
    response = get_and_expect_success(
        client, f'/v1/graphsnapshots/edge_health/{snapshot_id}', {},
        graph_api.EdgeHealthResponse)

    assert response.is_ready, "Edge stats not ready"
    assert len(response.statistics) == len(EXPECTED_EDGE_STATS), (
        f"Expected {len(EXPECTED_EDGE_STATS)} edges, "
        f"got {len(response.statistics)}")

    for spec in EXPECTED_EDGE_STATS:
        (exp_src, exp_fkey, exp_dst, exp_edge_count, exp_abs_src_in_dst,
         exp_abs_dst_in_src, exp_pct_src_in_dst, exp_pct_dst_in_src) = spec

        # V1 keys edges by Edge.fully_qualified_name.
        # Try both arrow and dot separator formats, and reversed direction.
        edge_stats = None
        reversed_edge = False
        for fqn in [
                f"{exp_src}.{exp_fkey}->{exp_dst}",
                f"{exp_src}.{exp_fkey}.{exp_dst}",
        ]:
            edge_stats = response.statistics.get(fqn)
            if edge_stats is not None:
                break
        if edge_stats is None:
            for fqn in [
                    f"{exp_dst}.{exp_fkey}->{exp_src}",
                    f"{exp_dst}.{exp_fkey}.{exp_src}",
            ]:
                edge_stats = response.statistics.get(fqn)
                if edge_stats is not None:
                    reversed_edge = True
                    break

        assert edge_stats is not None, (
            f"Missing edge {exp_src}.{exp_fkey}->{exp_dst}. "
            f"Got: {list(response.statistics.keys())}")

        prefix = f"{exp_src}->{exp_dst}"
        assert edge_stats.total_num_edges == exp_edge_count, (
            f"{prefix}: total_num_edges={edge_stats.total_num_edges}, "
            f"expected {exp_edge_count}")

        if not reversed_edge:
            abs_match = edge_stats.absolute_match_stats
            pct_match = edge_stats.percent_match_stats
        else:
            # When reversed, src/dst are swapped in the response
            abs_match = graph_api.EdgeMatches(
                src_in_dst=edge_stats.absolute_match_stats.dst_in_src,
                dst_in_src=edge_stats.absolute_match_stats.src_in_dst)
            pct_match = graph_api.EdgeMatches(
                src_in_dst=edge_stats.percent_match_stats.dst_in_src,
                dst_in_src=edge_stats.percent_match_stats.src_in_dst)

        assert _approx_eq(abs_match.src_in_dst, exp_abs_src_in_dst), (
            f"{prefix}: abs src_in_dst={abs_match.src_in_dst}, "
            f"expected {exp_abs_src_in_dst}")
        assert _approx_eq(abs_match.dst_in_src, exp_abs_dst_in_src), (
            f"{prefix}: abs dst_in_src={abs_match.dst_in_src}, "
            f"expected {exp_abs_dst_in_src}")
        assert _approx_eq(pct_match.src_in_dst, exp_pct_src_in_dst), (
            f"{prefix}: pct src_in_dst={pct_match.src_in_dst}, "
            f"expected {exp_pct_src_in_dst}")
        assert _approx_eq(pct_match.dst_in_src, exp_pct_dst_in_src), (
            f"{prefix}: pct dst_in_src={pct_match.dst_in_src}, "
            f"expected {exp_pct_dst_in_src}")

    log.info("V1 edge stats validated: %d edges OK", len(response.statistics))


# -- Job polling helpers (mirrors SDK polling) ------------------------------


def _wait_for_gen_train_table_job(
    client: TestClient,
    job_id: str,
) -> api.GenerateTrainTableJobResource:
    """Poll GET /v1/gentraintable_jobs/{id} until terminal.
    Mirrors TrainingTableJob._poll_job() in the SDK.
    """
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        job = get_and_expect_success(client, f'/v1/gentraintable_jobs/{job_id}',
                                     {}, api.GenerateTrainTableJobResource)
        if job.job_status_report.status.is_terminal:
            return job
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Gen-train-table job {job_id} did not complete in {MAX_POLL_SECONDS}s")


def _wait_for_gen_pred_table_job(
    client: TestClient,
    job_id: str,
) -> api.GeneratePredictionTableJobResource:
    """Poll GET /v1/genpredtable_jobs/{id} until terminal.
    Mirrors PredictionTableJob._poll_job() in the SDK.
    """
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        job = get_and_expect_success(client, f'/v1/genpredtable_jobs/{job_id}',
                                     {}, api.GeneratePredictionTableJobResource)
        if job.job_status_report.status.is_terminal:
            return job
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Gen-pred-table job {job_id} did not complete in {MAX_POLL_SECONDS}s")


def _wait_for_training_job_v1(
    client: TestClient,
    job_id: str,
) -> api.TrainingJobResource:
    """Poll GET /v1/training_jobs/{id} until terminal.
    Mirrors TrainingJob._poll_training() in the SDK.
    """
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        job = get_and_expect_success(client, f'/v1/training_jobs/{job_id}', {},
                                     api.TrainingJobResource)
        if job.job_status_report.status.is_terminal:
            return job
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Training job {job_id} did not complete in {MAX_POLL_SECONDS}s")


def _wait_for_prediction_job_v1(
    client: TestClient,
    job_id: str,
) -> api.BatchPredictionJobResource:
    """Poll GET /v1/prediction_jobs/{id} until terminal.
    Mirrors BatchPredictionJob._poll() in the SDK.
    """
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        job = get_and_expect_success(client, f'/v1/prediction_jobs/{job_id}',
                                     {}, api.BatchPredictionJobResource)
        if job.job_status_report.status.is_terminal:
            return job
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Prediction job {job_id} did not complete in {MAX_POLL_SECONDS}s")


# -- Diskgraph-specific filesystem checks ----------------------------------
# These supplement the API-level checks with implementation-level validation.


def _check_job_config_diskgraph(job_id: str) -> None:
    """Assert JobConfig has a populated diskgraph_index_path."""
    import os.path as osp

    from kumo_core.job_config import JobConfig

    from io_utils import fsutil
    from io_utils.json import load_json as _load_json
    from service_framework.deploy_env_config import deploy_env

    s3_root = deploy_env().data_plane_storage_root
    home_dir = osp.join(s3_root, 'jobs', str(job_id))
    config_path = osp.join(home_dir, 'jobs', f'{job_id}.json')

    assert fsutil.exists(config_path), f"JobConfig not found at {config_path}"

    job_config = _load_json(config_path, JobConfig)
    assert job_config.diskgraph_index_path
    log.info("JobConfig verified: diskgraph_index_path=%s",
             job_config.diskgraph_index_path)


def _check_prediction_config_diskgraph(job_id: str) -> None:
    """Assert PredictionConfig has a populated diskgraph_index_path."""
    import os.path as osp

    from kumo_core.job_config import PredictionConfig

    from io_utils import fsutil
    from service_framework.deploy_env_config import deploy_env

    s3_root = deploy_env().data_plane_storage_root
    home_dir = osp.join(s3_root, 'jobs', str(job_id))
    config_path = osp.join(home_dir, 'jobs', f'{job_id}.json')

    assert fsutil.exists(config_path), (
        f"PredictionConfig not found at {config_path}")

    pred_config = load_json(config_path, PredictionConfig)
    assert pred_config.diskgraph_index_path
    log.info("PredictionConfig verified: diskgraph_index_path=%s",
             pred_config.diskgraph_index_path)


# -- Train table parquet validation via API URL -----------------------------


def _check_train_table_parquet_via_url(
    client: TestClient,
    train_table_job_id: str,
) -> int:
    """Verify the train table parquet via the API-returned URL.

    Returns the row count.
    """
    import pyarrow.parquet as pq

    urls = get_and_expect_success(
        client, f'/v1/gentraintable_jobs/{train_table_job_id}/table_data', {},
        List[str])
    assert len(urls) > 0, "No train table data URLs"

    parquet_refs = flatten_public_parquet_refs(urls)
    assert parquet_refs, urls

    schema_names = None
    total_rows = 0
    for path in parquet_refs:
        # Never pass a signed URL to fsutil. Read each public transport through
        # the URL-aware boundary and validate every returned parquet part.
        pf = pq.ParquetFile(io.BytesIO(read_public_response_bytes(path)))
        part_schema_names = pf.schema_arrow.names
        if schema_names is None:
            schema_names = part_schema_names
        else:
            assert part_schema_names == schema_names, (path, part_schema_names,
                                                       schema_names)
        assert pf.metadata.num_rows > 0, f'Empty parquet partition: {path}'
        total_rows += pf.metadata.num_rows

    assert schema_names is not None
    # TT-NG DiskGraph-path signature: the producer tt.parquet carries the
    # slot-0 ``ENTITY1`` (and ``TARGET1``) columns. It does NOT materialize a
    # physical ``arange`` row-index column (that legacy signature is gone; the
    # row-index is the contiguous row position, synthesized by the split-mask
    # consumers -- see kumo/task/split.py).
    assert 'ENTITY1' in schema_names, (
        f"Missing 'ENTITY1' column (columns: {schema_names}) — "
        f"diskgraph path was likely not used")
    assert total_rows > 0, "Diskgraph train table has 0 rows"
    log.info("Train table parquet verified via %d API URLs: %d rows, cols=%s",
             len(parquet_refs), total_rows, schema_names)
    return total_rows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_cleanup_options():
    """Preserve state across the multi-phase test."""
    from rest.testing.config import TestFunctionCleanupOptions
    return TestFunctionCleanupOptions(
        should_reset_rest_db=False,
        should_reset_ml_db=False,
        should_reset_data_backend_cache=True,
        should_delete_data_persist_dir=True,
    )


# ---------------------------------------------------------------------------
# Test 1: Full decoupled training + prediction (with train table reuse)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(3600)
async def test_diskgraph_decoupled_training_and_prediction(
    public_api_client: TestClient,
    temporal_client: TemporalClient,
):
    """Full SDK-style diskgraph flow through /v1 endpoints."""
    client = public_api_client

    # Load and randomize fixture
    fixture = load_selected_fixture(FIXTURE_PATH, PQL_TEST_CASE)

    phase_started = time.monotonic()
    previous_phase = 'setup'

    def _record_phase(next_phase: str) -> None:
        nonlocal phase_started, previous_phase
        now = time.monotonic()
        print(
            f"CI_PHASE public-api {previous_phase} elapsed_seconds={now - phase_started:.3f}",
            flush=True)
        previous_phase, phase_started = next_phase, now

    # ==== Phase 1: Setup ====
    # SDK: connector = kumo.S3Connector(...); graph = kumo.Graph(...);
    #      graph.save(); pquery = kumo.PredictiveQuery(graph, query);
    #      pquery.save()
    _record_phase("provision")
    log.info("Phase 1: Creating connectors, PQuery, Graph via V1")
    await _create_connectors_v1(client, fixture)
    pquery_id, graph_id, args = await _setup_pquery_and_graph(client, fixture)
    log.info("  pquery_id=%s, graph_id=%s", pquery_id, graph_id)

    # ==== Phase 2: Graph Snapshot ====
    # SDK: graph.snapshot(force_refresh=True) — called explicitly or
    # implicitly by trainer.fit(). We create it here for stats validation
    # and reuse it in trainer.fit() / trainer.predict() below.
    _record_phase("snapshot")
    log.info("Phase 2: Creating graph snapshot via V1")
    snapshot_id = GraphSnapshotID(
        post_and_expect_success(
            client,
            f'/v1/graphsnapshots?graph_id={graph_id}&refresh_source=true', {},
            IDResponse).id)
    log.info("  snapshot_id=%s", snapshot_id)

    snapshot_resource = _wait_for_graph_snapshot(client, snapshot_id)
    assert set(snapshot_resource.table_ids.keys()) == EXPECTED_TABLE_NAMES, (
        f"Expected tables {EXPECTED_TABLE_NAMES}, "
        f"got {set(snapshot_resource.table_ids.keys())}")

    # Validate column stats via V1 table snapshot endpoints
    _check_column_stats_v1(client, snapshot_resource.table_ids)

    # Validate edge stats: Graph.get_edge_stats()
    _check_edge_stats_v1(client, snapshot_id)

    # ==== Phase 3: Train Table Generation ====
    # SDK: pquery.generate_training_table(plan=plan)
    # Note: SDK passes graph_snapshot_id=None — server triggers its own
    # graph refresh internally.
    _record_phase("training_table")
    log.info("Phase 3: Generating training table via V1")

    plan_payload = plan_api.SuggestModelPlanRequest(
        query_string=args.query_string,
        graph_id=graph_id,
        run_mode=RunMode.DEBUG,
    )
    train_table_plan = post_and_expect_success(
        client,
        '/v1/predictive_queries/train_table_generation_plan?dialect=PQLANG_V2',
        plan_payload, api.TrainingTableGenerationPlan)

    # SDK passes graph_snapshot_id=None (server handles graph refresh)
    train_table_job_id = post_and_expect_success(
        client, '/v1/gentraintable_jobs',
        api.GenerateTrainTableRequest({}, pquery_id, train_table_plan, None),
        IDResponse).id
    log.info("  train_table_job_id=%s", train_table_job_id)

    tt_job = _wait_for_gen_train_table_job(client, train_table_job_id)
    assert tt_job.job_status_report.status == JobStatus.DONE, (
        f"Train table gen failed: {tt_job.job_status_report.status}")

    # Verify table_data via API (TrainingTable.data_urls())
    train_table_rows = _check_train_table_parquet_via_url(
        client, train_table_job_id)
    assert train_table_rows == EXPECTED_TRAIN_TABLE_ROWS, (
        f"Train table rows {train_table_rows} != {EXPECTED_TRAIN_TABLE_ROWS}")

    # Verify the internal per-stage split artifacts independently of the
    # public signed URL. TT-NG no longer re-materializes the
    # split into ``{stage}_mask.pt`` files (the split-materialization step was
    # removed; ``train_table_materialization_path`` is empty), so the
    # ``/split_masks`` endpoint returns ``{}``. The per-stage split assignment
    # now lives beside the exact immutable ``tt.parquet`` path returned by the
    # generation workflow. Assert those seed parts exist.
    from io_utils import fsutil
    from rest.backends.ml_backend.workflows import GenerateTrainTableResult
    from statedb.config.common import GenerateTrainTableJobID

    train_table_result: GenerateTrainTableResult = await temporal_client.result(
        GenerateTrainTableJobID(train_table_job_id).wf_id)
    from kumo.data.generic.diskgraph_train_table import (
        ttng_bundle_dir_for_train_table,
    )
    staging_dir = ttng_bundle_dir_for_train_table(
        train_table_result.train_table_path)
    for stage in ('train', 'val', 'test'):
        seed_part = os.path.join(staging_dir, f'{stage}_seeds')
        assert fsutil.exists(seed_part), (
            f"missing {stage}_seeds in TT-NG bundle: {seed_part}")
    log.info("  split seed parts verified in bundle: %s", staging_dir)

    # ==== Phase 4a: Training Job A (seed=42) ====
    # SDK: model_plan = pquery.suggest_model_plan()
    #      trainer = kumo.Trainer(model_plan)
    #      trainer.fit(graph=graph, train_table=train_table)
    # Internally, fit() calls graph.snapshot() and passes the snapshot_id
    # along with train_table.job_id to TrainingJobRequest.
    _record_phase("training_A")
    log.info("Phase 4a: Training with seed=42 via V1")

    model_plan_info = post_and_expect_success(
        client, '/v1/predictive_queries/model_plan?dialect=PQLANG_V2',
        plan_payload, plan_api.ModelPlanInfo)
    model_plan = model_plan_info.model_plan
    clamp_e2e_training_budget(model_plan)
    model_plan.optimization.max_test_steps = 20
    model_plan.training_job.manual_seed = 42

    training_job_id_a = post_and_expect_success(
        client, '/v1/training_jobs',
        api.TrainingJobRequest({}, pquery_id=pquery_id, model_plan=model_plan,
                               graph_snapshot_id=snapshot_id,
                               train_table_job_id=train_table_job_id),
        IDResponse).id
    log.info("  training_job_id_a=%s", training_job_id_a)

    training_job_a = _wait_for_training_job_v1(client, training_job_id_a)
    assert training_job_a.job_status_report.status == JobStatus.DONE, (
        f"Training A failed: {training_job_a.job_status_report.status}")

    # Validate metrics: TrainingJobResult.metrics()
    assert training_job_a.result is not None
    eval_metrics_a = training_job_a.result.eval_metrics
    assert len(eval_metrics_a.test_metrics) > 0, "No test metrics"
    assert len(eval_metrics_a.validation_metrics) > 0, "No val metrics"
    assert len(eval_metrics_a.training_metrics) > 0, "No train metrics"
    rmse_found = False
    for metric in eval_metrics_a.test_metrics:
        if 'rmse' in metric.name.lower():
            rmse_found = True
            log.info("  Train A test RMSE: %s = %s", metric.name, metric.value)
            assert metric.value is not None, "RMSE value is None"
            assert 0 <= metric.value < EXPECTED_TEST_RMSE_MAX, (
                f"Test RMSE {metric.value} outside expected range "
                f"[0, {EXPECTED_TEST_RMSE_MAX})")
    assert rmse_found, (f"No RMSE metric found in test_metrics: "
                        f"{[m.name for m in eval_metrics_a.test_metrics]}")

    # Validate progress endpoint: TrainingJob.progress()
    # May fail in local test env due to job path format constraints.
    try:
        progress_resp = client.get(
            f'/v1/training_jobs/{training_job_id_a}/progress')
        log.info("  Training progress endpoint: %d", progress_resp.status_code)
    except Exception as e:
        log.warning(
            "  Training progress endpoint error (expected in local "
            "test env): %s", e)

    # Validate holdout endpoint: TrainingJobResult.holdout_df()
    try:
        holdout_resp = client.get(
            f'/v1/training_jobs/{training_job_id_a}/holdout')
        log.info("  Holdout endpoint: %d", holdout_resp.status_code)
    except Exception as e:
        log.warning(
            "  Holdout endpoint error (expected in local "
            "test env): %s", e)

    # Diskgraph backend checks (filesystem — supplementary). The
    # functional assertion that diskgraph was used lives in
    # _check_job_config_diskgraph above: the JobConfig file
    # has a populated diskgraph_index_path.
    # Auditing Temporal-history activity names was tried earlier and
    # was brittle (every rename / wrap broke the test); functional
    # config check is stable.
    _check_job_config_diskgraph(training_job_id_a)

    # ==== Phase 4b: Training Job B (seed=123, reuse train table) ====
    # SDK: Same trainer.fit() with different model_plan seed but same
    #      train_table.job_id — both jobs share the same train table.
    _record_phase("training_B")
    log.info("Phase 4b: Training with seed=123, reusing train table via V1")

    model_plan_b = copy.deepcopy(model_plan)
    model_plan_b.training_job.manual_seed = 123

    training_job_id_b = post_and_expect_success(
        client, '/v1/training_jobs',
        api.TrainingJobRequest({}, pquery_id=pquery_id, model_plan=model_plan_b,
                               graph_snapshot_id=snapshot_id,
                               train_table_job_id=train_table_job_id),
        IDResponse).id
    log.info("  training_job_id_b=%s", training_job_id_b)

    training_job_b = _wait_for_training_job_v1(client, training_job_id_b)
    assert training_job_b.job_status_report.status == JobStatus.DONE, (
        f"Training B failed: {training_job_b.job_status_report.status}")

    assert training_job_b.result is not None
    eval_metrics_b = training_job_b.result.eval_metrics
    assert len(eval_metrics_b.test_metrics) > 0, "No test metrics"
    rmse_found = False
    for metric in eval_metrics_b.test_metrics:
        if 'rmse' in metric.name.lower():
            rmse_found = True
            log.info("  Train B test RMSE: %s = %s", metric.name, metric.value)
            assert metric.value is not None, "RMSE value is None"
            assert 0 <= metric.value < EXPECTED_TEST_RMSE_MAX, (
                f"Test RMSE {metric.value} outside expected range "
                f"[0, {EXPECTED_TEST_RMSE_MAX})")
    assert rmse_found, (f"No RMSE metric found in test_metrics: "
                        f"{[m.name for m in eval_metrics_b.test_metrics]}")

    _check_job_config_diskgraph(training_job_id_b)

    # Verify train table was reused: same parquet, same row count
    reuse_rows = _check_train_table_parquet_via_url(client, train_table_job_id)
    assert reuse_rows == train_table_rows, (
        f"Train table row count changed: {train_table_rows} -> {reuse_rows}")
    log.info("  Train table reuse confirmed: %d rows, job=%s", reuse_rows,
             train_table_job_id)

    # ==== Phase 5: Prediction Table Generation ====
    # SDK: pquery.generate_prediction_table()
    # Note: SDK passes graph_snapshot_id=None — server handles graph refresh.
    _record_phase("prediction_table")
    log.info("Phase 5: Generating prediction table via V1")

    pred_table_job_id = post_and_expect_success(
        client, '/v1/genpredtable_jobs',
        api.GeneratePredictionTableRequest(
            {}, pquery_id=pquery_id, plan=api.PredictionTableGenerationPlan(),
            graph_snapshot_id=None), IDResponse).id
    log.info("  pred_table_job_id=%s", pred_table_job_id)

    pt_job = _wait_for_gen_pred_table_job(client, pred_table_job_id)
    assert pt_job.job_status_report.status == JobStatus.DONE, (
        f"Pred table gen failed: {pt_job.job_status_report.status}")
    assert pt_job.job_status_report.start_time is not None
    assert pt_job.job_status_report.end_time is not None
    assert (pt_job.job_status_report.end_time
            > pt_job.job_status_report.start_time)

    # Verify pred table data: PredictionTable.data_urls()
    pred_table_urls = get_and_expect_success(
        client, f'/v1/genpredtable_jobs/{pred_table_job_id}/table_data', {},
        List[str])
    assert len(pred_table_urls) >= 1, (
        f"Expected at least 1 pred table URL, got {len(pred_table_urls)}")
    log.info("  pred table URLs: %d", len(pred_table_urls))

    # Verify anchor time: PredictionTable.anchor_time
    anchor_resp = client.get(
        f'/v1/genpredtable_jobs/{pred_table_job_id}/get_anchor_time')
    assert anchor_resp.status_code == 200, (
        f"Anchor time endpoint returned {anchor_resp.status_code}")
    # Response is a JSON datetime string or null — either is valid for
    # this temporal regression task.
    anchor_body = anchor_resp.json()
    log.info("  Anchor time: %s", anchor_body)

    # ==== Phase 6: Batch Prediction ====
    # SDK: trainer.predict(graph=graph, prediction_table=prediction_table,
    #                      output_config=OutputConfig(...))
    # Internally, predict() calls graph.snapshot() and passes snapshot_id +
    # pred_table_job_id to BatchPredictionRequest.
    _record_phase("prediction_and_artifacts")
    log.info("Phase 6: Running batch prediction via V1")

    from kumo.validation import ValidationResponse
    bp_result = post_and_expect_success(
        client, '/v1/prediction_jobs',
        api.BatchPredictionRequest({}, model_training_job_id=training_job_id_a,
                                   predict_options=api.BatchPredictionOptions(),
                                   outputs=[
                                       api.LocalPredictionOutput(
                                           artifact_type=api.
                                           PredictionArtifactType.PREDICTIONS)
                                   ], graph_snapshot_id=snapshot_id,
                                   pred_table_job_id=pred_table_job_id),
        Tuple[Optional[str], ValidationResponse])
    bp_job_id = bp_result[0]
    assert bp_job_id is not None
    log.info("  bp_job_id=%s", bp_job_id)

    bp_job = _wait_for_prediction_job_v1(client, bp_job_id)
    assert bp_job.job_status_report.status == JobStatus.DONE, (
        f"BP failed: {bp_job.job_status_report.status}")

    # Validate results — exact entity count for this dataset
    assert bp_job.result is not None, "BP job completed but has no result"
    assert bp_job.result.num_entities_predicted == EXPECTED_BP_PREDICTION_ROWS, (
        f"Expected {EXPECTED_BP_PREDICTION_ROWS} entities predicted, "
        f"got {bp_job.result.num_entities_predicted}")
    log.info("  BP output: %d entities predicted",
             bp_job.result.num_entities_predicted)

    # Verify prediction URLs endpoint is reachable.
    # Local predictions are generated to the dataplane but not exported to a
    # customer path. The endpoint should still return 200
    # with download URLs, or 422 if no exportable artifacts exist.
    pred_urls_resp = client.get(
        f'/v1/prediction_jobs/{bp_job_id}/get_prediction_df_urls')
    assert pred_urls_resp.status_code in (200, 422), (
        f"get_prediction_df_urls returned unexpected "
        f"{pred_urls_resp.status_code}: {pred_urls_resp.text[:200]}")
    log.info("  get_prediction_df_urls: %d", pred_urls_resp.status_code)

    # Verify prediction config has diskgraph backend (supplementary)
    _check_prediction_config_diskgraph(bp_job_id)

    _record_phase("complete")
    log.info("Test 1 PASSED: full diskgraph decoupled flow via V1.")


# ---------------------------------------------------------------------------
# Test 2: Snapshot reuse across train table gen jobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(1800)
async def test_diskgraph_snapshot_reuse(public_api_client: TestClient, ):
    """Single graph snapshot reused for two train-table-gen jobs."""
    client = public_api_client

    fixture = load_selected_fixture(FIXTURE_PATH, PQL_TEST_CASE)

    await _create_connectors_v1(client, fixture)
    pquery_id, graph_id, args = await _setup_pquery_and_graph(client, fixture)

    # Create one snapshot
    snapshot_id = GraphSnapshotID(
        post_and_expect_success(
            client,
            f'/v1/graphsnapshots?graph_id={graph_id}&refresh_source=true', {},
            IDResponse).id)
    _wait_for_graph_snapshot(client, snapshot_id)

    plan_payload = plan_api.SuggestModelPlanRequest(
        query_string=args.query_string,
        graph_id=graph_id,
        run_mode=RunMode.DEBUG,
    )
    train_table_plan = post_and_expect_success(
        client,
        '/v1/predictive_queries/train_table_generation_plan?dialect=PQLANG_V2',
        plan_payload, api.TrainingTableGenerationPlan)

    # Job A — uses snapshot explicitly
    job_id_a = post_and_expect_success(
        client, '/v1/gentraintable_jobs',
        api.GenerateTrainTableRequest({}, pquery_id, train_table_plan,
                                      snapshot_id), IDResponse).id
    job_a = _wait_for_gen_train_table_job(client, job_id_a)
    assert job_a.job_status_report.status == JobStatus.DONE

    rows_a = _check_train_table_parquet_via_url(client, job_id_a)

    # Job B — same snapshot
    job_id_b = post_and_expect_success(
        client, '/v1/gentraintable_jobs',
        api.GenerateTrainTableRequest({}, pquery_id, train_table_plan,
                                      snapshot_id), IDResponse).id
    job_b = _wait_for_gen_train_table_job(client, job_id_b)
    assert job_b.job_status_report.status == JobStatus.DONE

    rows_b = _check_train_table_parquet_via_url(client, job_id_b)

    # Both reference same snapshot
    assert job_a.config.graph_snapshot_id == snapshot_id
    assert job_b.config.graph_snapshot_id == snapshot_id

    # Both produce exact expected row count
    assert rows_a == EXPECTED_TRAIN_TABLE_ROWS, (
        f"Job A rows {rows_a} != {EXPECTED_TRAIN_TABLE_ROWS}")
    assert rows_b == EXPECTED_TRAIN_TABLE_ROWS, (
        f"Job B rows {rows_b} != {EXPECTED_TRAIN_TABLE_ROWS}")
    assert rows_a == rows_b, (
        f"Row count mismatch: job A={rows_a}, job B={rows_b}")

    log.info("Test 2 PASSED: snapshot reuse across train table gen jobs.")


# ---------------------------------------------------------------------------
# Test 3: Graph refresh idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(1800)
async def test_diskgraph_graph_refresh_idempotency(
    public_api_client: TestClient, ):
    """Re-creating a graph snapshot produces consistent stats."""
    client = public_api_client

    fixture = load_selected_fixture(FIXTURE_PATH, PQL_TEST_CASE)

    await _create_connectors_v1(client, fixture)
    _, graph_id, _ = await _setup_pquery_and_graph(client, fixture)

    # Snapshot 1
    snapshot_id_1 = GraphSnapshotID(
        post_and_expect_success(
            client,
            f'/v1/graphsnapshots?graph_id={graph_id}&refresh_source=true', {},
            IDResponse).id)
    resource_1 = _wait_for_graph_snapshot(client, snapshot_id_1)

    # Snapshot 2 (re-ingest)
    snapshot_id_2 = GraphSnapshotID(
        post_and_expect_success(
            client,
            f'/v1/graphsnapshots?graph_id={graph_id}&refresh_source=true', {},
            IDResponse).id)
    resource_2 = _wait_for_graph_snapshot(client, snapshot_id_2)

    # Both should have the exact expected tables
    assert set(resource_1.table_ids.keys()) == EXPECTED_TABLE_NAMES, (
        f"Snapshot 1 tables: {set(resource_1.table_ids.keys())}")
    assert set(resource_2.table_ids.keys()) == EXPECTED_TABLE_NAMES, (
        f"Snapshot 2 tables: {set(resource_2.table_ids.keys())}")

    # Validate column stats identical for both
    _check_column_stats_v1(client, resource_1.table_ids)
    _check_column_stats_v1(client, resource_2.table_ids)

    # Validate edge stats identical for both
    _check_edge_stats_v1(client, snapshot_id_1)
    _check_edge_stats_v1(client, snapshot_id_2)

    log.info("Test 3 PASSED: graph refresh idempotency confirmed.")
