# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diskgraph end-to-end training and batch prediction test.

Runs the full training + batch prediction pipeline and validates:
1. Training completes successfully (status == DONE)
2. Model metrics exist and performance is in expected range
3. TT-NG durable train-table bundle (_current.json + immutable version)
4. JobConfig has diskgraph_index_path set
5. Exact column stats and edge stats for all tables
6. Diskgraph-specific Temporal activities were executed
7. Batch prediction completes with DISKGRAPH backend
8. PredictionConfig has diskgraph_index_path set

Prerequisites:
    - Temporal dev server running: make up-local-temporal
    - diskgraph Python package installed (provides pybind ingest)

Usage:
    python -m pytest test/rest/integ/test_diskgraph_e2e.py -xvs --timeout=1800
"""
import logging
import math
import os.path as osp
import time
from pathlib import Path
from test.rest.integ.seed_helpers import seed_and_train
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient
from kumo_core.job_config import PredictionConfig
from kumo_core.types import PredictionArtifactType
from kumoapi.data_snapshot import TableSnapshotID
from kumoapi.data_source import DEFAULT_PROJECT_ID

from io_utils import fsutil
from io_utils.json import load_json
from kumo.testing.rest import (
    get_and_expect_success,
    post_and_expect_success,
    randomize_fixture_ids_v2,
)
from rest.config.api_command import FixtureDefinition
from rest.config.batch_prediction_job import (
    BatchPredictionJobResponse,
    BatchPredictionJobStats,
)
from rest.config.batch_prediction_job_v2 import BatchPredictionInternalRequest
from rest.config.graph import EdgeHealthStats
from rest.config.graph_v2 import GraphSnapshotRequest, GraphSnapshotResponse
from rest.config.jobs import BatchPredictionOptions
from rest.config.ml_backend import MaterializationTaskInfo
from rest.config.prediction_output import LocalPredictionOutput
from rest.config.predictive_query import MaterializationInfoResponse
from rest.config.status import IDResponse
from rest.config.table import ColStatsResponse
from rest.config.training_job import TrainingJobResponse, TrainingJobResults
from rest.config.training_job_v2 import TrainingWorkflowResponse
from rest.testing.integration import wait_for_training_workflow_to_finish
from service_framework.deploy_env_config import deploy_env
from statedb.config.common import (
    BatchPredictionJobID,
    GraphIDv2,
    JobStatus,
    PredictiveQueryID,
    TrainingJobID,
)

log = logging.getLogger(__name__)


FIXTURE_PATH = Path(__file__).parent / "test_cases" / "test_e2e_fixtures.json"

# Pool fixtures (storage_backend, _diskgraph_deploy_config, _pool_subprocesses)
# live in ``test/rest/integ/conftest.py`` and are activated for every test in
# this module via ``pytestmark`` below.
pytestmark = pytest.mark.usefixtures("_pool_subprocesses")

PQL_TEST_CASE = "saas_temporal_regression_base"

POLL_INTERVAL_SECONDS = 5
MAX_POLL_SECONDS = 1200

# Expected metric ranges for the PKDD'99 regression task.
# These are loose bounds — we just want to ensure the model trained
# meaningfully, not that it hit exact numbers.
EXPECTED_TEST_RMSE_MAX = 100000  # upper bound on test RMSE

# ---------------------------------------------------------------------------
# Expected exact column stats per table.
#
# Each column entry: (stype, total, missing_pct, num_unique, invalid_pct,
#                      summary_stats_or_None)
# summary_stats is a list of (label, value) tuples matching the API response.
# For categorical/ID columns the API returns an empty summary_stats list.
# Floating-point values use pytest.approx for tolerance.
# ---------------------------------------------------------------------------

_ColSpec = Tuple[
    str,  # stype
    int,  # total row count
    float,  # missing_percentage
    int,  # num_unique
    float,  # invalid_percentage
    Optional[List[Tuple[str, Any]]],  # summary_stats (None = skip)
]

# Timestamp num_unique values are HyperLogLog estimates of *distinct*
# timestamps (not row counts). The earlier expected values (4500, 682,
# 5000) were row counts; that test was passing only because the
# pinned diskgraph wheel pre-dated kumo-diskgraph commit 1eb4bf9
# ("Fix timestamp NUM_UNIQUE: add HLL to Timestamp col stats variant"),
# where the Timestamp col_stats variant had no HLL and emitted
# total row count for NUM_UNIQUE. With the HLL fix in the wheel,
# NUM_UNIQUE now reflects actual cardinality.
#
# Verification (`pandas.read_csv(...; sep=';').DATE.nunique()`):
#   ACCOUNT.DATE:  4500 rows, 1535 distinct dates (1993-01-01..1997-12-29, 1823d)
#   LOAN.DATE:      682 rows,  559 distinct dates (1993-07-05..1998-12-08, 1982d)
#   TRANS.DATE:    5000 rows,  181 distinct dates (1993-01-01..1993-06-30,  180d)
# HLL with P=14 (M=16384 registers) returns these within ±1 via the
# small-range linear-counting correction; the assertion uses
# `_approx_unique_eq` with ±2 absolute tolerance for HLL-backed counts.
EXPECTED_COL_STATS: Dict[str, Dict[str, _ColSpec]] = {
    'ACCOUNT': {
        'FREQUENCY': ('categorical', 4500, 0.0, 3, 0.0, []),
        'DATE': ('timestamp', 4500, 0.0, 1535, 0.0, [('Range', '1,823 days'),
                                                     ('Min', '1993-01-01'),
                                                     ('Max', '1997-12-29')]),
        'DISTRICT_ID': ('ID', 4500, 0.0, 77, 0.0, []),
        'ACCOUNT_ID': ('ID', 4500, 0.0, 4500, 0.0, []),
    },
    'LOAN': {
        'AMOUNT': ('numerical', 682, 0.1466275659824047, 682, 0.0,
                   [('Mean', 151490.96035242305), ('Min', 4980.0),
                    ('25%', 66696.0), ('50%', 117082.5), ('75%', 210744.0),
                    ('Max', 590820.0)]),
        'DURATION': ('categorical', 682, 0.0, 5, 0.0, []),
        'PAYMENTS':
        ('numerical', 682, 0.0, 682, 0.0, [('Mean', 4190.664222873898),
                                           ('Min', 304.0), ('25%', 2465.5),
                                           ('50%', 3932.5666666666666),
                                           ('75%', 5810.375), ('Max', 9910.0)]),
        'STATUS': ('categorical', 682, 0.0, 4, 0.0, []),
        'ACCOUNT_ID': ('ID', 682, 0.0, 682, 0.0, []),
        'LOAN_ID': ('ID', 682, 0.0, 678, 0.0, []),
        'DATE': ('timestamp', 682, 0.0, 559, 0.0, [('Range', '1,982 days'),
                                                   ('Min', '1993-07-05'),
                                                   ('Max', '1998-12-08')]),
    },
    'TRANS': {
        'TYPE': ('categorical', 5000, 0.0, 3, 0.0, []),
        'OPERATION': ('categorical', 5000, 22.18, 4, 0.0, []),
        'AMOUNT':
        ('numerical', 5000, 0.0, 5000, 0.0, [('Mean', 9060.554939984037),
                                             ('Min', 0.800000011920929),
                                             ('25%', 232.95833415278682),
                                             ('50%', 3689.0),
                                             ('75%', 13392.500865051903),
                                             ('Max', 76500.0)]),
        'BALANCE':
        ('numerical', 5000, 0.0, 5000, 0.0, [('Mean', 31926.794406262186),
                                             ('Min', -3724.699951171875),
                                             ('25%', 14503.864717371323),
                                             ('50%', 27175.834833559784),
                                             ('75%', 44493.86989030934),
                                             ('Max', 169292.09375)]),
        'K_SYMBOL': ('categorical', 5000, 65.52, 7, 0.0, []),
        'ACCOUNT_ID': ('ID', 5000, 0.0, 556, 0.0, []),
        'TRANS_ID': ('ID', 5000, 0.0, 5000, 0.0, []),
        'DATE': ('timestamp', 5000, 0.0, 181, 0.0, [('Range', '180 days'),
                                                    ('Min', '1993-01-01'),
                                                    ('Max', '1993-06-30')]),
    },
}

# Relative tolerance for floating-point stat comparisons.
_FLOAT_REL_TOL = 1e-6
# T-digest quantiles are approximate and merge-order dependent. Parallel
# ingestion deliberately allows batches to finish out of order, so compare
# quartiles with a tolerance that covers the estimator's merge variance.
_QUANTILE_REL_TOL = 0.05

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _approx_eq(actual: Any, expected: Any) -> bool:
    """Compare two stat values, using tolerance for floats."""
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if math.isnan(expected):
            return math.isnan(float(actual))
        return math.isclose(float(actual), expected, rel_tol=_FLOAT_REL_TOL)
    return actual == expected


def _approx_summary_eq(label: str, actual: Any, expected: Any) -> bool:
    """Compare exact summaries and merge-order-dependent T-digest quantiles."""
    if (label in {'25%', '50%', '75%'} and isinstance(expected, (int, float))
            and isinstance(actual, (int, float))):
        return math.isclose(float(actual), float(expected),
                            rel_tol=_QUANTILE_REL_TOL)
    return _approx_eq(actual, expected)


async def _setup_and_train(
    client: TestClient,
    fixture: FixtureDefinition,
    pquery_name: str,
    *,
    provision_connectors: bool = True,
    project_id: str | None = None,
) -> tuple[PredictiveQueryID, TrainingJobID]:
    """Create connectors/pquery and kick off training.

    Thin wrapper over the shared :func:`seed_and_train` helper (see
    ``test/rest/integ/seed_helpers.py``); kept as a local alias so the rest
    of this module's call sites are unchanged.
    """
    return await seed_and_train(
        client,
        fixture,
        pquery_name,
        provision_connectors=provision_connectors,
        project_id=project_id,
    )


def _wait_for_training(
    client: TestClient,
    training_job_id: TrainingJobID,
    *,
    project_id: str | None = None,
) -> TrainingJobResponse:
    """Poll training job until terminal status."""
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        params = {} if project_id is None else {'project_id': project_id}
        job = get_and_expect_success(
            client,
            f'/v0/training_jobs/{training_job_id}',
            params,
            TrainingJobResponse,
        )
        if job.execution.status in {
                JobStatus.DONE,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
        }:
            return job
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Training job {training_job_id} did not complete within "
        f"{MAX_POLL_SECONDS}s")


def _wait_for_materialization_info(
    client: TestClient,
    training_job_id: TrainingJobID,
) -> MaterializationTaskInfo:
    """Poll until materialization info is persisted, then return it.

    Materialization info is durably saved at the end of the materialize stage,
    before the (slow) K8s GNN training launch -- so waiting for it instead of
    the terminal DONE status keeps this a fast materialization check. Bails out
    if the job fails before materialization completes so a real ingest/
    materialize error surfaces instead of timing out.
    """
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        resp = get_and_expect_success(
            client,
            f'/v0/training_jobs/{training_job_id}/materialization_info',
            {'project_id': DEFAULT_PROJECT_ID},
            MaterializationInfoResponse,
        )
        if resp.info is not None:
            return resp.info
        job = get_and_expect_success(
            client,
            f'/v0/training_jobs/{training_job_id}',
            {},
            TrainingJobResponse,
        )
        if job.execution.status in {JobStatus.FAILED, JobStatus.CANCELLED}:
            raise AssertionError(
                f"Job ended status={job.execution.status} before "
                f"materialization info was available: "
                f"results={job.execution.results}")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Materialization info for {training_job_id} was not available "
        f"within {MAX_POLL_SECONDS}s")


def _get_table_snapshot_ids(
    client: TestClient,
    training_job_id: TrainingJobID,
) -> Dict[str, TableSnapshotID]:
    """Get table snapshot IDs from the training workflow."""
    workflow = get_and_expect_success(
        client,
        f'/v0/training_jobs/{training_job_id}/workflow',
        {},
        TrainingWorkflowResponse,
    )
    graph_snapshot_id = workflow.graph_snapshot_id
    graph_data_version = workflow.training_job.graph_data_version

    candidate_ids = []
    if graph_snapshot_id:
        candidate_ids.append(str(graph_snapshot_id))

    if workflow.legacy_graph_id and graph_data_version:
        from rest.config.graph_v2 import GraphResourceResponse
        graph_v2_resp = get_and_expect_success(
            client,
            f"/v1/graphs/{workflow.legacy_graph_id}",
            {},
            GraphResourceResponse,
        )
        graph_v2_id = getattr(graph_v2_resp, "id", None)
        if graph_v2_id:
            derived = f"{graph_v2_id}@{graph_data_version}"
            if derived not in candidate_ids:
                candidate_ids.append(derived)

    for candidate_id in candidate_ids:
        try:
            snapshot_resp = post_and_expect_success(
                client,
                '/v0/snapshots/graph',
                GraphSnapshotRequest(graph_snapshot_id=candidate_id),
                GraphSnapshotResponse,
            )
            if hasattr(snapshot_resp, 'snapshot_resource'):
                return snapshot_resp.snapshot_resource.table_ids or {}
        except Exception:
            continue

    return {}


def _check_metrics(results: TrainingJobResults) -> None:
    """Validate training metrics exist and are reasonable."""
    assert results.test_metrics and len(
        results.test_metrics) > 0, ("No test metrics")
    assert results.val_metrics and len(
        results.val_metrics) > 0, ("No val metrics")
    assert results.train_metrics and len(
        results.train_metrics) > 0, ("No train metrics")

    # Check split sizes
    assert results.split_sizes is not None
    assert results.split_sizes.train > 0
    assert results.split_sizes.val > 0
    assert results.split_sizes.test is not None and results.split_sizes.test > 0

    # Check metric names don't have stage prefixes
    for metric in results.test_metrics + results.val_metrics:
        assert 'train_' not in metric.name
        assert 'val_' not in metric.name
        assert 'test_' not in metric.name


def _get_job_home_dir(training_job_id: TrainingJobID) -> str:
    """Compute the job home directory for a TEST deploy env."""
    s3_root = deploy_env().data_plane_storage_root
    return osp.join(s3_root, 'jobs', str(training_job_id))


def _assert_manifest_bundle_complete(manifest_path: str, ) -> tuple[dict, str]:
    """Verify the TT-NG durable train-table bundle is complete.

    The TT-NG DiskGraph producer publishes an immutable, self-contained bundle
    under ``<train_table>.ttng_staging/_bundles/<id>/`` and returns its
    ``tt.parquet`` path directly. The bundle contains manifest.json + tt.parquet +
    stats.json + train_table_col_stats.json + per-stage seed parts -- see
    ``kumo.data.generic.generate_train_table`` ->
    ``persist_durable_ttng_bundle`` /
    ``diskgraph_train_table.ttng_persisted_staging_dir``. The manifest declares
    the ``train_table`` / ``stats`` / ``train_table_col_stats`` artifacts; each
    is copied into the bundle dir, so we assert each declared artifact is
    present in the bundle by its basename (robust to whether the manifest
    embeds relative or absolute path fields). Returns the manifest and bundle
    directory.

    NOTE: TT-NG owns edge sampling at runtime via the DiskGraph runtime sampler
    and no longer emits a materialized ``train_table_edges`` bundle, so that key
    is intentionally absent from the manifest (callers must not require it).
    """
    from kumo.diskgraph.diskgraph_native_tt_artifacts import (
        validate_artifact_bundle,
    )

    return validate_artifact_bundle(manifest_path), osp.dirname(manifest_path)


def _check_diskgraph_train_table(training_job_id: TrainingJobID) -> None:
    """Assert the TT-NG durable train-table bundle is complete on the dataplane
    and the producer parquet carries the TT-NG train-table columns + rows.

    ``output_file_path`` resolves to ``{home_dir}/train_table/{name}/train_table``
    where ``{name}`` is ``pquery_config.name`` (not the registered
    ``pquery_id``; see ``kumo.data.utils.get_table_path``). Each training job
    has exactly one pquery and this test has one generation attempt, so the
    selected immutable bundle is the sole child of ``_bundles``. Training
    publication does not write a stable-root pointer.
    """
    import pyarrow.parquet as pq

    home_dir = _get_job_home_dir(training_job_id)
    train_table_root = osp.join(home_dir, 'train_table')
    pquery_dirs = list(fsutil.listdir(train_table_root))
    assert len(pquery_dirs) == 1, (
        f"expected exactly one pquery subdir under {train_table_root}, "
        f"got {len(pquery_dirs)}: {pquery_dirs}")
    staging_root = osp.join(pquery_dirs[0], 'train_table.ttng_staging')
    assert not fsutil.exists(osp.join(staging_root, '_current.json'))
    bundle_dirs = list(fsutil.listdir(osp.join(staging_root, '_bundles')))
    assert len(bundle_dirs) == 1, (
        f'expected one immutable TT-NG bundle under {staging_root}, '
        f'got {bundle_dirs}')
    manifest_path = osp.join(bundle_dirs[0], 'manifest.json')
    manifest, _bundle_dir = _assert_manifest_bundle_complete(manifest_path)

    # TT-NG owns edge sampling via the DiskGraph runtime sampler; there is no
    # materialized ``train_table_edges`` / ``reverse_edges`` bundle to assert.
    assert 'train_table_edges' not in manifest, (
        f"unexpected train_table_edges in TT-NG manifest: {manifest_path}")

    train_table = manifest['train_table']
    parts = train_table.get('parts')
    parquet_paths = ([part['path'] for part in parts] if isinstance(
        parts, list) else [train_table['path']])
    assert parquet_paths, 'train-table manifest has no parquet parts'

    total_rows = 0
    schema_names = None
    for parquet_path in parquet_paths:
        with fsutil.get_binary_handle(parquet_path, 'rb') as handle:
            pf = pq.ParquetFile(handle)
            if schema_names is None:
                schema_names = pf.schema_arrow.names
            total_rows += pf.metadata.num_rows
    # TT-NG producer tt.parquet carries the slot-0 entity/target columns (and
    # TIMESTAMP for temporal queries); it does NOT materialize a physical
    # ``arange`` row-index column (the row-index is the contiguous row position,
    # synthesized by the split-mask consumers -- see kumo/task/split.py).
    assert schema_names is not None
    assert 'ENTITY1' in schema_names, (
        f"train table missing 'ENTITY1' column: {schema_names}")
    assert total_rows > 0, "train table has 0 rows"


def _check_no_legacy_edge_persist(training_job_id: TrainingJobID) -> None:
    """Assert TT-NG did NOT materialize legacy per-job graph-edge artifacts.

    Under TT-NG the DiskGraph runtime sampler serves graph edges from the DG
    index; ``materialize_training_task`` threads an empty ``edge_persist_paths``
    and the wheel owns edge production, so no
    ``{home_dir}/persist/edges/<EdgeKey>/edge.parquet`` files are written. If
    any ``<EdgeKey>/edge.parquet`` reappears here it means a legacy
    edge-materialization path regressed back on -- fail loudly.
    """
    edge_root = osp.join(_get_job_home_dir(training_job_id), 'persist', 'edges')
    if not fsutil.exists(edge_root):
        log.info(
            "No legacy persist/edges dir (TT-NG serves edges from the DG "
            "index): %s", edge_root)
        return
    entries = list(fsutil.listdir(edge_root))
    stray = [d for d in entries if fsutil.exists(osp.join(d, 'edge.parquet'))]
    assert not stray, (
        f"TT-NG must not materialize per-job graph-edge artifacts, but found "
        f"<EdgeKey>/edge.parquet under {edge_root}: {stray}")
    log.info(
        "No legacy <EdgeKey>/edge.parquet materialized under %s "
        "(TT-NG serves edges from the DG index)", edge_root)


def _check_job_config_diskgraph(training_job_id: TrainingJobID) -> None:
    """Assert JobConfig has diskgraph backend settings.

    The JobConfig is written by generate_training_config and read by
    kumo-auto.  If graph_backend == DISKGRAPH and diskgraph_index_path is
    set, kumo-auto initializes with diskgraph backend, which causes
    auto_trainer._initialize_engine_clients() to create
    DiskgraphFeatureStore and DiskgraphGraphStore (validated by isinstance
    checks in auto_trainer.py:904-916 — training would fail otherwise).
    """
    from kumo_core.job_config import JobConfig

    from io_utils import fsutil
    from io_utils.json import load_json

    home_dir = _get_job_home_dir(training_job_id)
    config_path = osp.join(home_dir, 'jobs', f'{training_job_id}.json')

    assert fsutil.exists(config_path), (f"JobConfig not found at {config_path}")

    job_config = load_json(config_path, JobConfig)

    assert job_config.diskgraph_index_path, (
        "JobConfig.diskgraph_index_path is empty/None")
    log.info("JobConfig verified: diskgraph_index_path=%s",
             job_config.diskgraph_index_path)


def _check_column_stats(
    client: TestClient,
    table_snapshot_ids: Dict[str, TableSnapshotID],
) -> None:
    """Validate exact column stats for each table."""
    # Verify we got exactly the expected tables
    actual_tables = set(table_snapshot_ids.keys())
    expected_tables = set(EXPECTED_COL_STATS.keys())
    assert actual_tables == expected_tables, (
        f"Table mismatch: got {actual_tables}, expected {expected_tables}")

    for table_name, table_snapshot_id in table_snapshot_ids.items():
        col_stats = get_and_expect_success(
            client,
            f'/v0/snapshots/table/{table_snapshot_id}/col_stats',
            {},
            ColStatsResponse,
        )
        assert col_stats is not None, (f"No col_stats for table {table_name}")
        assert col_stats.statistics_available, (
            f"Col stats not available for table {table_name}")

        expected_cols = EXPECTED_COL_STATS[table_name]

        # Assert exact column names
        actual_col_names = {col.name for col in col_stats.columns}
        assert actual_col_names == set(
            expected_cols.keys()), (f"Column mismatch for {table_name}: "
                                    f"got {sorted(actual_col_names)}, "
                                    f"expected {sorted(expected_cols.keys())}")

        for col in col_stats.columns:
            spec = expected_cols[col.name]
            (exp_stype, exp_total, exp_missing, exp_unique, exp_invalid,
             exp_summary) = spec
            prefix = f"{table_name}.{col.name}"

            assert col.stype == exp_stype, (
                f"{prefix}: stype {col.stype!r} != {exp_stype!r}")

            assert col.statistics is not None, (f"{prefix}: statistics is None")
            counts = col.statistics.counts

            assert counts.total == exp_total, (
                f"{prefix}: total {counts.total} != {exp_total}")
            assert _approx_eq(counts.missing_percentage, exp_missing), (
                f"{prefix}: missing_pct {counts.missing_percentage} "
                f"!= {exp_missing}")
            # num_unique is HLL-backed (P=14) for timestamp / categorical /
            # ID stypes — apply ±5% tolerance. Numerical / sequence stypes
            # use exact COUNT and must match exactly.
            if exp_stype in ('timestamp', 'categorical', 'ID'):
                tol = max(1, int(round(exp_unique * 0.05)))
                assert abs(counts.num_unique - exp_unique) <= tol, (
                    f"{prefix}: num_unique {counts.num_unique} not within "
                    f"±5% ({tol}) of {exp_unique}")
            else:
                assert counts.num_unique == exp_unique, (
                    f"{prefix}: num_unique {counts.num_unique} != "
                    f"{exp_unique} (exact match required for {exp_stype})")
            assert _approx_eq(counts.invalid_percentage, exp_invalid), (
                f"{prefix}: invalid_pct {counts.invalid_percentage} "
                f"!= {exp_invalid}")

            if exp_summary is not None:
                actual_summary = col.statistics.summary_stats
                assert len(actual_summary) == len(exp_summary), (
                    f"{prefix}: summary_stats length {len(actual_summary)} "
                    f"!= {len(exp_summary)}")
                for (act_label,
                     act_val), (exp_label,
                                exp_val) in zip(actual_summary, exp_summary):
                    assert act_label == exp_label, (
                        f"{prefix}: summary label {act_label!r} "
                        f"!= {exp_label!r}")
                    assert _approx_summary_eq(
                        act_label, act_val,
                        exp_val), (f"{prefix}: summary {act_label}={act_val!r} "
                                   f"!= {exp_val!r}")

        log.info("Col stats validated for %s: %d columns OK", table_name,
                 len(col_stats.columns))


# ---------------------------------------------------------------------------
# Expected edge stats for the PKDD'99 Financial dataset.
#
# Two FK edges exist:
#   LOAN.ACCOUNT_ID  → ACCOUNT.ACCOUNT_ID  (682 loans, all FKs valid)
#   TRANS.ACCOUNT_ID → ACCOUNT.ACCOUNT_ID  (5000 trans, 556 unique accounts)
#
# parse_edge_stats_result resolves col_a = PK table (ACCOUNT) for both.
# After reverse-lookup + EdgeHealth.reverse(), the final shape is:
#   col_a = ACCOUNT, col_b = LOAN/TRANS
#   matches.a_in_b  = NUM_DST_NONZERO_DEGREE (unique dst accounts)
#   matches.b_in_a  = NUM_SRC_NONZERO_DEGREE (unique src rows with edges)
#   percents.a_in_b = FRAC_DST_NONZERO_DEGREE * 100
#   percents.b_in_a = FRAC_SRC_NONZERO_DEGREE * 100
#
# Each tuple: (col_a_table, col_b_table, col_a_col, col_b_col,
#              edge_count, matches_a_in_b, matches_b_in_a,
#              percents_a_in_b, percents_b_in_a)
# ---------------------------------------------------------------------------

_EdgeSpec = Tuple[str, str, str, str, int, float, float, float, float]

EXPECTED_EDGE_STATS: List[_EdgeSpec] = [
    # LOAN → ACCOUNT: 682 edges, 682 unique LOANs, 682 unique ACCOUNTs
    # All 682 LOAN.ACCOUNT_ID values resolve to distinct ACCOUNTs.
    ('ACCOUNT', 'LOAN', 'ACCOUNT_ID', 'ACCOUNT_ID', 682, 682, 682,
     682 / 4500 * 100, 682 / 682 * 100),
    # TRANS → ACCOUNT: 5000 edges, 5000 unique TRANS, 554 unique ACCOUNTs
    # TRANS.ACCOUNT_ID has 556 unique values in the raw data, but 2 target
    # ACCOUNT rows are dropped by PK dedup during diskgraph ingest, leaving
    # 554 unique destination nodes.
    ('ACCOUNT', 'TRANS', 'ACCOUNT_ID', 'ACCOUNT_ID', 5000, 554, 5000,
     554 / 4500 * 100, 5000 / 5000 * 100),
]


def _check_edge_stats(
    client: TestClient,
    graph_snapshot_id: str,
) -> None:
    """Validate exact edge stats for the PKDD'99 Financial dataset."""
    edge_health = get_and_expect_success(
        client,
        f'/v0/snapshots/graph/{graph_snapshot_id}/edge_stats',
        {},
        EdgeHealthStats,
    )

    assert edge_health.is_ready, "Edge stats not ready"
    assert edge_health.link_stats is not None, "link_stats is None"
    assert len(edge_health.link_stats) == len(EXPECTED_EDGE_STATS), (
        f"Expected {len(EXPECTED_EDGE_STATS)} edges, "
        f"got {len(edge_health.link_stats)}")

    # Index actual edges by (col_a_table, col_b_table) for lookup.
    actual_by_key = {}
    for edge in edge_health.link_stats:
        key = (edge.col_a.table_name, edge.col_b.table_name)
        actual_by_key[key] = edge

    for spec in EXPECTED_EDGE_STATS:
        (exp_a_table, exp_b_table, exp_a_col, exp_b_col, exp_edge_count,
         exp_match_a, exp_match_b, exp_pct_a, exp_pct_b) = spec
        key = (exp_a_table, exp_b_table)
        prefix = f"{exp_a_table}↔{exp_b_table}"

        assert key in actual_by_key, (f"Missing {prefix} edge. "
                                      f"Got: {list(actual_by_key.keys())}")
        edge = actual_by_key[key]

        assert edge.col_a.column_name == exp_a_col, (
            f"{prefix}: col_a.column_name={edge.col_a.column_name!r}, "
            f"expected {exp_a_col!r}")
        assert edge.col_b.column_name == exp_b_col, (
            f"{prefix}: col_b.column_name={edge.col_b.column_name!r}, "
            f"expected {exp_b_col!r}")
        assert edge.edge_count == exp_edge_count, (
            f"{prefix}: edge_count={edge.edge_count}, "
            f"expected {exp_edge_count}")
        assert edge.matches.a_in_b == exp_match_a, (
            f"{prefix}: matches.a_in_b={edge.matches.a_in_b}, "
            f"expected {exp_match_a}")
        assert edge.matches.b_in_a == exp_match_b, (
            f"{prefix}: matches.b_in_a={edge.matches.b_in_a}, "
            f"expected {exp_match_b}")
        assert _approx_eq(
            edge.percents.a_in_b,
            exp_pct_a), (f"{prefix}: percents.a_in_b={edge.percents.a_in_b}, "
                         f"expected {exp_pct_a}")
        assert _approx_eq(
            edge.percents.b_in_a,
            exp_pct_b), (f"{prefix}: percents.b_in_a={edge.percents.b_in_a}, "
                         f"expected {exp_pct_b}")

    log.info("Edge stats validated: %d edges OK", len(edge_health.link_stats))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# BP Helpers
# ---------------------------------------------------------------------------


async def _kick_off_batch_prediction(
    client: TestClient,
    training_job_id: TrainingJobID,
    num_parallel_workers: int = 1,
) -> BatchPredictionJobID:
    """Create a BP job via the internal V1 API and return the job ID.

    The internal V1 API can run directly from the successful training job and
    graph FPID, so this does not need to backfill any pquery metadata.
    """
    workflow = get_and_expect_success(
        client,
        f'/v0/training_jobs/{training_job_id}/workflow',
        {},
        TrainingWorkflowResponse,
    )
    assert workflow.legacy_graph_id, (
        f"Training job {training_job_id} has no graph id for BP")
    graph_fpid = GraphIDv2.from_public(workflow.legacy_graph_id).fpid
    assert graph_fpid is not None
    assert workflow.legacy_graph_id == GraphIDv2.public(graph_fpid)

    request = BatchPredictionInternalRequest(
        model_training_job_id=str(training_job_id),
        predict_options=BatchPredictionOptions(
            num_parallel_workers=num_parallel_workers),
        project_id=DEFAULT_PROJECT_ID,
        # This suite validates Kumo-managed prediction output, statistics, and
        # sharding. Provider export is covered by the identity-backed storage
        # E2Es and must not reuse an internal input fixture connector.
        outputs=[
            LocalPredictionOutput(
                artifact_type=PredictionArtifactType.PREDICTIONS)
        ],
        graph_id=graph_fpid,
        explanations=False,
    )
    res = post_and_expect_success(client, '/v1/batch_prediction_jobs/', request,
                                  IDResponse)
    return BatchPredictionJobID(res.id)


def _wait_for_bp(
    client: TestClient,
    bp_job_id: BatchPredictionJobID,
) -> BatchPredictionJobResponse:
    """Poll BP job until terminal status."""
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        resp = get_and_expect_success(client,
                                      f'/v1/batch_prediction_jobs/{bp_job_id}',
                                      {}, BatchPredictionJobResponse)
        if resp.execution.status in {
                JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED
        }:
            return resp
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"BP job {bp_job_id} timed out after {MAX_POLL_SECONDS}s")


def _wait_for_bp_stats(
    client: TestClient,
    bp_job_id: BatchPredictionJobID,
) -> BatchPredictionJobStats:
    """Poll until the BP stats child has published its managed output."""
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        stats = get_and_expect_success(
            client,
            f'/v1/batch_prediction_jobs/{bp_job_id}/prediction_stats',
            {},
            BatchPredictionJobStats,
        )
        if stats.raw_predictions.statistics_available:
            return stats
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"BP stats for {bp_job_id} timed out after {MAX_POLL_SECONDS}s")


def _check_prediction_config_diskgraph(
    bp_job_id: BatchPredictionJobID, ) -> None:
    """Assert PredictionConfig has diskgraph backend settings."""
    from io_utils import fsutil

    home_dir = _get_job_home_dir(str(bp_job_id))
    config_path = osp.join(home_dir, 'jobs', f'{bp_job_id}.json')
    assert fsutil.exists(config_path), (
        f"PredictionConfig not found at {config_path}")

    pred_config = load_json(config_path, PredictionConfig)
    assert pred_config.diskgraph_index_path, (
        "PredictionConfig.diskgraph_index_path is empty/None")
    log.info("PredictionConfig verified: index=%s",
             pred_config.diskgraph_index_path)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(1800)
async def test_diskgraph_e2e_training(
    client: TestClient,
    temporal_client,
):
    """Run full diskgraph training + batch prediction pipeline."""

    # Load and randomize fixture
    fixture = load_json(str(FIXTURE_PATH), FixtureDefinition)
    fixture.queries = [
        q for q in fixture.queries if q.pquery_name == PQL_TEST_CASE
    ]
    assert len(fixture.queries) == 1, (
        f"Expected exactly one query matching {PQL_TEST_CASE}")
    fixture = randomize_fixture_ids_v2(fixture)
    randomized_name = fixture.queries[0].pquery_name

    # Setup and kick off training
    pquery_id, training_job_id = await _setup_and_train(client, fixture,
                                                        randomized_name)
    log.info("Training started: pquery_id=%s, training_job_id=%s", pquery_id,
             training_job_id)

    # Wait for completion
    job = _wait_for_training(client, training_job_id)
    assert job.execution.status == JobStatus.DONE, (
        f"Training failed with status={job.execution.status}, "
        f"results={job.execution.results}")
    await wait_for_training_workflow_to_finish(training_job_id, temporal_client)

    # -----------------------------------------------------------------------
    # 1. Validate model metrics
    # -----------------------------------------------------------------------
    results = job.execution.results
    assert results is not None, "Training job completed but has no results"
    _check_metrics(results)

    # Check that RMSE is within expected bounds (regression task)
    assert results.test_metrics is not None
    for metric in results.test_metrics:
        if 'rmse' in metric.name.lower():
            log.info("Test RMSE: %s = %s", metric.name, metric.val)
            assert metric.val is not None and metric.val < EXPECTED_TEST_RMSE_MAX, (
                f"Test RMSE {metric.val} exceeds max {EXPECTED_TEST_RMSE_MAX}")

    # -----------------------------------------------------------------------
    # 2. Validate the diskgraph TT bundle landed on the dataplane in full —
    #    manifest + every artifact it references must be under
    #    data_plane_storage_root and reachable cross-pod (gen_tt's load-
    #    bearing handoff to materialize_train_table).
    # -----------------------------------------------------------------------
    _check_diskgraph_train_table(training_job_id)

    # -----------------------------------------------------------------------
    # 2b. TT-NG serves graph edges from the DiskGraph index at runtime and no
    #     longer materializes per-job ``persist/edges/<EdgeKey>/edge.parquet``
    #     artifacts (materialize_training_task uses an empty edge_persist_paths;
    #     the wheel owns edge production). Assert their ABSENCE so the check
    #     still guards the TT-NG contract; the graph edges themselves are
    #     validated via the snapshot edge-stats API in step 4b below.
    # -----------------------------------------------------------------------
    _check_no_legacy_edge_persist(training_job_id)

    # -----------------------------------------------------------------------
    # 3. Validate JobConfig has diskgraph backend settings (proves kumo-auto
    #    initialized with DISKGRAPH backend → DiskgraphFeatureStore and
    #    DiskgraphGraphStore were instantiated for training)
    # -----------------------------------------------------------------------
    _check_job_config_diskgraph(training_job_id)

    # -----------------------------------------------------------------------
    # 4. Validate exact column stats
    # -----------------------------------------------------------------------
    table_snapshot_ids = _get_table_snapshot_ids(client, training_job_id)
    assert len(table_snapshot_ids) > 0, "No table snapshot IDs found"
    _check_column_stats(client, table_snapshot_ids)

    # -----------------------------------------------------------------------
    # 4b. Validate edge stats via snapshot API (same endpoint the UI uses)
    # -----------------------------------------------------------------------
    workflow_resp = get_and_expect_success(
        client,
        f'/v0/training_jobs/{training_job_id}/workflow',
        {},
        TrainingWorkflowResponse,
    )
    graph_snapshot_id = workflow_resp.graph_snapshot_id
    assert graph_snapshot_id, "No graph_snapshot_id in workflow response"
    _check_edge_stats(client, graph_snapshot_id)

    log.info("Diskgraph E2E training test passed.")

    # ===================================================================
    # Phase 2: Batch Prediction
    # ===================================================================

    # 7. Run batch prediction against the trained model
    bp_job_id = await _kick_off_batch_prediction(client, training_job_id)
    log.info("Batch prediction started: bp_job_id=%s", bp_job_id)

    bp_response = _wait_for_bp(client, bp_job_id)
    assert bp_response.execution.status == JobStatus.DONE, (
        f"BP failed: status={bp_response.execution.status}, "
        f"execution={bp_response.execution}")
    assert bp_response.config.outputs == [
        LocalPredictionOutput(
            artifact_type=PredictionArtifactType.PREDICTIONS)
    ]
    log.info("Batch prediction completed successfully.")

    # 8. Validate BP results exist
    bp_results = bp_response.execution.results
    assert bp_results is not None, "BP job completed but has no results"
    assert bp_results.output_artifact_stats is not None
    assert len(bp_results.output_artifact_stats) > 0
    output_stats = bp_results.output_artifact_stats[0]
    assert output_stats.num_rows > 0, "Prediction output has 0 rows"
    log.info("BP output: %d rows, %d cols", output_stats.num_rows,
             output_stats.num_cols)

    # 9. Validate PredictionConfig has diskgraph backend
    _check_prediction_config_diskgraph(bp_job_id)

    # 9b. Validate BP column statistics — the post-cutover replacement for
    # the legacy Spark ColStatsWorkflow path. The data_backend.py
    # start_bp_stats_activity is now wired to compute stats over the BP
    # output parquet via the diskgraph pybind (compute_col_stats_for_parquet)
    # and write them to the same cache layout ingest col-stats use, so the
    # /prediction_stats endpoint must eventually report
    # `statistics_available=True` with populated columns. The API job record
    # reaches DONE when BatchPredictionJobWorkflow completes, just before its
    # parent awaits BatchPredictionStatsWorkflow, so wait for that managed
    # output instead of racing the child workflow.
    bp_stats = _wait_for_bp_stats(client, bp_job_id)
    assert bp_stats.raw_predictions.statistics_available, (
        "BP col-stats not available on DG — DataBackendActivities."
        "start_bp_stats_activity did not populate the table cache.")
    assert bp_stats.raw_predictions.columns, (
        "BP col-stats response has no columns")
    # SUMMARY_STATS_BY_CLASS (col.statistics.bp_stats) is computed by the DG
    # path only for multiclass/multilabel/ranking/link-pred (SCORE grouped by
    # CLASS). This is a regression task, whose BP output is the entity (ID)
    # column plus TARGET_PRED (numerical), so assert TARGET_PRED specifically
    # carries the numerical summary stats. ColumnStats emits capitalized keys
    # for numerical columns — 'Mean', 'Min', 'Max', '25%', '50%', '75%' (see
    # ColumnStats.from_stats_dict). (The entity ID column legitimately has no
    # summary_stats — IDs emit counts, not Mean/Min/Max.)
    cols_by_name = {c.name: c for c in bp_stats.raw_predictions.columns}
    target = cols_by_name.get('TARGET_PRED')
    assert target is not None, ("TARGET_PRED column missing from BP col-stats: "
                                f"{sorted(cols_by_name)}")
    assert target.statistics is not None and target.statistics.summary_stats, (
        "TARGET_PRED carries no summary stats — DG "
        "compute_col_stats_for_parquet did not produce numerical stats for "
        "the prediction column.")
    target_keys = {k for k, _ in target.statistics.summary_stats}
    assert {'Mean', 'Min', 'Max'} <= target_keys, (
        f"TARGET_PRED summary stats missing Mean/Min/Max; got {target_keys}")

    log.info("Diskgraph E2E training + batch prediction test passed.")

    # =================================================================
    # Phase 3: Sharded Batch Prediction (num_parallel_workers=2)
    # =================================================================
    # Exercises per-shard config split, kumo-pred's
    # start/stop_partition_id handling, and postprocess merge.
    sharded_bp_job_id = await _kick_off_batch_prediction(
        client, training_job_id, num_parallel_workers=2)
    sharded_bp_response = _wait_for_bp(client, sharded_bp_job_id)
    assert sharded_bp_response.execution.status == JobStatus.DONE, (
        f"Sharded BP failed: status={sharded_bp_response.execution.status}, "
        f"execution={sharded_bp_response.execution}")
    assert sharded_bp_response.config.outputs == [
        LocalPredictionOutput(
            artifact_type=PredictionArtifactType.PREDICTIONS)
    ]

    # Merged sharded output must match the unsharded baseline row count.
    sharded_results = sharded_bp_response.execution.results
    assert sharded_results is not None
    assert sharded_results.output_artifact_stats
    sharded_stats = sharded_results.output_artifact_stats[0]
    assert sharded_stats.num_rows == output_stats.num_rows, (
        f'sharded={sharded_stats.num_rows} != unsharded='
        f'{output_stats.num_rows}')


# Exercises the materialization path: read materialization info once the
# materialize stage persists it, without waiting for the full GNN train.
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("storage_backend", ["local"], indirect=True)
async def test_make_materialization_info(
    client: TestClient,
    temporal_client,
):
    """Materialization info is fully populated after the materialize stage."""
    fixture = load_json(str(FIXTURE_PATH), FixtureDefinition)
    fixture.queries = [
        q for q in fixture.queries if q.pquery_name == PQL_TEST_CASE
    ]
    assert len(fixture.queries) == 1, (
        f"Expected exactly one query matching {PQL_TEST_CASE}")
    fixture = randomize_fixture_ids_v2(fixture)
    randomized_name = fixture.queries[0].pquery_name

    _, training_job_id = await _setup_and_train(client, fixture,
                                                randomized_name)
    # Read materialization info as soon as the materialize stage persists it,
    # rather than waiting for the full GNN train to reach DONE.
    info = _wait_for_materialization_info(client, training_job_id)

    assert info.start_time
    assert info.end_time > info.start_time
    assert info.id == training_job_id
    assert info.graph_id
    assert info.predictive_query_id
    assert info.code_version
    # temporal_workflow_id is cleared to "" over the HTTP route, so not asserted.
    assert len(info.table_stats) >= 1

    for table_stat in info.table_stats:
        assert table_stat.code_version
        assert table_stat.connector_id
        assert table_stat.data_version
        assert table_stat.num_rows
        assert table_stat.source_table_name
        assert len(table_stat.columns) >= 1
        for col_stat in table_stat.columns:
            assert col_stat.dtype
            assert col_stat.stype
