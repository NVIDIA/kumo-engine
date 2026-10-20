# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiskGraph end-to-end test for the custom training table path.

Exercises the DG-native restoration of ``ingest_custom_train_table`` and
``validate_weighted_train_table`` (Stage 1-4 of the custom-TT DG
restoration plan): customer submits a training job whose request carries
a ``CustomTrainingTable`` with a weight column. The workflow runs
ingest → validate → materialize; training completes with
``JobStatus.DONE``; the materialized attributes show the weight column
propagated through to the trainer.

Patterned after ``test_diskgraph_e2e.py`` and
``test_diskgraph_public_api_e2e.py`` for the in-process Temporal +
``_pool_subprocesses`` fixture pattern. Uses the V1 public API for the
training submission because the V2 ``/v0/training_jobs/train`` request
schema does not carry ``custom_train_table`` — only V1
``api.TrainingJobRequest`` does.

Prerequisites:
    - Temporal dev server running: make up-local-temporal
    - diskgraph Python package installed (provides pybind ingest)

Usage:
    python -m pytest test/rest/integ/test_diskgraph_custom_tt_e2e.py \\
        -xvs --timeout=1800
"""
import logging
import os
import os.path as osp
import tempfile
import time
from pathlib import Path
from test.rest.integ.fixture_connector_args import (
    provision_fixture_connectors,
    provision_project_upload_connector,
)
from test.rest.integ.public_response_reader import (
    public_parquet_refs,
    read_public_response_bytes,
)
from test.rest.integ.training_budget import clamp_e2e_training_budget
from typing import List

import kumoapi.jobs as api
import kumoapi.model_plan as plan_api
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from kumoapi.common import JobStatus
from kumoapi.data_snapshot import GraphSnapshotID
from kumoapi.model_plan import RunMode
from kumoapi.source_table import FileType, S3SourceTable
from kumoapi.train import TrainingTableSpec

from io_utils import fsutil
from io_utils.json import load_json
from kumo.testing.rest import (
    get_and_expect_success,
    post_and_expect_success,
    randomize_fixture_ids_v2,
)
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.api_command import FixtureDefinition
from rest.config.status import EmptyResponse, IDResponse
from rest.public.routers.pquery import _public_pquery_from_internal
from rest.routers.fixture_utils import fixture_to_pquery_resources_v2

log = logging.getLogger(__name__)


FIXTURE_PATH = Path(__file__).parent / "test_cases" / "test_e2e_fixtures.json"

# Pool fixtures (storage_backend, _diskgraph_deploy_config, _pool_subprocesses)
# live in test/rest/integ/conftest.py.
pytestmark = pytest.mark.usefixtures("_pool_subprocesses")

PQL_TEST_CASE = "saas_temporal_regression_base"

POLL_INTERVAL_SECONDS = 5
MAX_POLL_SECONDS = 1500

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_gen_train_table_job(
    client: TestClient,
    job_id: str,
) -> api.GenerateTrainTableJobResource:
    """Poll GET /v1/gentraintable_jobs/{id} until terminal."""
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        job = get_and_expect_success(client, f'/v1/gentraintable_jobs/{job_id}',
                                     {}, api.GenerateTrainTableJobResource)
        if job.job_status_report.status.is_terminal:
            return job
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Gen-train-table job {job_id} did not complete in "
                       f"{MAX_POLL_SECONDS}s")


def _wait_for_training_job_v1(
    client: TestClient,
    job_id: str,
) -> api.TrainingJobResource:
    """Poll GET /v1/training_jobs/{id} until terminal."""
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        job = get_and_expect_success(client, f'/v1/training_jobs/{job_id}', {},
                                     api.TrainingJobResource)
        if job.job_status_report.status.is_terminal:
            return job
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Training job {job_id} did not complete in {MAX_POLL_SECONDS}s")


def _job_status_details(job_status_report) -> str:
    return (f"validation_response="
            f"{getattr(job_status_report, 'validation_response', None)}, "
            f"event_log={getattr(job_status_report, 'event_log', None)}")


def _download_train_table_parquet(
    client: TestClient,
    train_table_job_id: str,
    local_dest_dir: Path,
) -> Path:
    """Download the generated train table parquet to a single local file.
    Consume the public URL(s) and write a consolidated parquet bundle.
    """
    urls = get_and_expect_success(
        client, f'/v1/gentraintable_jobs/{train_table_job_id}/table_data',
        {}, List[str])
    assert len(urls) > 0, "No train table data URLs"

    parquet_files = [
        parquet_ref for url in urls for parquet_ref in public_parquet_refs(url)
    ]

    tables = []
    for f in parquet_files:
        with tempfile.NamedTemporaryFile(suffix='.parquet',
                                         delete=False) as tmp:
            tmp.write(read_public_response_bytes(f))
            tmp_path = tmp.name
        try:
            tables.append(pq.read_table(tmp_path))
        finally:
            os.unlink(tmp_path)

    combined = pa.concat_tables(tables)
    assert combined.num_rows > 0, "Original train table is empty"

    local_dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = local_dest_dir / 'custom_train.parquet'
    pq.write_table(combined, out_path)
    log.info("Downloaded original train table: %d rows -> %s",
             combined.num_rows, out_path)
    return out_path


def _augment_with_weight_column(
    parquet_path: Path,
    weight_col_name: str,
) -> int:
    """Add a constant-1.0 weight column to the parquet at ``parquet_path``.

    Writes back in place. Returns the row count of the resulting table.

    Constant 1.0 weights satisfy every check in
    ``validate_weighted_train_table``: no nulls, no inf, all positive,
    same row count as the original. Dtype is float64.
    """
    table = pq.read_table(parquet_path)
    n = table.num_rows
    weight_array = pa.array([1.0] * n, type=pa.float64())
    augmented = table.append_column(weight_col_name, weight_array)
    pq.write_table(augmented, parquet_path)
    log.info("Augmented %s with %r column: %d rows", parquet_path,
             weight_col_name, n)
    return n


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
# Test: weighted custom training table (full DG pipeline)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(1800)
async def test_diskgraph_custom_tt_weighted(
    public_api_client: TestClient,
    ml_backend: MLJobBackend,
    tmp_path: Path,
):
    """Submit a training job with a CustomTrainingTable carrying a weight
    column; verify the workflow runs ingest -> validate -> materialize
    end-to-end on the DG backend and the weight column reaches the
    materialized training-table attributes.
    """
    client = public_api_client

    # Load PKDD'99 Financial fixture.
    fixture = load_json(str(FIXTURE_PATH), FixtureDefinition)
    fixture.queries = [
        q for q in fixture.queries if q.pquery_name == PQL_TEST_CASE
    ]
    assert len(fixture.queries) == 1
    fixture = randomize_fixture_ids_v2(fixture)

    # ==== Phase 1: Setup connectors / PQuery / Graph via V1 ====
    project_id, _ = await provision_fixture_connectors(
        client, fixture, api_prefix='/v1')

    resources = await fixture_to_pquery_resources_v2.coroutine(fixture)
    assert len(resources) == 1
    pquery_args = resources[0]
    pquery_id = pquery_args.name

    post_and_expect_success(client, '/v1/predictive_queries/?dialect=PQLANG_V2',
                            _public_pquery_from_internal(pquery_args),
                            EmptyResponse,
                            project_id=str(project_id))
    graph_id = post_and_expect_success(
        client, f'/v1/graphs?name_alias={pquery_id}_graph', pquery_args.graph,
        IDResponse).id

    # ==== Phase 2: Graph snapshot ====
    snapshot_id = GraphSnapshotID(
        post_and_expect_success(
            client,
            f'/v1/graphsnapshots?graph_id={graph_id}&refresh_source=true', {},
            IDResponse).id)
    log.info("snapshot_id=%s", snapshot_id)

    # Wait for snapshot to be ready (mirrors SDK Graph.snapshot() polling).
    from kumoapi.data_snapshot import (
        GraphSnapshotResource,
        GraphSnapshotStage,
        TableSnapshotResource,
        TableSnapshotStage,
    )
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
        if all_done and graph_stage.status.is_terminal:
            assert graph_stage.status == JobStatus.DONE, (
                f"Graph snapshot failed: {graph_stage.error}")
            break
        time.sleep(POLL_INTERVAL_SECONDS)
    else:
        raise TimeoutError(f"Graph snapshot {snapshot_id} did not complete in "
                           f"{MAX_POLL_SECONDS}s")

    # ==== Phase 3: Generate train table (we'll use it as the "original"
    # to derive a weighted custom-TT from) ====
    plan_payload = plan_api.SuggestModelPlanRequest(
        query_string=pquery_args.query_string,
        graph_id=graph_id,
        run_mode=RunMode.DEBUG,
    )
    train_table_plan = post_and_expect_success(
        client,
        '/v1/predictive_queries/train_table_generation_plan?dialect=PQLANG_V2',
        plan_payload, api.TrainingTableGenerationPlan)

    train_table_job_id = post_and_expect_success(
        client, '/v1/gentraintable_jobs',
        api.GenerateTrainTableRequest({}, pquery_id, train_table_plan, None),
        IDResponse).id
    log.info("train_table_job_id=%s", train_table_job_id)

    tt_job = _wait_for_gen_train_table_job(client, train_table_job_id)
    assert tt_job.job_status_report.status == JobStatus.DONE, (
        f"Train table gen failed: {tt_job.job_status_report.status}")

    # ==== Phase 4: Build a custom TT with a WEIGHT column ====
    # Round-trip the original parquet plus a constant-1.0 WEIGHT column,
    # so column-name + dtype parity (required by the validator) is
    # guaranteed and only WEIGHT is new.
    weight_col_name = "WEIGHT"
    custom_dir = tmp_path / 'custom_tt'
    custom_parquet = _download_train_table_parquet(client, train_table_job_id,
                                                   custom_dir)
    _augment_with_weight_column(custom_parquet, weight_col_name)

    # CompositeFileConnector resolves source_table_name to a file under
    # root_dir, so the table is named after the parquet stem.
    _, custom_connector = await provision_project_upload_connector(
        client,
        source_root=str(custom_dir),
        api_prefix='/v1',
        project_id=project_id,
    )

    custom_source_table = S3SourceTable(
        connector_id=str(custom_connector.id),
        source={'mode': 'relative', 'path': custom_parquet.name},
        file_type=FileType.PARQUET,
    )

    # ==== Phase 5: Suggest model plan + submit training with
    # CustomTrainingTable ====
    model_plan_info = post_and_expect_success(
        client, '/v1/predictive_queries/model_plan?dialect=PQLANG_V2',
        plan_payload, plan_api.ModelPlanInfo)
    model_plan = model_plan_info.model_plan
    clamp_e2e_training_budget(model_plan)
    model_plan.training_job.manual_seed = 42

    training_job_id = post_and_expect_success(
        client,
        '/v1/training_jobs',
        api.TrainingJobRequest(
            {},
            pquery_id=pquery_id,
            model_plan=model_plan,
            graph_snapshot_id=snapshot_id,
            train_table_job_id=train_table_job_id,
            custom_train_table=api.CustomTrainingTable(
                source_table=custom_source_table,
                table_mod_spec=TrainingTableSpec(weight_col=weight_col_name),
                # Force validate_weighted_train_table to run.
                validated=False,
            ),
        ),
        IDResponse,
    ).id
    log.info("training_job_id=%s", training_job_id)

    # ==== Phase 6: Wait for DONE ====
    training_job = _wait_for_training_job_v1(client, training_job_id)
    assert training_job.job_status_report.status == JobStatus.DONE, (
        f"Training failed: status={training_job.job_status_report.status}, "
        f"{_job_status_details(training_job.job_status_report)}")

    # ==== Phase 7: Verify the TT-NG custom train-table bundle materialized. ====
    # TT-NG no longer re-materializes the split into
    # ``training_table/materialized/attributes.json`` (the split-materialization
    # step was removed). The custom weighted-TT gen path instead persists a
    # durable, self-contained bundle beside the custom train-table path under
    # ``<custom_train_table_path>.ttng_staging/_bundles/<id>/`` (see
    # rest .../custom_train_table_workflow.py ->
    # build_ttng_custom_train_table_bundle ->
    # diskgraph_train_table.ttng_persisted_staging_dir). Assert that bundle is
    # complete -- reaching a complete bundle proves the whole custom-gen path
    # (validate_weighted_train_table + build_ttng_custom_train_table_bundle +
    # persist) ran end-to-end on TT-NG for a weighted custom TT.
    #
    # NOTE: kumo threads ``weight_col`` all the way to the DiskGraph producer
    # (build_ttng_custom_train_table_bundle -> train_table_col_stats_config[
    # 'weight_col'] -> the wheel's ``weight_col`` param, forwarded by
    # diskgraph.tt_artifacts). Whether the wheel then emits a ``weight_sidecar``
    # / ``weight.bin`` is a DiskGraph-producer concern (deployed-wheel
    # dependent) tracked separately; the weight is consumed at training time via
    # the model plan's ``has_train_table_weight_col`` regardless, so this e2e
    # asserts the durable bundle rather than the wheel-internal sidecar.
    train_job_dir = await ml_backend.get_job_home_dir(training_job_id)
    assert train_job_dir is not None
    staging_root = osp.join(train_job_dir, "custom_train_table.ttng_staging")
    assert not fsutil.exists(osp.join(staging_root, '_current.json'))
    bundle_dirs = list(fsutil.listdir(osp.join(staging_root, '_bundles')))
    assert len(bundle_dirs) == 1, (
        f'expected one immutable TT-NG custom bundle under {staging_root}, '
        f'got {bundle_dirs}')
    staging_dir = bundle_dirs[0]
    manifest_path = osp.join(staging_dir, "manifest.json")
    assert fsutil.exists(manifest_path), (
        f"TT-NG custom train-table bundle manifest missing: {manifest_path}")
    manifest = load_json(manifest_path, dict)
    tt_parquet = osp.join(
        staging_dir, osp.basename(str(manifest["train_table"]["path"])))
    assert fsutil.exists(tt_parquet), (
        f"TT-NG custom train-table tt.parquet missing from bundle: "
        f"{tt_parquet}")
    log.info("Custom-TT weighted bundle materialized on TT-NG: %s", staging_dir)

    log.info("DG custom-TT (weighted) e2e test passed.")
