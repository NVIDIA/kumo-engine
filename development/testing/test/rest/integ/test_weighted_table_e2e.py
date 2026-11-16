# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from pathlib import Path
from test.rest.integ import fixture_connector_args
from test.rest.integ.training_budget import clamp_e2e_training_budget
from typing import Tuple

import kumoapi.jobs as jobs_api
import kumoapi.model_plan as plan_api
import kumoapi.pquery as pquery_api
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from kumoapi.common import ValidationResponse
from kumoapi.graph import GraphDefinition
from kumoapi.source_table import FileType, S3SourceTable
from kumoapi.train import TrainingTableSpec

from io_utils import logging, parquet_util
from io_utils.json import load_json
from kumo.testing.rest import get_and_expect_success, post_and_expect_success
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.status import IDResponse
from statedb.config.common import GenerateTrainTableJobID, TrainingJobID

log = logging.getLogger(__name__)

# Requires the pool subprocesses (controller + singlehost daemon) so
# DiskgraphIngestWorkflow's place_snapshot_activity dispatch doesn't
# stall on an unattended AutoRPC queue. ``local`` storage_backend is
# fine — this isn't a storage-coverage suite.
pytestmark = [
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", ["local"], indirect=True),
]


async def generate_train_table(
    public_api_client: TestClient,
    graph_definition_v2: GraphDefinition,
    wait_for_completion: bool = True,
    pquery_str: str = 'PREDICT customers.age FOR EACH customers.customer_id',
) -> Tuple[str, str, str, str]:

    project_id = str(
        await fixture_connector_args.ensure_test_project(public_api_client))

    graph_id = post_and_expect_success(
        public_api_client,
        '/v1/graphs',
        graph_definition_v2,
        IDResponse,
        project_id=project_id,
    )
    plan_request = plan_api.SuggestModelPlanRequest(
        query_string=pquery_str,
        graph_id=graph_id.id,
        run_mode=plan_api.RunMode.FAST,
    )
    pquery_request = pquery_api.PQueryResource(
        query_string=pquery_str,
        graph=graph_definition_v2,
    )
    pquery_id = post_and_expect_success(public_api_client,
                                        '/v1/predictive_queries/',
                                        pquery_request, IDResponse,
                                        project_id=project_id).id

    train_table_plan = post_and_expect_success(
        public_api_client,
        ('/v1/predictive_queries/train_table_generation_plan'), plan_request,
        jobs_api.TrainingTableGenerationPlan, project_id=project_id)
    train_table_job_id = post_and_expect_success(
        public_api_client, '/v1/gentraintable_jobs',
        jobs_api.GenerateTrainTableRequest({}, pquery_id, train_table_plan,
                                           None), IDResponse,
        project_id=project_id).id

    deadline = asyncio.get_running_loop().time() + 120
    train_table_job = None
    while asyncio.get_running_loop().time() < deadline:
        response = public_api_client.get(
            f'/v1/gentraintable_jobs/{train_table_job_id}',
            params={'project_id': project_id},
        )
        if response.status_code == 200:
            train_table_job = jobs_api.GenerateTrainTableJobResource(
                **response.json())
            break
        await asyncio.sleep(1)
    assert train_table_job is not None, (
        f'Generate train table job {train_table_job_id} was not visible '
        'after create.')

    if wait_for_completion:
        status = train_table_job.job_status_report.status
        while not status.is_terminal:
            await asyncio.sleep(10)
            status = get_and_expect_success(
                public_api_client,
                f'/v1/gentraintable_jobs/{train_table_job_id}', {
                    'project_id': project_id
                },
                jobs_api.GenerateTrainTableJobResource).job_status_report.status

    return pquery_id, train_table_job_id, graph_id.id, project_id


async def _write_weighted_train_table(
    ml_backend: MLJobBackend,
    train_table_job_id: str,
    root_dir: Path,
    table_name: str,
    *,
    drop_last_row: bool = False,
) -> Path:
    """Write a weighted copy of the generated table under ``root_dir``.

    Building this fixture from the job output keeps row-count validation
    meaningful when the source graph fixture grows or changes shape.
    """
    result = await ml_backend.get_train_table_result(
        GenerateTrainTableJobID(train_table_job_id))
    table = parquet_util.read_parquet(
        result.train_table_path, return_arrow=True)
    assert isinstance(table, pa.Table)
    assert table.num_rows > 1
    if drop_last_row:
        table = table.slice(0, table.num_rows - 1)
    table = table.append_column(
        'WEIGHT', pa.array([1.0] * table.num_rows, type=pa.float64()))

    root_dir.mkdir(parents=True, exist_ok=True)
    path = root_dir / f'{table_name}.parquet'
    pq.write_table(table, path)
    return path


# Custom weighted train table flow: validate bad custom table, then train with
# a valid custom weighted table and verify weight column is used.
@pytest.mark.asyncio
async def test_weighted_train_table(
    public_api_client: TestClient,
    graph_definition_v2: GraphDefinition,
    ml_backend: MLJobBackend,
    tmp_path: Path,
):

    # Generate train table
    pquery_id, train_table_job_id, graph_id, project_id = await generate_train_table(
        public_api_client,
        graph_definition_v2,
    )
    # Reuse an existing table from the project-owned graph fixture. It is
    # intentionally not a valid weighted train table because it lacks the
    # requested weight column.
    source_table = graph_definition_v2.tables['transactions'].source_table
    validate_request = {
        'custom_table': source_table,
        'train_table_mod': TrainingTableSpec(weight_col='weight', )
    }
    response = post_and_expect_success(
        public_api_client, (f'/v1/gentraintable_jobs/{train_table_job_id}/'
                            'validate_custom_train_table'), validate_request,
        ValidationResponse, project_id=project_id)
    assert not response.ok
    assert response.errors[0].title == "Weight column not found"

    # Derive the custom table from this job's generated table so it stays
    # semantically aligned with the graph fixture.
    weighted_root = tmp_path / 'weighted_train_table'
    await _write_weighted_train_table(
        ml_backend,
        train_table_job_id,
        weighted_root,
        'weighted_train',
    )
    _, weight_connector = await fixture_connector_args.provision_project_upload_connector(
        public_api_client,
        source_root=str(weighted_root),
        project_id=project_id,
        api_prefix='/v1',
    )
    source_table = S3SourceTable(
        connector_id=str(weight_connector.id),
        source={'mode': 'relative', 'path': "weighted_train"},
        file_type=FileType.PARQUET,
    )

    model_plan = post_and_expect_success(
        public_api_client,
        '/v1/predictive_queries/model_plan',
        plan_api.SuggestModelPlanRequest(
            query_string=('PREDICT customers.age '
                          'FOR EACH customers.customer_id'),
            graph_id=graph_id,
            run_mode=plan_api.RunMode.FAST,
        ),
        plan_api.ModelPlanInfo,
        project_id=project_id,
    ).model_plan
    clamp_e2e_training_budget(model_plan)

    # Trigger Trainer with the weighted train table
    training_job_config = jobs_api.TrainingJobRequest(
        {}, pquery_id, model_plan, None, train_table_job_id,
        custom_train_table=jobs_api.CustomTrainingTable(
            source_table=source_table,
            table_mod_spec=jobs_api.TrainingTableSpec(weight_col="WEIGHT", ),
            validated=True,
        ))
    train_job_id = post_and_expect_success(
        public_api_client,
        '/v1/training_jobs',
        training_job_config,
        IDResponse,
        project_id=project_id,
    ).id

    # Check the status of the training job
    status = get_and_expect_success(
        public_api_client, f'/v1/training_jobs/{train_job_id}', {
            'project_id': project_id
        }, jobs_api.TrainingJobResource).job_status_report.status
    while not status.is_terminal:
        await asyncio.sleep(10)
        status = get_and_expect_success(
            public_api_client, f'/v1/training_jobs/{train_job_id}', {
                'project_id': project_id
            }, jobs_api.TrainingJobResource).job_status_report.status
    assert status == jobs_api.JobStatus.DONE

    train_job_dir = await ml_backend.get_job_home_dir(
        TrainingJobID(train_job_id))
    assert train_job_dir is not None

    # Assert the TT-NG custom weighted-TT bundle materialized. TT-NG no longer
    # re-materializes the split into
    # ``training_table/materialized/attributes.json`` (that step was removed);
    # the custom weighted-TT gen path persists a durable, self-contained bundle
    # under ``<custom_train_table_path>.ttng_staging/_bundles/<id>/``. Reaching
    # a complete bundle proves the whole custom weighted-TT gen path ran
    # end-to-end on TT-NG.
    #
    # NOTE: kumo threads ``weight_col`` all the way to the DiskGraph producer;
    # whether the wheel emits a ``weight_sidecar`` / ``weight.bin`` is a
    # deployed-wheel-dependent DiskGraph-producer concern tracked separately.
    # The weight is consumed at training time via the model plan's
    # ``has_train_table_weight_col``, so this e2e asserts the durable bundle
    # rather than the wheel-internal sidecar.
    staging_root = os.path.join(train_job_dir,
                                "custom_train_table.ttng_staging")
    assert not os.path.exists(os.path.join(staging_root, '_current.json'))
    bundle_dirs = list(Path(staging_root, '_bundles').iterdir())
    assert len(bundle_dirs) == 1, (
        f'expected one immutable TT-NG custom bundle under {staging_root}, '
        f'got {bundle_dirs}')
    staging_dir = str(bundle_dirs[0])
    manifest_path = os.path.join(staging_dir, "manifest.json")
    assert os.path.exists(manifest_path), (
        f"TT-NG custom train-table bundle manifest missing: {manifest_path}")
    manifest = load_json(manifest_path, dict)
    tt_parquet = os.path.join(
        staging_dir, os.path.basename(str(manifest["train_table"]["path"])))
    assert os.path.exists(tt_parquet), (
        f"TT-NG custom train-table tt.parquet missing from bundle: {tt_parquet}")


# Extensive validation for custom weighted train tables: row-count checks with
# and without extensive_validation enabled.
@pytest.mark.asyncio
async def test_extensive_validation_weighted_train_table(
    public_api_client: TestClient,
    graph_definition_v2: GraphDefinition,
    ml_backend: MLJobBackend,
    tmp_path: Path,
):
    _, train_table_job_id, _, project_id = await generate_train_table(
        public_api_client, graph_definition_v2)

    # Materialize matching and mismatching tables from the generated output.
    # This exercises row-count semantics without pinning a fixture row count.
    weighted_root = tmp_path / 'weighted_validation_tables'
    await _write_weighted_train_table(
        ml_backend,
        train_table_job_id,
        weighted_root,
        'weighted_train',
    )
    await _write_weighted_train_table(
        ml_backend,
        train_table_job_id,
        weighted_root,
        'wrong_weighted_train',
        drop_last_row=True,
    )
    _, weight_connector = await fixture_connector_args.provision_project_upload_connector(
        public_api_client,
        source_root=str(weighted_root),
        project_id=project_id,
        api_prefix='/v1',
    )
    correct_source_table = S3SourceTable(
        connector_id=str(weight_connector.id),
        source={'mode': 'relative', 'path': "weighted_train"},
        file_type=FileType.PARQUET,
    )

    validate_request = {
        'custom_table': correct_source_table,
        'train_table_mod': TrainingTableSpec(weight_col='WEIGHT', ),
        'extensive_validation': True,
    }

    response = post_and_expect_success(
        public_api_client, (f'/v1/gentraintable_jobs/{train_table_job_id}/'
                            'validate_custom_train_table'), validate_request,
        ValidationResponse, project_id=project_id)

    assert response.ok

    # Case 2: Weighted training table doesn't have the correct number of rows
    wrong_num_rows_source_table = S3SourceTable(
        connector_id=str(weight_connector.id),
        source={'mode': 'relative', 'path': "wrong_weighted_train"},
        file_type=FileType.PARQUET,
    )

    validate_request = {
        'custom_table': wrong_num_rows_source_table,
        'train_table_mod': TrainingTableSpec(weight_col='WEIGHT', ),
        'extensive_validation': True,
    }

    response = post_and_expect_success(
        public_api_client, (f'/v1/gentraintable_jobs/{train_table_job_id}/'
                            'validate_custom_train_table'), validate_request,
        ValidationResponse, project_id=project_id)

    assert not response.ok
    assert "Your weighted training table and original training table don't contain the same number of rows." in response.errors[
        0].message

    # Case 3: Weighted training table doesn't have the correct number of rows
    # but extensive_validation is disabled
    validate_request['extensive_validation'] = False

    response = post_and_expect_success(
        public_api_client, (f'/v1/gentraintable_jobs/{train_table_job_id}/'
                            'validate_custom_train_table'), validate_request,
        ValidationResponse, project_id=project_id)
    assert response.ok
