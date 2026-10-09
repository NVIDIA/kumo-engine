# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for GCS connector flow using the v2 public API.

V2 sibling of `test_gcs_e2e.py`. Same GCS-connector coverage, but the
training and batch-prediction phases use the explicit multi-step v2 public
endpoints (`/v1/gentraintable_jobs`, `/v1/training_jobs`, `/v1/prediction_jobs`)
instead of the legacy auto-cascade training/prediction routes.
"""
import dataclasses
import os
import time
import uuid
from test.rest.integ import fixture_connector_args
from test.rest.integ.gcs_connector_helpers import (
    create_gcs_connector_in_project,
    create_identity_backed_gcs_connector,
    get_gcs_source_table,
    load_gcs_test_identity,
)
from typing import List, Optional, Tuple

import kumoapi.jobs as api
import kumoapi.model_plan as plan_api
import kumoapi.pquery as pquery_api
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from kumoapi.graph import ColumnKey, ColumnKeyGroup, GraphDefinition
from kumoapi.model_plan import RunMode
from kumoapi.source_table import (
    SourceTableDataResponse,
    SourceTableListResponse,
)

from io_utils import fsutil
from kumo.testing.gcs import ENV_GCS_TEST_BUCKET, withGCSCredentials
from kumo.testing.rest import get_and_expect_success, post_and_expect_success
from kumo.validation import ValidationResponse
from rest.config.graph_v2 import GraphDefinition as RestGraphDefinition
from rest.config.status import IDResponse
from rest.config.table_v2 import Column, TableDefinition
from statedb.config.common import GraphSnapshotID, JobStatus, TableName

# Pool fixtures (storage_backend, _diskgraph_deploy_config, _pool_subprocesses)
# live in ``test/rest/integ/conftest.py``. Without ``_pool_subprocesses`` the
# pool controller / worker subprocesses never spawn, so ``place_snapshot_activity``
# (routed via ``DiskgraphIngestWorkflow`` -> ``pool.wf_coordinator``) hangs on a
# no-worker AutoRPC queue and the workflow times out.
pytestmark = [
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", ["local"], indirect=True),
]

# GCS test bucket and paths
GCS_TEST_BUCKET = os.getenv(ENV_GCS_TEST_BUCKET, 'kumo-test-bucket')
GCS_INTEG_TEST_PATH = 'integ_tests'

# Polling cadence for v2 job-status endpoints.
MAX_POLL_SECONDS = 1800
POLL_INTERVAL_SECONDS = 10


def _get_gcs_financial_path() -> str:
    return f"gs://{GCS_TEST_BUCKET}/{GCS_INTEG_TEST_PATH}/FINANCIAL"


def _get_gcs_financial_e2e_path() -> str:
    return f"gs://{GCS_TEST_BUCKET}/{GCS_INTEG_TEST_PATH}/FINANCIAL_E2E"


def _generate_unique_suffix() -> str:
    return uuid.uuid4().hex[:8]


def _wait_for_source_tables(
    client: TestClient,
    project_id: str,
    connector_id: str,
) -> SourceTableListResponse:
    return get_and_expect_success(
        client,
        f'/v0/projects/{project_id}/connectors/{connector_id}/source-tables',
        {},
        SourceTableListResponse,
    )


async def _create_table_from_connector(
    client: TestClient,
    project_id: str,
    connector_id: str,
    source_table_name: str,
    table_name: str,
    pkey: str,
    time_col: str = None,
) -> TableDefinition:
    columns_response = get_and_expect_success(
        client,
        f'/v0/projects/{project_id}/connectors/{connector_id}/columns',
        {'table_name': source_table_name},
        SourceTableDataResponse,
    )
    cols = [
        Column(name=col.name, stype=col.stype, dtype=col.dtype)
        for col in columns_response.cols
    ]
    table_def = TableDefinition(
        cols=cols,
        source_table=get_gcs_source_table(
            client,
            project_id=project_id,
            connector_id=connector_id,
            source_table_name=source_table_name,
        ),
        pkey=pkey,
        time_col=time_col,
        end_time_col=None,
    )
    assert table_def is not None
    post_and_expect_success(
        client,
        '/v1/tables/',
        table_def,
        IDResponse,
        name_alias=table_name,
        project_id=project_id,
    )
    return table_def


def _wait_for_gen_train_table_job(
    client: TestClient,
    job_id: str,
) -> api.GenerateTrainTableJobResource:
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        job = get_and_expect_success(client, f'/v1/gentraintable_jobs/{job_id}',
                                     {}, api.GenerateTrainTableJobResource)
        if job.job_status_report.status.is_terminal:
            return job
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Gen-train-table job {job_id} did not complete in {MAX_POLL_SECONDS}s")


def _wait_for_training_job_v1(
    client: TestClient,
    job_id: str,
) -> api.TrainingJobResource:
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
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        job = get_and_expect_success(client, f'/v1/prediction_jobs/{job_id}',
                                     {}, api.BatchPredictionJobResource)
        if job.job_status_report.status.is_terminal:
            return job
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Prediction job {job_id} did not complete in {MAX_POLL_SECONDS}s")


@pytest.mark.asyncio
class TestGCSPublicAPIE2EFlow:
    """End-to-end GCS coverage exercising the v2 public training/BP API."""
    @withGCSCredentials
    async def test_gcs_full_e2e_flow_v2(
        self,
        client: TestClient,
        public_api_client: TestClient,
    ):
        """GCS e2e using `/v1/{gentraintable,training,prediction}_jobs/`."""
        suffix = _generate_unique_suffix()

        # Step 1: Create GCS connectors
        financial_connector = f'gcs_financial_{suffix}'
        financial_e2e_connector = f'gcs_financial_e2e_{suffix}'

        identity_config, identity_credentials = load_gcs_test_identity()
        financial_setup = create_identity_backed_gcs_connector(
            client,
            name=financial_connector,
            root_dir=_get_gcs_financial_path(),
            identity_config=identity_config,
            identity_credentials=identity_credentials,
        )
        project_id = financial_setup.project_id
        financial_connector = financial_setup.connector_id
        financial_e2e_connector = create_gcs_connector_in_project(
            client,
            name=financial_e2e_connector,
            root_dir=_get_gcs_financial_e2e_path(),
            project_id=project_id,
            identity_id=financial_setup.identity_id,
        )
        fixture_connector_args.bind_test_client_project(
            public_api_client, project_id)

        # Step 2: Wait for source tables to be discovered
        source_tables = _wait_for_source_tables(client, project_id,
                                                financial_connector)
        table_names = set(source_tables.table_names)
        assert 'ACCOUNT' in table_names
        assert 'TRANS' in table_names

        source_tables_e2e = _wait_for_source_tables(client, project_id,
                                                    financial_e2e_connector)
        e2e_table_names = set(source_tables_e2e.table_names)
        assert 'LOAN' in e2e_table_names

        # Step 3: Create tables from connectors
        account_name = TableName(f'ACCOUNT_{suffix}')
        trans_name = TableName(f'TRANS_{suffix}')
        loan_name = TableName(f'LOAN_{suffix}')
        account_table_def = await _create_table_from_connector(
            client, project_id, financial_connector, 'ACCOUNT',
            account_name, pkey='ACCOUNT_ID')
        trans_table_def = await _create_table_from_connector(
            client, project_id, financial_connector, 'TRANS',
            trans_name, pkey='TRANS_ID', time_col='DATE')
        loan_table_def = await _create_table_from_connector(
            client, project_id, financial_e2e_connector, 'LOAN', loan_name,
            pkey='LOAN_ID', time_col='DATE')

        # Step 4: Create graph linking the tables (v2)
        graph_def = GraphDefinition(
            tables={
                account_name: account_table_def,
                trans_name: trans_table_def,
                loan_name: loan_table_def,
            },
            col_groups=[
                ColumnKeyGroup(columns=(
                    ColumnKey(table_name=account_name, col_name='ACCOUNT_ID'),
                    ColumnKey(table_name=trans_name, col_name='ACCOUNT_ID'),
                    ColumnKey(table_name=loan_name, col_name='ACCOUNT_ID'),
                )),
            ],
        )
        graph_id = post_and_expect_success(
            client,
            '/v1/graphs/',
            graph_def,
            IDResponse,
            project_id=project_id,
        ).id

        # Step 5: Create predictive query (v2)
        pquery_name = f'gcs_pquery_{suffix}'
        graph_def_internal = RestGraphDefinition(
            **dataclasses.asdict(graph_def))
        query_string = (f"PREDICT LAST(LOAN_{suffix}.STATUS, 0, 30) = 'A' "
                        f"FOR EACH ACCOUNT_{suffix}.ACCOUNT_ID")
        post_and_expect_success(
            client,
            '/v1/predictive_queries/',
            pquery_api.PQueryResource(
                name=pquery_name,
                query_string=query_string,
                graph=graph_def_internal,
                desc='gcs e2e (v2)',
            ),
            IDResponse,
            project_id=project_id,
        )

        # Step 6: Graph snapshot for the pquery's graph
        snapshot_id = GraphSnapshotID(
            post_and_expect_success(
                public_api_client,
                f'/v1/graphsnapshots?graph_id={graph_id}&refresh_source=true',
                {}, IDResponse, project_id=project_id).id)

        # Step 7: Generate the training table (v2)
        plan_payload = plan_api.SuggestModelPlanRequest(
            query_string=query_string,
            graph_id=graph_id,
            run_mode=RunMode.FAST,
        )
        train_table_plan = post_and_expect_success(
            public_api_client,
            '/v1/predictive_queries/train_table_generation_plan?'
            'dialect=PQLANG_V2',
            plan_payload,
            api.TrainingTableGenerationPlan,
            project_id=project_id,
        )
        train_table_job_id = post_and_expect_success(
            public_api_client, '/v1/gentraintable_jobs',
            api.GenerateTrainTableRequest({}, pquery_name, train_table_plan,
                                          snapshot_id), IDResponse,
            project_id=project_id).id
        tt_job = _wait_for_gen_train_table_job(public_api_client,
                                               train_table_job_id)
        assert tt_job.job_status_report.status == JobStatus.DONE, (
            f"Train table gen failed: {tt_job.job_status_report.status}")

        # Step 7b: Verify train table presigned URLs work for GCS
        train_table_urls = get_and_expect_success(
            public_api_client,
            f'/v1/gentraintable_jobs/{train_table_job_id}/table_data',
            {},
            List[str],
        )
        assert len(train_table_urls) > 0

        # Step 8: Launch training (v2)
        model_plan_info = post_and_expect_success(
            public_api_client,
            '/v1/predictive_queries/model_plan?dialect=PQLANG_V2', plan_payload,
            plan_api.ModelPlanInfo, project_id=project_id)
        model_plan = model_plan_info.model_plan
        training_job_id = post_and_expect_success(
            public_api_client, '/v1/training_jobs',
            api.TrainingJobRequest({}, pquery_id=pquery_name,
                                   model_plan=model_plan,
                                   graph_snapshot_id=snapshot_id,
                                   train_table_job_id=train_table_job_id),
            IDResponse, project_id=project_id).id
        training_job = _wait_for_training_job_v1(public_api_client,
                                                 training_job_id)
        assert training_job.job_status_report.status == JobStatus.DONE, (
            f"Training failed: {training_job.job_status_report.status}")

        # Step 9: Batch prediction with GCS output (predictions + embeddings)
        gcs_predictions_path = (f"gs://{GCS_TEST_BUCKET}/{GCS_INTEG_TEST_PATH}/"
                                f"predictions/{suffix}/")
        gcs_embeddings_path = (f"gs://{GCS_TEST_BUCKET}/{GCS_INTEG_TEST_PATH}/"
                               f"embeddings/{suffix}/")

        bp_result = post_and_expect_success(
            public_api_client,
            '/v1/prediction_jobs',
            api.BatchPredictionRequest(
                {},
                model_training_job_id=training_job_id,
                predict_options=api.BatchPredictionOptions(
                    binary_classification_threshold=0.5,
                    num_workers=1,
                ),
                outputs=[
                    api.GCSPredictionOutput(
                        artifact_type=api.PredictionArtifactType.PREDICTIONS,
                        connector_id=financial_connector,
                        file_path=gcs_predictions_path,
                    ),
                    api.GCSPredictionOutput(
                        artifact_type=api.PredictionArtifactType.EMBEDDINGS,
                        connector_id=financial_connector,
                        file_path=gcs_embeddings_path,
                    ),
                ],
                graph_snapshot_id=snapshot_id,
                pred_table_job_id=None,
            ),
            Tuple[Optional[str], ValidationResponse],
            project_id=project_id,
        )
        bp_job_id = bp_result[0]
        assert bp_job_id is not None
        bp_job = _wait_for_prediction_job_v1(public_api_client, bp_job_id)
        assert bp_job.job_status_report.status == JobStatus.DONE, (
            f"BP failed: {bp_job.job_status_report.status}")

        # Step 10: Verify prediction output exists in customer-specified GCS path
        assert fsutil.exists(gcs_predictions_path), (
            f"Prediction output not found: {gcs_predictions_path}")

        prediction_files = fsutil.listdir(gcs_predictions_path)
        parquet_files = [f for f in prediction_files if f.endswith('.parquet')]
        assert len(parquet_files) > 0

        df = pd.read_parquet(parquet_files[0])
        assert len(df) > 0

        # Step 11: Get prediction results via presigned URLs from internal storage
        prediction_urls_response = get_and_expect_success(
            public_api_client,
            f'/v1/prediction_jobs/{bp_job_id}/get_prediction_df_urls',
            {},
            api.GetPredictionsDfUrlResponse,
        )
        assert len(prediction_urls_response.prediction_partitions) > 0

        for url in prediction_urls_response.prediction_partitions[:1]:
            df = pd.read_parquet(url)
            assert len(df) > 0

        # Step 12: Verify embeddings output via public API
        embeddings_urls_response = get_and_expect_success(
            public_api_client,
            f'/v1/prediction_jobs/{bp_job_id}/get_embedding_df_urls',
            {},
            api.GetEmbeddingsDfUrlResponse,
        )
        assert len(embeddings_urls_response.embedding_partitions) > 0

        for url in embeddings_urls_response.embedding_partitions[:1]:
            df = pd.read_parquet(url)
            assert len(df) > 0
            assert len(df.columns) > 1
