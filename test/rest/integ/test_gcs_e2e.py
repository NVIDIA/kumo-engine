# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for GCS connector flow.

Tests the full GCS workflow: connector creation, table ingestion, training,
and batch prediction with output to GCS through the internal REST API.
"""
import dataclasses
import os
import uuid
from http import HTTPStatus
from test.rest.integ import fixture_connector_args
from test.rest.integ.gcs_connector_helpers import (
    create_gcs_connector_in_project,
    create_identity_backed_gcs_connector,
    get_gcs_source_table,
    load_gcs_test_identity,
)
from typing import List, Optional, Tuple

import kumoapi.data_source as data_source_api
import kumoapi.jobs as api
import kumoapi.pquery as pquery_api
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from kumoapi.graph import ColumnKey, ColumnKeyGroup, GraphDefinition
from kumoapi.source_table import (
    SourceTableDataResponse,
    SourceTableListResponse,
)

from io_utils import fsutil
from kumo.testing.gcs import ENV_GCS_TEST_BUCKET, withGCSCredentials
from kumo.testing.rest import (
    create_identity_binding,
    create_project_for_group,
    get_and_expect_success,
    post_and_expect_error,
    post_and_expect_success,
)
from kumo.validation import ValidationResponse
from rest.config.graph_v2 import GraphDefinition as RestGraphDefinition
from rest.config.status import IDResponse
from rest.config.table_v2 import Column, TableDefinition
from rest.testing.integration import (
    wait_for_prediction_job_to_finish,
    wait_for_training_job_to_finish,
)
from statedb.config.common import (
    BatchPredictionJobID,
    GenerateTrainTableJobID,
    GraphSnapshotID,
    JobStatus,
    TableName,
    TrainingJobID,
)
from temporalio_commons import TemporalClient

# Pool fixtures (storage_backend, _diskgraph_deploy_config, _pool_subprocesses)
# live in ``test/rest/integ/conftest.py``. Without ``_pool_subprocesses`` the
# pool controller / worker subprocesses never spawn, so ``place_snapshot_activity``
# (routed via ``DiskgraphIngestWorkflow`` -> ``pool.wf_coordinator``) hangs on a
# no-worker AutoRPC queue and the workflow times out.
# This suite covers GCS-connector e2e, not storage-backend coverage —
# only test_diskgraph_e2e.py needs the [s3] parametrize.
pytestmark = [
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", ["local"], indirect=True),
]

# GCS test bucket and paths
GCS_TEST_BUCKET = os.getenv(ENV_GCS_TEST_BUCKET, 'kumo-test-bucket')
GCS_INTEG_TEST_PATH = 'integ_tests'


def _get_gcs_financial_path() -> str:
    """Get the GCS path for FINANCIAL test data."""
    return f"gs://{GCS_TEST_BUCKET}/{GCS_INTEG_TEST_PATH}/FINANCIAL"


def _get_gcs_financial_e2e_path() -> str:
    """Get the GCS path for FINANCIAL_E2E test data."""
    return f"gs://{GCS_TEST_BUCKET}/{GCS_INTEG_TEST_PATH}/FINANCIAL_E2E"


def _generate_unique_suffix() -> str:
    """Generate a unique suffix for test resources."""
    return uuid.uuid4().hex[:8]


def _wait_for_source_tables(
    client: TestClient,
    project_id: str,
    connector_id: str,
) -> SourceTableListResponse:
    """Wait for source tables to be available."""
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
    """Create a v2 table from a connector using its inferred schema."""
    # Get column schema from connector
    columns_response = get_and_expect_success(
        client,
        f'/v0/projects/{project_id}/connectors/{connector_id}/columns',
        {'table_name': source_table_name},
        SourceTableDataResponse,
    )

    # Create table with inferred columns
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
    assert table_def is not None, (
        "TableDefinition._build_from_inputs returned None "
        f"for {source_table_name!r}")
    post_and_expect_success(
        client,
        '/v1/tables/',
        table_def,
        IDResponse,
        name_alias=table_name,
        project_id=project_id,
    )
    return table_def


@pytest.mark.asyncio
class TestGCSE2EFlow:
    """End-to-end tests for GCS using internal API flow."""
    @withGCSCredentials
    async def test_gcs_full_e2e_flow(
        self,
        client: TestClient,
        public_api_client: TestClient,
        temporal_client: TemporalClient,
    ):
        """Test full GCS e2e flow: connector -> tables -> graph -> train -> predict.

        This test uses the internal API with GCS data.
        """
        suffix = _generate_unique_suffix()

        identity_config, identity_credentials = load_gcs_test_identity()
        output_setup = create_identity_backed_gcs_connector(
            client,
            identity_config=identity_config,
            identity_credentials=identity_credentials,
            name=f'gcs_output_{suffix}',
            root_dir=f'gs://{GCS_TEST_BUCKET}/{GCS_INTEG_TEST_PATH}',
        )
        project_id = output_setup.project_id
        output_connector_id = output_setup.connector_id

        # The public client uses an owner-bound user API key. Grant that
        # authenticated test principal access through the project's immutable
        # group before exercising public training and prediction routes.
        membership_response = client.put(
            f'/v0/groups/{output_setup.group_id}/all-authenticated-users')
        assert membership_response.status_code == HTTPStatus.NO_CONTENT

        # Step 1: Create GCS connectors
        financial_connector = f'gcs_financial_{suffix}'
        financial_e2e_connector = f'gcs_financial_e2e_{suffix}'

        financial_connector = create_gcs_connector_in_project(
            client,
            name=financial_connector,
            root_dir=_get_gcs_financial_path(),
            project_id=project_id,
            identity_id=output_setup.identity_id,
        )
        financial_e2e_connector = create_gcs_connector_in_project(
            client,
            name=financial_e2e_connector,
            root_dir=_get_gcs_financial_e2e_path(),
            project_id=project_id,
            identity_id=output_setup.identity_id,
        )
        project_id = str(project_id)
        fixture_connector_args.bind_test_client_project(public_api_client,
                                                        project_id)

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
            client,
            project_id,
            financial_connector,
            'ACCOUNT',
            account_name,
            pkey='ACCOUNT_ID',
        )
        trans_table_def = await _create_table_from_connector(
            client,
            project_id,
            financial_connector,
            'TRANS',
            trans_name,
            pkey='TRANS_ID',
            time_col='DATE',
        )
        loan_table_def = await _create_table_from_connector(
            client,
            project_id,
            financial_e2e_connector,
            'LOAN',
            loan_name,
            pkey='LOAN_ID',
            time_col='DATE',
        )

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
        # Note: Table names in query must match the created table names (with suffix)
        pquery_name = f'gcs_pquery_{suffix}'
        graph_def_internal = RestGraphDefinition(
            **dataclasses.asdict(graph_def))
        post_and_expect_success(
            client,
            '/v1/predictive_queries/',
            pquery_api.PQueryResource(
                name=pquery_name,
                query_string=f"PREDICT LAST(LOAN_{suffix}.STATUS, 0, 30) = 'A' "
                f"FOR EACH ACCOUNT_{suffix}.ACCOUNT_ID",
                graph=graph_def_internal,
                desc='gcs e2e',
            ),
            IDResponse,
            project_id=project_id,
        )

        # Step 6: Start training via public API and wait for completion
        training_job_id = post_and_expect_success(
            public_api_client,
            '/v1/training_jobs',
            api.TrainingJobRequest({}, pquery_name),
            IDResponse,
            project_id=project_id,
        ).id
        training_job = get_and_expect_success(
            public_api_client,
            f'/v1/training_jobs/{training_job_id}',
            {},
            api.TrainingJobResource,
        )
        assert training_job.job_status_report.status.is_submitted

        await wait_for_training_job_to_finish(TrainingJobID(training_job_id),
                                              temporal_client)

        # Verify training completed
        training_job = get_and_expect_success(
            public_api_client,
            f'/v1/training_jobs/{training_job_id}',
            {},
            api.TrainingJobResource,
        )
        assert training_job.job_status_report.status == JobStatus.DONE

        # Step 6b: Verify train table presigned URLs work for GCS
        # This exercises presign_parquet_table_data() with a GCS path
        train_table_job_id = GenerateTrainTableJobID.as_child_of(
            TrainingJobID(training_job.job_id))
        train_table_urls = get_and_expect_success(
            public_api_client,
            f'/v1/gentraintable_jobs/{train_table_job_id}/table_data',
            {},
            List[str],
        )
        assert len(train_table_urls) > 0, \
            "Expected at least one presigned URL for train table data"

        # Step 8: Run batch prediction with GCS output (predictions + embeddings)
        gcs_predictions_path = (f"gs://{GCS_TEST_BUCKET}/{GCS_INTEG_TEST_PATH}/"
                                f"predictions/{suffix}/")
        gcs_embeddings_path = (f"gs://{GCS_TEST_BUCKET}/{GCS_INTEG_TEST_PATH}/"
                               f"embeddings/{suffix}/")

        # The v2 batch-prediction path requires an explicit graph snapshot.
        bp_snapshot_id = GraphSnapshotID(
            post_and_expect_success(
                public_api_client,
                f'/v1/graphsnapshots?graph_id={graph_id}&refresh_source=true',
                {}, IDResponse, project_id=project_id).id)

        predict_request = api.BatchPredictionRequest(
            {},
            model_training_job_id=training_job.job_id,
            graph_snapshot_id=bp_snapshot_id,
            predict_options=api.BatchPredictionOptions(
                binary_classification_threshold=0.5,
                num_workers=1,
            ),
            outputs=[
                api.GCSPredictionOutput(
                    artifact_type=api.PredictionArtifactType.PREDICTIONS,
                    connector_id=output_connector_id,
                    file_path=gcs_predictions_path,
                ),
                api.GCSPredictionOutput(
                    artifact_type=api.PredictionArtifactType.EMBEDDINGS,
                    connector_id=output_connector_id,
                    file_path=gcs_embeddings_path,
                ),
            ],
        )

        bp_result = post_and_expect_success(
            public_api_client,
            '/v1/prediction_jobs',
            predict_request,
            Tuple[Optional[str], ValidationResponse],
            project_id=project_id,
        )
        predict_job_id = bp_result[0]
        assert predict_job_id is not None

        # The v2 path does not set `current_batch_prediction_job_id` on the
        # pquery, so wait on the returned job id directly.
        await wait_for_prediction_job_to_finish(
            BatchPredictionJobID(predict_job_id), temporal_client)

        # Verify prediction completed
        predict_job = get_and_expect_success(
            public_api_client,
            f'/v1/prediction_jobs/{predict_job_id}',
            {},
            api.BatchPredictionJobResource,
        )
        assert predict_job.job_status_report.status == JobStatus.DONE

        # Step 9: Verify prediction output exists in customer-specified GCS path
        assert fsutil.exists(gcs_predictions_path), \
            f"Prediction output not found: {gcs_predictions_path}"

        # Read and verify prediction output from GCS
        prediction_files = fsutil.listdir(gcs_predictions_path)
        parquet_files = [f for f in prediction_files if f.endswith('.parquet')]
        assert len(parquet_files) > 0, "Expected at least one parquet file"

        # Read the first parquet file to verify content (listdir returns full paths)
        df = pd.read_parquet(parquet_files[0])
        assert len(df) > 0, "Prediction output should have rows"

        # Step 10: Get prediction results via presigned URLs from internal storage
        # Kumo stores predictions internally, then copies to customer location.
        # This endpoint returns presigned URLs for the internal storage copy.
        prediction_urls_response = get_and_expect_success(
            public_api_client,
            f'/v1/prediction_jobs/{predict_job_id}/get_prediction_df_urls',
            {},
            api.GetPredictionsDfUrlResponse,
        )
        assert len(prediction_urls_response.prediction_partitions) > 0, \
            "Expected at least one prediction URL from internal storage"

        # Verify we can read predictions via presigned URL
        for url in prediction_urls_response.prediction_partitions[:1]:
            df = pd.read_parquet(url)
            assert len(
                df) > 0, "Prediction from internal storage should have rows"

        # Step 11: Verify embeddings output via public API
        # This tests the embeddings functionality that SDK uses via predictions_job.embeddings_df()
        embeddings_urls_response = get_and_expect_success(
            public_api_client,
            f'/v1/prediction_jobs/{predict_job_id}/get_embedding_df_urls',
            {},
            api.GetEmbeddingsDfUrlResponse,
        )
        assert len(embeddings_urls_response.embedding_partitions) > 0, \
            "Expected at least one embeddings URL from internal storage"

        # Verify we can read embeddings via presigned URL
        for url in embeddings_urls_response.embedding_partitions[:1]:
            df = pd.read_parquet(url)
            assert len(
                df) > 0, "Embeddings from internal storage should have rows"
            # Embeddings should have many columns (one per dimension)
            assert len(
                df.columns) > 1, "Embeddings should have multiple columns"

    @withGCSCredentials
    async def test_gcs_connector_invalid_bucket(
        self,
        client: TestClient,
    ):
        """Test that invalid GCS bucket fails connector creation atomically."""
        identity_config, identity_credentials = load_gcs_test_identity()
        binding = create_identity_binding(
            client,
            identity_config=identity_config,
            credentials=identity_credentials,
        )
        project_id = create_project_for_group(client, binding.group_id)
        post_and_expect_error(
            client,
            f'/v0/projects/{project_id}/connectors/',
            data_source_api.CreateConnectorArgs(
                identity_id=binding.identity_id,
                name='gcs_invalid_bucket',
                config=data_source_api.GCSConnectorResourceConfig(
                    bucket='nonexistent-bucket-xyz-99999',
                    prefix='path',
                ),
            ),
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "Connector target validation failed",
        )


@pytest.mark.asyncio
class TestGCSConnectorOnly:
    """Tests for GCS connector creation and source table operations."""
    @withGCSCredentials
    async def test_gcs_connector_source_tables(
        self,
        client: TestClient,
    ):
        """Test GCS connector creation and source table listing."""
        suffix = _generate_unique_suffix()
        connector_name = f'gcs_connector_test_{suffix}'

        identity_config, identity_credentials = load_gcs_test_identity()
        setup = create_identity_backed_gcs_connector(
            client,
            name=connector_name,
            root_dir=_get_gcs_financial_path(),
            identity_config=identity_config,
            identity_credentials=identity_credentials,
        )

        source_tables = _wait_for_source_tables(client, setup.project_id,
                                                setup.connector_id)
        assert len(source_tables.table_names) > 0

        table_names = set(source_tables.table_names)
        expected_tables = {'ACCOUNT', 'LOAN', 'TRANS', 'CLIENT', 'DISP'}
        assert expected_tables.issubset(table_names), \
            f"Missing tables: {expected_tables - table_names}"

    @withGCSCredentials
    async def test_gcs_connector_sample_rows(
        self,
        client: TestClient,
    ):
        """Test fetching sample rows from GCS connector."""
        suffix = _generate_unique_suffix()
        connector_name = f'gcs_sample_test_{suffix}'

        identity_config, identity_credentials = load_gcs_test_identity()
        setup = create_identity_backed_gcs_connector(
            client,
            name=connector_name,
            root_dir=_get_gcs_financial_path(),
            identity_config=identity_config,
            identity_credentials=identity_credentials,
        )

        sample_response = get_and_expect_success(
            client,
            (f'/v0/projects/{setup.project_id}/connectors/'
             f'{setup.connector_id}/sample'),
            {'table_name': 'LOAN'},
            SourceTableDataResponse,
        )
        assert len(sample_response.cols) > 0
        assert sample_response.sample_rows

        col_names = {c.name for c in sample_response.cols}
        expected_cols = {'LOAN_ID', 'ACCOUNT_ID', 'AMOUNT', 'STATUS'}
        assert expected_cols.issubset(col_names), \
            f"Missing columns: {expected_cols - col_names}"
