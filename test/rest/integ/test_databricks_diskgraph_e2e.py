# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live Databricks training and output test.

A physical table is staged as Kumo parquet during ingest; a Custom SQL table
is read straight from Databricks by DiskGraph. The same run trains over both
and writes predictions and embeddings back to Databricks.
"""
import asyncio
import dataclasses
import os
import time
import uuid
from pathlib import Path
from test.rest.integ.seed_helpers import seed_and_train

import kumoapi.data_source as data_source_api
import kumoapi.jobs as jobs_api
import kumoapi.table as table_api
import pytest
from fastapi.testclient import TestClient
from kumoapi.typing import Dtype, Stype

from io_utils import fsutil
from io_utils.databricks import DatabricksConnectionInfo
from kumo.data import MaterializationStage
from kumo.data.utils import get_materialize_path
from kumo.testing.databricks import withDatabricksOauthCredentials
from kumo.testing.rest import (
    create_identity_backed_connector,
    databricks_test_identity,
    get_and_expect_success,
    hm_databricks_config,
    post_and_expect_success,
    randomize_fixture_ids_v2,
    remap_fixture_connector_ids_v2,
)
from rest.config.api_command import FixtureDefinition, FixturePQueryRecord
from rest.config.batch_prediction_job import BatchPredictionJobResponse
from rest.config.batch_prediction_job_v2 import BatchPredictionInternalRequest
from rest.config.connector_db import _api_external_config
from rest.config.graph_v2 import (
    GraphDefinition,
    GraphSnapshotRequest,
    GraphSnapshotResponse,
)
from rest.config.jobs import BatchPredictionOptions
from rest.config.status import IDResponse
from rest.config.table_v2 import (
    CustomSQLSourceTable,
    DatabricksSourceTable,
    TableDefinition,
)
from rest.config.training_job import TrainingJobResponse
from rest.config.training_job_v2 import TrainingWorkflowResponse
from rest.testing.integration import wait_for_training_workflow_to_finish
from statedb.config.common import (
    BatchPredictionJobID,
    GraphIDv2,
    JobStatus,
    ProjectID,
    TrainingJobID,
)

if 'DEPLOY_ENV_CONFIG_FILE' not in os.environ:
    os.environ['DEPLOY_ENV_CONFIG_FILE'] = str(
        Path(__file__).resolve().parents[3] / 'rest' / 'devserver' / 'configs' /
        'diskgraph-test.json')

pytestmark = pytest.mark.usefixtures("_pool_subprocesses")

_SOURCE_CONNECTOR = 'databricks_source'
_CUSTOMERS = 'hm_customers_small'
_ARTICLES = 'hm_articles_small'
_TRANSACTIONS = 'hm_transactions_small'
_PQUERY = 'databricks_ingestion'
_POLL_SECONDS = 5
_TIMEOUT_SECONDS = 1800


@pytest.fixture(scope='session')
def storage_backend():
    """This live connector test does not depend on Kumo's S3 test backend."""
    return 'local'


def _fixture(connector_id: str) -> FixtureDefinition:
    customers = TableDefinition(
        cols=[
            table_api.Column(name='customer_id', stype=Stype.ID,
                             dtype=Dtype.string),
            table_api.Column(name='age', stype=Stype.numerical,
                             dtype=Dtype.float),
        ],
        source_table=CustomSQLSourceTable(
            connector_id=connector_id,
            connector_type=data_source_api.ConnectorType.DATABRICKS,
            table=f'{_CUSTOMERS}_query',
            sql=('SELECT customer_id, age FROM '
                 f'hive_mapped_uc.test_schema.{_CUSTOMERS}'),
            referenced_tables=[
                f'hive_mapped_uc.test_schema.{_CUSTOMERS}',
            ],
        ),
        pkey='customer_id',
    )
    articles = TableDefinition(
        cols=[
            table_api.Column(name='article_id', stype=Stype.ID,
                             dtype=Dtype.int),
        ],
        source_table=DatabricksSourceTable(
            databricks_connector_id=connector_id,
            table=_ARTICLES,
        ),
        pkey='article_id',
    )
    transactions = TableDefinition(
        cols=[
            table_api.Column(name='t_dat', stype=Stype.timestamp,
                             dtype=Dtype.time),
            table_api.Column(name='customer_id', stype=Stype.ID,
                             dtype=Dtype.string),
            table_api.Column(name='article_id', stype=Stype.ID,
                             dtype=Dtype.int),
            table_api.Column(name='price', stype=Stype.numerical,
                             dtype=Dtype.float),
            table_api.Column(name='sales_channel_id', stype=Stype.numerical,
                             dtype=Dtype.int),
        ],
        source_table=DatabricksSourceTable(
            databricks_connector_id=connector_id,
            table=_TRANSACTIONS,
        ),
        time_col='t_dat',
    )

    from kumoapi.graph import ColumnKey, ColumnKeyGroup
    graph = GraphDefinition(
        tables={
            'customers': customers,
            'articles': articles,
            'transactions': transactions,
        },
        col_groups=[
            ColumnKeyGroup(columns=(
                ColumnKey(table_name='customers', col_name='customer_id'),
                ColumnKey(table_name='transactions', col_name='customer_id'),
            )),
            ColumnKeyGroup(columns=(
                ColumnKey(table_name='articles', col_name='article_id'),
                ColumnKey(table_name='transactions', col_name='article_id'),
            )),
        ],
    )
    return FixtureDefinition(
        tables=[customers, articles, transactions],
        graphs=[graph],
        queries=[
            FixturePQueryRecord(
                query_str=('PREDICT customers.age '
                           'FOR EACH customers.customer_id'),
                graph_fpid=graph.fpid,
                pquery_name=_PQUERY,
            )
        ],
    )


def _wait_for_training(
    client: TestClient,
    training_job_id: TrainingJobID,
    project_id: str,
) -> TrainingJobResponse:
    deadline = time.time() + _TIMEOUT_SECONDS
    while time.time() < deadline:
        response = get_and_expect_success(
            client,
            f'/v0/training_jobs/{training_job_id}',
            {'project_id': project_id},
            TrainingJobResponse,
        )
        if response.execution.status.is_terminal:
            return response
        time.sleep(_POLL_SECONDS)
    raise TimeoutError(f'training job {training_job_id} did not finish')


def _wait_for_prediction(
    client: TestClient,
    prediction_job_id: BatchPredictionJobID,
    project_id: str,
) -> BatchPredictionJobResponse:
    deadline = time.time() + _TIMEOUT_SECONDS
    while time.time() < deadline:
        response = get_and_expect_success(
            client,
            f'/v1/batch_prediction_jobs/{prediction_job_id}',
            {'project_id': project_id},
            BatchPredictionJobResponse,
        )
        if response.execution.status.is_terminal:
            return response
        time.sleep(_POLL_SECONDS)
    raise TimeoutError(f'prediction job {prediction_job_id} did not finish')


def _assert_every_source_is_staged_as_parquet(
    client: TestClient,
    workflow: TrainingWorkflowResponse,
    project_id: str,
) -> None:
    """Every Databricks source, query or table, is staged as Kumo Parquet:
    nothing is left for the engine to read out of the warehouse itself."""
    assert workflow.graph_snapshot_id is not None
    snapshot = post_and_expect_success(
        client,
        '/v0/snapshots/graph',
        GraphSnapshotRequest(graph_snapshot_id=workflow.graph_snapshot_id),
        GraphSnapshotResponse,
        project_id=project_id,
    )
    staged = []
    for table_name, api_definition in snapshot.definition.tables.items():
        definition = TableDefinition(**dataclasses.asdict(api_definition))
        snapshot_id = str(snapshot.snapshot_resource.table_ids[table_name])
        data_version = snapshot_id.rsplit('@', 1)[1]
        ingest_path = get_materialize_path(
            definition.fingerprint(),
            data_version,
            MaterializationStage.INGEST,
        )
        parquet_files = ([
            path for path in fsutil.listdir(ingest_path)
            if path.endswith('.parquet')
        ] if fsutil.exists(ingest_path) else [])
        staged.append(table_name)
        assert parquet_files, (
            f'Databricks source {table_name!r} left no Parquet under '
            f'{ingest_path}')
        for path in parquet_files:
            assert (fsutil.filesize(path) or 0) > 0, f'empty shard {path}'
    assert sorted(staged) == sorted(snapshot.definition.tables)


def _read_nonempty_table(connector, table_name: str):
    deadline = time.time() + 120
    last_error = None
    while time.time() < deadline:
        try:
            frame = connector.read_table(table_name)
            if len(frame) > 0:
                return frame
        except Exception as error:  # The export may still be committing.
            last_error = error
        time.sleep(_POLL_SECONDS)
    if last_error is not None:
        raise last_error
    raise AssertionError(f'Databricks table {table_name!r} is empty')


@pytest.mark.asyncio
@pytest.mark.timeout(2400)
@withDatabricksOauthCredentials
async def test_databricks_table_and_query_train_and_export(
    client: TestClient,
    temporal_client,
):
    fixture = randomize_fixture_ids_v2(_fixture(_SOURCE_CONNECTOR))
    randomized_source_id = str(
        next(iter(fixture.graphs[0].tables.values())).get_connector_id())
    source_identity, credentials = databricks_test_identity()
    source_setup = create_identity_backed_connector(
        client,
        identity_config=source_identity,
        credentials=credentials,
        connector_config=dataclasses.replace(
            hm_databricks_config(_SOURCE_CONNECTOR),
            schema_name='test_schema',
        ),
    )
    fixture = remap_fixture_connector_ids_v2(
        fixture, {randomized_source_id: source_setup.connector_id})

    suffix = uuid.uuid4().hex[:12]
    output_config = dataclasses.replace(
        hm_databricks_config(f'databricks_output_{suffix}'),
        schema_name='predictions',
    )
    output_resource = post_and_expect_success(
        client,
        f'/v0/projects/{source_setup.project_id}/connectors/',
        data_source_api.CreateConnectorArgs(
            identity_id=source_setup.identity_id,
            name=output_config.name,
            config=_api_external_config(output_config),
        ),
        data_source_api.ConnectorResponse,
    )

    query_name = fixture.queries[0].pquery_name
    _, training_job_id = await seed_and_train(
        client,
        fixture,
        query_name,
        provision_connectors=False,
        project_id=source_setup.project_id,
    )
    training = _wait_for_training(client, training_job_id,
                                  source_setup.project_id)
    assert training.execution.status == JobStatus.DONE, training.execution
    await wait_for_training_workflow_to_finish(training_job_id, temporal_client)

    workflow = get_and_expect_success(
        client,
        f'/v0/training_jobs/{training_job_id}/workflow',
        {'project_id': source_setup.project_id},
        TrainingWorkflowResponse,
    )
    _assert_every_source_is_staged_as_parquet(client, workflow,
                                              source_setup.project_id)
    assert workflow.legacy_graph_id is not None
    graph_fpid = GraphIDv2.from_public(workflow.legacy_graph_id).fpid
    assert graph_fpid is not None

    prediction_table = f'predictions_{suffix}'
    embedding_table = f'embeddings_{suffix}'
    output_id = str(output_resource.id)
    prediction_request = BatchPredictionInternalRequest(
        model_training_job_id=str(training_job_id),
        predict_options=BatchPredictionOptions(),
        project_id=ProjectID(source_setup.project_id),
        outputs=[
            jobs_api.DatabricksPredictionOutput(
                artifact_type=jobs_api.PredictionArtifactType.PREDICTIONS,
                connector_id=output_id,
                table_name=f'predictions__kumo__{prediction_table}',
            ),
            jobs_api.DatabricksPredictionOutput(
                artifact_type=jobs_api.PredictionArtifactType.EMBEDDINGS,
                connector_id=output_id,
                table_name=f'predictions__kumo__{embedding_table}',
            ),
        ],
        graph_id=graph_fpid,
        explanations=False,
    )
    response = post_and_expect_success(
        client,
        '/v1/batch_prediction_jobs/',
        prediction_request,
        IDResponse,
    )
    prediction_job_id = BatchPredictionJobID(response.id)
    prediction = _wait_for_prediction(client, prediction_job_id,
                                      source_setup.project_id)
    assert prediction.execution.status == JobStatus.DONE, prediction.execution

    info = DatabricksConnectionInfo.get()
    verifier = output_config.as_connector(info.host,
                                          require_explicit_credentials=True)
    verifier.set_credentials(info.credentials)
    await asyncio.to_thread(verifier.open)
    try:
        predictions = await asyncio.to_thread(_read_nonempty_table, verifier,
                                              prediction_table)
        embeddings = await asyncio.to_thread(_read_nonempty_table, verifier,
                                             embedding_table)
        assert len(predictions) > 0
        assert len(embeddings) > 0
        assert any('EMBED' in column.upper() for column in embeddings.columns)
    finally:
        try:
            for table_name in (prediction_table, embedding_table):
                await asyncio.to_thread(
                    verifier._get_df_result,
                    verifier.context,
                    f'DROP TABLE IF EXISTS '
                    f'{verifier._fully_qualify_table_name(table_name)}',
                )
        finally:
            await asyncio.to_thread(verifier.close)
