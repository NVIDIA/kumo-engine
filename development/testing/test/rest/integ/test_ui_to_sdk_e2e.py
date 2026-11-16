# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import time
from test.rest.integ.training_budget import clamp_e2e_training_budget

import kumoapi.model_plan as plan_api
import pytest
from fastapi.testclient import TestClient
from kumoapi.graph import GraphDefinition as apiGraphDefinition

from kumo.testing.rest import get_and_expect_success, post_and_expect_success
from rest.config.batch_prediction_job_v2 import BatchPredictionInternalRequest
from rest.config.graph_v2 import GraphDefinition
from rest.config.jobs import BatchPredictionOptions
from rest.config.predictive_query_v2 import ModelPlanResponse
from rest.config.status import IDResponse
from rest.config.training_job import TrainingJobResponse
from rest.config.training_job_v2 import TrainingJobRequest
from rest.testing.integration import wait_for_training_job_to_finish
from statedb.config.common import (
    ConnectorID,
    GraphIDv2,
    JobStatus,
    ProjectID,
    TrainingJobID,
)
from statedb.db.services import StateServices
from temporalio_commons import TemporalClient

# Poll cadence for confirming the persisted training-job status reached a
# terminal state before kicking off batch prediction. Mirrors the proven-green
# s3 sibling test_diskgraph_e2e.py (_wait_for_training).
POLL_INTERVAL_SECONDS = 5
MAX_POLL_SECONDS = 1200


def _wait_for_training_terminal(
    client: TestClient,
    training_job_id: TrainingJobID,
    project_id: str,
) -> TrainingJobResponse:
    """Poll the persisted training job until it reaches a terminal status.

    The Temporal workflow result returning does not guarantee the StateDB
    training-job status has been committed as terminal, so poll the status
    endpoint until terminal before kicking off batch prediction (otherwise the
    BP request can fire while the status is still RUNNING).
    """
    start = time.time()
    while time.time() - start < MAX_POLL_SECONDS:
        job = get_and_expect_success(
            client,
            f'/v0/training_jobs/{training_job_id}',
            {'project_id': project_id},
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
        f"Training job {training_job_id} did not reach a terminal status "
        f"within {MAX_POLL_SECONDS}s")


# This e2e drives the real DG-native training pipeline via /v0/training_jobs/
# train, which self-bootstraps the graph snapshot + DiskGraph ingest +
# DG-native train-table generation (no legacy generic materialization). That
# work runs on the diskgraph pool task queue, so bring up the pool worker for
# this module (same as test_diskgraph_e2e). The earlier skip reason claimed a
# fall-through to deleted legacy materialization, which is inaccurate.
#
# Pin ``storage_backend`` to ``local``: the session-scoped ``storage_backend``
# fixture is parametrized [local, s3], but pytest does not run session-scoped
# fixture finalizers between parametrize values, so the diskgraph pool worker
# spawned during [local] survives into [s3] with stale env -- it writes the
# DiskGraph ingest artifacts to the local tmp persist-dir while ensure_snapshot
# reads them back from ``s3://kumo-dev-dataplane-bucket/test/`` and 404s on
# HeadObject (the cross-parametrization stale-worker mismatch documented on the
# conftest ``_pool_subprocesses`` fixture). The local backend exercises the
# identical DG train -> batch-prediction pipeline end to end, and the s3 storage
# path is covered by ``test_diskgraph_e2e[s3]``, so pinning to local keeps the
# DG coverage without the stale-worker dependency -- same scoping as
# ``test_warm_start_training_e2e``, ``test_e2e`` and ``test_e2e_public_api``.
pytestmark = [
    pytest.mark.usefixtures("_pool_subprocesses"),
    pytest.mark.parametrize("storage_backend", ["local"], indirect=True),
]


@pytest.mark.asyncio
async def test_prediction_ui_to_sdk_e2e(
    client: TestClient,
    temporal_client: TemporalClient,
    graph_definition_v2: apiGraphDefinition,
    default_project_id: str,
    state_services: StateServices,
):
    # The shared graph fixture uses an arbitrary test-only internal connector.
    # Generated SDK replay supports only public external connectors and the one
    # canonical project upload connector, so remap the same local fixture root
    # to the real project upload resource before creating the graph.
    source_connector_id = ConnectorID(
        next(iter(
            graph_definition_v2.tables.values())).source_table.connector_id)
    source_connector = await state_services.connectors.get_internal(
        ProjectID(default_project_id), source_connector_id)
    upload_connector_id = (
        await state_services.connectors.register_project_upload_connector(
            ProjectID(default_project_id), source_connector.config.root_dir))
    graph_definition_v2 = dataclasses.replace(
        graph_definition_v2,
        tables={
            table_name:
            dataclasses.replace(
                table,
                source_table=dataclasses.replace(
                    table.source_table,
                    connector_id=str(upload_connector_id),
                ),
            )
            for table_name, table in graph_definition_v2.tables.items()
        },
    )
    graph_definition_internal = GraphDefinition(
        **dataclasses.asdict(graph_definition_v2))
    graph_id = post_and_expect_success(
        client,
        '/v1/graphs/',
        graph_definition_v2,
        IDResponse,
        project_id=default_project_id,
    )

    graph_fpid = GraphIDv2(graph_id.id).fpid

    assert graph_fpid is not None

    pquery_str = (
        'PREDICT COUNT(transactions.*, 0, 30) FOR EACH customers.customer_id')
    payload = plan_api.SuggestModelPlanRequest(
        query_string=pquery_str,
        graph_id=graph_id.id,
        run_mode=plan_api.RunMode.FAST,
        graph_definition=graph_definition_internal,
    )

    model_plan = post_and_expect_success(
        client=client,
        path='/v1/predictive_queries/model_plan',
        payload=payload,
        response_type=ModelPlanResponse,
        project_id=default_project_id,
    )
    clamp_e2e_training_budget(model_plan.suggested_model_plan.model_plan)

    training_request = TrainingJobRequest(
        predictive_query=pquery_str,
        graph_definition=graph_definition_internal,
        training_table_gen_plan=model_plan.suggested_model_plan.model_plan.
        training_table_generation,
        model_plan=model_plan.suggested_model_plan.model_plan,
        project_id=ProjectID(default_project_id),
    )

    training_job_id = post_and_expect_success(
        client,
        "/v0/training_jobs/train",
        training_request,
        TrainingJobID,
    )
    assert training_job_id is not None

    await wait_for_training_job_to_finish(training_job_id, temporal_client)

    # The Temporal workflow result returning does not guarantee the persisted
    # StateDB training-job status has been committed as terminal-DONE. On the
    # slower [s3] backend there is a real gap, and firing batch prediction
    # while status is still RUNNING yields HTTP 400. Confirm the persisted
    # status reached terminal-DONE before kicking off batch prediction
    # (mirrors test_diskgraph_e2e._wait_for_training).
    training_job = _wait_for_training_terminal(
        client,
        training_job_id,
        default_project_id,
    )
    assert training_job.execution.status == JobStatus.DONE, (
        f"Training failed with status={training_job.execution.status}, "
        f"results={training_job.execution.results}")

    prediction_request = BatchPredictionInternalRequest(
        training_job_id,
        BatchPredictionOptions(num_parallel_workers=1),
        ProjectID(default_project_id),
        graph_id=graph_fpid,
    )

    bp_job_id = post_and_expect_success(
        client,
        "/v1/batch_prediction_jobs/",
        prediction_request,
        IDResponse,
    )

    predict_sdk_script = get_and_expect_success(
        client,
        f"/v1/batch_prediction_jobs/{bp_job_id.id}/sdk",
        {'project_id': default_project_id},
        str,
    )
    predict_sdk_script = json.loads(predict_sdk_script)

    assert predict_sdk_script != ""
    compile(predict_sdk_script, "<generated-prediction-sdk>", "exec")
    assert f'project="{default_project_id}"' in predict_sdk_script
    assert "root_dir=" not in predict_sdk_script
    components = [
        # Load existing model (not retrain)
        "kumo.PredictiveQuery.load_from_training_job(",
        "kumo.Trainer.load(",
        "kumo.TrainingJob(",
        # Prediction components
        "prediction_table_config",
        "pred_table = pquery.generate_prediction_table(",
        "batch_prediction_options",
        "output_configurations",
        "prediction_job = trainer.predict(",
    ]

    for component in components:
        assert component in predict_sdk_script

    # Should NOT contain full training pipeline
    assert "trainer.fit(" not in predict_sdk_script
