# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import pathlib
from http import HTTPStatus
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest
from fastapi.testclient import TestClient
from kumo_core.types.schema import FPID, ColumnArgs
from kumoapi.graph import GraphDefinition as apiGraphDefinition

from kumo.testing.rest import (
    get_and_expect_error,
    get_and_expect_success,
    post_and_expect_success,
)
from kumo.validation.validation import (
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)
from rest.backends.ml_backend.data_util import get_job_dir
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.batch_prediction_job import BatchPredictionJobResponse
from rest.config.batch_prediction_job_v2 import BatchPredictionInternalRequest
from rest.config.graph_v2 import GraphDefinition
from rest.config.jobs import BatchPredictionOptions
from rest.config.status import IDResponse, RestStatusCode
from rest.driver.pquery_api_driver import ForecastingParams, PQueryApiDriver
from statedb.config.common import (
    BatchPredictionJobID,
    ErrWarnArgs,
    ErrWarnResponse,
    JobStatus,
    ProjectID,
    TrainingJobID,
)
from statedb.db.identities import ConnectorService


def _make_bp_internal_request(
    training_job_id: TrainingJobID,
    graph_fpid: FPID,
    tmp_path: pathlib.Path,
    project_id: Optional[ProjectID] = None,
) -> BatchPredictionInternalRequest:
    root_dir = tmp_path / "bp_output"
    root_dir.mkdir(exist_ok=True, parents=True)
    return BatchPredictionInternalRequest(
        model_training_job_id=str(training_job_id),
        predict_options=BatchPredictionOptions(),
        graph_id=graph_fpid,
        project_id=project_id,
        # An empty output list is the connector-free, Kumo-managed download
        # mode. Object-store outputs must name a project-owned connector.
        outputs=[],
    )


def test_get_batch_prediction_warnings(client: TestClient, job_base_dir: str,
                                       ml_backend: MLJobBackend, event_loop,
                                       fake_training_job_v2: TrainingJobID,
                                       graph_definition_v2: apiGraphDefinition,
                                       tmp_path: pathlib.Path,
                                       default_project_id: str):
    graph_fpid = GraphDefinition(**dataclasses.asdict(graph_definition_v2)).fpid

    # Mark the training job DONE so a prediction job can be created. (Sync test:
    # it calls the blocking validation.save, so drive the async completion via
    # the session event loop rather than making the test async.)
    event_loop.run_until_complete(
        ml_backend.mock_training_job_complete(fake_training_job_v2))
    response = post_and_expect_success(
        client, "/v1/batch_prediction_jobs/",
        _make_bp_internal_request(fake_training_job_v2, graph_fpid, tmp_path,
                                  ProjectID(default_project_id)), IDResponse)
    validation = ValidationResponse(
        errors=[ValidationError("title", "error 1")],
        warnings=[ValidationWarning("title", "warning 1")])
    job_id = TrainingJobID(response.id)
    home_dir = get_job_dir(job_base_dir, job_id)
    validation.save(home_dir, job_id)

    payload = ErrWarnArgs(ids=[job_id])
    # Should have 1 err and 1 warn
    warn_response = post_and_expect_success(
        client,
        f"/v1/batch_prediction_jobs/get_warnings?project_id={default_project_id}",
        payload, Dict[str, ErrWarnResponse])
    errs = warn_response[job_id].errors
    assert len(errs) > 0
    assert 'error 1' == errs[0].description
    warns = warn_response[job_id].warnings
    assert len(warns) > 0
    assert 'warning 1' == warns[0].description


@pytest.mark.asyncio
async def test_get_batch_prediction_job(client: TestClient,
                                        ml_backend: MLJobBackend,
                                        fake_training_job_v2: TrainingJobID,
                                        graph_definition_v2: apiGraphDefinition,
                                        tmp_path: pathlib.Path,
                                        default_project_id: str):
    graph_fpid = GraphDefinition(**dataclasses.asdict(graph_definition_v2)).fpid
    # Should validate job id and return 422 if invalid
    get_and_expect_error(
        client,
        f"/v1/batch_prediction_jobs/invalid_id/?project_id={default_project_id}",
        {}, HTTPStatus.UNPROCESSABLE_ENTITY,
        'identifier may only contain alphanumeric characters and dashes')
    # Should return error if batch prediction job does not exist
    get_and_expect_error(
        client,
        f"/v1/batch_prediction_jobs/missing-id/?project_id={default_project_id}",
        {}, HTTPStatus.NOT_FOUND,
        RestStatusCode.PREDICTION_JOB_DOES_NOT_EXIST.value)

    # Mark the training job DONE so a prediction job can be created.
    await ml_backend.mock_training_job_complete(fake_training_job_v2)
    response1 = post_and_expect_success(
        client, "/v1/batch_prediction_jobs/",
        _make_bp_internal_request(fake_training_job_v2, graph_fpid, tmp_path,
                                  ProjectID(default_project_id)), IDResponse)
    response2 = get_and_expect_success(
        client,
        f"/v1/batch_prediction_jobs/{response1.id}/?project_id={default_project_id}",
        {}, BatchPredictionJobResponse)
    assert response2.config.training_job_id == fake_training_job_v2
    assert response2.id == response1.id
    # A freshly-created prediction job presents as RUNNING / "Not Started".
    assert response2.execution.status == JobStatus.RUNNING
    assert response2.execution.log.stage_name == 'Not Started'
    assert response2.download_supported is False

    # Mark the prediction job DONE.
    await ml_backend.mock_bp_job_complete(BatchPredictionJobID(response1.id))
    response5 = get_and_expect_success(
        client,
        f"/v1/batch_prediction_jobs/{response1.id}/?project_id={default_project_id}",
        {}, BatchPredictionJobResponse)
    assert response5.execution.status == JobStatus.DONE
    assert response5.execution.results is not None
    assert response5.execution.log.stage_name == 'Done'
    assert response5.download_supported is False


@pytest.mark.asyncio
async def test_get_batch_prediction_sdk_script(
        client: TestClient, ml_backend: MLJobBackend,
        fake_training_job_v2: TrainingJobID,
        graph_definition_v2: apiGraphDefinition, tmp_path: pathlib.Path,
        default_project_id: str, monkeypatch: pytest.MonkeyPatch):
    """Verifies that the BP SDK script loads an existing trained model
    rather than regenerating the full training pipeline.

    The script must reference the real training_job_id (round-tripped) in
    all three load calls and must NOT contain training-from-scratch code
    paths.
    """
    graph_fpid = GraphDefinition(**dataclasses.asdict(graph_definition_v2)).fpid

    async def non_forecasting_params(_self, _training_job_id):
        return ForecastingParams.default()

    # This test owns the executable non-forecasting SDK composition contract.
    # Forecasting's intentional pre-work skip has dedicated coverage.
    monkeypatch.setattr(PQueryApiDriver, '_get_forecasting_params',
                        non_forecasting_params)

    # Mark the fake training job DONE so a prediction job can be created.
    await ml_backend.mock_training_job_complete(fake_training_job_v2)

    bp_response = post_and_expect_success(
        client, "/v1/batch_prediction_jobs/",
        _make_bp_internal_request(fake_training_job_v2, graph_fpid, tmp_path,
                                  ProjectID(default_project_id)), IDResponse)

    # The shared HNM fixture is deliberately an internal local-file connector
    # so tests run without cloud credentials. The production boundary still
    # rejects arbitrary internal connectors; expose only this fixture's safe
    # name so the endpoint test can exercise prediction-script composition.
    async def visible_test_connector(self, project_id, connector_id, principal):
        connector = await self.get_internal(project_id, connector_id)
        return SimpleNamespace(id=connector.id, name=connector.name)

    monkeypatch.setattr(ConnectorService, 'source_table_connector_for_project',
                        visible_test_connector)

    # The endpoint returns a JSON-encoded string body (a quoted Python source
    # with escapes); decode it back into raw Python source for assertions.
    script_encoded = get_and_expect_success(
        client,
        f"/v1/batch_prediction_jobs/{bp_response.id}/sdk?project_id={default_project_id}",
        {}, str)
    script = json.loads(script_encoded)
    assert f'project="{default_project_id}"' in script
    assert 'root_dir=' not in script

    tjid = str(fake_training_job_v2)
    assert f'PredictiveQuery.load_from_training_job("{tjid}")' in script, (
        f'missing load_from_training_job call for {tjid}; script:\n{script}')
    assert f'Trainer.load("{tjid}")' in script, (
        f'missing Trainer.load call for {tjid}; script:\n{script}')
    assert f'TrainingJob("{tjid}")' in script, (
        f'missing TrainingJob(...) construction for {tjid}; script:\n{script}')
    # Must NOT take the retrain path.
    assert "trainer.fit(" not in script, (
        f'unexpected trainer.fit(...) in prediction-only script:\n{script}')
    assert "pquery = kumo.PredictiveQuery(" not in script, (
        f'unexpected full PredictiveQuery(...) constructor in prediction-only '
        f'script:\n{script}')


def test_get_prediction_table_schema(client: TestClient, fake_training_job):
    """The v1 prediction-table schema endpoint resolves a predictive query by
    id and returns its batch-prediction output columns (replaces the deleted
    v0 `batch_prediction_workflows` `prediction_table_schema` endpoint).
    """
    cols = get_and_expect_success(
        client,
        f"/v1/batch_prediction_jobs/{fake_training_job.pquery_id}"
        f"/prediction_table_schema?project_id={fake_training_job.project_id}",
        {},
        List[ColumnArgs],
    )
    assert len(cols) == 4
    assert cols[0].name == 'ENTITY'
    assert cols[1].name == 'TARGET_PRED'
    assert cols[2].name == 'False_PROB'
    assert cols[3].name == 'True_PROB'
