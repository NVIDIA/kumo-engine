# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from pool.failure_output import POOL_FAILURE_OUTPUT_PATCH
from rest.backends.ml_backend.activities import TrainingJobActivities
from rest.backends.ml_backend.settings import MLJobSettings
from rest.backends.ml_backend.workflows import training_job_workflow as module
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.ml_backend import JobState, TrainingJobInput
from statedb.config.common import ConnectorID, ProjectID, TrainingJobID


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


@pytest.mark.asyncio
async def test_pool_dispatch_timeout_and_postprocess_connector_refs(
    monkeypatch, ) -> None:
    connector_id = ConnectorID('connector_' + 'a' * 32)
    connectors = {
        connector_id:
        ConnectorExecutionRef(
            connector_id=connector_id,
            project_id=ProjectID('project_' + 'b' * 32),
            authorization_token='child-workflow-token',
        ),
    }
    job_input = SimpleNamespace(
        id=TrainingJobID('trainingjob-test'),
        spec=SimpleNamespace(predictive_query=SimpleNamespace(
            name='query', graph=SimpleNamespace(name='training-graph'))),
        graph_data_compute_input=SimpleNamespace(
            graph_key='training-graph',
            connectors=connectors,
        ),
        graph_version_info=SimpleNamespace(graph_data_version='version-1'),
    )
    expected_output = object()

    async def execute_activity(activity_fn, *, args=None, **_kwargs):
        if activity_fn is TrainingJobActivities.postprocess_job_output:
            return expected_output
        return None

    execute = AsyncMock(side_effect=execute_activity)
    monkeypatch.setattr(module.workflow, 'execute_activity', execute)
    monkeypatch.setattr(module.workflow, 'now', lambda: datetime.now())
    run_pooled = AsyncMock(return_value=JobState.SUCCEEDED)
    monkeypatch.setattr(module, 'run_pooled_activity', run_pooled)

    workflow = module.TrainingJobWorkflow()
    workflow._save_output = AsyncMock()  # type: ignore[method-assign]

    typed_job_input = cast(TrainingJobInput, job_input)
    await workflow.run(
        typed_job_input,
        cast(MLJobSettings, SimpleNamespace(raise_fake_error=False)),
        '/tmp/job-config.json',
    )

    dispatch = run_pooled.await_args_list[0].kwargs
    assert dispatch['schedule_to_close_timeout'] == timedelta(days=21)
    assert dispatch['heartbeat_timeout'] == timedelta(minutes=5)
    postprocess_call = next(
        call for call in execute.await_args_list
        if call.args[0] is TrainingJobActivities.postprocess_job_output)
    assert postprocess_call.kwargs['args'][2] is connectors
    assert postprocess_call.kwargs['args'][2][
        connector_id].authorization_token == 'child-workflow-token'
    workflow._save_output.assert_awaited_with(expected_output)

    # Histories recorded before the failure-output patch keep the old text.
    monkeypatch.setattr(module.workflow, 'patched', lambda _patch_id: False)
    run_pooled.return_value = JobState.FAILED
    failed = await workflow._run_via_pool(typed_job_input,
                                          '/tmp/job-config.json')
    assert failed.err_msg == (
        'kumo-auto subprocess exited non-zero for job trainingjob-test; '
        'see pool worker logs for details')
    assert failed.stacktrace is None


def _pool_training_workflow(monkeypatch, read_output: AsyncMock):
    job_input = SimpleNamespace(
        id=TrainingJobID('trainingjob-test'),
        spec=SimpleNamespace(predictive_query=SimpleNamespace(
            name='query', graph=SimpleNamespace(name='training-graph'))),
        graph_data_compute_input=SimpleNamespace(graph_key='training-graph'),
        graph_version_info=SimpleNamespace(graph_data_version='version-1'),
    )
    monkeypatch.setattr(module.workflow, 'patched',
                        lambda patch_id: patch_id == POOL_FAILURE_OUTPUT_PATCH)
    monkeypatch.setattr(module, 'run_pooled_activity',
                        AsyncMock(return_value=JobState.FAILED))
    monkeypatch.setattr(module, 'read_saved_pool_failure_output', read_output)
    workflow = module.TrainingJobWorkflow()
    workflow._input = job_input  # type: ignore[assignment]
    workflow._settings = SimpleNamespace(  # type: ignore[assignment]
        raise_fake_error=False)
    workflow._save_output = AsyncMock()  # type: ignore[method-assign]
    return workflow, cast(TrainingJobInput, job_input)


@pytest.mark.asyncio
async def test_pool_failure_reports_saved_child_output(monkeypatch) -> None:
    output = ('Traceback (most recent call last):\n'
              "ValueError: column 'amount' is missing")
    read_output = AsyncMock(return_value=output)
    workflow, job_input = _pool_training_workflow(monkeypatch, read_output)

    failed = await workflow._run_via_pool(job_input, '/tmp/job-config.json')

    assert failed.state == JobState.FAILED
    assert failed.err_msg == (
        'kumo-auto subprocess exited non-zero for job trainingjob-test: '
        "ValueError: column 'amount' is missing")
    assert failed.stacktrace == output
    read_output.assert_awaited_once_with(
        'trainingjob-test', ['/tmp/job-config.json.kumo-auto.failure.txt'])


@pytest.mark.asyncio
async def test_pool_failure_without_saved_output_keeps_pointer(
        monkeypatch) -> None:
    workflow, job_input = _pool_training_workflow(monkeypatch,
                                                  AsyncMock(return_value=None))

    failed = await workflow._run_via_pool(job_input, '/tmp/job-config.json')

    assert failed.err_msg == (
        'kumo-auto subprocess exited non-zero for job trainingjob-test; '
        'see pool worker logs for details')
    assert failed.stacktrace is None
