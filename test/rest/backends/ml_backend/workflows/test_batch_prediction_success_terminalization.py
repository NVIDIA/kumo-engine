# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch-prediction success is independent of detailed MLOps statistics."""

from types import SimpleNamespace
from unittest import mock

import pytest
from kumo_core.types import PredictionArtifactType, TaskType
from temporalio.client import WorkflowExecutionStatus
from temporalio.exceptions import CancelledError

from rest.backends.ml_backend.validation import ValidationResponse
from rest.backends.ml_backend.workflows import batch_prediction_workflow as bpw
from rest.config.batch_prediction_job import (
    BatchPredictionJobOutput,
    BatchPredictionJobResults,
)
from statedb.config.common import JobEventLogEntry, JobStatus


class _DirectParentWorkflow(bpw.BatchPredictionParentWorkflow):
    async def run_until_canceled(self, coro):
        return await coro


def _job_input():
    return SimpleNamespace(
        id='batchpredictionjob-terminal-order',
        training_job_id='trainingjob-source',
        config=SimpleNamespace(
            num_parallel_workers_limit=1,
            task_type=TaskType.FORECASTING,
            explanations=False,
            outputs=[SimpleNamespace(
                artifact_type=PredictionArtifactType.PREDICTIONS)],
        ),
        generate_pred_table_job_id=None,
        graph_version_info=SimpleNamespace(graph_data_version='version-1'),
        graph_data_compute_input=SimpleNamespace(graph_key='graph'),
        predictive_query=SimpleNamespace(graph=SimpleNamespace(name='graph')),
        stop_after_table_gen=False,
    )


def _success_output():
    return BatchPredictionJobOutput(
        status=JobStatus.DONE,
        log=JobEventLogEntry.done(),
        results=BatchPredictionJobResults(key_cols=['entity_id']),
    )


def _configure_success_path(monkeypatch, workflow, output, stats_start_error=None):
    events = []
    stats_starts = []

    async def execute_activity(activity, **_kwargs):
        if activity is bpw.init_home_dir_activity:
            return None
        if activity is bpw.cleanup_diskgraph_staging:
            events.append('cleanup')
            return None
        raise AssertionError(f'Unexpected activity: {activity}')

    async def execute_child_workflow(child, **_kwargs):
        if child is bpw.GraphRefreshWorkflow.run:
            return (SimpleNamespace(validation=ValidationResponse()),
                    SimpleNamespace())
        if child is bpw.MaterializePredictionTaskWorkflow.run:
            return '/tmp/prediction-config.json', ValidationResponse()
        if child is bpw.BatchPredictionJobWorkflow.run:
            return output
        raise AssertionError(f'Unexpected child workflow: {child}')

    async def start_child_workflow(child, **kwargs):
        assert child is bpw.BatchPredictionStatsWorkflow.run
        events.append('stats-start')
        stats_starts.append(kwargs)
        if stats_start_error is not None:
            raise stats_start_error
        return SimpleNamespace()

    async def save_output(saved_output):
        events.append(f'save-{saved_output.status.value}')
        saved.append(saved_output)

    async def save_validation(*_args, **_kwargs):
        events.append('validation')

    saved = []
    workflow._save_output = save_output
    workflow._generate_pred_table = mock.AsyncMock(return_value=SimpleNamespace(
        validation=ValidationResponse(),
        pred_table_path='/tmp/pred-table',
    ))
    monkeypatch.setattr(
        bpw,
        'deploy_env',
        lambda: SimpleNamespace(is_diskgraph=False),
    )
    monkeypatch.setattr(bpw, 'DataBackendContext', lambda *_args: object())
    monkeypatch.setattr(bpw, 'init_job_generic', mock.AsyncMock())
    monkeypatch.setattr(bpw, 'enqueue_job', mock.AsyncMock())
    monkeypatch.setattr(bpw, 'dequeue_job', mock.AsyncMock())
    monkeypatch.setattr(bpw, 'get_job_base_dir', lambda: '/tmp/jobs')
    monkeypatch.setattr(bpw, 'get_job_dir', lambda *_args: '/tmp/jobs/bp')
    monkeypatch.setattr(bpw, 'update_graph_config_from_graph_input',
                        lambda graph, _graph_input: graph)
    monkeypatch.setattr(bpw.GraphRefreshWorkflow, 'compose_wf_id',
                        lambda *_args: 'graph-refresh')
    monkeypatch.setattr(bpw.MaterializePredictionTaskWorkflow, 'generate_wf_id',
                        lambda *_args: 'materialize')
    monkeypatch.setattr(bpw.BatchPredictionJobWorkflow, 'generate_wf_id',
                        lambda *_args: 'predict')
    monkeypatch.setattr(bpw.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(bpw.workflow, 'execute_child_workflow',
                        execute_child_workflow)
    monkeypatch.setattr(bpw.workflow, 'start_child_workflow',
                        start_child_workflow)
    monkeypatch.setattr(bpw, 'save_validation_response', save_validation)
    return events, saved, stats_starts


@pytest.mark.asyncio
async def test_done_is_published_after_required_tail_without_waiting_for_stats(
        monkeypatch):
    workflow = _DirectParentWorkflow()
    output = _success_output()
    events, saved, stats_starts = _configure_success_path(
        monkeypatch, workflow, output)

    await workflow.run(_job_input(), SimpleNamespace())

    assert events == [
        'save-RUNNING', 'cleanup', 'validation', 'stats-start', 'save-DONE'
    ]
    assert [saved_output.status
            for saved_output in saved] == [JobStatus.RUNNING, JobStatus.DONE]
    assert saved[-1] is output
    assert output.workflow_end_state == WorkflowExecutionStatus.COMPLETED
    assert len(stats_starts) == 1
    assert stats_starts[0]['cancellation_type'] == (
        bpw.ChildWorkflowCancellationType.ABANDON)
    assert stats_starts[0]['parent_close_policy'] == (
        bpw.ParentClosePolicy.ABANDON)
    bpw.dequeue_job.assert_awaited_once_with(
        'batchpredictionjob-terminal-order')


@pytest.mark.asyncio
async def test_stats_launch_failure_does_not_fail_terminal_bp(monkeypatch):
    workflow = _DirectParentWorkflow()
    output = _success_output()
    events, saved, stats_starts = _configure_success_path(
        monkeypatch,
        workflow,
        output,
        stats_start_error=RuntimeError('stats failed'),
    )

    await workflow.run(_job_input(), SimpleNamespace())

    assert events == [
        'save-RUNNING', 'cleanup', 'validation', 'stats-start', 'save-DONE'
    ]
    assert [saved_output.status
            for saved_output in saved] == [JobStatus.RUNNING, JobStatus.DONE]
    assert output.workflow_end_state == WorkflowExecutionStatus.COMPLETED
    assert len(stats_starts) == 1
    bpw.dequeue_job.assert_awaited_once_with(
        'batchpredictionjob-terminal-order')


@pytest.mark.asyncio
async def test_validation_failure_prevents_stats_launch_and_done(monkeypatch):
    workflow = _DirectParentWorkflow()
    output = _success_output()
    events, saved, stats_starts = _configure_success_path(
        monkeypatch, workflow, output)
    validation_attempts = 0

    async def fail_validation_once(*_args, **_kwargs):
        nonlocal validation_attempts
        validation_attempts += 1
        events.append('validation')
        if validation_attempts == 1:
            raise RuntimeError('validation persistence failed')

    monkeypatch.setattr(bpw, 'save_validation_response', fail_validation_once)

    with pytest.raises(RuntimeError, match='validation persistence failed'):
        await workflow.run(_job_input(), SimpleNamespace())

    assert events == [
        'save-RUNNING', 'cleanup', 'validation', 'validation', 'save-FAILED'
    ]
    assert all(saved_output.status != JobStatus.DONE for saved_output in saved)
    assert saved[-1].workflow_end_state == WorkflowExecutionStatus.COMPLETED
    assert stats_starts == []
    bpw.dequeue_job.assert_awaited_once_with(
        'batchpredictionjob-terminal-order')


@pytest.mark.asyncio
async def test_cancellation_while_starting_stats_does_not_publish_done(
        monkeypatch):
    workflow = _DirectParentWorkflow()
    output = _success_output()
    events, saved, _ = _configure_success_path(
        monkeypatch,
        workflow,
        output,
        stats_start_error=CancelledError('cancelled while starting stats'),
    )

    with pytest.raises(CancelledError):
        await workflow.run(_job_input(), SimpleNamespace())

    assert events == ['save-RUNNING', 'cleanup', 'validation', 'stats-start']
    assert [saved_output.status for saved_output in saved] == [JobStatus.RUNNING]
    assert output.workflow_end_state is None
    bpw.dequeue_job.assert_awaited_once_with(
        'batchpredictionjob-terminal-order')
