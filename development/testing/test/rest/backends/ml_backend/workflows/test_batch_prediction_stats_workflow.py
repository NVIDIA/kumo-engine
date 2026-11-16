# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kumo_core.types import PredictionArtifactType, TaskType

from rest.backends.ml_backend.validation import ValidationResponse
from rest.backends.ml_backend.workflows import \
    batch_prediction_stats_workflow as module
from rest.backends.ml_backend.workflows import \
    batch_prediction_workflow as parent_module
from rest.config.prediction_output import LocalPredictionOutput
from statedb.config.common import JobStatus


def _job_input(*artifact_types: PredictionArtifactType) -> SimpleNamespace:
    return SimpleNamespace(
        id='bp-job-preserved-02',
        config=SimpleNamespace(outputs=[
            SimpleNamespace(artifact_type=artifact_type)
            for artifact_type in artifact_types
        ]),
        transformation=None,
    )


@pytest.mark.asyncio
async def test_embeddings_only_never_opens_prediction_stats(monkeypatch):
    monkeypatch.setattr(
        module,
        'DataBackendContext',
        lambda *_args: pytest.fail('must not construct prediction backend'),
    )
    execute_activity = AsyncMock(
        side_effect=AssertionError('must not request prediction stats'))
    monkeypatch.setattr(module.workflow, 'execute_activity', execute_activity)
    job_input = SimpleNamespace(
        id='bp-job-87205d3553204ac7ab1c2ba10bbfb48c',
        config=SimpleNamespace(outputs=[
            LocalPredictionOutput(
                artifact_type=PredictionArtifactType.EMBEDDINGS,
                extra_fields=[],
            ),
        ]),
        transformation=None,
    )

    await module.BatchPredictionStatsWorkflow().run(
        job_input,
        SimpleNamespace(),
    )

    execute_activity.assert_not_awaited()


@pytest.mark.asyncio
async def test_predictions_still_request_raw_stats(monkeypatch):
    context = SimpleNamespace()
    execute_activity = AsyncMock()
    monkeypatch.setattr(module, 'deploy_env', SimpleNamespace)
    monkeypatch.setattr(module, 'DataBackendContext', lambda *_args: context)
    monkeypatch.setattr(module.workflow, 'execute_activity', execute_activity)

    job_input = _job_input(PredictionArtifactType.PREDICTIONS)
    await module.BatchPredictionStatsWorkflow().run(
        job_input,
        SimpleNamespace(),
    )

    execute_activity.assert_awaited_once()
    assert execute_activity.await_args.kwargs['args'] == [
        context,
        job_input,
        False,
    ]


@pytest.mark.asyncio
async def test_no_output_fails_before_backend_construction(monkeypatch):
    monkeypatch.setattr(
        module,
        'DataBackendContext',
        lambda *_args: pytest.fail('must fail before backend construction'),
    )

    with pytest.raises(ValueError, match='at least one requested output'):
        await module.BatchPredictionStatsWorkflow().run(
            _job_input(),
            SimpleNamespace(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('artifact_types', 'expect_stats_child'),
    [
        ((PredictionArtifactType.EMBEDDINGS, ), False),
        ((PredictionArtifactType.PREDICTIONS, ), True),
        ((PredictionArtifactType.PREDICTIONS,
          PredictionArtifactType.EMBEDDINGS), True),
    ],
)
async def test_parent_dispatches_stats_only_for_requested_predictions(
        monkeypatch, artifact_types, expect_stats_child):
    outputs = [
        LocalPredictionOutput(artifact_type=artifact_type, extra_fields=[])
        for artifact_type in artifact_types
    ]
    job_input = SimpleNamespace(
        id=('bp-job-87205d3553204ac7ab1c2ba10bbfb48c'
            if artifact_types == (PredictionArtifactType.EMBEDDINGS, ) else
            'bp-job-parent-output-contract'),
        training_job_id='trainingjob-parent-output-contract',
        generate_pred_table_job_id=None,
        graph_version_info=SimpleNamespace(graph_data_version='version'),
        graph_data_compute_input=SimpleNamespace(graph_key='graph'),
        predictive_query=SimpleNamespace(graph=SimpleNamespace()),
        stop_after_table_gen=False,
        config=SimpleNamespace(
            outputs=outputs,
            task_type=TaskType.REGRESSION,
            explanations=False,
            num_parallel_workers_limit=1,
        ),
    )
    ok = ValidationResponse()
    child_workflows = []
    detached_children = []

    async def execute_child_workflow(child, **_kwargs):
        child_workflows.append(child)
        if child is parent_module.GraphRefreshWorkflow.run:
            return SimpleNamespace(validation=ok), SimpleNamespace()
        if child is parent_module.MaterializePredictionTaskWorkflow.run:
            return '/tmp/prediction-config.json', ok
        if child is parent_module.BatchPredictionJobWorkflow.run:
            return SimpleNamespace(status=JobStatus.DONE)
        raise AssertionError(f'unexpected child workflow: {child}')

    async def start_child_workflow(child, **kwargs):
        detached_children.append((child, kwargs))
        return None

    async def no_op(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        parent_module,
        'deploy_env',
        lambda: SimpleNamespace(is_diskgraph=False),
    )
    monkeypatch.setattr(parent_module, 'DataBackendContext',
                        lambda *_args: SimpleNamespace())
    monkeypatch.setattr(parent_module, 'init_job_generic', no_op)
    monkeypatch.setattr(parent_module, 'enqueue_job', no_op)
    monkeypatch.setattr(parent_module, 'save_validation_response', no_op)
    monkeypatch.setattr(parent_module, 'get_job_base_dir', lambda: '/tmp')
    monkeypatch.setattr(parent_module, 'get_job_dir', lambda *_args: '/tmp/job')
    monkeypatch.setattr(parent_module, 'update_graph_config_from_graph_input',
                        lambda graph, _graph_input: graph)
    monkeypatch.setattr(parent_module.workflow, 'execute_activity', no_op)
    monkeypatch.setattr(parent_module.workflow, 'execute_child_workflow',
                        execute_child_workflow)
    monkeypatch.setattr(parent_module.workflow, 'start_child_workflow',
                        start_child_workflow)

    parent = parent_module.BatchPredictionParentWorkflow()
    parent._save_output = AsyncMock()  # type: ignore[method-assign]
    parent._generate_pred_table = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            pred_table_path='/tmp/pred-table',
            validation=ok,
        ))

    result = await parent._run(job_input, SimpleNamespace())

    assert result.ok
    stats_children = [
        (child, kwargs) for child, kwargs in detached_children
        if child is parent_module.BatchPredictionStatsWorkflow.run
    ]
    assert bool(stats_children) is expect_stats_child
    if expect_stats_child:
        assert len(stats_children) == 1
        _, kwargs = stats_children[0]
        assert kwargs['cancellation_type'] == (
            parent_module.ChildWorkflowCancellationType.ABANDON)
        assert kwargs['parent_close_policy'] == (
            parent_module.ParentClosePolicy.ABANDON)
