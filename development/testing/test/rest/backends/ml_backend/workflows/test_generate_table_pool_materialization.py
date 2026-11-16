# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType

from rest.backends.ml_backend.validation import ValidationResponse
from rest.backends.ml_backend.workflows import generate_table_workflows as mod


@pytest.mark.asyncio
async def test_generated_table_materialization_uses_durable_ttng_bundle(
        monkeypatch):
    validation = ValidationResponse()
    execute_activity = AsyncMock(return_value=('materialized', validation))
    monkeypatch.setattr(mod.workflow, 'execute_activity', execute_activity)

    workflow = mod.GenerateTrainingTableWorkflow()
    workflow._job_input = cast(Any, SimpleNamespace(
        graph_data_compute_input=SimpleNamespace(
            graph_key='NA-22222222222222222222222222222222'),
        graph_version_info=SimpleNamespace(graph_data_version='v2'),
        weight_col_name=None,
    ))
    workflow._ml_job_settings = cast(Any, SimpleNamespace())

    result = await workflow._materialize_training_table('s3://bucket/tt', True)

    assert result == ('materialized', validation)
    execute_activity.assert_awaited_once()
    args, kwargs = execute_activity.await_args
    assert args[0] is mod.materialize_training_table
    assert kwargs['args'][0] is workflow._job_input
    assert kwargs['args'][2] == 's3://bucket/tt'
    assert kwargs['args'][-1] is False
    assert kwargs['task_queue'] == mod.DATA_BACKEND_TASK_QUEUE


def _registering_job_input(time_col_name=None):
    predictive_query = SimpleNamespace(
        query_type=QueryType.STATIC,
        list_distinct_target_is_array=False,
        validated_predictive_query=SimpleNamespace(entity_table='users'),
        graph=SimpleNamespace(tables=[
            SimpleNamespace(
                table_name='users',
                time_col=(None if time_col_name is None else SimpleNamespace(
                    name=time_col_name)),
            ),
        ]),
    )
    return SimpleNamespace(
        id='gen-traintable-job-registered',
        spec=SimpleNamespace(
            task_type=TaskType.REGRESSION,
            predictive_query=predictive_query,
        ),
    )


def _registering_spec():
    return SimpleNamespace(
        source_table=SimpleNamespace(data_source_type='s3'),
        weight_col_name='WEIGHT',
        split_col_name='SPLIT',
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(('time_col_name', 'anchored'), [
    (None, False),
    ('event_ts', True),
])
async def test_registering_reads_the_supplied_rows_instead_of_generating(
        monkeypatch, time_col_name, anchored):
    r"""A job that records rows reads them, checks them, and generates
    nothing.

    No pooled ``gen_tt`` and no snapshot residency: the rows are the
    caller's. The copy the ingest writes is what the job stands for, so it is
    both paths the job reports -- a supplied table is split and bundled by
    the run that trains on it, against this copy -- and it is that copy the
    contract is checked against, before the job is DONE and every later run
    trains on it unchecked. There is no generated table to compare with, so
    the validation is asked for none and told what the anchor would have
    been. A static node-pred query resolves its anchor from the entity
    table's own time column when it has one, and gets the sentinel
    otherwise -- both are exercised here.
    """
    validation = ValidationResponse()
    execute_activity = AsyncMock(side_effect=[
        ('s3://example-bucket/custom_train_table', validation),
        validation,
    ])
    run_pooled = AsyncMock()
    monkeypatch.setattr(mod.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(mod, 'run_pooled_activity', run_pooled)
    monkeypatch.setattr(mod, 'deploy_env', SimpleNamespace)
    monkeypatch.setattr(mod, 'get_job_base_dir',
                        lambda: 's3://example-bucket/jobs')

    workflow = mod.GenerateTrainingTableWorkflow()
    workflow._job_input = cast(Any, _registering_job_input(time_col_name))
    workflow._ml_job_settings = cast(Any, SimpleNamespace())
    workflow.job_id = 'gen-traintable-job-registered'
    custom_spec = _registering_spec()

    result = await workflow._register_supplied_table(cast(Any, custom_spec),
                                                     ValidationResponse())

    run_pooled.assert_not_awaited()
    assert execute_activity.await_count == 2
    args, kwargs = execute_activity.await_args_list[0]
    assert args[0] is mod.ingest_custom_train_table
    assert kwargs['args'][0] is custom_spec
    # The copy lands under this job's own home directory, not a training
    # job's: the registration owns it.
    assert kwargs['args'][1].startswith(
        's3://example-bucket/jobs/gen-traintable-job-registered')
    assert kwargs['args'][1].endswith('custom_train_table')
    assert kwargs['args'][-1] == time_col_name
    assert kwargs['task_queue'] == mod.DATA_BACKEND_TASK_QUEUE

    args, kwargs = execute_activity.await_args_list[1]
    assert args[0] is mod.validate_weighted_train_table
    assert kwargs['args'][0] is None
    assert kwargs['args'][1] == 's3://example-bucket/custom_train_table'
    assert kwargs['args'][2] == 'WEIGHT'
    assert kwargs['args'][-2:] == ['SPLIT', anchored]

    assert result.train_table_path == 's3://example-bucket/custom_train_table'
    assert result.train_table_materialization_path == (
        's3://example-bucket/custom_train_table')
    assert result.validation.ok


@pytest.mark.asyncio
async def test_a_supplied_table_the_platform_cannot_read_fails_the_job(
        monkeypatch):
    r"""The ingest's own errors are the job's, rather than a path of ''."""
    refused = ValidationResponse()
    refused.errors.append(
        mod.ValidationError(title='Source table not found',
                            message='no such table'))
    monkeypatch.setattr(mod.workflow, 'execute_activity',
                        AsyncMock(return_value=('', refused)))
    monkeypatch.setattr(mod, 'deploy_env', SimpleNamespace)
    monkeypatch.setattr(mod, 'get_job_base_dir',
                        lambda: 's3://example-bucket/jobs')

    workflow = mod.GenerateTrainingTableWorkflow()
    workflow._job_input = cast(Any, _registering_job_input())
    workflow._ml_job_settings = cast(Any, SimpleNamespace())
    workflow.job_id = 'gen-traintable-job-registered'

    result = await workflow._register_supplied_table(
        cast(Any, SimpleNamespace(source_table=SimpleNamespace())),
        ValidationResponse())

    assert not result.validation.ok
    assert result.train_table_path == ''
    assert result.train_table_materialization_path == ''


@pytest.mark.asyncio
async def test_a_registering_job_never_dispatches_generation(monkeypatch):
    r"""The branch is what makes a job record rather than generate.

    `gen_tt` is the pooled activity that builds a training table out of the
    graph. A job carrying supplied rows must not reach it: there is nothing
    to generate, and dispatching it would both cost a snapshot lease and
    overwrite the caller's rows with generated ones.
    """
    ingested = ('s3://example-bucket/jobs/gen-traintable-job-registered/'
                'custom_train_table')
    execute_activity = AsyncMock(
        side_effect=[None, (ingested, ValidationResponse()),
                     ValidationResponse()])
    run_pooled = AsyncMock()
    monkeypatch.setattr(mod.workflow, 'execute_activity', execute_activity)
    monkeypatch.setattr(mod, 'run_pooled_activity', run_pooled)
    monkeypatch.setattr(mod, 'deploy_env', SimpleNamespace)
    monkeypatch.setattr(mod, 'get_job_base_dir',
                        lambda: 's3://example-bucket/jobs')
    monkeypatch.setattr(mod, '_apply_graph_version_to_table_configs',
                        lambda *args: None)
    monkeypatch.setattr(mod, 'DataBackendContext',
                        lambda *args, **kwargs: SimpleNamespace())

    job_input = _registering_job_input()
    job_input.spec.custom_train_table = _registering_spec()
    job_input.spec.predictive_query.graph = SimpleNamespace(tables=[])
    job_input.graph_version_info = SimpleNamespace(graph_data_version='v3')
    job_input.generate_train_table_job_id = 'gen-traintable-job-registered'

    workflow = mod.GenerateTrainingTableWorkflow()
    workflow._job_input = cast(Any, job_input)
    workflow._ml_job_settings = cast(Any, SimpleNamespace())
    workflow._standalone_mode = False
    workflow.job_id = 'gen-traintable-job-registered'
    monkeypatch.setattr(workflow, 'check_cancelled', lambda: None)

    result = await workflow._generate_table()

    # Never generation, and never the snapshot lease generation needs.
    run_pooled.assert_not_awaited()
    dispatched = [call.args[0] for call in execute_activity.await_args_list]
    assert mod.ingest_custom_train_table in dispatched
    assert mod.validate_weighted_train_table in dispatched
    assert result.train_table_path == ingested
    assert result.validation.ok
