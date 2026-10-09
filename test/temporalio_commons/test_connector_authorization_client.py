# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from kumoapi.model_plan import HopConfig, NumNeighborsConfig
from temporalio import activity, workflow
from temporalio.api.enums.v1 import EventType
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from rest.backends.data_backend.activities.connector_authorization import (
    _actions_for_child_workflow,
)
from rest.backends.data_backend.connector_authorization_client import (
    replace_connector_authorization_tokens,
)
from rest.config.data_backend import ConnectorExecutionRef
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.config.common import ConnectorID, ProjectID
from temporalio_commons.temporal_client import (
    ClientOutboundInterceptor,
    TemporalClient,
)
from temporalio_commons.worker_interceptors import (
    KumoActivityInterceptor,
    KumoWorkflowInterceptor,
    KumoWorkflowOutboundInterceptor,
)

PROJECT_ID = ProjectID('project_' + '1' * 32)
CONNECTOR_ID = ConnectorID('connector_' + '2' * 32)
PARENT_TOKEN = 'parent-opaque-capability'
CHILD_TOKEN = 'child-opaque-capability'


def _reference(token: str = PARENT_TOKEN) -> ConnectorExecutionRef:
    return ConnectorExecutionRef(
        project_id=PROJECT_ID,
        connector_id=CONNECTOR_ID,
        authorization_token=token,
    )


@dataclass
class _ConnectorSamplingEnvelope:
    reference: ConnectorExecutionRef
    num_neighbors: NumNeighborsConfig


def test_replace_connector_authorization_tokens_preserves_pydantic_extras(
) -> None:
    envelope = _ConnectorSamplingEnvelope(
        reference=_reference(),
        num_neighbors=NumNeighborsConfig(
            hop1=HopConfig(
                default=0,
                **{
                    'impressions.CONSUMER_ID->users.CONSUMER_ID': 1,
                    'impressions.TEMPLATE_ID->templates.TEMPLATE_ID': 1,
                },
            ),
            hop2=HopConfig(
                default=0,
                **{
                    'users.CONSUMER_ID->orders.CONSUMER_ID': 32,
                },
            ),
        ),
    )

    replaced = replace_connector_authorization_tokens(
        envelope, {PARENT_TOKEN: CHILD_TOKEN})

    assert replaced.reference.authorization_token == CHILD_TOKEN
    hop1 = replaced.num_neighbors.hop1
    hop2 = replaced.num_neighbors.hop2
    assert hop1.__pydantic_extra__ == {
        'impressions.CONSUMER_ID->users.CONSUMER_ID': 1,
        'impressions.TEMPLATE_ID->templates.TEMPLATE_ID': 1,
    }
    assert hop2.__pydantic_extra__ == {
        'users.CONSUMER_ID->orders.CONSUMER_ID': 32,
    }


@pytest.mark.parametrize(
    'workflow_type',
    [
        'ArtifactExportWorkflow',
        'BatchPredictionJobWorkflow',
        'BatchPredictionStatsWorkflow',
        'CustomTrainTableWorkflow',
        'GeneratePredictionTableWorkflow',
        'GenerateTrainingTableWorkflow',
        'GraphRefreshWorkflow',
        'MaterializePredictionTaskWorkflow',
        'MaterializeTrainingTaskWorkflow',
        'SplitTrainingTableWorkflow',
        'TableIngestionWorkflow',
        'TrainingJobWorkflow',
    ],
)
def test_connector_child_workflow_has_closed_action_scope(
        workflow_type) -> None:
    assert _actions_for_child_workflow(workflow_type)


def test_unknown_child_workflow_cannot_receive_connector_authority() -> None:
    with pytest.raises(ValueError, match='cannot receive connector authority'):
        _actions_for_child_workflow('UnknownWorkflow')


def _client_config():
    return SimpleNamespace(
        workflow_id=lambda workflow_id: workflow_id,
        task_queue=lambda task_queue: task_queue,
        orchestration=object(),
    )


def _start_input():
    return SimpleNamespace(
        args=[_reference()],
        id='root-workflow',
        task_queue='task-queue',
        memo=None,
        search_attributes=None,
        rpc_timeout=None,
        task_timeout=None,
    )


@pytest.mark.asyncio
async def test_root_ambiguous_start_failure_retains_bound_capabilities(
    monkeypatch, ) -> None:
    state_db = object()
    services = object()
    bind = AsyncMock(return_value=(state_db, services, (PARENT_TOKEN, )))
    revoke = AsyncMock(side_effect=RuntimeError('cleanup failed'))
    close = AsyncMock()
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_authorization_client.'
        'bind_connector_authorizations_for_start', bind)
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_authorization_client.'
        'revoke_bound_connector_authorizations', revoke)
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_authorization_client.'
        'close_connector_authorization_state', close)
    next_interceptor = SimpleNamespace(start_workflow=AsyncMock(
        side_effect=RuntimeError('start failed')))
    interceptor = ClientOutboundInterceptor(_client_config(), next_interceptor)

    with pytest.raises(RuntimeError, match='start failed'):
        await interceptor.start_workflow(_start_input())

    revoke.assert_not_awaited()
    close.assert_awaited_once_with(state_db)


@pytest.mark.asyncio
async def test_root_already_started_revokes_loser_capabilities(
    monkeypatch, ) -> None:
    state_db = object()
    services = object()
    bind = AsyncMock(return_value=(state_db, services, (PARENT_TOKEN, )))
    revoke = AsyncMock(side_effect=RuntimeError('cleanup failed'))
    close = AsyncMock()
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_authorization_client.'
        'bind_connector_authorizations_for_start', bind)
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_authorization_client.'
        'revoke_bound_connector_authorizations', revoke)
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_authorization_client.'
        'close_connector_authorization_state', close)
    next_interceptor = SimpleNamespace(start_workflow=AsyncMock(side_effect=(
        WorkflowAlreadyStartedError('root-workflow', 'Workflow'))))
    interceptor = ClientOutboundInterceptor(_client_config(), next_interceptor)

    with pytest.raises(WorkflowAlreadyStartedError):
        await interceptor.start_workflow(_start_input())

    revoke.assert_awaited_once_with(state_db, services, (PARENT_TOKEN, ))
    close.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_root_start_success_closes_state_without_revoking(
    monkeypatch, ) -> None:
    state_db = object()
    services = object()
    bind = AsyncMock(return_value=(state_db, services, (PARENT_TOKEN, )))
    revoke = AsyncMock()
    close = AsyncMock()
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_authorization_client.'
        'bind_connector_authorizations_for_start', bind)
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_authorization_client.'
        'revoke_bound_connector_authorizations', revoke)
    monkeypatch.setattr(
        'rest.backends.data_backend.connector_authorization_client.'
        'close_connector_authorization_state', close)
    expected = object()
    next_interceptor = SimpleNamespace(start_workflow=AsyncMock(
        return_value=expected))
    interceptor = ClientOutboundInterceptor(_client_config(), next_interceptor)

    assert await interceptor.start_workflow(_start_input()) is expected

    close.assert_awaited_once_with(state_db)
    revoke.assert_not_awaited()


def _child_input():
    return SimpleNamespace(
        args=[{
            'connector': _reference()
        }],
        id='child-workflow-id',
        workflow='TableRefreshWorkflow',
        task_queue='task-queue',
        search_attributes=None,
        memo=None,
        task_timeout=None,
    )


@pytest.mark.asyncio
async def test_child_start_delegates_via_activity_and_replaces_token(
    monkeypatch, ) -> None:
    execute_activity = AsyncMock(return_value={PARENT_TOKEN: CHILD_TOKEN})
    monkeypatch.setattr(
        'temporalio_commons.worker_interceptors.workflow.execute_activity',
        execute_activity,
    )
    expected = object()
    next_interceptor = SimpleNamespace(start_child_workflow=AsyncMock(
        return_value=expected))
    interceptor = KumoWorkflowOutboundInterceptor(next_interceptor)
    interceptor.info = MagicMock(return_value=SimpleNamespace(
        search_attributes=None,
        raw_memo={},
        task_timeout=timedelta(seconds=10),
    ))
    child_input = _child_input()

    assert await interceptor.start_child_workflow(child_input) is expected

    assert execute_activity.await_args.args[0] == (
        'delegate_connector_authorizations')
    delegated = child_input.args[0]['connector']
    assert delegated.authorization_token == CHILD_TOKEN
    assert PARENT_TOKEN not in repr(delegated)


@pytest.mark.asyncio
async def test_child_ambiguous_start_failure_retains_delegated_tokens(
    monkeypatch, ) -> None:
    execute_activity = AsyncMock(side_effect=[{
        PARENT_TOKEN: CHILD_TOKEN
    }, None])
    monkeypatch.setattr(
        'temporalio_commons.worker_interceptors.workflow.execute_activity',
        execute_activity,
    )
    next_interceptor = SimpleNamespace(start_child_workflow=AsyncMock(
        side_effect=RuntimeError('child start failed')))
    interceptor = KumoWorkflowOutboundInterceptor(next_interceptor)
    interceptor.info = MagicMock(return_value=SimpleNamespace(
        search_attributes=None,
        raw_memo={},
        task_timeout=timedelta(seconds=10),
    ))

    with pytest.raises(RuntimeError, match='child start failed'):
        await interceptor.start_child_workflow(_child_input())

    assert execute_activity.await_count == 1


@pytest.mark.asyncio
async def test_child_already_started_revokes_loser_delegated_tokens(
    monkeypatch, ) -> None:
    execute_activity = AsyncMock(side_effect=[{
        PARENT_TOKEN: CHILD_TOKEN
    }, RuntimeError('cleanup failed')])
    monkeypatch.setattr(
        'temporalio_commons.worker_interceptors.workflow.execute_activity',
        execute_activity,
    )
    next_interceptor = SimpleNamespace(start_child_workflow=AsyncMock(
        side_effect=WorkflowAlreadyStartedError('child', 'Workflow')))
    interceptor = KumoWorkflowOutboundInterceptor(next_interceptor)
    interceptor.info = MagicMock(return_value=SimpleNamespace(
        search_attributes=None,
        raw_memo={},
        task_timeout=timedelta(seconds=10),
    ))

    with pytest.raises(WorkflowAlreadyStartedError):
        await interceptor.start_child_workflow(_child_input())

    assert execute_activity.await_count == 2
    assert execute_activity.await_args_list[1].args[0] == (
        'revoke_connector_authorizations')
    assert execute_activity.await_args_list[1].kwargs['args'] == [[CHILD_TOKEN]]


@activity.defn(name='delegate_connector_authorizations')
async def _history_delegate_connector_authorizations(
    references: list[ConnectorExecutionRef],
    _child_workflow_id: str,
    _child_workflow_type: str,
) -> dict[str, str]:
    return {
        reference.authorization_token: CHILD_TOKEN
        for reference in references
    }


@workflow.defn(name='TrainingJobWorkflow')
class _CapabilityDelegationChildWorkflow:
    @workflow.run
    async def run(self, reference: ConnectorExecutionRef) -> str:
        return await workflow.execute_activity(
            _history_postprocess_training_job_output,
            {reference.connector_id: reference},
            schedule_to_close_timeout=timedelta(seconds=10),
        )


@activity.defn(name='postprocess_training_job_output')
async def _history_postprocess_training_job_output(
    references: dict[ConnectorID, ConnectorExecutionRef],
) -> str:
    reference, = references.values()
    return reference.authorization_token


@workflow.defn
class _CapabilityDelegationParentWorkflow:
    @workflow.run
    async def run(self) -> str:
        reference = ConnectorExecutionRef(
            project_id=PROJECT_ID,
            connector_id=CONNECTOR_ID,
            authorization_token=PARENT_TOKEN,
        )
        return await workflow.execute_child_workflow(
            _CapabilityDelegationChildWorkflow.run,
            reference,
            id='capability-child',
        )


@pytest.mark.asyncio
async def test_child_delegation_activity_precedes_child_start_in_history(
    temporal_config: DeployEnvConfig, ) -> None:
    raw_task_queue = f'capability-child-history-{uuid4().hex}'
    client = await TemporalClient(temporal_config).get()
    worker = Worker(
        client,
        task_queue=temporal_config.task_queue(raw_task_queue),
        workflows=[
            _CapabilityDelegationParentWorkflow,
            _CapabilityDelegationChildWorkflow,
        ],
        activities=[
            _history_delegate_connector_authorizations,
            _history_postprocess_training_job_output,
        ],
        interceptors=[KumoActivityInterceptor(),
                      KumoWorkflowInterceptor()],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    async with worker:
        handle = await client.start_workflow(
            _CapabilityDelegationParentWorkflow.run,
            id=f'capability-parent-{uuid4().hex}',
            task_queue=raw_task_queue,
            execution_timeout=timedelta(seconds=20),
        )
        assert await handle.result() == CHILD_TOKEN

    events = [event async for event in handle.fetch_history_events()]
    delegation_index = next(
        index for index, event in enumerate(events)
        if (event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
            and event.activity_task_scheduled_event_attributes.activity_type.
            name == 'delegate_connector_authorizations'))
    child_index = next(
        index for index, event in enumerate(events) if event.event_type == (
            EventType.EVENT_TYPE_START_CHILD_WORKFLOW_EXECUTION_INITIATED))
    assert delegation_index < child_index

    child_handle = client.get_workflow_handle(
        temporal_config.workflow_id('capability-child'))
    child_events = [
        event async for event in child_handle.fetch_history_events()
    ]
    scheduled = next(
        event for event in child_events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED)
    scheduled_input = await client.data_converter.decode(
        scheduled.activity_task_scheduled_event_attributes.input.payloads,
        [dict[ConnectorID, ConnectorExecutionRef]],
    )
    assert scheduled_input == [
        {
            CONNECTOR_ID:
            ConnectorExecutionRef(
                project_id=PROJECT_ID,
                connector_id=CONNECTOR_ID,
                authorization_token=CHILD_TOKEN,
            )
        }
    ]
    child_history = b''.join(
        event.SerializeToString() for event in child_events)
    assert PARENT_TOKEN.encode() not in child_history
