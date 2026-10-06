# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import kumoapi.data_source as api
import pytest
from google.api_core.exceptions import GoogleAPIError
from pydantic import SecretStr
from temporalio import activity, workflow
from temporalio.api.enums.v1 import EventType
from temporalio.client import WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from kumo.config.connector import ConnectorConfig
from kumo.config.graph import GraphConfig, TableConfig
from rest.backends.data_backend.activities.graph_config_activity import (
    assert_public_connector_target_config,
)
from rest.backends.data_backend.activities.ingest_table_data_activity_v2 import (
    SAFE_TABLE_INGESTION_ERROR_MESSAGE,
    _raise_sanitized_table_ingestion_error,
)
from rest.config.credentials import (
    AWSFederatedCredentials,
    AzureFederatedCredentials,
    GoogleCloudCredentials,
    UsernamePassword,
)
from rest.config.data_backend import (
    ConnectorExecutionRef,
    ConnectorSpec,
    ResolvedConnectorSpec,
    bind_graph_config_to_execution_refs,
)
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.config.common import ConnectorID, ProjectID
from temporalio_commons.dataclass_converter import KumoPayloadConverter
from temporalio_commons.temporal_client import TemporalClient
from temporalio_commons.worker_interceptors import (
    KumoActivityInterceptor,
    KumoWorkflowInterceptor,
)

_CONNECTOR_ID = ConnectorID('connector_' + 'b' * 32)
_PROVIDER_HISTORY_SECRET = 'provider-token-stack-cause-must-not-enter-history'


@dataclass
class _NestedPayload:
    connector: object


@dataclass
class _ConnectorExecutionRefEnvelope:
    connectors: dict[str, ConnectorExecutionRef]


@activity.defn
async def _use_connector_reference(reference: ConnectorExecutionRef) -> str:
    if not isinstance(reference, ConnectorExecutionRef):
        raise TypeError('expected a project connector reference')
    return f'{reference.project_id}:{reference.connector_id}'


@workflow.defn
class _ConnectorReferenceWorkflow:
    @workflow.run
    async def run(self, reference: ConnectorExecutionRef) -> str:
        return await workflow.execute_activity(
            _use_connector_reference,
            reference,
            schedule_to_close_timeout=timedelta(seconds=10),
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )


@activity.defn
async def _resolve_activity_local_connector(
    reference: ConnectorExecutionRef, ) -> str:
    from rest.backends.data_backend.connector_broker import (
        connector_credential_broker,
    )

    resolved = await connector_credential_broker.resolve_async(reference)
    assert isinstance(resolved, ResolvedConnectorSpec)
    return resolved.connector_config['generation']


@workflow.defn
class _RepeatedConnectorAccessWorkflow:
    @workflow.run
    async def run(self, reference: ConnectorExecutionRef) -> list[str]:
        results = []
        for _ in range(2):
            results.append(await workflow.execute_activity(
                _resolve_activity_local_connector,
                reference,
                schedule_to_close_timeout=timedelta(seconds=10),
                start_to_close_timeout=timedelta(seconds=10),
                retry_policy=RetryPolicy(maximum_attempts=2),
            ))
        return results


@activity.defn
async def _sanitized_ingestion_failure() -> None:
    try:
        raise GoogleAPIError(_PROVIDER_HISTORY_SECRET)
    except Exception as error:
        _raise_sanitized_table_ingestion_error(error)


@workflow.defn
class _SanitizedIngestionFailureWorkflow:
    @workflow.run
    async def run(self) -> None:
        await workflow.execute_activity(
            _sanitized_ingestion_failure,
            schedule_to_close_timeout=timedelta(seconds=10),
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=2),
        )


def _resolved() -> ResolvedConnectorSpec:
    return ResolvedConnectorSpec(
        connector_id=_CONNECTOR_ID,
        connector_config={'name': 'warehouse'},
        credentials=UsernamePassword(
            username=SecretStr('service-account'),
            password=SecretStr('must-never-enter-history'),
        ),
    )


@pytest.mark.parametrize('value', [_resolved(), _NestedPayload(_resolved())])
def test_activity_local_resolution_cannot_enter_temporal_payload(value) -> None:
    with pytest.raises(TypeError, match='activity-local'):
        KumoPayloadConverter().to_payloads([value])


def test_project_connector_reference_is_temporal_serializable() -> None:
    reference = ConnectorExecutionRef(
        connector_id=_CONNECTOR_ID,
        project_id=ProjectID('project_' + 'a' * 32),
    )

    payloads = KumoPayloadConverter().to_payloads([reference])

    assert len(payloads) == 1
    assert b'must-never-enter-history' not in payloads[0].data


def test_resolved_internal_connector_config_cannot_enter_temporal_payload(
) -> None:
    value = ConnectorSpec(
        connector_id=_CONNECTOR_ID,
        connector_config={'target': 'must-remain-in-process'},
    )

    with pytest.raises(TypeError, match='activity-local'):
        KumoPayloadConverter().to_payloads([value])


def test_workflow_graph_strips_resolved_connector_configuration() -> None:
    project_id = ProjectID('project_' + 'a' * 32)
    reference = ConnectorExecutionRef(
        connector_id=_CONNECTOR_ID,
        project_id=project_id,
    )
    graph = GraphConfig(
        name='graph',
        data_source_dict={
            str(_CONNECTOR_ID):
            ConnectorConfig(name='resolved-target-must-not-enter-history')
        },
        tables=[
            TableConfig(
                table_name='events',
                source_name='events',
                data_source=str(_CONNECTOR_ID),
            )
        ],
    )

    safe_graph = bind_graph_config_to_execution_refs(
        graph,
        {_CONNECTOR_ID: reference},
    )
    payload = KumoPayloadConverter().to_payloads([
        _ConnectorExecutionRefEnvelope(connectors={'warehouse': reference}),
        safe_graph,
    ])
    payload_bytes = b''.join(value.data for value in payload)

    assert safe_graph.data_source_dict == {}
    assert b'resolved-target-must-not-enter-history' not in payload_bytes


def test_one_shot_management_config_allows_only_public_target_metadata(
) -> None:
    # Unlike long-running jobs, a one-shot management workflow may return the
    # same public target metadata exposed on a connector resource. It must
    # still fail closed on any identity, principal, or authentication field.
    assert_public_connector_target_config({
        'host': 'workspace.example.com',
        'catalog': 'analytics',
        'schema_name': 'events',
        'bucket': 'customer-data',
        'prefix': 'production/',
    })

    for forbidden in (
            'identity_id',
            'principal_id',
            'group_ids',
            'credentials',
            'aws_access_key_id',
            'client_secret',
            'private_key',
            'access_token',
            'session_token',
    ):
        with pytest.raises(TypeError, match='non-public field'):
            assert_public_connector_target_config(
                {'nested': {
                    forbidden: 'must-not-enter-history'
                }})


def test_nested_connector_reference_round_trips() -> None:
    converter = KumoPayloadConverter()
    reference = ConnectorExecutionRef(
        connector_id=_CONNECTOR_ID,
        project_id=ProjectID('project_' + 'a' * 32),
    )
    envelope = _ConnectorExecutionRefEnvelope(
        connectors={'warehouse': reference})

    decoded = converter.from_payloads(
        converter.to_payloads([envelope]),
        [_ConnectorExecutionRefEnvelope],
    )

    assert decoded == [envelope]
    assert isinstance(decoded[0].connectors['warehouse'], ConnectorExecutionRef)


@pytest.mark.parametrize(
    'credentials',
    [
        UsernamePassword(
            username=SecretStr('service-account'),
            password=SecretStr('password'),
        ),
        api.DatabricksOAuthCredentials(
            client_secret=SecretStr('client-secret')),
        AWSFederatedCredentials(
            access_key_id=SecretStr('access-key'),
            secret_access_key=SecretStr('secret-key'),
            session_token=SecretStr('session-token'),
        ),
        GoogleCloudCredentials(
            provider='gcs',
            credential={'access_token': 'gcs-access-token'},
        ),
        AzureFederatedCredentials(
            credential={'access_token': 'azure-access-token'}, ),
    ],
)
def test_bare_credentials_cannot_enter_temporal_payload(credentials) -> None:
    with pytest.raises(TypeError, match='activity-local'):
        KumoPayloadConverter().to_payloads([credentials])


def test_workflow_modules_cannot_resolve_connector_credentials() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    forbidden = (
        'get_connector_from_spec',
        'get_credentials_from_transports',
        'connector_credential_broker',
        'resolve_for_activity',
        'ResolvedConnectorSpec',
    )
    violations = []
    for path in (repository_root / 'rest' /
                 'backends').glob('**/workflows/*.py'):
        source = path.read_text()
        for marker in forbidden:
            if marker in source:
                violations.append(
                    f'{path.relative_to(repository_root)}: {marker}')
    assert violations == []


def test_runtime_modules_do_not_transport_or_recheck_submitter_membership(
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    forbidden = (
        'AuthPrincipal',
        'UserAuthorization',
        'user_authorization',
        'api_user_auth',
        'kumo_group_ids',
        'membership',
    )
    violations = []
    for backend in ('data_backend', 'ml_backend'):
        for runtime_kind in ('workflows', 'activities'):
            directory = (repository_root / 'rest' / 'backends' / backend /
                         runtime_kind)
            for path in directory.glob('*.py'):
                source = path.read_text()
                for marker in forbidden:
                    if marker in source:
                        violations.append(
                            f'{path.relative_to(repository_root)}: {marker}')
    assert violations == []


def test_backend_jobs_do_not_read_legacy_credential_payload_fields() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    forbidden = (
        'spec.credentials',
        'job_input.credentials',
        'connector.credentials',
    )
    violations = []
    for relative_root in ('rest/backends', 'pool'):
        for path in (repository_root / relative_root).rglob('*.py'):
            source = path.read_text(encoding='utf-8')
            for marker in forbidden:
                if marker in source:
                    violations.append(
                        f'{path.relative_to(repository_root)}: {marker}')
    assert violations == []


@pytest.mark.asyncio
async def test_temporal_history_contains_only_project_connector_ids(
    temporal_config: DeployEnvConfig, ) -> None:
    """Integration proof: no principal, binding, or secret enters history."""
    raw_task_queue = f'identity-boundary-{uuid4().hex}'
    client = await TemporalClient(temporal_config).get()
    reference = ConnectorExecutionRef(
        connector_id=_CONNECTOR_ID,
        project_id=ProjectID('project_' + 'a' * 32),
    )
    forbidden_markers = (
        b'api-user-auth',
        b'warehouse-password-must-never-enter-history',
        b'kumo_group_ids',
        b'access_token',
        b'credentials',
    )

    worker = Worker(
        client,
        task_queue=temporal_config.task_queue(raw_task_queue),
        workflows=[_ConnectorReferenceWorkflow],
        activities=[_use_connector_reference],
        interceptors=[KumoActivityInterceptor(),
                      KumoWorkflowInterceptor()],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    async with worker:
        handle = await client.start_workflow(
            _ConnectorReferenceWorkflow.run,
            reference,
            id=f'identity-boundary-{uuid4().hex}',
            task_queue=raw_task_queue,
            execution_timeout=timedelta(seconds=20),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        assert await handle.result() == (
            f'{reference.project_id}:{reference.connector_id}')

    events = [event async for event in handle.fetch_history_events()]
    history_bytes = b''.join(event.SerializeToString() for event in events)
    for marker in forbidden_markers:
        assert marker not in history_bytes

    started = next(
        event for event in events
        if event.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_STARTED)
    assert 'api-user-auth' not in (
        started.workflow_execution_started_event_attributes.header.fields)

    scheduled = next(
        event for event in events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED)
    inputs = await client.data_converter.decode(
        scheduled.activity_task_scheduled_event_attributes.input.payloads,
        [ConnectorExecutionRef],
    )
    assert inputs == [reference]


@pytest.mark.asyncio
async def test_ingestion_provider_failure_is_sanitized_in_history_and_result(
    temporal_config: DeployEnvConfig, ) -> None:
    raw_task_queue = f'ingestion-error-boundary-{uuid4().hex}'
    client = await TemporalClient(temporal_config).get()
    worker = Worker(
        client,
        task_queue=temporal_config.task_queue(raw_task_queue),
        workflows=[_SanitizedIngestionFailureWorkflow],
        activities=[_sanitized_ingestion_failure],
        interceptors=[KumoActivityInterceptor(),
                      KumoWorkflowInterceptor()],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    async with worker:
        handle = await client.start_workflow(
            _SanitizedIngestionFailureWorkflow.run,
            id=f'ingestion-error-boundary-{uuid4().hex}',
            task_queue=raw_task_queue,
            execution_timeout=timedelta(seconds=20),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        with pytest.raises(WorkflowFailureError) as caught:
            await handle.result()

    failure_text = str(caught.value)
    cause = caught.value.__cause__
    cause_types = []
    cause_messages = []
    while cause is not None:
        cause_types.append(getattr(cause, 'type', type(cause).__name__))
        cause_messages.append(str(cause))
        cause = cause.__cause__
    assert 'TableIngestionError' in cause_types
    result_envelope = '\n'.join([failure_text, *cause_messages])
    assert SAFE_TABLE_INGESTION_ERROR_MESSAGE in result_envelope
    assert _PROVIDER_HISTORY_SECRET not in result_envelope

    events = [event async for event in handle.fetch_history_events()]
    history_bytes = b''.join(event.SerializeToString() for event in events)
    assert _PROVIDER_HISTORY_SECRET.encode() not in history_bytes


@pytest.mark.asyncio
async def test_each_activity_access_reresolves_rotated_credentials_without_history_leak(
    temporal_config: DeployEnvConfig,
    monkeypatch,
) -> None:
    """A queued job re-resolves at each activity, never at submission."""
    from rest.backends.data_backend.connector_broker import (
        connector_credential_broker,
    )

    raw_task_queue = f'identity-reresolve-{uuid4().hex}'
    client = await TemporalClient(temporal_config).get()
    reference = ConnectorExecutionRef(
        connector_id=_CONNECTOR_ID,
        project_id=ProjectID('project_' + 'a' * 32),
    )
    secrets = ('rotation-one-never-in-history', 'rotation-two-never-in-history')
    resolved = [
        ResolvedConnectorSpec(
            connector_id=_CONNECTOR_ID,
            connector_config={'generation': generation},
            credentials=UsernamePassword(
                username=SecretStr('service-account'),
                password=SecretStr(secret),
            ),
        ) for generation, secret in zip(('one', 'two'), secrets)
    ]

    class _MutableConnectorService:
        def __init__(self) -> None:
            self.calls = []

        async def resolve_for_activity(
            self,
            project_id,
            connector_id,
        ):
            self.calls.append((project_id, connector_id))
            return resolved[len(self.calls) - 1]

    service = _MutableConnectorService()
    previous_services = connector_credential_broker._services
    connector_credential_broker._services = SimpleNamespace(connectors=service)
    monkeypatch.setattr(
        connector_credential_broker,
        '_authorize_reference',
        AsyncMock(return_value=object()),
    )
    worker = Worker(
        client,
        task_queue=temporal_config.task_queue(raw_task_queue),
        workflows=[_RepeatedConnectorAccessWorkflow],
        activities=[_resolve_activity_local_connector],
        interceptors=[KumoActivityInterceptor(),
                      KumoWorkflowInterceptor()],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    try:
        async with worker:
            handle = await client.start_workflow(
                _RepeatedConnectorAccessWorkflow.run,
                reference,
                id=f'identity-reresolve-{uuid4().hex}',
                task_queue=raw_task_queue,
                execution_timeout=timedelta(seconds=20),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            assert await handle.result() == ['one', 'two']
    finally:
        connector_credential_broker._services = previous_services

    assert service.calls == [
        (reference.project_id, reference.connector_id),
        (reference.project_id, reference.connector_id),
    ]
    events = [event async for event in handle.fetch_history_events()]
    history_bytes = b''.join(event.SerializeToString() for event in events)
    for secret in secrets:
        assert secret.encode() not in history_bytes


@pytest.mark.asyncio
async def test_binding_removal_is_observed_before_the_next_activity_access(
    temporal_config: DeployEnvConfig,
    monkeypatch,
) -> None:
    from rest.backends.data_backend.connector_broker import (
        connector_credential_broker,
    )

    raw_task_queue = f'identity-binding-removal-{uuid4().hex}'
    client = await TemporalClient(temporal_config).get()
    reference = ConnectorExecutionRef(
        connector_id=_CONNECTOR_ID,
        project_id=ProjectID('project_' + 'a' * 32),
    )
    secret = 'removed-binding-secret-never-in-history'
    first = ResolvedConnectorSpec(
        connector_id=_CONNECTOR_ID,
        connector_config={'generation': 'before-removal'},
        credentials=UsernamePassword(
            username=SecretStr('service-account'),
            password=SecretStr(secret),
        ),
    )

    class _RemovedBindingService:
        def __init__(self) -> None:
            self.calls = 0

        async def resolve_for_activity(
            self,
            _project_id,
            _connector_id,
        ):
            self.calls += 1
            if self.calls == 1:
                return first
            raise PermissionError('identity binding is no longer active')

    service = _RemovedBindingService()
    previous_services = connector_credential_broker._services
    connector_credential_broker._services = SimpleNamespace(connectors=service)
    monkeypatch.setattr(
        connector_credential_broker,
        '_authorize_reference',
        AsyncMock(return_value=object()),
    )
    worker = Worker(
        client,
        task_queue=temporal_config.task_queue(raw_task_queue),
        workflows=[_RepeatedConnectorAccessWorkflow],
        activities=[_resolve_activity_local_connector],
        interceptors=[KumoActivityInterceptor(),
                      KumoWorkflowInterceptor()],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    try:
        async with worker:
            handle = await client.start_workflow(
                _RepeatedConnectorAccessWorkflow.run,
                reference,
                id=f'identity-binding-removal-{uuid4().hex}',
                task_queue=raw_task_queue,
                execution_timeout=timedelta(seconds=20),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            with pytest.raises(WorkflowFailureError):
                await handle.result()
    finally:
        connector_credential_broker._services = previous_services

    assert service.calls == 2  # first access + immediate denied next access
    events = [event async for event in handle.fetch_history_events()]
    history_bytes = b''.join(event.SerializeToString() for event in events)
    assert secret.encode() not in history_bytes
