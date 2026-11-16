# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TypeVar

from temporalio import activity

from io_utils import logging
from rest.config.data_backend import (
    ActivityAuthorizationRef,
    ConnectorExecutionRef,
    ConnectorOpenSpec,
)
from rest.settings import get_db_storage_root
from service_framework.deploy_env_config import deploy_env
from statedb.db import StateDB, StateServices, open_state_db_from_deploy_config
from statedb.rest.connector_authorization import (
    ConnectorAuthorizationDBEntry,
    ConnectorAuthorizationResourceKind,
    ConnectorCapabilityAction,
    connector_capability_action_for_activity_type,
)

T = TypeVar('T')

_DEFAULT_BROKER_TIMEOUT_SECONDS = 30.0
log = logging.getLogger(__name__)


class ConnectorCredentialBrokerTimeout(RuntimeError):
    """A credential resolution exceeded its bounded activity-local wait."""


class ConnectorCredentialBroker:
    """Resolve a project-owned identity only inside an activity process."""
    def __init__(
        self,
        timeout_seconds: float = _DEFAULT_BROKER_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError('Connector broker timeout must be positive')
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._services_init_lock: asyncio.Lock | None = None
        self._state_db: StateDB | None = None
        self._services: StateServices | None = None
        self._principal_authority = None

    def resolve(self, reference: ConnectorExecutionRef) -> ConnectorOpenSpec:
        activity_info = activity.info()

        async def resolve_connector(
                services: StateServices) -> ConnectorOpenSpec:
            await self._authorize_reference(services, reference, activity_info)
            return await services.connectors.resolve_for_activity(
                reference.project_id,
                reference.connector_id,
            )

        return self.run(resolve_connector)

    async def resolve_async(
        self,
        reference: ConnectorExecutionRef,
    ) -> ConnectorOpenSpec:
        """Resolve without blocking an async Temporal activity event loop."""
        activity_info = activity.info()

        async def resolve_connector(
                services: StateServices) -> ConnectorOpenSpec:
            await self._authorize_reference(services, reference, activity_info)
            return await services.connectors.resolve_for_activity(
                reference.project_id,
                reference.connector_id,
            )

        return await self.run_async(resolve_connector)

    async def _authorize_reference(
        self,
        services: StateServices,
        reference: ConnectorExecutionRef,
        info,
    ) -> ConnectorAuthorizationDBEntry:
        if (info.workflow_id is None or info.workflow_run_id is None
                or not info.activity_id):
            raise RuntimeError(
                'Connector resolution requires workflow activity provenance')
        action = connector_capability_action_for_activity_type(
            info.activity_type)
        deploy_config = deploy_env()
        authorization = await services.connector_authorizations.authorize_activity(
            reference.authorization_token,
            deployment_id=(f'{deploy_config.instance_id}:'
                           f'{deploy_config.client_id}'),
            temporal_namespace=info.namespace,
            workflow_id=info.workflow_id,
            workflow_run_id=info.workflow_run_id,
            activity_id=info.activity_id,
            action=action,
            resource_kind=ConnectorAuthorizationResourceKind.CONNECTOR,
            resource_id=str(reference.connector_id),
            project_id=str(reference.project_id),
        )
        return authorization

    def delegate_connector_references(
        self,
        references: list[ConnectorExecutionRef],
        *,
        child_workflow_id: str,
        child_actions: set[ConnectorCapabilityAction],
    ) -> dict[str, str]:
        """Validate every reference before minting exact child capabilities.

        A second live check immediately precedes each mint. If that check or a
        later mint fails, already-created children are revoked best-effort so
        a multi-connector delegation cannot leave partial authority until TTL.
        """
        info = activity.info()

        async def delegate(services: StateServices) -> dict[str, str]:
            prepared = []
            for reference in references:
                parent = await self._authorize_reference(
                    services, reference, info)
                await services.connectors.require_delegatable_for_project(
                    reference.project_id, reference.connector_id)
                requested = child_actions.intersection(parent.allowed_actions)
                if not requested:
                    from statedb.db.connector_authorizations import (
                        ConnectorAuthorizationError,
                    )
                    raise ConnectorAuthorizationError(
                        'Child workflow has no authorized connector actions')
                prepared.append((reference, parent, requested))

            replacements: dict[str, str] = {}
            minted_tokens: list[str] = []
            try:
                for reference, parent, requested in prepared:
                    # Close the phase-one race as far as possible. Revocation
                    # below handles a later reference becoming unavailable.
                    await services.connectors.require_delegatable_for_project(
                        reference.project_id,
                        reference.connector_id,
                    )
                    _, token = (
                        await
                        services.connector_authorizations.delegate_to_child(
                            parent,
                            child_workflow_id=child_workflow_id,
                            child_actions=requested,
                        ))
                    minted_tokens.append(token)
                    replacements[reference.authorization_token] = token
            except BaseException:
                for token in minted_tokens:
                    try:
                        await services.connector_authorizations.revoke(token)
                    except BaseException as cleanup_error:
                        log.warning(
                            'Child connector capability rollback failed '
                            '(error_type=%s)',
                            type(cleanup_error).__name__,
                        )
                raise
            return replacements

        return self.run(delegate)

    def authorize_activity_scope(
        self,
        reference: ActivityAuthorizationRef,
        *,
        resource_kind: ConnectorAuthorizationResourceKind,
        resource_id: str,
        project_id: str,
        request_digest: str,
        action: ConnectorCapabilityAction,
    ):
        """Consume a management capability and recheck its current owner."""
        info = activity.info()

        async def authorize(services: StateServices):
            deploy_config = deploy_env()
            authorization = (
                await services.connector_authorizations.authorize_activity(
                    reference.authorization_token,
                    deployment_id=(f'{deploy_config.instance_id}:'
                                   f'{deploy_config.client_id}'),
                    temporal_namespace=info.namespace,
                    workflow_id=info.workflow_id or '',
                    workflow_run_id=info.workflow_run_id or '',
                    activity_id=info.activity_id,
                    action=action,
                    resource_kind=resource_kind,
                    resource_id=resource_id,
                    project_id=project_id,
                    request_digest=request_digest,
                ))
            if self._principal_authority is None:
                from rest.backends.data_backend.principal_authority import (
                    WorkerPrincipalAuthorityResolver,
                )
                self._principal_authority = WorkerPrincipalAuthorityResolver(
                    services)
            return await self._principal_authority.resolve(authorization)

        return self.run(authorize)

    def authorize_project_activity_scope(
        self,
        reference: ActivityAuthorizationRef,
        *,
        resource_kind: ConnectorAuthorizationResourceKind,
        resource_id: str,
        project_id: str,
        request_digest: str,
        action: ConnectorCapabilityAction,
    ):
        """Consume accepted project-work authority without user revalidation.

        The opaque capability is the non-forgeable admission decision. The
        activity service must still reload every project-owned resource and
        relationship before using credentials or committing a mutation.
        """
        info = activity.info()

        async def authorize(services: StateServices):
            deploy_config = deploy_env()
            return await services.connector_authorizations.authorize_activity(
                reference.authorization_token,
                deployment_id=(f'{deploy_config.instance_id}:'
                               f'{deploy_config.client_id}'),
                temporal_namespace=info.namespace,
                workflow_id=info.workflow_id or '',
                workflow_run_id=info.workflow_run_id or '',
                activity_id=info.activity_id,
                action=action,
                resource_kind=resource_kind,
                resource_id=resource_id,
                project_id=project_id,
                request_digest=request_digest,
            )

        return self.run(authorize)

    def run(self, operation: Callable[[StateServices], Awaitable[T]]) -> T:
        """Run a StateServices operation only from an activity process."""
        if not activity.in_activity():
            raise RuntimeError(
                'Credential-bearing StateServices operations may run only '
                'inside a Temporal activity.')
        loop = self._ensure_loop()
        future: Future = asyncio.run_coroutine_threadsafe(
            self._run(operation), loop)
        try:
            return future.result(timeout=self._timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            raise ConnectorCredentialBrokerTimeout(
                'Connector credential resolution timed out') from None

    async def run_async(
        self,
        operation: Callable[[StateServices], Awaitable[T]],
    ) -> T:
        """Await a service operation with bounded, cancellation-safe waiting."""
        if not activity.in_activity():
            raise RuntimeError(
                'Credential-bearing StateServices operations may run only '
                'inside a Temporal activity.')
        loop = self._ensure_loop()
        future: Future = asyncio.run_coroutine_threadsafe(
            self._run(operation), loop)
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.wait_for(
                wrapped,
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            future.cancel()
            raise ConnectorCredentialBrokerTimeout(
                'Connector credential resolution timed out') from None
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def _run(
        self,
        operation: Callable[[StateServices], Awaitable[T]],
    ) -> T:
        services = await self._get_services()
        return await operation(services)

    async def _get_services(self) -> StateServices:
        if self._services is not None:
            return self._services
        if self._services_init_lock is None:
            self._services_init_lock = asyncio.Lock()
        async with self._services_init_lock:
            if self._services is None:
                self._state_db = await open_state_db_from_deploy_config(
                    deploy_env(),
                    state_root=get_db_storage_root(),
                )
                self._services = StateServices(self._state_db)
        services = self._services
        assert services is not None
        return services

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None:
                return self._loop
            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def run() -> None:
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_forever()

            thread = threading.Thread(
                target=run,
                name='connector-credential-broker',
                daemon=True,
            )
            thread.start()
            if not ready.wait(timeout=self._timeout_seconds):
                raise ConnectorCredentialBrokerTimeout(
                    'Connector credential broker failed to start')
            self._loop = loop
            return loop


connector_credential_broker = ConnectorCredentialBroker()
