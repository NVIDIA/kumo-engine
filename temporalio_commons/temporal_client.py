# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import dataclasses
import os
import random
import threading
import weakref
from asyncio.exceptions import TimeoutError
from datetime import timedelta
from typing import Any, Dict, Optional, Union

import temporalio.api.workflowservice.v1 as wsv1
from google.protobuf.duration_pb2 import Duration
from temporalio.client import (
    Client,
    Interceptor,
    OutboundInterceptor,
    StartWorkflowInput,
    WorkflowExecutionDescription,
    WorkflowExecutionStatus,
    WorkflowHandle,
)
from temporalio.common import SearchAttributeValues, TypedSearchAttributes
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.converter import DataConverter
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.service import (
    RetryConfig,
    RPCError,
    RPCStatusCode,
    ServiceClient,
    TLSConfig,
)
from temporalio.service import WorkflowService as TemporalService

from async_utils.instrument import measure_blocking_code
from io_utils import fsutil, logging
from io_utils.secrets import get_secrets
from service_framework.deploy_env import DeployStage
from service_framework.deploy_env_config import (
    DeployEnvConfig,
    TemporalCloudConfig,
    deploy_env,
)
from service_framework.env_var import EnvVar
from temporalio_commons.compression_codec import CompressionCodec
from temporalio_commons.dataclass_converter import KumoPayloadConverter
from temporalio_commons.heartbeat_utils import auto_heartbeater

log = logging.getLogger(__name__)

_REGISTER_NAMESPACE_POLL_INTERVAL_SEC = 2.0
_NEW_NAMESPACE_WAIT_SEC = 10.0
_TEMPORAL_CLUSTER_CONNECTION_TIMEOUT_SEC = 5.0
# Includes waiting for another initializer on the same event loop as well as
# connection setup, so a wedged initializer cannot strand every later caller.
_TEMPORAL_CLIENT_ACQUISITION_TIMEOUT_SEC = 20.0
# Deadline for a single gRPC call to Temporal service
_TEMPORAL_RPC_TIMEOUT = timedelta(seconds=2.0)
# Total deadline for gRPC call to Temporal service including retries.
_TEMPORAL_RPC_TOTAL_TIMEOUT = timedelta(seconds=20.0)

# Secret name in AWS Secret Manager for Temporal Cloud certificate.
_TEMPORAL_CERT_SECRET_NAME_TEMPLATE = 'kumo-{namespace}-temporal'

# Field name in workflow memo (metadata) dict for workflow id suffix.
ID_SUFFIX_MEMO_FIELD = "id_suffix"

DEFAULT_WORKFLOW_TASK_TIMEOUT_SEC = EnvVar('WORKFLOW_TASK_TIMEOUT_SEC',
                                           type=int,
                                           default_value=10).parse_value()


# pylint: disable=too-few-public-methods
@dataclasses.dataclass
class _LoopServiceClientState:
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    service_clients: Dict[str, ServiceClient] = dataclasses.field(
        default_factory=dict)


class _TemporalServiceClientManager:
    def __init__(self) -> None:
        # Temporal's Rust service clients and asyncio synchronization
        # primitives are event-loop-affine. A process-global asyncio.Lock can
        # be acquired on one loop and then strand callers on another loop
        # forever (notably pytest's session loop vs TestClient's AnyIO portal).
        # Weak keys let completed short-lived loops and their clients be
        # reclaimed. The small threading lock protects only state creation;
        # it is never held across async work.
        self._loop_states: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop,
            _LoopServiceClientState] = (weakref.WeakKeyDictionary())
        self._loop_states_lock = threading.Lock()

    def _loop_state(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        create: bool,
    ) -> Optional[_LoopServiceClientState]:
        with self._loop_states_lock:
            state = self._loop_states.get(loop)
            if state is None and create:
                state = _LoopServiceClientState()
                self._loop_states[loop] = state
            return state

    async def get_service_client(
        self,
        deploy_env: DeployEnvConfig,
        cert_secret_name: Optional[str] = None,
        cert_dir: Optional[str] = None,
        re_initialize: bool = False,
    ) -> ServiceClient:
        loop = asyncio.get_running_loop()
        state = self._loop_state(loop, create=True)
        assert state is not None
        target_address = deploy_env.temporal_server_address
        client_id = deploy_env.client_id
        key = f'{target_address}-{client_id}'

        async def get_or_connect() -> tuple[ServiceClient, bool]:
            connected = False
            async with state.lock:
                if key not in state.service_clients or re_initialize:
                    old_service_client = state.service_clients.get(key)
                    new_service_client = await self._connect(
                        deploy_env,
                        cert_secret_name=cert_secret_name,
                        cert_dir=cert_dir,
                    )
                    # Do not invalidate the usable cached connection until its
                    # replacement has connected successfully.
                    state.service_clients[key] = new_service_client
                    if (old_service_client is not None
                            and old_service_client is not new_service_client
                            and hasattr(old_service_client, '_bridge_client')):
                        setattr(old_service_client, '_bridge_client', None)
                    connected = True
                return state.service_clients[key], connected

        try:
            service_client, connected = await asyncio.wait_for(
                get_or_connect(),
                timeout=_TEMPORAL_CLIENT_ACQUISITION_TIMEOUT_SEC,
            )
        except TimeoutError as error:
            raise TimeoutError(
                'Timed out acquiring Temporal service client after '
                f'{_TEMPORAL_CLIENT_ACQUISITION_TIMEOUT_SEC:.1f}s '
                f'(target={target_address}, loop_id={id(loop)}, '
                f'initializer_locked={state.lock.locked()})') from error

        if connected:
            log.info(
                'Connected client %s to %s, key=%s, re_initialize=%s, '
                'loop_id=%s', client_id, target_address, key, re_initialize,
                id(loop))
        return service_client

    async def _connect(
        self,
        deploy_env: DeployEnvConfig,
        *,
        cert_secret_name: Optional[str],
        cert_dir: Optional[str],
    ) -> ServiceClient:
        tls: Union[bool, TLSConfig] = False
        if cert_dir:
            with open(f'{cert_dir}/tls.crt', 'rb') as f:
                client_cert = f.read()
            with open(f'{cert_dir}/tls.key', 'rb') as f:
                client_key = f.read()
            with open(f'{cert_dir}/ca.crt', 'rb') as f:
                server_cert = f.read()
            tls = TLSConfig(
                client_cert=client_cert,
                client_private_key=client_key,
                server_root_ca_cert=server_cert,
            )
        elif cert_secret_name:
            # Retrieve TLS certificate secrets without blocking this loop.
            loop = asyncio.get_running_loop()
            cert_secret = await loop.run_in_executor(None, get_secrets,
                                                     cert_secret_name)
            tls = TLSConfig(
                client_cert=base64.b64decode(cert_secret['ca-cert']),
                client_private_key=base64.b64decode(cert_secret['ca-key']))

        client = await asyncio.wait_for(
            Client.connect(
                deploy_env.temporal_server_address,
                identity=deploy_env.client_id,
                tls=tls,
                retry_config=RetryConfig(
                    max_elapsed_time_millis=int(
                        _TEMPORAL_RPC_TOTAL_TIMEOUT.total_seconds() * 1000),
                    max_retries=10,  # the default value
                ),
            ),
            timeout=_TEMPORAL_CLUSTER_CONNECTION_TIMEOUT_SEC)
        return client.service_client

    async def close(self) -> None:
        """Release cached gRPC connections owned by the current event loop.

        The temporalio SDK does not expose a close method on ServiceClient.
        Dropping references to the underlying Rust ClientRef (via
        ``_bridge_client``) is the only way to release gRPC channels and
        prevent anyio's BlockingPortal teardown from blocking on the Rust
        gRPC GOAWAY handshake.
        """
        loop = asyncio.get_running_loop()
        state = self._loop_state(loop, create=False)
        if state is None:
            return
        async with state.lock:
            for sc in state.service_clients.values():
                if hasattr(sc, '_bridge_client'):
                    setattr(sc, '_bridge_client', None)
            state.service_clients.clear()
        with self._loop_states_lock:
            self._loop_states.pop(loop, None)

    def close_all_loop_states(self) -> None:
        """Release every loop's clients from a quiescent process owner.

        This is intentionally synchronous and rejects active initializers. It
        is for process/test shutdown after the owning event loops have stopped;
        normal application lifecycle code must use ``close()`` on its loop.
        """
        with self._loop_states_lock:
            loop_states = list(self._loop_states.items())
            if any(loop.is_running() or state.lock.locked()
                   for loop, state in loop_states):
                raise RuntimeError(
                    'Cannot close all Temporal clients until every owning '
                    'event loop is stopped and every initializer is idle')
            self._loop_states.clear()
        for _, state in loop_states:
            for service_client in state.service_clients.values():
                if hasattr(service_client, '_bridge_client'):
                    setattr(service_client, '_bridge_client', None)
            state.service_clients.clear()


_SERVICE_CLIENT_MANAGER = _TemporalServiceClientManager()


class TemporalClient:
    """
    Helper class that initializes connection and creates clients to talk with
    Temporal Cluster.  It provides both async and sync interfaces for
    convenience.  It also plugs in the data converter that works well for
    pydantic dataclasses.
    This should be a global singleton instance of TemporalClient per temporal
    cluster address per application process, so that client objects share the
    same underlying gRPC service connection to that cluster.

    Args:
        config (TemporalConfig): configuration of the temporal client.
    """
    def __init__(self, config: Optional[DeployEnvConfig] = None) -> None:
        self._config = config or deploy_env()
        # Keep the public SDK wrapper loop-local for the same reason as the
        # underlying ServiceClient. ``_client`` remains a best-effort alias for
        # older tooling that inspects it after initialization; async code must
        # always select from ``_loop_clients``.
        self._loop_clients: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, Client] = weakref.WeakKeyDictionary()
        self._loop_clients_lock = threading.Lock()
        self._client: Optional[Client] = None
        self._client_manager = _SERVICE_CLIENT_MANAGER

    def _client_for_loop(self,
                         loop: asyncio.AbstractEventLoop) -> Optional[Client]:
        with self._loop_clients_lock:
            return self._loop_clients.get(loop)

    def _set_client_for_loop(self, loop: asyncio.AbstractEventLoop,
                             client: Client) -> None:
        with self._loop_clients_lock:
            self._loop_clients[loop] = client
            self._client = client

    @property
    def deploy_mode(self) -> DeployStage:
        return self._config.deploy_mode

    @property
    def config(self) -> DeployEnvConfig:
        return self._config

    async def close(self) -> None:
        """Close this loop's client and release its gRPC connections."""
        loop = asyncio.get_running_loop()
        with self._loop_clients_lock:
            removed = self._loop_clients.pop(loop, None)
            if removed is self._client:
                self._client = next(iter(self._loop_clients.values()), None)
        # Capability binding opens a loop-local SQLite StateDB.  Keep its
        # lifecycle aligned with this loop's Temporal client so aiosqlite
        # pools are never shared across loops or left behind at shutdown.
        # yapf disagrees with isort on wrapping a single-name parenthesized
        # import; isort wins here.
        # yapf: disable
        from rest.backends.data_backend.connector_authorization_client import (
            close_cached_authorization_state_for_current_loop,
        )

        # yapf: enable
        await close_cached_authorization_state_for_current_loop()
        await self._client_manager.close()

    def close_all_loop_clients(self) -> None:
        """Release all loop-local clients from a quiescent process owner."""
        self._client_manager.close_all_loop_states()
        with self._loop_clients_lock:
            self._loop_clients.clear()
            self._client = None

    async def init(self, re_initialize: bool = False) -> None:
        """
        Initialize the underlying gRPC service connection.

        Args:
            re_initialize (bool, optional): Force re-initialize the connection.
        """
        loop = asyncio.get_running_loop()
        if self._client_for_loop(loop) is None or re_initialize:
            log.info('Initialize connection to temporal service at %s',
                     self._config.temporal_server_address)

            # Retrieve TLS certificate from AWS Secret Manager when connecting
            # to Temporal Cloud.
            cert_secret_name = None
            if isinstance(self._config.orchestration, TemporalCloudConfig):
                cert_secret_name = _TEMPORAL_CERT_SECRET_NAME_TEMPLATE.format(
                    namespace=self._config.orchestration.namespace)

            service_client = await self._client_manager.get_service_client(
                self._config, cert_secret_name=cert_secret_name,
                cert_dir=self._config.tls_cert_dir, re_initialize=re_initialize)

            self._set_client_for_loop(loop, self._create_client(service_client))

    @measure_blocking_code
    async def get(self, re_initialize: bool = False) -> Client:
        """
        Get an initialized temporal Client object for the config.  If the
        namespace has not been registered before, this call will block for 10+
        seconds until the new namespace is fully registered.
        """
        await self.init(re_initialize=re_initialize)
        client = self._client_for_loop(asyncio.get_running_loop())
        assert client
        return client

    async def get_workflow_handle(
            self, workflow_id: str, *,
            run_id: Optional[str] = None) -> WorkflowHandle:
        """
        Replacement for temporalio.client.Client.get_workflow_handle().
        """
        client = await self.get()
        workflow_id = self._config.workflow_id(workflow_id)
        return client.get_workflow_handle(workflow_id, run_id=run_id)

    async def get_workflow_desc(
            self, workflow_id: str, *, run_id: Optional[str] = None
    ) -> Optional[WorkflowExecutionDescription]:
        wf_handle = await self.get_workflow_handle(workflow_id, run_id=run_id)
        try:
            return await wf_handle.describe()
        except RPCError as rpc_error:
            if rpc_error.status == RPCStatusCode.NOT_FOUND:
                return None
            raise

    def format_workflow_web_url(self, workflow_id: str) -> str:
        base_url = 'http://localhost:7233'
        if isinstance(self._config.orchestration, TemporalCloudConfig):
            base_url = 'https://cloud.temporal.io'

        full_wf_id = self._config.workflow_id(workflow_id)
        return (f'{base_url}/namespaces/'
                f'{self._config.temporal_namespace}/workflows/'
                f'{full_wf_id}')

    async def cancel(self, workflow_id: str, *, max_retries: int = 3) -> bool:
        """Cancel a workflow with retries on RPC error.

        Args:
            workflow_id (str): ID of the workflow.
            max_retries (int, optional): Max number of retries upon unexpected
            RPCError. Defaults to 3.

        Returns:
            `True` if the workflow exists and was canceled, `False` if it
            doesn't exist.
        """
        wf_handle = await self.get_workflow_handle(workflow_id)
        backoff_delay_initial = 1.0
        backoff_delay_max = 30.0
        backoff_delay = backoff_delay_initial
        num_retries = 0
        while True:
            try:
                await wf_handle.cancel()
                log.info('Cancellation request sent for %s', workflow_id)
                return True
            except RPCError as rpc_error:
                log.error(f'(attempt #{num_retries}) '
                          f'Failed to cancel workflow: {workflow_id}, '
                          f'rpc error: {rpc_error}')
                if rpc_error.status == RPCStatusCode.NOT_FOUND:
                    # Valid if workflow is NOT FOUND, which could happen if the
                    # workflow is older than retention period.
                    return False

                if num_retries < max_retries:
                    num_retries += 1
                    await asyncio.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 2.0, backoff_delay_max)
                    continue

                raise

    async def kill_wf(self, wf_id: str, reason: str):
        """Force kill a workflow if it's still running."""
        wf_handle = await self.get_workflow_handle(wf_id)
        wf_desc = await wf_handle.describe()
        if wf_desc.status == WorkflowExecutionStatus.RUNNING:
            client = await self.get()
            wf_exec = wf_desc.raw_description.workflow_execution_info.execution
            await client.workflow_service.terminate_workflow_execution(
                wsv1.TerminateWorkflowExecutionRequest(
                    namespace=self.config.temporal_namespace,
                    workflow_execution=wf_exec, reason=reason))
            log.warning('Force terminated wf=%s, reason=%s', wf_id, reason)

    @auto_heartbeater
    async def result(
        self,
        workflow_id: str,
        *,
        run_id: Optional[str] = None,
        disable_backoff: bool = False,
    ):
        """
        Get a workflow result while awaiting for it to complete. Raise
        exception if workflow finished unsuccessfully.
        If called from a Temporal Activity execution and heartbeat internal is
        specified, this will periodically report heartbeats while awaiting for
        the workflow result.
        `first_poll_timeout` specifies the initial "long-poll" timeout for
        waiting on workflow result; after it expries, we'll switch to use
        short-polls with exponential backoff logic to reduce the global total
        of concurrenct pollers on Temporal service.  If negative, we'll always
        use long-poll and never perform any backoff (mostly for testing).
        """
        # Pin callers that already resolved a concrete execution to that run.
        # Looking up only by workflow ID can attach to a later execution when
        # a completed reusable child races with its foster parent.
        wf_handle = await self.get_workflow_handle(workflow_id, run_id=run_id)
        if disable_backoff:
            return await wf_handle.result()

        # Initial polling on workflow result.
        long_poll_timeout = 10.0
        try:
            return await asyncio.wait_for(wf_handle.result(), long_poll_timeout)
        except TimeoutError:
            log.debug(
                'Workflow %s has no result after %s seconds, start polling '
                'workflow status with exponential backoff.', workflow_id,
                long_poll_timeout)

        # Start randomized exponential backoff.
        backoff_delay_initial = 1.0
        backoff_delay_max = 30.0
        backoff_delay = backoff_delay_initial
        while True:
            sleep_seconds = backoff_delay
            if backoff_delay < backoff_delay_max:
                # Randomize backoff within 0.5x~1.5x, but not exceeding max
                sleep_seconds = min(backoff_delay_max,
                                    backoff_delay * random.uniform(0.5, 1.5))
                # Exponential backoff with 2.0 factor
                backoff_delay = min(backoff_delay * 2.0, backoff_delay_max)

            await asyncio.sleep(sleep_seconds)

            # Short-poll of workflow status (not result)
            try:
                wf_status = (await wf_handle.describe()).status
            except RPCError as rpc_err:
                log.warning('Workflow status polling failed for %s, err: %s',
                            workflow_id, rpc_err)
                continue  # Ignore transient rpc error.

            if wf_status and wf_status != WorkflowExecutionStatus.RUNNING:
                # Workflow is no longer running!
                log.debug('Workflow %s is no longer running, result=%s',
                          workflow_id, wf_status)
                return await wf_handle.result()

    async def _namespace_exists(self) -> bool:
        namespace = self._config.temporal_namespace
        temporal_service = self._get_workflow_service()
        try:
            resp = await temporal_service.describe_namespace(
                wsv1.DescribeNamespaceRequest(namespace=namespace))
            log.debug('Found temporal namespace: %s', resp)
            assert resp.namespace_info.name == namespace
            return True
        except RPCError as err:
            if err.status == RPCStatusCode.NOT_FOUND:
                log.warning('Temporal namespace: %s not found.', namespace)
                return False
            raise err

    def _get_workflow_service(self) -> TemporalService:
        client = self._client_for_loop(asyncio.get_running_loop())
        assert client
        return client.workflow_service

    async def create_namespace(self) -> None:
        """
        Create namespace if not exists, only applicable when not using
        Temporal Cloud.
        """
        if isinstance(self._config.orchestration, TemporalCloudConfig):
            log.warning('Create namespace is not needed for Temporal Cloud.')
            return

        await self.init()
        if await self._namespace_exists():
            return

        namespace = self._config.temporal_namespace
        log.warning('Register new namespace=%s with Temporal Cluster.',
                    namespace)
        try:
            await self._get_workflow_service().register_namespace(
                wsv1.RegisterNamespaceRequest(
                    namespace=namespace,
                    description=f'Created by {self._config.client_id}',
                    workflow_execution_retention_period=Duration(
                        seconds=int(timedelta(days=3).total_seconds())),
                ))
        except RPCError as rpc_error:
            if rpc_error.status == RPCStatusCode.ALREADY_EXISTS:
                log.warning(
                    'Namespace=%s already exists, wait for it to take effect',
                    namespace)
            else:
                raise
        # There could be a delay, we need to verify by polling every 2sec.
        while True:
            log.info('Waiting for namespace=%s to be created...', namespace)
            await asyncio.sleep(_REGISTER_NAMESPACE_POLL_INTERVAL_SEC)
            if await self._namespace_exists():
                break
        log.info('New namespace=%s created, waiting for it to take effects...',
                 namespace)
        await asyncio.sleep(_NEW_NAMESPACE_WAIT_SEC)
        assert await self._namespace_exists()

    def _create_client(self, service_client: ServiceClient) -> Client:
        namespace = self._config.temporal_namespace

        tmp_payloads_dir = _tmp_payloads_dir(self._config)

        return Client(
            service_client=service_client,
            namespace=namespace,
            interceptors=[
                TracingInterceptor(),
                ClientInterceptor(self._config),
                DeployConfigMemoInterceptor(),
            ],
            data_converter=DataConverter(
                payload_converter_class=KumoPayloadConverter,
                payload_codec=CompressionCodec(tmp_payloads_dir)),
        )


def _tmp_payloads_dir(config: DeployEnvConfig) -> Optional[str]:
    """Return/create the offload prefix only with explicit lifecycle proof."""
    if not config.storage.lifecycle_ready:
        return None
    path = os.path.join(config.data_plane_storage_root, 'tmp_payloads')
    fsutil.makedirs(path, exist_ok=True)
    return path


def add_search_attributes_to_input(
    input: StartWorkflowInput,
    config: DeployEnvConfig,
) -> StartWorkflowInput:
    if not isinstance(config.orchestration, TemporalCloudConfig):
        # Custom search attributes are supported only on Temporal Cloud.
        input.search_attributes = None
    else:
        search_attributes: Dict[str, SearchAttributeValues] = {
            'instance_id': [config.instance_id],
        }
        if input.search_attributes:
            assert not isinstance(input.search_attributes,
                                  TypedSearchAttributes)
            search_attributes.update(input.search_attributes)
        input.search_attributes = search_attributes

    return input


class ClientOutboundInterceptor(OutboundInterceptor):
    """
    Client interceptor to append appropriate suffixes to task queue and
    workflow id, based on the config.
    """
    def __init__(self, config: DeployEnvConfig,
                 next: OutboundInterceptor) -> None:
        super().__init__(next)
        self._config = config

    async def start_workflow(self, input: StartWorkflowInput):
        id_suffix = self._config.workflow_id(workflow_id='')
        memo: Dict[str, Any] = {ID_SUFFIX_MEMO_FIELD: id_suffix.encode()}
        if input.memo:
            memo.update(input.memo)
        input.memo = memo

        input.task_queue = self._config.task_queue(input.task_queue)
        input.id = self._config.workflow_id(input.id)

        # Add search attributes so we can find it in the Temporal UI
        input = add_search_attributes_to_input(input, self._config)

        # Set timeout/deadline for a single gRPC call to Temporal service.
        if input.rpc_timeout is None:
            input.rpc_timeout = _TEMPORAL_RPC_TIMEOUT

        # Set default workflow task timeout if not specified already.
        if input.task_timeout is None:
            input.task_timeout = timedelta(
                seconds=DEFAULT_WORKFLOW_TASK_TIMEOUT_SEC)
        # Import locally to preserve the temporalio_commons -> rest layering at
        # module import time. Binding runs in the trusted client process before
        # any capability-bearing payload is accepted by Temporal.
        from rest.backends.data_backend.connector_authorization_client import (
            bind_connector_authorizations_for_start,
            close_connector_authorization_state,
            revoke_bound_connector_authorizations,
        )
        state_db, _services, _newly_bound = (
            await bind_connector_authorizations_for_start(input, self._config))
        try:
            return await self.next.start_workflow(input)
        except WorkflowAlreadyStartedError:
            try:
                await revoke_bound_connector_authorizations(
                    state_db, _services, _newly_bound)
            except BaseException:
                log.exception(
                    'Failed best-effort cleanup of duplicate-start capability')
            state_db = None
            raise
        finally:
            # A start exception can be a transport ambiguity after Temporal
            # accepted the workflow. Keep workflow-bound capabilities alive;
            # unused secrets are forgotten locally and expire by their TTL.
            await close_connector_authorization_state(state_db)

    async def cancel_workflow(self, input) -> None:
        input.id = self._config.workflow_id(input.id)
        if input.rpc_timeout is None:
            input.rpc_timeout = _TEMPORAL_RPC_TIMEOUT
        await self.next.cancel_workflow(input)
        # yapf disagrees with isort on wrapping a single-name parenthesized
        # import; isort wins here.
        # yapf: disable
        from rest.backends.data_backend.connector_authorization_client import (
            revoke_connector_authorizations_for_workflow,
        )

        # yapf: enable
        await revoke_connector_authorizations_for_workflow(
            self._config, input.id)

    async def describe_workflow(self, input):
        input.id = self._config.workflow_id(input.id)
        if input.rpc_timeout is None:
            input.rpc_timeout = _TEMPORAL_RPC_TIMEOUT
        return await self.next.describe_workflow(input)

    async def query_workflow(self, input):
        input.id = self._config.workflow_id(input.id)
        if input.rpc_timeout is None:
            # Unlike other client operations, "query" will not be retried by
            # Temporal client upon gRPC error such as timeout; therefore we
            # need to relax the rpc timeout.  Also see #8338.
            input.rpc_timeout = _TEMPORAL_RPC_TOTAL_TIMEOUT
        return await self.next.query_workflow(input)

    async def signal_workflow(self, input):
        input.id = self._config.workflow_id(input.id)
        if input.rpc_timeout is None:
            input.rpc_timeout = _TEMPORAL_RPC_TIMEOUT
        await self.next.signal_workflow(input)

    async def terminate_workflow(self, input):
        input.id = self._config.workflow_id(input.id)
        if input.rpc_timeout is None:
            input.rpc_timeout = _TEMPORAL_RPC_TIMEOUT
        await self.next.terminate_workflow(input)
        # yapf disagrees with isort on wrapping a single-name parenthesized
        # import; isort wins here.
        # yapf: disable
        from rest.backends.data_backend.connector_authorization_client import (
            revoke_connector_authorizations_for_workflow,
        )

        # yapf: enable
        await revoke_connector_authorizations_for_workflow(
            self._config, input.id)


class ClientInterceptor(Interceptor):  # pylint: disable=too-few-public-methods
    def __init__(self, config: DeployEnvConfig) -> None:
        self._config = config

    def intercept_client(self,
                         next: OutboundInterceptor) -> OutboundInterceptor:
        """See base."""
        return ClientOutboundInterceptor(self._config, next)


class DeployConfigMemoOutboundInterceptor(OutboundInterceptor):
    """Attaches deploy_env() snapshot as workflow memo for observability."""
    async def start_workflow(self, input: StartWorkflowInput):
        memo = dict(input.memo or {})
        memo['deploy_env_config'] = dataclasses.asdict(deploy_env())
        input.memo = memo
        return await self.next.start_workflow(input)


class DeployConfigMemoInterceptor(Interceptor):  # pylint: disable=too-few-public-methods
    def intercept_client(self,
                         next: OutboundInterceptor) -> OutboundInterceptor:
        return DeployConfigMemoOutboundInterceptor(next)
