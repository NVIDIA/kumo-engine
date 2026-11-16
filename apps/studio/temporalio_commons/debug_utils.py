# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Utilities related to debug and troubleshooting Temporal workflow.

import asyncio
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

import temporalio.api.workflowservice.v1 as wsv1
from temporalio.api.enums.v1 import EventType
from temporalio.api.enums.v1.workflow_pb2 import TimeoutType
from temporalio.client import Client, WorkflowExecutionStatus
from temporalio.service import RPCError, RPCStatusCode

from io_utils import kumo_dataclass, logging
from rest.constants import DATA_BACKEND_WORKER_NAME, ML_WORKFLOW_WORKER
from service_framework.deploy_env import DeployStage
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_env_config import deploy_env as get_deploy_env
from temporalio_commons.temporal_client import TemporalClient
from temporalio_commons.worker_main import build_workers
from temporalio_commons.worker_registry import (
    GLOBAL_WORKER_REGISTRY,
    RunSingleActivityWorkflow,
    WorkerInfo,
    WorkerInfoWorkflow,
)

log = logging.getLogger(__name__)

WORKER_NAME_TO_CONTAINER = {
    ML_WORKFLOW_WORKER: 'kumo-temporalworker-mlworkflow',
    DATA_BACKEND_WORKER_NAME: 'kumo-temporalworker-dataworkflow',
}


@kumo_dataclass
class DecodeFailure:
    message: str


DecodedData = Union[List[Any], DecodeFailure]


# TODO(siyang): retry is not delicately handled yet. Upon retries, this will
# record information from the last finished attempt.
@kumo_dataclass
class ActivityDebugInfo:
    """
    Debug information about an activity execution, as derived from workflow
    execution event history.
    """
    activity_id: str
    activity_type: str
    scheduled_at: datetime
    task_queue: str
    inputs: DecodedData

    # All following fields are absent if the activity is still scheduled or
    # actively running.
    started_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    worker_info: Optional[WorkerInfo] = None
    final_status: Optional[str] = None
    outputs: Optional[DecodedData] = None
    failure: Optional[str] = None

    @property
    def range_query_from(self) -> str:
        return str(int(self.scheduled_at.timestamp() * 1000))

    @property
    def range_query_to(self) -> str:
        end_time = (self.closed_at + timedelta(minutes=1) if self.closed_at else
                    self.scheduled_at + timedelta(days=1))
        end_time = min(end_time, datetime.now(timezone.utc))
        return str(int(end_time.timestamp() * 1000))

    @property
    def schedule_delay(self) -> Optional[timedelta]:
        return self.started_at - self.scheduled_at if self.started_at else None

    @property
    def worker_container(self) -> Optional[str]:
        if self.worker_info:
            return WORKER_NAME_TO_CONTAINER.get(self.worker_info.worker_name)
        return None

    @property
    def execution_duration(self) -> Optional[timedelta]:
        if self.started_at and self.closed_at:
            return self.closed_at - self.started_at
        return None

    @property
    def summary(self) -> dict:
        timing_dict = {'scheduled': self.scheduled_at.astimezone().isoformat()}
        if self.started_at:
            timing_dict['started'] = self.started_at.astimezone().isoformat()
            if (self.schedule_delay
                    and self.schedule_delay > timedelta(seconds=1)):
                timing_dict['schedule_delay'] = str(self.schedule_delay)
            if self.closed_at:
                timing_dict['closed'] = self.closed_at.astimezone().isoformat()
                if (self.execution_duration
                        and self.execution_duration > timedelta(seconds=1)):
                    timing_dict['exec_duration'] = str(self.execution_duration)

        worker_summary = {}
        if self.worker_info:
            worker_summary = {
                'worker_name':
                self.worker_info.worker_name,
                'instance_id':
                self.worker_info.instance_id,
                'container':
                WORKER_NAME_TO_CONTAINER.get(self.worker_info.worker_name),
                'started_at':
                self.worker_info.start_time.isoformat(),
                'kumo_version':
                self.worker_info.kumo_version,
            }

        return {
            'activity_type': self.activity_type,
            'task_queue': self.task_queue,
            'worker': worker_summary,
            'timing (local timezone)': timing_dict,
            'final status': self.final_status
        }

    def get_log_url(self, deploy_config: DeployEnvConfig) -> str:
        if not self.worker_info:
            return 'Log url unknown: no worker id for this activity.'

        worker_name = self.worker_info.worker_name
        worker_container = WORKER_NAME_TO_CONTAINER[worker_name]
        instance_id = self.worker_info.instance_id

        if deploy_config.deploy_mode.is_dev:
            return f'View logs in {worker_container} panel on {instance_id}'

        if deploy_config.deploy_mode == DeployStage.PREPROD:
            log_command = (
                f'aws ssm start-session --target {instance_id} --region '
                'us-west-2 --document-name AWS-StartInteractiveCommand '
                '--parameters \'{"command": ["'
                f'sudo docker logs {worker_container}'
                '"]}\'')
            return ('Loki logs is not available on PREPROD. To view the docker '
                    'container logs for the temporal worker, run the '
                    f'following command:\n{log_command}')

        return 'Log URL unavailable for this deployment.'

    # Re-execute this activity locally with the simulated deploy env config.
    async def rerun_local(self, deploy_env_config: DeployEnvConfig):
        assert self.worker_info
        assert isinstance(self.inputs, list)
        worker_names = [self.worker_info.worker_name]
        _, worker_envs = await build_workers(deploy_env_config, worker_names)
        assert len(worker_envs) == 1
        worker_env = worker_envs[0]
        # Switch the task queue to a separate, unique task queue name to
        # ensure the workflow/activity task is private to this debug session.
        debug_task_queue = self.task_queue + f'-debug-{uuid.uuid4().hex}'
        worker_env.options.task_queue = debug_task_queue
        workers = await GLOBAL_WORKER_REGISTRY.build_worker(worker_env)
        assert len(workers) == 1

        # Spin up the local worker(s) as async task(s).
        for worker in workers:
            # TODO(frg100): Save a reference to the result of this function
            # https://textual.textualize.io/blog/2023/02/11/
            # the-heisenbug-lurking-in-your-async-code/
            asyncio.create_task(worker.run())

        # Execute the activity by submitting a single-activity workflow.
        temporal_client = TemporalClient(deploy_env_config)
        debug_wf_id = f'debug-{self.activity_type}-{uuid.uuid4().hex}'
        client = await temporal_client.get()
        wf_handle = await client.start_workflow(
            RunSingleActivityWorkflow.run,
            args=[self.activity_type, self.inputs],
            id=debug_wf_id,
            task_queue=debug_task_queue,
        )
        debug_wf_url = temporal_client.format_workflow_web_url(debug_wf_id)
        log.info(
            'Started debug workflow for %s in:\n%s\n'
            'Please use the url to terminate the workflow if not needed.',
            self.activity_type, debug_wf_url)
        result = await wf_handle.result()

        # Shutdown the worker and associated executor.
        for worker in workers:
            await worker.shutdown()
        worker_env.shutdown()

        return result


@kumo_dataclass
class WorkflowDebugInfo:
    status: WorkflowExecutionStatus
    task_queue: str
    web_url: str
    inputs: DecodedData
    activities: List[
        ActivityDebugInfo]  # ordered by activity schedule timestamp
    # Present if the workflow failed due to a criticial failure in an activity
    critical_activity: Optional[ActivityDebugInfo] = None


class WorkflowNotFoundException(Exception):
    pass


async def _safe_decode(client: Client, payloads) -> DecodedData:
    try:
        return await client.data_converter.decode(payloads)
    except Exception as ex:  # pylint: disable=broad-except
        log.exception('Failed to decode payloads due to ex: %s', ex)
        return DecodeFailure(traceback.format_exc())


# Produce detailed debug information about a Temporal workflow given its id and
# deploy environment by querying and examining the full workflow event
# history stored in the Temporal cluster.
#
# WorkflowNotFoundException will be raised if the workflow is not found, either
# because the wf_id or deploy_env is incorrect, or the Temporal Cluster has
# garbage collected the workflow event history past the retention period. As
# of now, Temporal Cloud has 30d retention for all namespaces while the local
# Temporal cluster only has 3d retention.
async def debug_temporal_workflow(
        wf_id: str,
        deploy_env: Optional[DeployEnvConfig] = None) -> WorkflowDebugInfo:
    deploy_env = deploy_env or get_deploy_env()
    temporal_client = TemporalClient(deploy_env)
    client = await temporal_client.get()
    wf_handle = await temporal_client.get_workflow_handle(wf_id)

    try:
        wf_desc = await wf_handle.describe()
    except RPCError as rpc_error:
        if rpc_error.status == RPCStatusCode.NOT_FOUND:
            log.error(
                'Workflow is not found, please make sure workflow id and '
                'deploy_env is correctly specified and the workflow is not '
                'older than the retention period. Please make sure the '
                'workflow %s can be found in Temporal Web UI: %s', wf_id,
                deploy_env.web_url)
            raise WorkflowNotFoundException() from rpc_error
        raise

    wf_exec = wf_desc.raw_description.workflow_execution_info.execution
    assert wf_desc.status

    ws_client = client.workflow_service
    wf_exec_history = await ws_client.get_workflow_execution_history(
        wsv1.GetWorkflowExecutionHistoryRequest(
            namespace=deploy_env.temporal_namespace, execution=wf_exec))

    events = wf_exec_history.history.events
    event_dict = {event.event_id: event for event in events}
    all_activities: Dict[str, ActivityDebugInfo] = {}

    wf_start_event = events[0]
    assert (wf_start_event.event_type ==
            EventType.EVENT_TYPE_WORKFLOW_EXECUTION_STARTED)
    event_attrs = wf_start_event.workflow_execution_started_event_attributes
    task_queue = event_attrs.task_queue.name
    encoded_inputs = event_attrs.input.payloads
    inputs = await _safe_decode(client, encoded_inputs)

    log.debug('Workflow complete event history:\n%s', wf_exec_history)

    def _get_activity_info(attrs):
        schedule_event_id = attrs.scheduled_event_id
        schedule_event = event_dict[schedule_event_id]
        attrs = schedule_event.activity_task_scheduled_event_attributes
        return all_activities[attrs.activity_id]

    for event in events:
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            attrs = event.activity_task_scheduled_event_attributes
            activity_info = ActivityDebugInfo(
                activity_id=attrs.activity_id,
                activity_type=attrs.activity_type.name,
                scheduled_at=event.event_time.ToDatetime(timezone.utc),
                task_queue=attrs.task_queue.name,
                inputs=await _safe_decode(client, attrs.input.payloads),
            )
            all_activities[activity_info.activity_id] = activity_info
            continue

        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_STARTED:
            attrs = event.activity_task_started_event_attributes
            activity_info = _get_activity_info(attrs)
            activity_info.started_at = event.event_time.ToDatetime(timezone.utc)
            worker_identity = attrs.identity
            if ':' in worker_identity:
                worker_name, instance_id = worker_identity.split(':', 1)
                worker_info = await client.execute_workflow(
                    WorkerInfoWorkflow.run,
                    id=f'get-worker-info-{worker_identity}',
                    task_queue=activity_info.task_queue,
                )
                if worker_info.instance_id != instance_id:
                    log.warning(
                        'Worker %s was hosted on %s when executing %s, '
                        'but is now running on %s', worker_name, instance_id,
                        activity_info.activity_type, worker_info.instance_id)
            else:
                worker_name = worker_identity
                worker_info = WorkerInfo(
                    worker_name, 'n/a',
                    start_time=datetime.fromtimestamp(0, timezone.utc),
                    workflows=[], activities=[])
            activity_info.worker_info = worker_info

        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED:
            attrs = event.activity_task_completed_event_attributes
            activity_info = _get_activity_info(attrs)
            activity_info.final_status = 'COMPLETED'
            activity_info.closed_at = event.event_time.ToDatetime(timezone.utc)
            activity_info.outputs = await client.data_converter.decode(
                attrs.result.payloads)
            continue

        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_CANCELED:
            attrs = event.activity_task_canceled_event_attributes
            activity_info = _get_activity_info(attrs)
            activity_info.final_status = 'CANCELED'
            activity_info.closed_at = event.event_time.ToDatetime(timezone.utc)
            activity_info.failure = f'Cancelled, details: {attrs.details}'
            continue

        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_TIMED_OUT:
            attrs = event.activity_task_timed_out_event_attributes
            activity_info = _get_activity_info(attrs)
            activity_info.final_status = TimeoutType.Name(
                attrs.failure.timeout_failure_info.timeout_type)
            activity_info.closed_at = event.event_time.ToDatetime(timezone.utc)
            activity_info.failure = str(attrs.failure)
            continue

        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_FAILED:
            attrs = event.activity_task_failed_event_attributes
            activity_info = _get_activity_info(attrs)
            activity_info.final_status = 'FAILED'
            activity_info.closed_at = event.event_time.ToDatetime(timezone.utc)
            activity_info.failure = str(attrs.failure)
            continue

    critical_activity_id = _infer_critical_activity_id(wf_desc, events[-1])
    return WorkflowDebugInfo(
        wf_desc.status,
        task_queue,
        temporal_client.format_workflow_web_url(wf_id),
        inputs,
        list(all_activities.values()),
        all_activities[critical_activity_id] if critical_activity_id else None,
    )


# Get the id of activity that caused workflow execution failure.
def _infer_critical_activity_id(wf_desc, last_wf_event) -> Optional[str]:
    if (wf_desc.status == WorkflowExecutionStatus.FAILED
            and last_wf_event.event_type
            == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_FAILED):
        attrs = last_wf_event.workflow_execution_failed_event_attributes
        failure = attrs.failure
        if failure.HasField('activity_failure_info'):
            log.info('Workflow failed due to activity: %s',
                     failure.activity_failure_info.activity_type)
            return failure.activity_failure_info.activity_id
    return None
