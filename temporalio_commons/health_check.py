# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Script to run health check of temporal worker. Usage example:

python temporalio_commons/health_check.py --worker data-backend-worker

By default, this will parse DeployEnvConfig using env variables. Alternatively
the following args can be optionally specified to override DeployEnvConfig:
python temporalio_commons/health_check.py \
    --worker data-backend-worker \
    -d dev -i i-0c36d1f8c668db02 -c test

By default, this health check will detect "stuck" activity executor that has
been unresponsive for more than 2 minutes. This threshold can be specified as:
python temporalio_commons/health_check.py \
    --worker data-backend-worker \
    --unresponsive_seconds 60

Use '-h' to see documentation on all required and optional arguments.
"""
import argparse
import asyncio
import dataclasses
import time
from datetime import timedelta
from typing import List, Optional

import requests
from temporalio.client import WorkflowFailureError
from temporalio.common import RetryPolicy
from temporalio.service import RPCError, RPCStatusCode

from io_utils import logging
from service_framework.deploy_env import _SLACK_URL, DeployStage
from service_framework.deploy_env_config import DeployEnvConfig, deploy_env
from service_framework.env_var import EnvVar
from temporalio_commons.temporal_client import TemporalClient
from temporalio_commons.worker_registry import WorkerInfoWorkflow

log = logging.getLogger(__name__)

# Map from worker name to task queues hosted on each worker.
WORKER_TASK_QUEUES = {
    'data-backend-worker': ['data-backend-task-queue'],
    'ml-workflow-worker': ['ml-workflow-task-queue'],
    'pool-controller': ['PoolController-task-queue'],
    'pool-worker': ['diskgraph-daemon-tq'],

    # Test-only
    'test-worker': 'test-healthcheck-task-queue',
}

_DEFAULT_UNRESPONSIVE_SECONDS = 120
_DEFAULT_TIMEOUT_SECONDS = 10

unresponsive_seconds_var = EnvVar('HEALTH_CHECK_UNRESPONSIVE_SECONDS', type=int,
                                  required=False,
                                  default_value=_DEFAULT_UNRESPONSIVE_SECONDS)

timeout_seconds_var = EnvVar('HEALTH_CHECK_TIMEOUT_SECONDS', type=int,
                             required=False,
                             default_value=_DEFAULT_TIMEOUT_SECONDS)


def send_slack_message(message: str, url: Optional[str] = None):
    """
    This function makes a blocking call to post a slack message for notifying
    failures, be careful with using this in async functions.
    """
    url = url or _SLACK_URL.parse_value()
    try:
        if url:
            data = {"text": message, "link_names": True}
            headers = {'Content-type': 'application/json'}
            requests.post(url, json=data, headers=headers)
    except Exception:
        log.exception("Failed to send slack message.")


async def health_check(
        deploy_config: DeployEnvConfig,
        task_queues: List[str],
        *,
        unresponsive_threshold: timedelta = timedelta(minutes=5),
        timeout: timedelta = timedelta(seconds=10.0),
):
    temporal_client = TemporalClient(deploy_config)
    try:
        client = await temporal_client.get()
    except TimeoutError:  # raised from Temporal client/connection creation
        # skip health check
        log.error('Connecting to Temporal cloud timed out, skip health check')
        return
    except BaseException as ex:
        log.error(
            'Connecting to Temporal cloud failed due to %s("%s"), '
            'skip health check',
            type(ex).__name__, ex)
        return

    for task_queue in task_queues:
        wf_id = f'health-check-{task_queue}@{int(time.time())}'
        log.debug('Start health check workflow in %s for task queue: %s',
                  deploy_config.deploy_mode,
                  deploy_config.task_queue(task_queue))
        try:
            worker_info = await client.execute_workflow(
                WorkerInfoWorkflow.run,
                id=wf_id,
                task_queue=task_queue,
                execution_timeout=timeout,
                retry_policy=RetryPolicy(maximum_attempts=1),  # no retry
            )
        except RPCError as rpc_error:  # gRPC error from Temopral service
            log.exception('Encountered grpc error when connecting to Temporal!',
                          stack_info=True)
            if rpc_error.status in [
                    RPCStatusCode.DEADLINE_EXCEEDED,
                    RPCStatusCode.INTERNAL,
                    RPCStatusCode.UNKNOWN,
                    RPCStatusCode.RESOURCE_EXHAUSTED,
            ]:
                log.warning(
                    'Skip health check upon transient Temporal gRPC error.')
            elif deploy_config.deploy_mode == DeployStage.PROD:
                send_slack_message(
                    (f"[WARNING] Health check for {task_queue} encountered gRPC "
                     f"error from Temporal service: {rpc_error}, "
                     f"status code: {rpc_error.status.name}"),
                    _SLACK_URL.parse_value(),
                )
            return
        except WorkflowFailureError as ex:
            log.exception('Health check failed for %s in env: %s due to: %s',
                          task_queue, deploy_config, ex, stack_info=True)
            if deploy_config.deploy_mode == DeployStage.PROD:
                send_slack_message(
                    (f"[WARNING] Health check for {task_queue} FAILED due to "
                     f"{ex.cause}, url: "
                     f"{temporal_client.format_workflow_web_url(wf_id)}"),
                    _SLACK_URL.parse_value(),
                )
            raise

        log.info('Worker info on task queue %s: %s', task_queue, worker_info)
        if worker_info.is_stuck(unresponsive_threshold):
            if deploy_config.deploy_mode == DeployStage.PROD:
                send_slack_message(
                    (f"[WARNING] Health check for {task_queue} FAILED because "
                     f"worker activity executor is unresponsive for more than "
                     f"{unresponsive_threshold.total_seconds():.2f} seconds, "
                     f"worker_info: {worker_info}"),
                    _SLACK_URL.parse_value(),
                )
            raise RuntimeError(
                f'Health check failed, worker {worker_info.worker_name} '
                f'is stuck!  Worker info: {worker_info}')

    log.info('Health check passed for %s', task_queues)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--worker', required=False, default=None, type=str,
        help=('Worker name(s) to perform health check. '
              'Comma separated names if checking multiple workers.'))

    # Below are optional args to override DeployEnvConfig options that are
    # parsed from env variables by default.
    parser.add_argument(
        '-d',
        '--deploy_stage',
        dest='deploy_stage',
        choices=[stage.value for stage in DeployStage],
        required=False,
        default=None,
    )
    parser.add_argument('-i', '--instance_id', dest='instance_id', default=None,
                        required=False,
                        help='Instance ID (only useful for test/dev/preprod).')

    parser.add_argument(
        '--unresponsive_seconds', default=None, required=False, type=int,
        help=('Activity executor unresponsive time threshold in seconds, i.e, '
              'the health check will fail if the activity executor has become '
              'unresponsive longer than this threshold value.'))

    parser.add_argument('--timeout', default=None, required=False, type=int,
                        help=('Health check timeout in seconds.'))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Parse DeployEnvConfig and optionally override by args.
    deploy_config = deploy_env()
    overrides = {}
    if args.deploy_stage is not None:
        overrides['deploy_mode'] = DeployStage(args.deploy_stage)
    if args.instance_id is not None:
        overrides['instance_id'] = args.instance_id
    if overrides:
        deploy_config = dataclasses.replace(deploy_config, **overrides)

    logging.initialize_logging(
        logging.LoggingMode.PRODUCTION if deploy_config.deploy_mode
        == DeployStage.PROD else logging.LoggingMode.DEBUG, log_to_file=False)

    log.debug(f'Performing health check in deploy env: {deploy_config}')

    if args.worker:
        if args.worker == 'ALL':
            worker_names = list(WORKER_TASK_QUEUES.keys())
        else:
            worker_names = args.worker.split(',')
    else:
        worker_names = EnvVar('WORKERS', str, required=False,
                              default_value='').parse_value().split(',')

    unresponsive_seconds = unresponsive_seconds_var.parse_value()
    if args.unresponsive_seconds:
        unresponsive_seconds = args.unresponsive_seconds

    timeout_seconds = timeout_seconds_var.parse_value()
    if args.timeout:
        timeout_seconds = args.timeout

    for worker_name in worker_names:
        task_queues = WORKER_TASK_QUEUES[worker_name]
        log.debug(
            f'Checking worker {worker_name} on task queues: {task_queues}')
        asyncio.run(
            health_check(
                deploy_config,
                task_queues,
                unresponsive_threshold=timedelta(seconds=unresponsive_seconds),
                timeout=timedelta(seconds=timeout_seconds),
            ))
